"""Cohort and seed checks for the registered v2 analyses."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .data import CITIES

AGENT_KEYS = ["scenario_id", "agent_id"]
AGENT_META = ["city", "focal", "scored", "planner_relevant", "n_valid", "speed"]
REGISTERED_COHORTS = {
    "reserve": (10000, "286a9423ffd907eebf7d67a6c7f66b2edfc6d1c4309e2a7f03e2633f8aac52ed"),
    "q_closed": (6000, "b35794e0ed5c14b08f5420e800bbc29ec7addfeea5fe6a349059b73d1b495084"),
}


def registered_cohort(rows: pd.DataFrame, name: str) -> None:
    ids = sorted(rows.scenario_id.astype(str).unique())
    n, digest = REGISTERED_COHORTS[name]
    if len(ids) != n or hashlib.sha256("\n".join(ids).encode()).hexdigest() != digest:
        raise ValueError(f"incomplete or unexpected registered {name} cohort")


def scenario_cohort(rows: pd.DataFrame, expected: list[str] | None = None) -> None:
    if rows.empty or not {"scenario_id", "city"} <= set(rows):
        raise ValueError("empty or missing scenario cohort")
    if rows[["scenario_id", "city"]].isna().any().any() or rows.scenario_id.duplicated().any():
        raise ValueError("missing or duplicate scenario identity")
    if set(rows.city) != set(CITIES):
        raise ValueError("scenario cohort must contain all six registered cities")
    if expected is not None and set(rows.scenario_id.astype(str)) != set(expected):
        raise ValueError("incomplete or unexpected registered scenario cohort")


def matched_scenarios(reference: pd.DataFrame, other: pd.DataFrame) -> None:
    scenario_cohort(reference)
    scenario_cohort(other, reference.scenario_id.astype(str).tolist())
    a = reference.set_index("scenario_id").city.sort_index()
    b = other.set_index("scenario_id").city.sort_index()
    if not a.equals(b):
        raise ValueError("inconsistent city metadata between paired arms")


def agent_cohort(rows: pd.DataFrame, predictors: list[str], expected_ids: list[str] | None = None) -> None:
    required = set(AGENT_KEYS + AGENT_META + ["predictor", "miss"])
    if rows.empty or not required <= set(rows):
        raise ValueError("missing agent rows or metadata")
    if rows[list(required)].isna().any().any():
        raise ValueError("missing agent values")
    if set(rows.predictor) != set(predictors):
        raise ValueError("all registered predictors and three seeds per family are required")
    if rows.duplicated(AGENT_KEYS + ["predictor"]).any():
        raise ValueError("duplicate agent/predictor rows")
    if not np.isfinite(rows[["miss", "speed", "n_valid"]].to_numpy(float)).all():
        raise ValueError("non-finite agent metrics")
    if not rows.miss.isin([0, 1]).all():
        raise ValueError("miss outcomes must be binary")
    groups = rows.groupby(AGENT_KEYS, sort=False)
    if not (groups.size() == len(predictors)).all():
        raise ValueError("agent cohorts differ between predictors or seeds")
    if (groups[AGENT_META].nunique(dropna=False) != 1).any().any():
        raise ValueError("inconsistent per-agent metadata between predictors")
    if expected_ids is not None and set(rows.scenario_id.astype(str)) != set(expected_ids):
        raise ValueError("incomplete registered agent scenario cohort")
