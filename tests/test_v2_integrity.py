"""Regression checks for incomplete-cohort and seed publication failures."""

import numpy as np
import pandas as pd
import pytest

from cityshift.analysis_v2 import S_FAMILIES, arm_column, q_open, r_derived, s_open
from cityshift.data import CITIES
from cityshift.study_q import validate_q_parts
from cityshift.v2_validation import agent_cohort, matched_scenarios, scenario_cohort


def agents(predictors):
    rows = []
    for c in CITIES:
        for i in range(3):
            for p in predictors:
                rows.append({"scenario_id": f"{c}-{i}", "agent_id": "agent", "city": c, "focal": i == 0,
                             "scored": False, "planner_relevant": True, "n_valid": 60, "speed": 0.0 if i < 2 else 3.0,
                             "predictor": p, "miss": bool(i == 1), "min_fde": 1.25, "min_ade": 0.72,
                             "brier_min_fde": 1.87, "moving_mode_mass": 0.1})
    return pd.DataFrame(rows)


def test_missing_seed_cannot_be_averaged_as_a_complete_s_study():
    rows = agents([f"{f}_s{s}" for f in S_FAMILIES for s in range(3)] + ["CV"])
    with pytest.raises(ValueError, match="three seeds"):
        s_open(rows[rows.predictor != "SEL_s2"], n_boot=10)
    with pytest.raises(ValueError, match="cohorts differ"):
        agent_cohort(rows.iloc[1:], sorted(rows.predictor.unique()))


def test_s_closed_pairing_cannot_silently_intersect_scenarios():
    rows = pd.DataFrame({"scenario_id": CITIES, "city": CITIES})
    changed = rows.copy()
    changed.loc[0, "scenario_id"] = "different"
    with pytest.raises(ValueError, match="incomplete"):
        matched_scenarios(rows, changed)
    changed = rows.copy()
    changed.loc[0, "city"] = CITIES[1]
    with pytest.raises(ValueError):
        matched_scenarios(rows, changed)


def test_partial_registered_scenario_cohort_is_rejected():
    rows = pd.DataFrame({"scenario_id": CITIES, "city": CITIES})
    with pytest.raises(ValueError, match="incomplete"):
        scenario_cohort(rows, list(CITIES) + ["missing"])


def test_partial_seeded_closed_loop_or_nan_outcome_is_rejected():
    with pytest.raises(ValueError, match="three seeds"):
        arm_column(pd.DataFrame({"ALL_s0_collision": [False]}), "ALL", "collision")
    with pytest.raises(ValueError, match="non-finite"):
        arm_column(pd.DataFrame({"ALL_collision": [np.nan]}), "ALL", "collision")


def test_zero_patch_effect_has_an_undefined_ratio_instead_of_crashing():
    with np.errstate(divide="ignore", invalid="ignore"):
        out = r_derived({"PATCH_reduction": 0.0, "PROB_reduction": 0.0, "GEO_reduction": 0.0})
    assert np.isnan(out["G_ratio_PROB_over_PATCH"])


def test_q_cannot_silently_drop_a_registered_agent():
    base = agents(["CV", "ALL_s0", "ALL_s1", "ALL_s2"])
    q = agents(["QCNET"]).iloc[:-1]
    with pytest.raises(ValueError, match="cohort differs"):
        q_open(q, base, n_boot=10)


def test_q_resume_cannot_skip_a_wrong_or_incomplete_chunk(tmp_path):
    rows = pd.DataFrame({"scenario_id": ["unexpected"]})
    rows.to_parquet(tmp_path / "part_00000.parquet", index=False)
    with pytest.raises(ValueError, match="membership"):
        validate_q_parts(str(tmp_path), ["expected"], 1, "closed")
    rows["scenario_id"] = "expected"
    rows.to_parquet(tmp_path / "part_00000.parquet", index=False)
    with pytest.raises(ValueError, match="invalid closed-loop"):
        validate_q_parts(str(tmp_path), ["expected"], 1, "closed")
