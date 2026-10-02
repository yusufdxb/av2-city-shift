"""Synthetic checks of the v2 study R/G and Q analyses: effects, the G ratio, gates and verdict wiring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cityshift.analysis_stage3 import CITIES
from cityshift.analysis_v2 import R_ARMS, analyze_r, q_closed


def r_rows(rates: dict[str, float], n_per_city: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for c in CITIES:
        for k in range(n_per_city):
            row = {"scenario_id": f"{c}-{k}", "city": c}
            for a in R_ARMS:
                for s in range(3):
                    p = f"{a}_s{s}"
                    row[f"{p}_unnecessary_hard_brake"] = bool(rng.random() < rates[a])
                    row[f"{p}_collision"] = bool(rng.random() < 0.01)
                    row[f"{p}_dose_by_replan"] = [1, 0, 0, 0, 0, 0]
                    row[f"{p}_trigger_count"] = 1
                    row[f"{p}_substituted_count"] = 1 if a in ("PATCH", "SHAM2", "GEO", "PROB") else 0
                    row[f"{p}_trim_fallback_count"] = 1 if a == "TRIM" else 0
            for a in ("cv", "oracle", "static", "log"):
                row[f"{a}_unnecessary_hard_brake"] = bool(rng.random() < 0.01)
                row[f"{a}_collision"] = bool(rng.random() < 0.03)
            rows.append(row)
    return pd.DataFrame(rows)


def test_r_supports_a_large_reduction_and_reads_the_g_ratio() -> None:
    rates = {"ALL": 0.20, "PATCH": 0.10, "SHAM2": 0.19, "TRIM": 0.11, "GEO": 0.195, "PROB": 0.11}
    out = analyze_r(r_rows(rates), "v4", n_boot=400)
    assert out["verdicts"]["R1 PATCH"] == "supported"
    assert out["verdicts"]["R2 sham gap"] == "supported"
    assert out["verdicts"]["R3 TRIM"] == "supported"
    g = out["effects"]["G_ratio_PROB_over_PATCH"]
    assert 0.6 < g["point"] < 1.2 and out["verdicts"]["G1 PROB/PATCH"] in ("probability sufficient", "inconclusive")


def test_r_brake_control_failure_makes_verdicts_uninterpretable() -> None:
    rates = {a: 0.02 for a in R_ARMS}
    d = r_rows(rates)
    d["oracle_unnecessary_hard_brake"] = True  # oracle brakes more than the model: control fails
    out = analyze_r(d, "v4", n_boot=200)
    assert set(out["verdicts"].values()) == {"uninterpretable"}


def test_q_closed_gates_on_q0() -> None:
    rng = np.random.default_rng(1)
    rows = []
    for c in CITIES:
        for k in range(300):
            row = {"scenario_id": f"{c}-{k}", "city": c}
            for a, p in (("QBASE", 0.2), ("QPATCH", 0.1), ("QSHAM2", 0.19)):
                row[f"{a}_unnecessary_hard_brake"] = bool(rng.random() < p)
                row[f"{a}_collision"] = False
                row[f"{a}_dose_by_replan"] = [1, 0, 0, 0, 0, 0]
                row[f"{a}_qcnet_missing_count"] = 0
                row[f"{a}_trigger_count"] = 1
            for s in range(3):
                row[f"ALL_s{s}_unnecessary_hard_brake"] = bool(rng.random() < 0.2)
                row[f"ALL_s{s}_collision"] = False
            for a in ("cv", "oracle", "static", "log"):
                row[f"{a}_unnecessary_hard_brake"] = False
                row[f"{a}_collision"] = False
            row["oracle_unnecessary_hard_brake"] = bool(rng.random() < 0.01)
            rows.append(row)
    d = pd.DataFrame(rows)
    assert q_closed(d, True, n_boot=300)["verdicts"]["Q2 QPATCH"] == "supported"
    assert q_closed(d, False, n_boot=300)["verdicts"]["Q2 QPATCH"] == "uninterpretable"
