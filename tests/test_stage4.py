"""Synthetic Stage 4 dose, mechanism, leakage, calibration and decision tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cityshift.analysis_stage4 import analyze, decide_gap, decide_noninferiority, decide_reduction, summarize_calibration
from cityshift.closedloop import REPLANS
from cityshift.closedloop_v3 import calibration_rows, intervention, sham_indices, trim_predictions
from cityshift.closedloop_v2 import init_ego
from cityshift.preprocess import OBJECT_TYPES
from cityshift.scene import Scene
from cityshift.train_mix import sample_indices


def scene() -> Scene:
    pos = np.zeros((4, 110, 2))
    pos[0, :, 0] = (np.arange(110) - 49) * 0.5
    pos[1, :, 0] = 10.0
    pos[2, :, 0] = 20.0
    pos[3, :, 0] = 30.0
    vel = np.zeros_like(pos)
    vel[0, :, 0] = 5.0
    vel[3, :, 0] = 2.0
    head = np.zeros((4, 110))
    valid = np.ones((4, 110), bool)
    return Scene("synthetic", "austin", ["AV", "one", "two", "moving"],
                 np.full(4, OBJECT_TYPES.index("vehicle")), pos, vel, head, valid, 1, 0,
                 np.zeros((0, 20, 2)), np.zeros((0, 2), np.int8), np.zeros((0, 2)), np.zeros(0, np.int64))


def test_sham2_matches_patch_at_each_replan() -> None:
    sc = scene()
    agents = [1, 2, 3]
    traj = np.zeros((3, 6, 60, 2))
    prob = np.ones((3, 6)) / 6
    for t in REPLANS:
        _, _, trigger, patch_count, _ = intervention(sc, agents, t, "PATCH", 0, traj.copy(), prob.copy())
        assert trigger == patch_count == 2
        chosen = sham_indices(sc, agents, t, 0, patch_count)
        assert len(chosen) == 2 and len(set(chosen)) == 2
        _, _, sham_trigger, sham_count, _ = intervention(sc, agents, t, "SHAM2", 0,
                                                          traj.copy(), prob.copy(), patch_count)
        assert sham_trigger == sham_count == patch_count
        assert np.array_equal(chosen, sham_indices(sc, agents, t, 0, patch_count))
    # a diverged ego with fewer selected agents than PATCH's dose: capped, never a crash
    assert len(sham_indices(sc, [1], 49, 0, 2)) == 1


def test_trim_drops_moving_mass_and_renormalises_with_cv_fallback() -> None:
    sc = scene()
    traj = np.zeros((3, 3, 60, 2))
    traj[0, :, :, 0] = np.array([10.5, 11.5, 20.0])[:, None]
    traj[1, :, :, 0] = np.array([23.0, 24.0, 25.0])[:, None]
    traj[2, :, :, 0] = np.array([31.0, 35.0, 40.0])[:, None]
    prob = np.array([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5], [0.2, 0.3, 0.5]])
    original_moving = traj[2].copy()
    trimmed, mass, fallback = trim_predictions(sc, [1, 2, 3], 49, traj, prob)
    assert fallback == 1
    np.testing.assert_allclose(mass[0], [0.4, 0.6, 0.0])
    np.testing.assert_allclose(mass[1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(trimmed[1, 0, :, 0], 20.0)
    np.testing.assert_allclose(trimmed[2], original_moving)
    np.testing.assert_allclose(mass[2], [0.2, 0.3, 0.5])


def test_mix_fixed_proportion_and_scenario_exclusion() -> None:
    focal = pd.DataFrame({"scenario_id": ["pool", "dev", "a", "b", "c", "d", "e"]})
    multi = pd.DataFrame({"scenario_id": ["pool", "dev", "a", "b", "c", "d", "e"],
                          "focal": [False] * 7, "speed": [0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 2.0]})
    fi, mi = sample_indices(focal, multi, {"pool", "dev"}, 4, 3)
    assert len(fi) == 3 and len(mi) == 1
    assert not set(focal.scenario_id.iloc[fi]) & {"pool", "dev"}
    assert not set(multi.scenario_id.iloc[mi]) & {"pool", "dev"}
    assert (multi.speed.iloc[mi] < 0.5).all()
    with pytest.raises(ValueError, match="insufficient"):
        sample_indices(focal, multi, {"pool", "dev"}, 20, 3)


def test_calibration_records_candidate_risk_and_censoring() -> None:
    sc = scene()
    ego = init_ego(sc)
    traj = np.repeat(sc.pos[1, 49][None, None, None], 60, axis=2).reshape(1, 1, 60, 2)
    risk = calibration_rows(sc, ego, 49, [1], traj, np.ones((1, 1)), 7, "PATCH", 0)
    assert len(risk) == 1
    assert risk[0]["predicted_hit_probability"] == 1.0
    assert risk[0]["contact"] is True
    assert risk[0]["complete_future"] is True
    late = calibration_rows(sc, ego, 99, [1], traj, np.ones((1, 1)), 7, "PATCH", 0)
    assert late[0]["observed_steps"] == 10
    assert late[0]["complete_future"] is False


def synthetic_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    closed, focal = [], []
    for city in ("austin", "dearborn", "miami", "palo-alto", "pittsburgh", "washington-dc"):
        for i in range(200):
            sid = f"{city}-{i}"
            row = {"scenario_id": sid, "city": city, "static_collision": i < 8,
                   "log_collision": False}
            for seed in range(3):
                for arm in ("ALL", "PATCH", "TRIM", "SHAM2", "MIX"):
                    threshold = {"ALL": 80, "PATCH": 40, "TRIM": 40, "SHAM2": 72, "MIX": 40}[arm]
                    row[f"{arm}_s{seed}_unnecessary_hard_brake"] = i < threshold
                    row[f"{arm}_s{seed}_collision"] = i < 2
                    row[f"{arm}_s{seed}_trigger_count"] = 6
                    row[f"{arm}_s{seed}_substituted_count"] = 6 if arm == "SHAM2" else 0
                    row[f"{arm}_s{seed}_trim_fallback_count"] = 0
                    for t in REPLANS:
                        row[f"{arm}_s{seed}_dose_t{t}"] = 1
                for arm in ("ALL", "MIX"):
                    focal.append({"scenario_id": sid, "city": city, "agent_id": "focal", "focal": True,
                                  "predictor": f"{arm}_s{seed}", "miss": float(i < 40)})
            closed.append(row)
    return pd.DataFrame(closed), pd.DataFrame(focal)


def test_decisions_and_per_replan_dose_gate() -> None:
    closed, focal = synthetic_tables()
    result = analyze(closed, focal, 100)
    assert all(item["supported"] for item in result["claims"].values())
    assert result["controls"]["ALL_brake_events"] == 480
    assert decide_reduction(0.30, [0.01, 0.5])
    assert not decide_reduction(0.29, [0.01, 0.5])
    assert decide_noninferiority([-0.01, 0.0029], 0.003)
    assert not decide_noninferiority([-0.01, 0.003], 0.003)
    assert decide_gap(0.20, [0.01, 0.3])
    assert not decide_gap(0.19, [0.01, 0.3])
    assert result["controls"]["dose_dose_pass"] and result["controls"]["dose_SHAM2_over_PATCH"] == 1.0
    over = closed.copy()
    over.loc[0, "SHAM2_s0_dose_t49"] = 2  # a sham dose above PATCH's is a bug, not a cap
    with pytest.raises(ValueError, match="invalid SHAM2 dose"):
        analyze(over, focal, 10)
    short = closed.copy()
    for seed in range(3):
        short.loc[: len(short) // 10, f"SHAM2_s{seed}_dose_t49"] = 0  # realised dose falls below 99% of PATCH's
    res = analyze(short, focal, 50)
    assert not res["controls"]["dose_dose_pass"]
    assert res["claims"]["H11"]["status"].startswith("uninterpretable")


def test_calibration_summary_excludes_censored_rows(tmp_path) -> None:
    path = tmp_path / "risk.parquet"
    pd.DataFrame([{"arm": "PATCH", "stopped": True, "predicted_hit_probability": 0.9,
                   "contact": True, "complete_future": True},
                  {"arm": "PATCH", "stopped": True, "predicted_hit_probability": 0.1,
                   "contact": False, "complete_future": True},
                  {"arm": "PATCH", "stopped": True, "predicted_hit_probability": 1.0,
                   "contact": False, "complete_future": False}]).to_parquet(path)
    result = summarize_calibration(str(path))
    assert result["by_arm"]["PATCH"]["all"]["n"] == 2
    assert result["by_arm"]["PATCH"]["all"]["auroc"] == 1.0
    assert result["by_arm"]["PATCH"]["all"]["brier"] == pytest.approx(0.01)
    assert result["censored_rows_excluded"]["PATCH"] == 1
