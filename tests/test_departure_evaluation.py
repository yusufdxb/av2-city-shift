"""Protocol, pairing, calibration selection and frozen-artifact invariants."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cityshift import departure_evaluation as ev
from cityshift.departure_calibration import CALIBRATOR_FEATURE_NAMES, HISTORY_FEATURE_NAMES, fit_calibrator


def model(probability=0.08):
    fitted = fit_calibrator(np.zeros((4, len(HISTORY_FEATURE_NAMES))), [0.2, 0.4, 0.6, 0.8],
                            [0, 0, 1, 1], ["a", "b", "c", "d"])
    fitted["coefficients"] = [0.0] * len(CALIBRATOR_FEATURE_NAMES)
    fitted["intercept"] = float(np.log(probability / (1 - probability)))
    return fitted


def frame():
    rows = []
    for city in ev.CITIES:
        for index in range(2):
            for seed in ev.SEEDS:
                row = {"scenario_id": f"{city}-{index}", "city": city, "track_id": "vehicle", "t": 49,
                       "seed": seed, "label": bool(index), "native_mass": 0.8,
                       "native_fde4s": 2.0, "patch_mass": 0.0, "patch_fde4s": 6.0 if index else 0.1,
                       "calib_mass": 0.7 if index else 0.1, "calib_fde4s": 1.0,
                       "selective_mass": 0.8 if index else 0.1, "selective_fde4s": 1.5}
                for threshold in ev.THRESHOLDS:
                    prefix = ev._threshold_key(threshold)
                    row[f"{prefix}_mass"] = 0.8 if index else 0.1
                    row[f"{prefix}_fde4s"] = 1.5
                rows.append(row)
    return pd.DataFrame(rows)


def test_tuning_uses_recall_constraint_and_conservative_tie_break():
    rows = frame()
    result = ev.tune_selective_threshold(rows)
    assert result["threshold"] == 0.05
    assert result["status"] == "admissible_correction"
    assert result["n_independent_departure_scenes"] == 6
    assert all(item["admissible"] for item in result["candidates"])
    # Suppressing the true-departure mass improves aggregate Brier here, but
    # violates the predeclared departure-recall constraint.
    rows.loc[rows.label, "selective_05_mass"] = 0.01
    result = ev.tune_selective_threshold(rows)
    assert result["threshold"] == 0.10
    assert not result["candidates"][0]["admissible"]


def test_tuning_identity_when_no_admissible_improvement_or_no_departures():
    rows = frame()
    for threshold in ev.THRESHOLDS:
        rows[f"{ev._threshold_key(threshold)}_mass"] = rows.native_mass
    assert ev.tune_selective_threshold(rows)["threshold"] == 0
    assert ev.tune_selective_threshold(rows)["status"] == "no_admissible_improvement_identity"
    stationary = rows[~rows.label].copy()
    result = ev.tune_selective_threshold(stationary)
    assert result["threshold"] == 0
    assert result["native_departure_recall"] is None
    json.dumps(result, allow_nan=False)


def test_tuning_scene_equal_weighting_resists_repeated_actors():
    rows = frame()
    extra = rows[(rows.city == "austin") & ~rows.label].copy()
    extra["track_id"] = "extra-vehicle"
    # More actors in one scene do not change the Brier objective.
    first = ev.tune_selective_threshold(rows)
    second = ev.tune_selective_threshold(pd.concat([rows, extra], ignore_index=True))
    assert first["native_brier"] == second["native_brier"]
    assert first["candidates"] == second["candidates"]


def test_summary_independent_scenes_and_city_equal_seed_averages():
    rows = frame()
    result = ev.summarize_rows(rows, partition="pilot")
    assert result["row_inventory"]["rows"] == 36
    assert result["row_inventory"]["events"] == 12
    assert result["n_independent_departure_scenes"] == 6
    assert result["arms"]["NATIVE"]["departure_recall_at_0_1"] == 1
    assert result["arms"]["PATCH"]["departure_recall_at_0_1"] == 0
    assert result["arms"]["PATCH"]["by_label"]["departed"]["top1_fde4s_m"] == 6
    interval = result["arms"]["SELECTIVE"]["departure_recall_loss_vs_native"]
    assert interval["estimate"] == 0
    assert interval["ci"] == [0, 0]
    assert interval["coverage_complete"]
    assert interval["underpowered"]
    # Six cities, each with a scene-averaged metric, remain equally weighted
    # even when one city has several extra scenes.
    extra = rows[rows.city == "austin"].copy()
    extra["scenario_id"] += "-more"
    assert ev.summarize_rows(pd.concat([rows, extra]), partition="pilot")["arms"]["NATIVE"]["brier"] == pytest.approx(
        result["arms"]["NATIVE"]["brier"])
    json.dumps(result, allow_nan=False)


def test_missing_city_departures_are_coverage_gaps_and_bootstrap_is_paired():
    rows = frame()
    rows.loc[rows.city != "austin", "label"] = False
    rows.loc[(rows.city == "austin") & rows.label & (rows.seed == 0), "selective_mass"] = 0.01
    result = ev.summarize_rows(rows, partition="pilot")
    interval = result["arms"]["SELECTIVE"]["departure_recall_loss_vs_native"]
    assert not interval["coverage_complete"]
    assert len(interval["missing_cities"]) == 5
    assert interval["estimate"] == pytest.approx(1 / 3)
    assert interval["ci"] == pytest.approx([1 / 3, 1 / 3])
    assert interval["n_independent_departure_scenes"] == 1


def test_no_departures_are_unknown_recall_and_invalid_pairs_rejected():
    rows = frame()
    rows.label = False
    result = ev.summarize_rows(rows, partition="pilot")
    assert result["arms"]["NATIVE"]["departure_recall_at_0_1"] is None
    assert result["arms"]["SELECTIVE"]["departure_recall_loss_vs_native"]["ci"] is None
    with pytest.raises(ValueError, match="three predictor seeds"):
        ev.summarize_rows(rows.iloc[1:], partition="pilot")
    with pytest.raises(ValueError, match="duplicate"):
        ev.summarize_rows(pd.concat([rows, rows.iloc[:1]]), partition="pilot")
    rows.loc[0, "label"] = True
    with pytest.raises(ValueError, match="ground-truth"):
        ev.summarize_rows(rows, partition="pilot")


def test_corrected_top1_error_accounts_for_added_cv_mode_and_selective_identity():
    sc = SimpleNamespace(scenario_id="event", city="austin", track_ids=["vehicle"],
                         pos=np.zeros((1, 110, 2)), vel=np.zeros((1, 110, 2)), valid=np.ones((1, 110), bool))
    traj = np.zeros((1, 2, 60, 2))
    traj[0, 0, :, 0] = np.arange(1, 61) * 0.2
    traj[0, 1, :, 0] = np.arange(1, 61) * 0.3
    prob = np.array([[0.6, 0.4]])
    bundle = {"agents": [0], "t": 49, "labels": [False], "features": np.zeros((1, 10))}
    row = ev._score_forecasts(sc, bundle, 0, traj, prob, model(), (0.0, 0.2))[0]
    assert row["native_fde4s"] == pytest.approx(8)
    assert row["calib_fde4s"] == pytest.approx(0)
    assert row["calib_cv_admitted"]
    assert row["calib_mass"] == pytest.approx(0.08)
    assert row["selective_00_mass"] == row["native_mass"]
    assert row["selective_00_fde4s"] == row["native_fde4s"]
    assert row["selective_20_fde4s"] == pytest.approx(0)
    # Future mutations can alter scoring, but not realizable correction mass.
    sc.pos[0, 89, 0] = 9
    changed = ev._score_forecasts(sc, bundle, 0, traj, prob, model(), (0.2,))[0]
    assert changed["calib_mass"] == row["calib_mass"]
    assert changed["calib_fde4s"] == pytest.approx(9)


def test_logged_ego_selection_pose_is_exact_with_subfivecentimeter_motion():
    sc = SimpleNamespace(av=0, pos=np.zeros((1, 110, 2)), vel=np.zeros((1, 110, 2)), head=np.zeros((1, 110)))
    sc.pos[0, :, 0] = np.arange(110) * 0.01
    sc.pos[0, 57:60] = sc.pos[0, 56]  # duplicate logged vertices are harmless
    sc.vel[0, :, 0] = 0.1
    for t in ev.TIMES:
        ego = ev._logged_ego(sc, t)
        np.testing.assert_allclose(ego.path.at(np.array([ego.s[t]]))[0][0], sc.pos[0, t], atol=1e-12)
        assert ego.v[t] == pytest.approx(0.1)


def test_table_cache_checks_inventory_context_and_content(tmp_path):
    path = tmp_path / "openloop_pilot.parquet"
    context = {"model": "frozen", "partition": "pilot"}
    rows = frame()
    metadata = ev._write_table(path, rows, context, [{"scenario_id": "raw", "files": {"map.json": "digest"}}])
    cached, restored = ev._read_cached(path, context)
    pd.testing.assert_frame_equal(cached, rows)
    assert restored == metadata
    with pytest.raises(ValueError, match="context"):
        ev._read_cached(path, {"model": "changed"})
    with pytest.raises(ValueError, match="overwrite"):
        ev._write_table(path, rows, context, [])
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        ev._read_cached(path, context)


def test_cohort_overlap_and_checkpoint_fingerprint_checks(tmp_path):
    partitions = {name: {"ids": [name], "sha256": ev.ids_digest([name])}
                  for name in ("fit", "tune", "pilot", "test")}
    cohort = {"schema": 1, "partitions": partitions, "training_fingerprints": []}
    for seed in ev.SEEDS:
        path = tmp_path / f"seed{seed}" / "model.pt"
        path.parent.mkdir()
        path.write_bytes(f"model-{seed}".encode())
        cohort["training_fingerprints"].append({"seed": seed, "checkpoint_sha256": ev.file_sha256(path)})
    cohort_path = tmp_path / "cohort.json"
    cohort_path.write_text(json.dumps(cohort))
    assert ev.load_cohort(cohort_path) == cohort
    assert len(ev.checkpoint_manifest(tmp_path, cohort)) == 3
    mismatched = copy.deepcopy(cohort)
    mismatched["training_fingerprints"][0]["checkpoint_sha256"] = "wrong"
    with pytest.raises(ValueError, match="checkpoint differs"):
        ev.checkpoint_manifest(tmp_path, mismatched)
    cohort["partitions"]["test"] = copy.deepcopy(partitions["pilot"])
    cohort_path.write_text(json.dumps(cohort))
    with pytest.raises(ValueError, match="overlapping"):
        ev.load_cohort(cohort_path)


def test_json_artifact_cannot_be_overwritten_with_new_decision(tmp_path):
    path = tmp_path / "calibrator.json"
    original = {"model": model(), "tuning": {"threshold": 0.05}}
    ev._write_json_once(path, original)
    ev._write_json_once(path, original)
    original["tuning"]["threshold"] = 0.2
    with pytest.raises(ValueError, match="refusing to overwrite"):
        ev._write_json_once(path, original)
    assert json.loads(path.read_text())["tuning"]["threshold"] == 0.05
