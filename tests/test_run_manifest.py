import json

import pandas as pd
import pytest

from cityshift.run_manifest import file_sha256, run_settings, validate_manifest, validate_parts


def test_manifest_matching_resume_and_changed_settings(tmp_path):
    path, output = tmp_path / "manifest.json", tmp_path / "part_00000.parquet"
    settings = {"device": "cpu", "seeds": [0, 1, 2], "chunk": 500, "source_sha256": {"harness": "old"}}
    validate_manifest(path, settings, [output])
    before = path.read_bytes()
    output.write_bytes(b"rows")
    validate_manifest(path, settings, [output])
    assert path.read_bytes() == before
    for key, value in (("device", "cuda"), ("seeds", [0]), ("chunk", 100), ("source_sha256", {"harness": "new"})):
        with pytest.raises(SystemExit, match="different settings or inputs"):
            validate_manifest(path, settings | {key: value}, [output])
    assert path.read_bytes() == before


def test_legacy_outputs_and_manifest_are_not_adopted(tmp_path):
    path, output = tmp_path / "manifest.json", tmp_path / "rows.parquet"
    output.write_bytes(b"historical rows")
    with pytest.raises(SystemExit, match="without"):
        validate_manifest(path, {"device": "cpu"}, [output])
    assert not path.exists()
    path.write_text(json.dumps({"device": "cpu", "git": "old"}))
    with pytest.raises(SystemExit, match="different settings"):
        validate_manifest(path, {"device": "cpu"}, [output])


def test_settings_fingerprint_checkpoints_and_scenario_selection(tmp_path):
    pool, checkpoint = tmp_path / "pool.npy", tmp_path / "model.pt"
    pool.write_bytes(b"pool")
    checkpoint.write_bytes(b"checkpoint")
    args = ("closedloop_v3", "cpu", [0], str(pool), ["one", "two"], str(tmp_path), {"ALL_s0": str(checkpoint)})
    old = run_settings(*args, arms=["TRIMNF"], offset=0)
    assert old["checkpoints_sha256"]["ALL_s0"] == file_sha256(checkpoint)
    assert "closedloop_v3.py" in old["source_sha256"]
    assert str(tmp_path) not in json.dumps(old)
    checkpoint.write_bytes(b"replaced checkpoint at the same path")
    assert run_settings(*args, arms=["TRIMNF"], offset=0) != old
    checkpoint.write_bytes(b"checkpoint")
    changed = list(args)
    changed[4] = ["two", "one"]
    assert run_settings(*changed, arms=["TRIMNF"], offset=0) != old


def _part(path, ids):
    metrics = ("collision", "first_collision_step", "planner_hard_brake", "logged_hard_brake",
               "unnecessary_hard_brake", "progress", "accels")
    prefixes = [f"{a}_s0" for a in ("ALL", "PATCH", "SHAM2", "TRIM")] + ["cv", "oracle", "static", "log"]
    rows = pd.DataFrame({"scenario_id": ids, "city": ["austin"] * len(ids),
                         **{f"{a}_{m}": [0] * len(ids) for a in prefixes for m in metrics
                            if a != "log" or m != "accels"},
                         **{f"{a}_s0_dose_by_replan": [[0] * 6] * len(ids)
                            for a in ("ALL", "PATCH", "SHAM2", "TRIM")}})
    rows.to_parquet(path, index=False)
    return rows


def test_resume_validates_parts_before_skipping(tmp_path):
    path = tmp_path / "part_00000.parquet"
    good = _part(path, ["one", "two"])
    validate_parts(tmp_path, ["one", "two", "three"], 2, [0])
    _part(path, ["two", "one"])
    with pytest.raises(SystemExit, match="invalid scenarios"):
        validate_parts(tmp_path, ["one", "two", "three"], 2, [0])
    good.drop(columns="SHAM2_s0_dose_by_replan").to_parquet(path, index=False)
    with pytest.raises(SystemExit, match="invalid scenarios or columns"):
        validate_parts(tmp_path, ["one", "two", "three"], 2, [0])
    good.to_parquet(path, index=False)
    _part(tmp_path / "part_00004.parquet", ["other"])
    with pytest.raises(SystemExit, match="unexpected part"):
        validate_parts(tmp_path, ["one", "two", "three"], 2, [0])
