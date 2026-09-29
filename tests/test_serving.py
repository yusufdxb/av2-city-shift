import os

import numpy as np
import pytest

from cityshift.preprocess import LANE_PTS, OBJECT_TYPES
from cityshift.scene import Scene, build_input, load_scene
from cityshift.serving import STAGES, StageTimer, build_input_batch, decision_agreement


def example_scene(seed: int = 7) -> Scene:
    rng = np.random.default_rng(seed)
    n, p = 40, 145
    pos = rng.normal(size=(n, 110, 2)) * 20
    vel = rng.normal(size=(n, 110, 2))
    head = rng.normal(size=(n, 110))
    valid = rng.random((n, 110)) > 0.25
    valid[[2, 9, 18], [3, 3, 3]] = True
    valid[[2, 9, 18], [59, 59, 59]] = True
    orig = rng.normal(size=(p * 3, 2)) * 30
    return Scene(
        scenario_id="test", city="test", track_ids=[str(i) for i in range(n)],
        types=rng.integers(0, len(OBJECT_TYPES), n), pos=pos, vel=vel, head=head,
        valid=valid, focal=2, av=0, poly_res=rng.normal(size=(p, LANE_PTS, 2)) * 30,
        poly_attr=rng.integers(0, 2, (p, 2), dtype=np.int8), poly_orig=orig,
        poly_start=np.arange(0, p * 3, 3),
    )


@pytest.mark.parametrize("t", [3, 59])
def test_batch_matches_scalar_on_synthetic_scene(t):
    scene = example_scene()
    centers = [2, 9, 18]
    batch = build_input_batch(scene, centers, t)
    for row, center in enumerate(centers):
        scalar = build_input(scene, center, t)
        for key in scalar:
            np.testing.assert_array_equal(batch[key][row], scalar[key], err_msg=key)


def test_batch_with_no_map_or_centers():
    scene = example_scene()
    scene.poly_start = np.zeros(0, np.int64)
    scene.poly_orig = np.zeros((0, 2))
    scene.poly_res = np.zeros((0, LANE_PTS, 2))
    scene.poly_attr = np.zeros((0, 2), np.int8)
    batch = build_input_batch(scene, [2], 59)
    for key, value in build_input(scene, 2, 59).items():
        np.testing.assert_array_equal(batch[key][0], value)
    assert build_input_batch(scene, [], 59)["agent_hist"].shape[0] == 0


RAW_TRAIN = os.environ.get("CITYSHIFT_RAW_TRAIN", "data/raw/train")


@pytest.mark.skipif(not os.path.isdir(RAW_TRAIN), reason="raw TRAIN scenes unavailable")
def test_batch_matches_scalar_on_real_train_scene():
    with os.scandir(RAW_TRAIN) as entries:
        path = next((entry.path for entry in entries if entry.is_dir()), None)
    if path is None:
        pytest.skip("raw TRAIN directory is empty")
    scene = load_scene(path)
    centers = np.flatnonzero(scene.valid[:, 49])[:3].tolist()
    batch = build_input_batch(scene, centers, 49)
    for row, center in enumerate(centers):
        for key, value in build_input(scene, center, 49).items():
            np.testing.assert_array_equal(batch[key][row], value)


def test_timer_records_stages_and_syncs(monkeypatch):
    calls = []
    monkeypatch.setattr("torch.cuda.synchronize", lambda: calls.append("sync"))
    timer = StageTimer(cuda=True)
    for name in STAGES:
        with timer.stage(name):
            pass
    assert len(calls) == 2 * len(STAGES)
    assert set(timer.summary()) == set(STAGES)
    assert all(timer.summary()[name]["p99_ms"] >= timer.summary()[name]["p50_ms"] >= 0 for name in STAGES)


def test_decision_agreement_counts_paired_replans_and_outcomes():
    scene = example_scene()
    traj = np.zeros((1, 2, 60, 2))
    reference = [dict(t=t, accel=0.0, agents=[2], traj=traj) for t in (49, 59)]
    candidate = [dict(t=49, accel=-1.0, agents=[2], traj=traj + 0.5),
                 dict(t=59, accel=0.0, agents=[], traj=np.zeros((0, 2, 60, 2)))]
    ref_score = {"collision": False, "unnecessary_hard_brake": False}
    test_score = {"collision": True, "unnecessary_hard_brake": False}
    result = decision_agreement(iter([(scene, (ref_score, reference), (test_score, candidate))]))  # generators too
    assert result["acceleration_changed"] == 1
    assert result["acceleration_changed_share"] == 0.5
    assert result["collision_changed_share"] == 1.0
    assert result["unnecessary_hard_brake_changed_share"] == 0.0
    assert result["either_outcome_changed_share"] == 1.0
    assert len(result["largest_trajectory_deviations"]) == 1
    assert result["largest_trajectory_deviations"][0]["max_trajectory_deviation_m"] == pytest.approx(2**-0.5)
