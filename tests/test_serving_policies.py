import numpy as np
import pytest
import torch

from cityshift.preprocess import LANE_PTS, OBJECT_TYPES
from cityshift.scene import Scene
from cityshift.serving import StageTimer
from cityshift.serving_policies import STAGES, apply_policy, run_scenario


def policy_scene() -> Scene:
    """Straight ego path with one stopped and one moving nearby agent."""
    steps = np.arange(110)
    pos = np.zeros((3, 110, 2), np.float64)
    pos[0, :, 0] = steps * 0.5
    pos[1, :, 0] = 34.0
    pos[2, :, 0] = 35.0 + steps * 0.2
    vel = np.zeros_like(pos)
    vel[0, :, 0] = 5.0
    vel[2, :, 0] = 2.0
    return Scene(
        scenario_id="policy_test", city="synthetic", track_ids=["AV", "stopped", "moving"],
        types=np.array([OBJECT_TYPES.index("vehicle")] * 3), pos=pos, vel=vel,
        head=np.zeros((3, 110)), valid=np.ones((3, 110), bool), focal=1, av=0,
        poly_res=np.zeros((0, LANE_PTS, 2)), poly_attr=np.zeros((0, 2), np.int8),
        poly_orig=np.zeros((0, 2)), poly_start=np.zeros(0, np.int64),
    )


def test_trim_renormalizes_and_falls_back_to_cv():
    scene = policy_scene()
    t = 49
    traj = np.repeat(scene.pos[1:3, t, None, None], 6, axis=1)
    traj = np.repeat(traj, 60, axis=2)
    traj = traj.copy()
    traj[0, :, -1, 0] += np.array([0.0, 1.0, 2.0, 2.1, 3.0, 10.0])
    traj[1, :, -1, 0] += 10.0
    prob = np.array([[0.1, 0.2, 0.3, 0.1, 0.1, 0.2], [1 / 6] * 6], np.float32)
    moving_before = traj[1].copy()
    apply_policy(scene, [1, 2], t, traj, prob, "TRIM")
    np.testing.assert_allclose(prob[0], [1 / 6, 1 / 3, 1 / 2, 0, 0, 0])
    np.testing.assert_array_equal(traj[1], moving_before)
    np.testing.assert_allclose(prob[1], [1 / 6] * 6)

    traj[:] = scene.pos[1:3, t, None, None]
    traj[0, :, -1, 0] += 3.0
    apply_policy(scene, [1, 2], t, traj, prob, "TRIM")
    np.testing.assert_array_equal(prob[0], [1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(traj[0, 0, 0], scene.pos[1, t])
    np.testing.assert_array_equal(traj[0], np.broadcast_to(scene.pos[1, t], traj[0].shape))


class FakeBackend:
    device = torch.device("cpu")

    def __init__(self):
        self.batch_sizes = []

    def infer(self, inputs):
        size = len(inputs["agent_hist"])
        self.batch_sizes.append(size)
        traj = torch.zeros((size, 6, 60, 2), dtype=torch.float32)
        prob = torch.zeros((size, 6), dtype=torch.float32)
        prob[:, 0] = 1.0
        return traj, prob


def test_patch_skip_has_identical_plans_to_patch():
    scene = policy_scene()
    full, skipped = FakeBackend(), FakeBackend()
    full_score, full_trace = run_scenario(scene, full, StageTimer(False), "PATCH")
    skip_timer = StageTimer(False)
    skip_score, skip_trace = run_scenario(scene, skipped, skip_timer, "PATCH-skip-inference")
    assert [row["accel"] for row in full_trace] == [row["accel"] for row in skip_trace]
    assert [row["agents"] for row in full_trace] == [row["agents"] for row in skip_trace]
    for ref, result in zip(full_trace, skip_trace):
        np.testing.assert_array_equal(ref["traj"], result["traj"])
        assert result["inferred_agents"] < result["selected_agents"]
    assert all(a > b for a, b in zip(full.batch_sizes, skipped.batch_sizes))
    assert full_score["collision"] == skip_score["collision"]
    assert full_score["unnecessary_hard_brake"] == skip_score["unnecessary_hard_brake"]
    assert set(skip_timer.summary()) == set(STAGES) | {"end_to_end"}


def test_patch_skip_avoids_inference_when_every_agent_is_stopped():
    scene = policy_scene()
    scene.pos[2, :, 0] = 42.0
    scene.vel[2] = 0.0
    full, skipped = FakeBackend(), FakeBackend()
    full_score, full_trace = run_scenario(scene, full, StageTimer(False), "PATCH")
    skip_score, skip_trace = run_scenario(scene, skipped, StageTimer(False), "PATCH-skip-inference")
    assert skipped.batch_sizes == []
    assert [row["accel"] for row in full_trace] == [row["accel"] for row in skip_trace]
    assert all(row["inferred_agents"] == 0 for row in skip_trace)
    assert full_score["failure"] == skip_score["failure"]


@pytest.mark.parametrize("policy", ["none", "PATCH", "PATCH-skip-inference", "TRIM"])
def test_all_policies_score_with_stage3_path(policy):
    score, trace = run_scenario(policy_scene(), FakeBackend(), StageTimer(False), policy)
    assert len(trace) == 6
    assert "first_collision_step" in score
    assert "progress" in score
