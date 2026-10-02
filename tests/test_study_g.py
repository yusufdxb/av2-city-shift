"""v2 study G factorial arms: definitions, the PATCH identity, and the mode-permutation no-op (positive control)."""

from __future__ import annotations

import numpy as np

from cityshift import closedloop as base
from cityshift.closedloop_v2 import init_ego
from cityshift.closedloop_v3 import intervention, most_stationary_mode
from cityshift.preprocess import OBJECT_TYPES
from cityshift.scene import Scene


def scene() -> Scene:
    """Same synthetic scene as tests/test_stage4.py: ego driving +x, two stopped cars ahead, one moving."""
    pos = np.zeros((4, 110, 2))
    pos[0, :, 0] = (np.arange(110) - 49) * 0.5
    pos[1, :, 0], pos[2, :, 0], pos[3, :, 0] = 10.0, 20.0, 30.0
    vel = np.zeros_like(pos)
    vel[0, :, 0], vel[3, :, 0] = 5.0, 2.0
    return Scene("synthetic", "austin", ["AV", "one", "two", "moving"], np.full(4, OBJECT_TYPES.index("vehicle")),
                 pos, vel, np.zeros((4, 110)), np.ones((4, 110), bool), 1, 0,
                 np.zeros((0, 20, 2)), np.zeros((0, 2), np.int8), np.zeros((0, 2)), np.zeros(0, np.int64))


def forecasts(sc, agents, t):
    """Six modes per agent; mode 2 nearly stationary, the rest moving off at different speeds."""
    traj = np.zeros((len(agents), 6, 60, 2))
    steps = np.arange(1, 61)[:, None] * 0.1
    for j, i in enumerate(agents):
        for k, speed in enumerate((6.0, 4.0, 0.05, 3.0, 5.0, 8.0)):
            traj[j, k] = sc.pos[i, t] + steps * np.array([0.0, speed])
    prob = np.array([[0.3, 0.2, 0.05, 0.15, 0.2, 0.1]] * len(agents))
    return traj, prob


def test_geo_and_prob_change_only_stopped_agents_and_only_their_own_axis() -> None:
    sc, agents, t = scene(), [1, 2, 3], 49
    traj, prob = forecasts(sc, agents, t)
    assert most_stationary_mode(sc, 1, t, traj[0]) == 2
    g_traj, g_prob, trigger, n, _ = intervention(sc, agents, t, "GEO", 0, traj.copy(), prob.copy())
    p_traj, p_prob, trigger_p, n_p, _ = intervention(sc, agents, t, "PROB", 0, traj.copy(), prob.copy())
    assert trigger == trigger_p == n == n_p == 2  # agents 1 and 2 are stopped, 3 moves
    np.testing.assert_array_equal(g_prob, prob)  # GEO keeps every probability
    np.testing.assert_allclose(g_traj[0, 2], base.cv_forecast(sc, 1, t))  # ...and replaces the stationary mode
    np.testing.assert_array_equal(np.delete(g_traj[0], 2, 0), np.delete(traj[0], 2, 0))
    np.testing.assert_array_equal(p_traj, traj)  # PROB keeps every trajectory
    np.testing.assert_array_equal(p_prob[0], [0, 0, 1, 0, 0, 0])
    np.testing.assert_array_equal(g_traj[2], traj[2])  # the moving agent is untouched by both
    np.testing.assert_array_equal(p_prob[2], prob[2])


def test_mode_permutation_is_a_no_op_for_the_planner() -> None:
    """Positive control: the planner sums risk over modes, so reordering modes cannot change its choice."""
    sc, agents = scene(), [1, 2, 3]
    ego = init_ego(sc)
    for t in (49, 59):
        traj, prob = forecasts(sc, agents, t)
        order = np.array([4, 0, 5, 2, 1, 3])
        assert base.plan(sc, ego, t, agents, traj, prob) == base.plan(sc, ego, t, agents, traj[:, order], prob[:, order])


def test_geo_plus_prob_is_patch_for_the_planner() -> None:
    sc, agents, t = scene(), [1, 2, 3], 49
    ego = init_ego(sc)
    traj, prob = forecasts(sc, agents, t)
    both_traj, _, _, _, _ = intervention(sc, agents, t, "GEO", 0, traj.copy(), prob.copy())
    both_traj, both_prob, _, _, _ = intervention(sc, agents, t, "PROB", 0, both_traj, prob.copy())
    patch_traj, patch_prob, _, _, _ = intervention(sc, agents, t, "PATCH", 0, traj.copy(), prob.copy())
    assert base.plan(sc, ego, t, agents, both_traj, both_prob) == base.plan(sc, ego, t, agents, patch_traj, patch_prob)
