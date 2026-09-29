"""Geometry, set membership, partial scoring, and clustered inference for Stage 3a."""

import numpy as np
import pandas as pd
import pytest

from cityshift import analysis_stage3a, multiagent_eval
from cityshift.baselines import constant_velocity, lane_following
from cityshift.preprocess import FUT, LANE_PTS, MAX_AGENTS, MAX_LANES


def _input(velocity=(4.0, 0.0)):
    inp = {
        "agent_hist": np.zeros((MAX_AGENTS, 50, 6), np.float32),
        "lane_pts": np.zeros((MAX_LANES, LANE_PTS, 2), np.float32),
        "lane_attr": np.full((MAX_LANES, 2), -1, np.int8),
    }
    inp["agent_hist"][0, -1, 2:4] = velocity
    return inp


def test_cv_exact_straight_agent():
    inp = _input((3.0, 4.0))
    traj, prob = constant_velocity(inp)
    target = np.arange(1, FUT + 1)[:, None] * 0.1 * np.array([3.0, 4.0])
    np.testing.assert_allclose(traj[0], target, atol=2e-6)
    assert traj.shape == (6, 60, 2)
    assert prob.sum() == pytest.approx(1)


def test_lane_follows_curved_centerline():
    inp = _input((4.0, 0.0))
    radius = 25.0
    angle = np.linspace(0, 1.2, LANE_PTS)
    inp["lane_pts"][0] = np.column_stack((radius * np.sin(angle), radius * (1 - np.cos(angle))))
    inp["lane_attr"][0] = (0, 0)
    traj, prob = lane_following(inp)
    expected_angle = (4 * 0.1 * np.arange(1, FUT + 1)) / radius
    expected = np.column_stack((radius * np.sin(expected_angle), radius * (1 - np.cos(expected_angle))))
    assert np.linalg.norm(traj[0] - expected, axis=1).max() < 0.12
    assert prob[0] == pytest.approx(1)
    np.testing.assert_array_equal(traj[1:], np.repeat(traj[:1], 5, axis=0))


def test_lane_falls_back_when_heading_disagrees():
    inp = _input()
    inp["lane_pts"][0, :, 0] = np.linspace(0, -20, LANE_PTS)
    inp["lane_attr"][0] = (0, 0)
    actual = lane_following(inp)
    expected = constant_velocity(inp)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def test_memberships_exclude_av_and_require_full_scored_future(monkeypatch):
    class FakeScene:
        av = 0
        focal = 1
        track_ids = ["AV", "focal", "scored", "partial", "other"]
        valid = np.ones((5, 110), bool)

    sc = FakeScene()
    sc.valid[3, 90] = False
    monkeypatch.setattr(multiagent_eval, "init_ego", lambda scene: object())
    monkeypatch.setattr(multiagent_eval, "select_agents", lambda scene, ego, t: [0, 2, 3, 4])
    got = multiagent_eval.memberships(sc, {"AV": 2, "focal": 3, "scored": 2, "partial": 2})
    assert got == {1: (True, False, False), 2: (False, True, True),
                   3: (False, False, True), 4: (False, False, True)}


def test_partial_future_uses_last_valid_endpoint():
    traj = np.zeros((6, 60, 2), np.float32)
    traj[0, :, 0] = np.arange(1, 61) * 0.1
    target = traj[0].copy()
    target[3:, 0] = 100
    valid = np.zeros(60, bool)
    valid[:3] = True
    got = multiagent_eval.score(traj, np.array([1, 0, 0, 0, 0, 0]), target, valid)
    assert got["min_fde"] == pytest.approx(0)
    assert got["min_ade"] == pytest.approx(0)


def test_cluster_bootstrap_resamples_scenarios_not_agents():
    # A large cluster must retain all ten agents together. The point estimate is
    # agent weighted, while the bootstrap support has only two possible cluster mixes.
    clusters = np.array(["large"] * 10 + ["small"])
    treatment = np.array([0.0] * 10 + [1.0])
    control = np.ones(11)
    point, boots = analysis_stage3a.cluster_relative(treatment, control, clusters, np.random.default_rng(4), 1000)
    assert point == pytest.approx(-10 / 11)
    assert set(np.round(boots, 8)) <= {-1.0, round(-10 / 11, 8), 0.0}
    assert {-1.0, 0.0} <= set(np.round(boots, 8))


def test_analysis_seed_average_and_city_cluster(monkeypatch):
    monkeypatch.setattr(analysis_stage3a, "CITIES", ("austin", "miami"))
    rows = []
    for city in analysis_stage3a.CITIES:
        for scenario in range(110):
            for agent in range(2):
                common = {"scenario_id": f"{city}-{scenario}", "agent_id": str(agent), "city": city,
                          "agent_type": "vehicle", "speed": 4.0, "focal": agent == 0,
                          "scored": True, "planner_relevant": True, "n_valid": 60}
                for name, miss in [("CV", 0.4), ("LANE", 0.5), ("STATIC", 0.8),
                                   ("ALL_s0", 0.2), ("ALL_s1", 0.3), ("ALL_s2", 0.4),
                                   *[(f"LOCO-{city}_s{s}", 0.4) for s in range(3)]]:
                    # Binary misses with rates fixed across scenarios and paired by agent.
                    value = float((scenario + agent * 10) % 10 < round(10 * miss))
                    rows.append(common | {"predictor": name, "miss": value, "min_ade": value,
                                          "min_fde": value, "brier_min_fde": value})
    result = analysis_stage3a.analyze(pd.DataFrame(rows), n_boot=200)
    assert result["H5"]["pooled_reduction"] == pytest.approx(0.4)
    assert result["H1_planner_descriptive"]["pooled_rel_change"] == pytest.approx(1 / 3)
    assert result["H5"]["decision"] == "supported"


def test_h6_stopped_and_moving_agents():
    import numpy as np
    import pandas as pd

    from cityshift import analysis_stage3a

    rows = []
    for city in analysis_stage3a.CITIES:
        for scenario in range(60):
            for agent, speed in enumerate((0.0, 5.0)):
                common = {"scenario_id": f"{city}-{scenario}", "agent_id": str(agent), "city": city,
                          "agent_type": "vehicle", "speed": speed, "focal": False,
                          "scored": True, "planner_relevant": True, "n_valid": 60}
                # stopped: ALL misses 60%, CV 20% (ratio 3); moving: ALL 20%, CV 80% (reduction 0.75)
                rates = {"ALL": 0.6 if speed < 0.5 else 0.2, "CV": 0.2 if speed < 0.5 else 0.8}
                for name in ["CV", "LANE", "STATIC", "ALL_s0", "ALL_s1", "ALL_s2", *[f"LOCO-{city}_s{s}" for s in range(3)]]:
                    r = rates["ALL"] if name.startswith(("ALL", "LOCO")) else rates["CV"] if name == "CV" else 0.9
                    v = float(scenario % 10 < round(10 * r))
                    rows.append(common | {"predictor": name, "miss": v, "min_ade": v, "min_fde": v, "brier_min_fde": v})
    res = analysis_stage3a.analyze(pd.DataFrame(rows), n_boot=200)
    assert np.isclose(res["H6a"]["pooled_difference_ALL_minus_CV"], 0.4)
    assert res["H6a"]["decision"] == "supported"
    assert np.isclose(res["H6b"]["pooled_reduction_ALL_vs_CV"], 0.75)
    assert res["H6b"]["decision"] == "supported"
