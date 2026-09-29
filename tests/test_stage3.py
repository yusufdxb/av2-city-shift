"""Stage 3 geometry, leakage controls, and decision rules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cityshift.analysis_stage3 import decide_improvement, decide_noninferiority, decide_reduction
from cityshift.closedloop_v2 import contact_point, init_ego, score, speed_cap
from cityshift.preprocess import OBJECT_TYPES
from cityshift.scene import Scene
from cityshift.train_multi import sample_indices


def synthetic_scene(agent_pos: tuple[float, float], agent_heading: float, agent_type: str) -> Scene:
    pos = np.zeros((2, 110, 2))
    pos[1, 50] = agent_pos
    vel = np.zeros_like(pos)
    head = np.zeros((2, 110))
    head[1, 50] = agent_heading
    valid = np.zeros((2, 110), bool)
    valid[0] = True
    valid[1, 50] = True
    return Scene("synthetic", "austin", ["AV", "other"],
                 np.array([0, OBJECT_TYPES.index(agent_type)]), pos, vel, head, valid, 1, 0,
                 np.zeros((0, 20, 2)), np.zeros((0, 2), np.int8), np.zeros((0, 2)), np.zeros(0, np.int64))


def test_contact_front_bus_center_behind() -> None:
    sc = synthetic_scene((-5.0, -4.0), 0.3, "bus")
    point = contact_point(np.zeros(2), 0.0, sc.pos[1, 50], sc.head[1, 50], np.array([12.0, 2.6]))
    assert point is not None and point[0] > 0
    result = score(sc, sc.pos[0], sc.head[0], np.zeros(110), [])
    assert result["collision"] is True
    assert result["first_collision_step"] == 50


def test_contact_rear_end_excluded() -> None:
    sc = synthetic_scene((-3.0, 0.0), 0.0, "vehicle")
    point = contact_point(np.zeros(2), 0.0, sc.pos[1, 50], 0.0, np.array([4.5, 2.0]))
    assert point is not None and point[0] < 0
    result = score(sc, sc.pos[0], sc.head[0], np.zeros(110), [])
    assert result["collision"] is False
    assert result["first_collision_step"] == 50


def test_speed_cap_uses_only_handoff_speed() -> None:
    sc = synthetic_scene((30.0, 0.0), 0.0, "vehicle")
    sc.pos[0, :, 0] = np.arange(110) * 0.5
    sc.vel[0, 49, 0] = 8.0
    sc.vel[0, 50:, 0] = 100.0
    assert speed_cap(8.0) == 15.0
    assert init_ego(sc).vmax == 15.0
    sc.vel[0, 50:, 0] = 0.0
    assert init_ego(sc).vmax == 15.0
    assert speed_cap(20.0) == 23.0


def test_dev_excluded_by_scenario_id() -> None:
    meta = pd.DataFrame({"scenario_id": ["a", "a", "b", "c", "c", "d"]})
    train, dev = sample_indices(meta, {"a", "c"}, 2, 1000)
    assert set(meta.scenario_id.iloc[train]) <= {"b", "d"}
    assert set(meta.scenario_id.iloc[dev]) == {"a", "c"}
    assert len(dev) == 4


def test_analysis_thresholds() -> None:
    assert decide_reduction(0.31, [0.01, 0.55], 0.30)
    assert not decide_reduction(0.29, [0.01, 0.55], 0.30)
    assert not decide_reduction(0.31, [-0.01, 0.55], 0.30)
    assert decide_noninferiority(0.001, [-0.002, 0.0029])
    assert not decide_noninferiority(0.001, [-0.002, 0.003])
    assert decide_improvement(-0.16, [-0.25, -0.01], 0.15)
    assert not decide_improvement(-0.16, [-0.25, 0.01], 0.15)


def test_joint_closedloop_decisions() -> None:
    from cityshift.analysis_stage3 import CITIES, closedloop_analysis

    rows = []
    for city in CITIES:
        for i in range(200):
            row = {"city": city, "scenario_id": f"{city}-{i}",
                   "cv_collision": i < 10, "oracle_collision": i < 2,
                   "static_collision": i < 12, "log_collision": False,
                   "cv_unnecessary_hard_brake": i < 6,
                   "oracle_unnecessary_hard_brake": i < 2,
                   "static_unnecessary_hard_brake": i < 6}
            for seed in range(3):
                for arm in ("ALL", "MULTI", "PATCH", "SHAM"):
                    row[f"{arm}_s{seed}_collision"] = i < 4
                    row[f"{arm}_s{seed}_unnecessary_hard_brake"] = i < (40 if arm in ("ALL", "SHAM") else 15)
            rows.append(row)
    result = closedloop_analysis(pd.DataFrame(rows), np.random.default_rng(1), 200)
    assert result["H7"]["supported"]
    assert result["H8"]["supported"]
    assert result["controls"]["SHAM_pass"]


def test_joint_openloop_decision() -> None:
    from cityshift.analysis_stage3 import CITIES, openloop_analysis

    rows = []
    for city in CITIES:
        for i in range(10):
            for agent, focal in (("parked", False), ("focal", True)):
                for arm in ("ALL", "MULTI"):
                    for seed in range(3):
                        rows.append({"city": city, "scenario_id": f"{city}-{i}", "agent_id": agent,
                                     "focal": focal, "planner_relevant": not focal,
                                     "speed": 1.0 if focal else 0.0,
                                     "predictor": f"{arm}_s{seed}",
                                     "miss": float(agent == "parked" and arm == "ALL")})
    result = openloop_analysis(pd.DataFrame(rows), np.random.default_rng(2), 200)
    assert result["supported"]
    assert result["stopped_nonfocal_difference"] == -1.0
    assert result["focal_difference"] == 0.0


def test_patch_and_dose_matched_sham() -> None:
    from dataclasses import replace

    from cityshift.closedloop_v2 import patch_predictions

    base = synthetic_scene((5.0, 0.0), 0.0, "vehicle")
    # three tracks: AV (0), a stopped agent (1) and a moving agent (2), all observed at t=49
    pos = np.zeros((3, 110, 2))
    pos[1, 49], pos[2, 49] = (5.0, 3.0), (20.0, -3.0)
    vel = np.zeros((3, 110, 2))
    vel[2, 49] = (8.0, 0.0)
    valid = np.ones((3, 110), bool)
    sc = replace(base, track_ids=["AV", "stopped", "moving"], types=np.zeros(3, np.int64), pos=pos, vel=vel,
                 head=np.zeros((3, 110)), valid=valid)
    traj = np.ones((2, 6, 60, 2)) * 7.0
    prob = np.tile(np.array([0.4, 0.2, 0.1, 0.1, 0.1, 0.1]), (2, 1))
    p_traj, p_prob, p_n = patch_predictions(sc, [1, 2], 49, traj.copy(), prob.copy())
    s_traj, s_prob, s_n = patch_predictions(sc, [1, 2], 49, traj.copy(), prob.copy(), sham=True)
    assert p_n == s_n == 1  # identical dose
    # PATCH replaces the stopped agent with a constant-velocity (here: standing) forecast, leaves the moving one
    np.testing.assert_allclose(p_traj[0], np.broadcast_to(pos[1, 49], (6, 60, 2)))
    np.testing.assert_array_equal(p_prob[0], [1.0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(p_traj[1], traj[1])
    # SHAM leaves the stopped agent alone and substitutes the moving one instead
    np.testing.assert_array_equal(s_traj[0], traj[0])
    np.testing.assert_allclose(s_traj[1, 0, :, 0], 20.0 + 8.0 * 0.1 * np.arange(1, 61))
    np.testing.assert_array_equal(s_prob[1], [1.0, 0, 0, 0, 0, 0])

