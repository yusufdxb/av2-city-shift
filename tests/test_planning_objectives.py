import numpy as np
import pytest

from cityshift import closedloop as base
from cityshift import closedloop_v4 as v4
from cityshift.planning_objectives import box_clearance, choose_candidate, complete_future, departure_label, plan, replay_metrics
from cityshift.preprocess import OBJECT_TYPES
from cityshift.scene import Scene


def fixture_scene(obstacle_y=0.0):
    pos = np.zeros((2, 110, 2))
    pos[0, :, 0] = (np.arange(110) - 49) * 0.5
    pos[1, :, 0] = 14
    pos[1, :, 1] = obstacle_y
    vel = np.zeros_like(pos)
    vel[0, :, 0] = 5
    return Scene(
        "fixture",
        "austin",
        ["AV", "obstacle"],
        np.array([OBJECT_TYPES.index("vehicle")] * 2),
        pos,
        vel,
        np.zeros((2, 110)),
        np.ones((2, 110), bool),
        1,
        0,
        np.zeros((0, 20, 2)),
        np.zeros((0, 2), int),
        np.zeros((0, 2)),
        np.zeros(0, int),
    )


def test_objectives_are_structurally_distinct():
    risk = np.zeros(len(base.ACCELS))
    progress = np.linspace(0, 1, len(risk))
    risk[-1] = 0.04
    feasible = np.ones_like(risk, bool)
    weighted, _ = choose_candidate(risk, progress, feasible, "weighted")
    budget, _ = choose_candidate(risk, progress, feasible, "risk_budget")
    assert weighted != budget and budget == len(risk) - 1


def test_budget_fallback_prefers_least_risk_and_respects_stop_feasibility():
    risk = np.linspace(0.1, 0.9, len(base.ACCELS))
    progress = np.linspace(0, 1, len(risk))
    feasible = np.ones_like(risk, bool)
    feasible[0] = False
    candidate, diagnostic = choose_candidate(risk, progress, feasible, "risk_budget")
    assert candidate == 1 and diagnostic["risk_budget_fallback"]
    candidate, diagnostic = choose_candidate(risk, progress, np.zeros_like(feasible), "risk_budget")
    assert candidate == 0 and diagnostic["stopline_infeasible"]


def test_clearance_is_geometric_and_counts_rear_and_lateral_contacts():
    assert box_clearance(np.zeros(2), 0, 4, 2, np.array([6.0, 0]), 0, 4, 2) == pytest.approx(2)
    assert box_clearance(np.zeros(2), 0, 4, 2, np.array([0.0, 4]), 0, 4, 2) == pytest.approx(2)
    assert box_clearance(np.zeros(2), 0, 4, 2, np.array([5.0, 3]), 0, 4, 2) == pytest.approx(np.sqrt(2))
    assert box_clearance(np.zeros(2), 0, 4, 2, np.array([-1.0, 0]), 0, 4, 2) == 0
    assert box_clearance(np.zeros(2), 0, 4, 0.5, np.array([1.5, -1.5]), np.pi / 4, 4, 0.5) >= 0


def test_incomplete_futures_never_get_negative_departure_labels():
    sc = fixture_scene()
    sc.valid[1, 80] = False
    assert not complete_future(sc, [1], 49)
    assert departure_label(sc, 1, 49) is None
    assert departure_label(sc, 1, 79) is None


@pytest.mark.parametrize("objective", ["weighted", "risk_budget"])
def test_known_obstacle_control_good_forecasts_prevent_blind_contact(objective):
    from cityshift.closedloop_v2 import init_ego

    sc = fixture_scene()
    good, blind = init_ego(sc), init_ego(sc)
    for t in base.REPLANS:
        forecast = np.repeat(sc.pos[1, t][None, None, None], 60, axis=2)
        candidate, _ = plan(sc, good, t, [1], forecast, np.ones((1, 1)), objective)
        base.execute(good, t, candidate)
        candidate, _ = plan(sc, blind, t, [], np.zeros((0, 1, 60, 2)), np.zeros((0, 1)), objective)
        base.execute(blind, t, candidate)
    assert not replay_metrics(sc, good)["observed_contact"]
    assert replay_metrics(sc, blind)["observed_contact"]


def test_weighted_objective_matches_frozen_v4_without_installing_globals():
    from cityshift.closedloop_v2 import init_ego

    sc = fixture_scene(4)
    ego = init_ego(sc)
    rng = np.random.default_rng(12)
    forecast = np.repeat(sc.pos[1, 49][None, None, None], 60, axis=2) + rng.normal(0, 1, (1, 1, 60, 2))
    before = base.plan
    candidate, _ = plan(sc, ego, 49, [1], forecast, np.ones((1, 1)), "weighted")
    assert candidate == v4.plan(sc, ego, 49, [1], forecast, np.ones((1, 1)))
    assert base.plan is before
