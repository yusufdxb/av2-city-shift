"""Synthetic transfer controls for geometry and implementation, not natural-scene safety.

All controls keep the registered candidate accelerations, costs and 0.05 risk
budget fixed. Their geometry is constructed to make forecast information matter;
passing them cannot establish a natural-scene effect or reactive driving safety.
"""

import numpy as np
import pytest

from cityshift import closedloop as base
from cityshift import closedloop_v2 as v2
from cityshift import closedloop_v4 as v4
from cityshift.planning_objectives import (
    candidate_risks,
    complete_future,
    departure_label,
    plan,
    replay_metrics,
    true_candidate_contacts,
)
from cityshift.preprocess import OBJECT_TYPES
from cityshift.scene import Scene


OBJECTIVES = ("weighted", "risk_budget")


def control_scene(departing=False):
    """Ego: (0, 0), 5 m/s east, 30 m logged route after handoff.

    Actor: vehicle at (18, -8), facing north, stopped at handoff. The
    departing version waits 0.5 s, accelerates north at 3 m/s^2 to 4 m/s,
    then cruises. It crosses y=0 at 3 1/6 s, within the complete 4 s future.
    The stationary version remains eight meters lateral to the ego route.
    """
    seconds = (np.arange(base.END + 1) - base.HANDOFF) * base.DT
    moving_seconds = np.maximum(seconds - 0.5, 0)
    ramp_seconds = 4 / 3
    travel = np.where(
        moving_seconds <= ramp_seconds,
        1.5 * moving_seconds**2,
        1.5 * ramp_seconds**2 + 4 * (moving_seconds - ramp_seconds),
    )
    position = np.zeros((2, base.END + 1, 2))
    position[0, :, 0] = 5 * seconds
    position[1, :, 0] = 18
    position[1, :, 1] = -8 + travel if departing else -8
    velocity = np.zeros_like(position)
    velocity[0, :, 0] = 5
    if departing:
        velocity[1, :, 1] = np.minimum(3 * moving_seconds, 4)
    heading = np.zeros((2, base.END + 1))
    heading[1] = np.pi / 2
    return Scene(
        scenario_id="synthetic-transfer-control",
        city="austin",
        track_ids=["AV", "actor"],
        types=np.array([OBJECT_TYPES.index("vehicle")] * 2),
        pos=position,
        vel=velocity,
        head=heading,
        valid=np.ones((2, base.END + 1), bool),
        focal=1,
        av=0,
        poly_res=np.zeros((0, 20, 2)),
        poly_attr=np.zeros((0, 2), int),
        poly_orig=np.zeros((0, 2)),
        poly_start=np.zeros(0, int),
    )


def forecast(sc, t, source):
    if source == "ignored":
        return [], np.zeros((0, 1, base.PLAN_STEPS, 2)), np.zeros((0, 1))
    if source == "oracle":
        trajectory = base.oracle_forecast(sc, 1, t)
    elif source == "stationary":
        trajectory = base.static_forecast(sc, 1, t)
    elif source == "false_alarm":
        # A false departure: the truly parked actor is predicted to reach the
        # route in two seconds and remain there, creating a spurious obstacle.
        trajectory = base.static_forecast(sc, 1, t)
        trajectory[:, 1] = np.minimum(-8 + 4 * base.DT * np.arange(1, base.PLAN_STEPS + 1), 0)
    else:
        raise ValueError(source)
    return [1], trajectory[None, None], np.ones((1, 1))


def rollout(sc, objective, source):
    ego = v2.init_ego(sc)
    diagnostics = []
    for t in base.REPLANS:
        agents, trajectory, probability = forecast(sc, t, source)
        candidate, diagnostic = plan(sc, ego, t, agents, trajectory, probability, objective, risk_budget=0.05)
        assert v4.feasible(ego, t)[candidate]
        base.execute(ego, t, candidate)
        diagnostics.append(diagnostic)
    return ego, replay_metrics(sc, ego), diagnostics


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("blind_source", ("stationary", "ignored"))
def test_stopped_departure_oracle_avoids_contact_that_blind_forecasts_make(objective, blind_source):
    sc = control_scene(departing=True)
    t = base.HANDOFF
    assert np.linalg.norm(sc.vel[1, t]) == 0
    assert complete_future(sc, [1], t)
    assert departure_label(sc, 1, t) is True
    assert sc.pos[1, t, 1] < 0 < sc.pos[1, t + base.CHECK_STEPS, 1]

    ego = v2.init_ego(sc)
    true_agents, true_forecast, probability = forecast(sc, t, "oracle")
    still_agents, still_forecast, still_probability = forecast(sc, t, "stationary")
    oracle_risk, _ = candidate_risks(sc, ego, t, true_agents, true_forecast, probability)
    still_risk, _ = candidate_risks(sc, ego, t, still_agents, still_forecast, still_probability)
    coast = int(np.flatnonzero(base.ACCELS == 0)[0])
    assert oracle_risk[coast] == 1
    assert still_risk[coast] == 0
    assert true_candidate_contacts(sc, ego, t, [1])[coast]

    informed_ego, informed, informed_diagnostics = rollout(sc, objective, "oracle")
    blind_ego, blind, _ = rollout(sc, objective, blind_source)
    assert informed_ego.chosen[0] < blind_ego.chosen[0]
    assert blind_ego.chosen[0] == 0
    assert not informed["observed_contact"]
    assert not informed["observed_departing_contact"]
    assert informed["min_clearance_m"] > 0.5
    assert blind["observed_contact"]
    assert blind["observed_departing_contact"]
    assert blind["min_clearance_m"] == 0
    assert informed["observed_actor_steps"] == blind["observed_actor_steps"] == base.END - t
    assert not any(d["risk_budget_fallback"] or d["stopline_infeasible"] for d in informed_diagnostics)


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_false_departure_forecast_sacrifices_progress_without_true_contact(objective):
    sc = control_scene()
    assert departure_label(sc, 1, base.HANDOFF) is False
    assert not true_candidate_contacts(sc, v2.init_ego(sc), base.HANDOFF, [1]).any()
    informed_ego, informed, _ = rollout(sc, objective, "oracle")
    misled_ego, misled, _ = rollout(sc, objective, "false_alarm")
    assert informed_ego.chosen[0] == 0
    assert misled_ego.chosen[0] < 0
    assert informed["progress_m"] - misled["progress_m"] > 5
    assert not informed["observed_contact"] and not misled["observed_contact"]
    assert not informed["observed_near_conflict"] and not misled["observed_near_conflict"]
    assert informed["min_clearance_m"] > 1 and misled["min_clearance_m"] > 1


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_no_risk_plans_make_progress_and_respect_stopline(objective):
    sc = control_scene()
    ego, metrics, diagnostics = rollout(sc, objective, "ignored")
    assert metrics["progress_fraction"] > 0.9
    assert (ego.s[base.HANDOFF :] <= ego.path.length_logged + 1e-6).all()
    assert (ego.v[base.HANDOFF :] >= 0).all()
    assert metrics["route_overshoot_m"] <= 1e-6
    assert not metrics["planner_hard_brake"]
    assert not metrics["observed_contact"]
    assert all(d["chosen_forecast_risk"] == 0 for d in diagnostics)
    assert not any(d["risk_budget_fallback"] or d["stopline_infeasible"] for d in diagnostics)


@pytest.mark.parametrize("offset", ((-3, 0), (-0.25, 1.5), (-0.25, -1.5)), ids=("rear", "left-side", "right-side"))
def test_observed_contacts_include_rear_and_side_outside_front_risk_scope(offset):
    sc = control_scene()
    # Co-moving overlapping boxes isolate directional scoring. These are
    # deliberately intersecting geometry fixtures, not plausible initial states.
    sc.pos[1] = sc.pos[0] + np.asarray(offset)
    sc.vel[1] = sc.vel[0]
    sc.head[1] = 0
    ego = v2.init_ego(sc)
    steps = np.arange(base.HANDOFF, base.END + 1)
    ego.s[steps] = 5 * (steps - base.HANDOFF) * base.DT
    ego.v[steps] = 5
    ego.chosen = [0.0] * len(base.REPLANS)
    result = replay_metrics(sc, ego)
    assert result["observed_contact"]
    assert result["observed_near_conflict"]
    assert result["min_clearance_m"] == 0

    initial = v2.init_ego(sc)
    coast = int(np.flatnonzero(base.ACCELS == 0)[0])
    agents, trajectory, probability = forecast(sc, base.HANDOFF, "oracle")
    risk, _ = candidate_risks(sc, initial, base.HANDOFF, agents, trajectory, probability)
    assert true_candidate_contacts(sc, initial, base.HANDOFF, [1])[coast]
    # The inherited planning kernel filters to in-front centers. The new
    # observed-contact measure intentionally does not inherit that restriction.
    assert risk[coast] == 0


def test_missing_departure_future_is_unknown_and_not_scored_as_observed_clearance():
    sc = control_scene(departing=True)
    sc.valid[1, base.HANDOFF + 1 :] = False
    assert not complete_future(sc, [1], base.HANDOFF)
    assert departure_label(sc, 1, base.HANDOFF) is None
    with pytest.raises(ValueError, match="complete logged futures"):
        true_candidate_contacts(sc, v2.init_ego(sc), base.HANDOFF, [1])
    _, metrics, _ = rollout(sc, "weighted", "ignored")
    assert metrics["observed_actor_steps"] == 0
    assert np.isnan(metrics["min_clearance_m"])
    # This flag reports only observed contacts; the zero sample count above
    # prevents reading its value as evidence about the missing actor future.
    assert not metrics["observed_contact"]
