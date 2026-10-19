import numpy as np
import pytest

from cityshift import closedloop as base
from cityshift.planner_transfer import common_support_metrics, replay_commands
from test_planning_objectives import fixture_scene


def record(objective="weighted", policy="NATIVE", agents=None, commands=None):
    return {
        "objective": objective,
        "policy": policy,
        "seed": 0,
        "agents": agents or [[1]] * 6,
        "accels": commands or [0.0] * 6,
        "diagnostics": [{"stopline_infeasible": False, "risk_budget_fallback": False, "chosen_forecast_risk": 0.0}] * 6,
    }


def test_replay_requires_registered_full_command_sequence():
    scene = fixture_scene()
    with pytest.raises(ValueError):
        replay_commands(scene, [0.0])
    with pytest.raises(ValueError):
        replay_commands(scene, [0.123] * 6)
    ego = replay_commands(scene, [0.0] * 6)
    assert ego.s[109] == pytest.approx(30)


def test_missing_future_censors_union_for_every_policy():
    scene = fixture_scene()
    scene.valid[1, 80] = False
    records = [record(), record(policy="PATCH", agents=[[] for _ in base.REPLANS])]
    scored = common_support_metrics(scene, records)
    assert len(scored) == 2
    for row in scored:
        assert row["complete_decisions"] == 0
        assert row["eligible_decisions"] == 3
        assert row["genuine_conflict"] is None
        assert row["excessive_brake"] is None
        assert row["common_actor_counts"] == [1, 1, 1]
        # Missing future labels never erase observed actual contact elsewhere.
        assert row["observed_contact"]


def test_common_support_counts_observed_contact_and_safe_alternative():
    scene = fixture_scene(obstacle_y=10)
    result = common_support_metrics(scene, [record(commands=[-6.0, 0, 0, 0, 0, 0])])[0]
    assert result["complete_decisions"] == 3
    assert result["excessive_brake"]
    assert result["hard_brake"] and result["commanded_hard_brake"]
    assert not result["genuine_conflict"]
    assert not result["observed_contact"]
    assert result["progress_m"] < 30
    assert all(result["decision_nonhard_safe"])


def test_zero_speed_command_is_not_realized_hard_braking():
    scene = fixture_scene(obstacle_y=10)
    scene.vel[scene.av, 49] = 0
    result = common_support_metrics(scene, [record(commands=[-8.0] * 6)])[0]
    assert result["commanded_hard_brake"]
    assert not result["hard_brake"]
    assert not result["excessive_brake"]
    assert np.isfinite(result["min_clearance_m"])
