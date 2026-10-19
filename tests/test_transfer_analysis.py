"""Scientific safeguards for paired transfer inference on small synthetic cohorts."""

import json

import numpy as np
import pandas as pd
import pytest

from cityshift.transfer_analysis import (CITIES, LEARNED, POLICIES, analyze_rollouts, clopper_pearson_upper,
                                         harmful_discordance_upper)


def rollouts(city_sizes=None):
    city_sizes = city_sizes or {city: 2 for city in CITIES}
    rows = []
    for city, size in city_sizes.items():
        for i in range(size):
            for objective in ("weighted", "risk_budget"):
                for policy in POLICIES:
                    for seed in (range(3) if policy in LEARNED else [-1]):
                        rows.append({"scenario_id": f"{city}-{i}", "city": city, "objective": objective,
                                     "policy": policy, "seed": seed, "hard_brake": False,
                                     "excessive_brake": False, "genuine_conflict": False,
                                     "complete_decisions": 3, "eligible_decisions": 3,
                                     "observed_contact": False, "observed_near_conflict": False,
                                     "min_clearance_m": 5., "progress_fraction": 1., "progress_m": 20.,
                                     "mean_abs_jerk_mps3": 0.})
    return pd.DataFrame(rows)


def test_zero_events_do_not_establish_contact_noninferiority():
    result = analyze_rollouts(rollouts(), bootstrap_samples=30, departure_scenes=100)
    cell = result["objectives"]["weighted"]
    contact = cell["comparisons"]["SELECTIVE"]["NATIVE"]["observed_contact"]
    assert contact["ci98_75"] == [0., 0.]
    assert contact["harm_upper"] > .003
    assert not cell["joint"]["SELECTIVE"]["planning_gates"]["contact_harm_below_0_003"]
    assert not cell["joint"]["SELECTIVE"]["joint4"]
    assert result["full_test_gate"].startswith("stop")
    json.dumps(result, allow_nan=False)


def test_exact_bound_matches_zero_event_formula_and_is_monotone():
    assert clopper_pearson_upper(0, 10, .05) == pytest.approx(1 - .05 ** .1)
    assert clopper_pearson_upper(2, 10, .05) == pytest.approx(.5069013011)
    assert clopper_pearson_upper(3, 10, .05) > clopper_pearson_upper(2, 10, .05)
    assert clopper_pearson_upper(10, 10) == 1
    with pytest.raises(ValueError):
        clopper_pearson_upper(0, 0)


def test_city_equal_estimator_averages_training_seeds_before_cities():
    data = rollouts({"austin": 1, "miami": 3})
    mask = (data.city == "austin") & (data.policy == "NATIVE") & (data.seed == 0)
    data.loc[mask, ["hard_brake", "excessive_brake"]] = True
    result = analyze_rollouts(data, bootstrap_samples=20)
    rates = result["objectives"]["weighted"]["rates"]
    assert rates["NATIVE"]["excessive_brake"]["city_equal_mean"] == pytest.approx(1 / 6)
    effect = result["objectives"]["weighted"]["comparisons"]["SELECTIVE"]["NATIVE"]["excessive_brake"]
    assert effect["point"] == pytest.approx(-1 / 6)
    assert effect["ci98_75"] == pytest.approx([-1 / 6, -1 / 6])


def test_paired_improvement_and_progress_loss_have_correct_directions():
    data = rollouts()
    data.loc[data.policy == "NATIVE", ["hard_brake", "excessive_brake"]] = True
    data.loc[data.policy == "SELECTIVE", "progress_fraction"] = .97
    cell = analyze_rollouts(data, bootstrap_samples=20)["objectives"]["weighted"]
    effects = cell["comparisons"]["SELECTIVE"]["NATIVE"]
    assert effects["excessive_brake"]["ci98_75"] == pytest.approx([-1., -1.])
    assert effects["progress_loss"]["point"] == pytest.approx(.03)
    assert cell["joint"]["SELECTIVE"]["planning_gates"]["less_excessive_braking"]
    assert not cell["joint"]["SELECTIVE"]["planning_gates"]["progress_loss_below_0_02"]


@pytest.mark.parametrize("failure", ["missing_seed", "missing_scene", "duplicate", "different_support"])
def test_incomplete_or_unpaired_cohorts_are_rejected(failure):
    data = rollouts()
    if failure == "missing_seed":
        data = data[~((data.policy == "SELECTIVE") & (data.seed == 2))]
    elif failure == "missing_scene":
        data = data[~((data.policy == "PATCH") & (data.scenario_id == "austin-0"))]
    elif failure == "duplicate":
        data = pd.concat([data, data.iloc[:1]])
    else:
        data.loc[0, "progress_fraction"] = np.nan
    with pytest.raises(ValueError):
        analyze_rollouts(data, bootstrap_samples=5)


def test_common_missing_geometry_is_reported_without_nan_json():
    data = rollouts()
    data.loc[data.scenario_id == "austin-0", ["min_clearance_m", "progress_fraction"]] = np.nan
    result = analyze_rollouts(data, bootstrap_samples=10)
    rates = result["objectives"]["weighted"]["rates"]["NATIVE"]
    assert rates["progress_fraction"]["finite_scenarios_by_city"]["austin"] == 1
    json.dumps(result, allow_nan=False)


def test_conservative_exact_bound_weights_cities_equally():
    cities = np.array(["austin", *("miami" for _ in range(30))])
    bound = harmful_discordance_upper(np.zeros(len(cities), bool), cities)
    expected = np.mean([clopper_pearson_upper(0, 1, .00625 / 2), clopper_pearson_upper(0, 30, .00625 / 2)])
    assert bound["upper"] == pytest.approx(expected)


def test_pilot_controls_can_pass_without_claiming_joint_success():
    data = rollouts({city: 5 for city in CITIES})
    data.loc[data.policy == "NATIVE", ["hard_brake", "excessive_brake"]] = True
    data.loc[data.policy == "blind", "observed_contact"] = True
    result = analyze_rollouts(data, bootstrap_samples=15, departure_scenes=25, phase="pilot")
    assert result["feasibility_passed_both_objectives"]
    assert result["full_test_gate"] == "proceed"
    cell = result["objectives"]["weighted"]
    assert not cell["final_power"]["passed"]
    assert cell["joint"]["SELECTIVE"]["status"] == "development_only"
    assert cell["joint"]["SELECTIVE"]["joint_pending_departure"]


def test_complete_decision_support_cannot_differ_across_policies():
    data = rollouts()
    data.loc[0, "complete_decisions"] = 2
    with pytest.raises(ValueError, match="common"):
        analyze_rollouts(data, bootstrap_samples=5)


def test_no_complete_decisions_remain_unknown_on_common_support():
    data = rollouts()
    data = data.astype({"excessive_brake": float, "genuine_conflict": float})
    mask = data.scenario_id == "austin-0"
    data.loc[mask, "complete_decisions"] = 0
    data.loc[mask, ["excessive_brake", "genuine_conflict"]] = None
    result = analyze_rollouts(data, bootstrap_samples=10)
    cell = result["objectives"]["weighted"]
    assert cell["rates"]["NATIVE"]["excessive_brake"]["finite_scenarios_by_city"]["austin"] == 1
    assert cell["feasibility"]["native_brake_scored_scenes"] == 11
    json.dumps(result, allow_nan=False)
    data.loc[mask, "excessive_brake"] = 0.
    with pytest.raises(ValueError, match="unknown exactly"):
        analyze_rollouts(data, bootstrap_samples=5)


def test_planner_hard_brake_alias_is_supported():
    data = rollouts().rename(columns={"hard_brake": "planner_hard_brake"})
    assert analyze_rollouts(data, bootstrap_samples=5)["objectives"]["weighted"]["brake_metric"] == "excessive_brake"
