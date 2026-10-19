"""Offline invariants for departure calibration and geometry-preserving correction."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from cityshift.departure_calibration import (
    CALIBRATOR_FEATURE_NAMES,
    HISTORY_FEATURE_NAMES,
    correct_predictions,
    fit_calibrator,
    history_features,
    native_departure_mass,
    predict_departure,
)


def scene():
    pos = np.zeros((3, 110, 2))
    pos[1, :, 0] = 10
    pos[2, :, 1] = np.arange(110) * 0.2
    vel = np.zeros_like(pos)
    vel[2, :, 1] = 2.0
    return SimpleNamespace(pos=pos, vel=vel, valid=np.ones((3, 110), bool))


def forecasts(sc, t=49):
    traj = np.repeat(sc.pos[:, t, None, None], 3, axis=1)
    traj = np.repeat(traj, 60, axis=2)
    traj[:, 1, :, 0] += np.arange(1, 61) * 0.2
    traj[:, 2, :, 1] += np.arange(1, 61) * 0.3
    return traj, np.tile([0.1, 0.3, 0.6], (3, 1))


def calibrator(probability=0.08):
    features = np.zeros((8, len(HISTORY_FEATURE_NAMES)))
    model = fit_calibrator(features, np.linspace(0.1, 0.9, 8), np.array([0, 0, 0, 0, 1, 1, 1, 1]),
                           [f"scenario-{i}" for i in range(8)])
    model["coefficients"] = [0.0] * len(CALIBRATOR_FEATURE_NAMES)
    model["intercept"] = float(np.log(probability / (1 - probability)))
    return model


def test_future_mutation_cannot_change_features_or_correction():
    sc = scene()
    traj, prob = forecasts(sc)
    before = history_features(sc, [0, 1, 2], 49)
    output = correct_predictions(sc, [0, 1, 2], 49, traj, prob, calibrator())
    sc.pos[:, 50:] = np.nan
    sc.vel[:, 50:] = np.inf
    sc.valid[:, 50:] = False
    np.testing.assert_array_equal(history_features(sc, [0, 1, 2], 49), before)
    after = correct_predictions(sc, [0, 1, 2], 49, traj, prob, calibrator())
    np.testing.assert_array_equal(after[0], output[0])
    np.testing.assert_array_equal(after[1], output[1])
    assert after[2] == output[2]


def test_invalid_past_is_masked_and_history_has_no_pose_dependence():
    sc = scene()
    sc.valid[0, 45:48] = False
    baseline = history_features(sc, [0], 49)
    sc.pos[0, 45:48] = np.nan
    sc.vel[0, 45:48] = np.inf
    np.testing.assert_array_equal(history_features(sc, [0], 49), baseline)
    rotated = copy.deepcopy(sc)
    with np.errstate(invalid="ignore"):
        rotated.pos = sc.pos @ np.array([[0, -1], [1, 0]]) + 123
        rotated.vel = sc.vel @ np.array([[0, -1], [1, 0]])
    np.testing.assert_allclose(history_features(rotated, [0], 49), baseline)
    assert baseline[0, HISTORY_FEATURE_NAMES.index("consecutive_stopped_s")] == pytest.approx(0.2)


def test_native_mass_uses_maximum_displacement_and_complete_horizon():
    traj = np.zeros((1, 2, 60, 2))
    traj[0, 1, 8, 0] = 2.1  # departs and returns before the endpoint
    traj[0, 0, 45, 0] = 10  # outside the 4 s event
    assert native_departure_mass(traj, [[0.7, 0.3]], [[0, 0]])[0] == pytest.approx(0.3)
    with pytest.raises(ValueError, match="T>=horizon"):
        native_departure_mass(traj[:, :, :39], [[0.7, 0.3]], [[0, 0]])


def test_preserves_native_geometry_prob_ratios_and_moving_agents():
    sc = scene()
    traj, prob = forecasts(sc)
    original_traj, original_prob = traj.copy(), prob.copy()
    new_traj, new_prob, metadata = correct_predictions(sc, [0, 1, 2], 49, traj, prob, calibrator())
    np.testing.assert_array_equal(new_traj[:, :3], original_traj)
    np.testing.assert_array_equal(new_prob[2, :3], original_prob[2])
    assert new_prob[2, -1] == 0
    np.testing.assert_array_equal(traj, original_traj)
    np.testing.assert_array_equal(prob, original_prob)
    assert new_prob[0, 1] / new_prob[0, 2] == pytest.approx(0.5)
    assert metadata["realized_departure_mass"][0] == pytest.approx(0.08)
    assert metadata["corrected"] == [True, True, False]
    np.testing.assert_allclose(new_prob.sum(1), 1)


def test_mode_permutation_equivariance():
    sc = scene()
    traj, prob = forecasts(sc)
    permutation = [2, 0, 1]
    first = correct_predictions(sc, [0, 1, 2], 49, traj, prob, calibrator())
    second = correct_predictions(sc, [0, 1, 2], 49, traj[:, permutation], prob[:, permutation], calibrator())
    np.testing.assert_allclose(second[0][:, :3], first[0][:, permutation])
    np.testing.assert_allclose(second[1][:, :3], first[1][:, permutation])
    np.testing.assert_allclose(second[1][:, -1], first[1][:, -1])


@pytest.mark.parametrize("prediction,native", [(0.3, [0.1, 0.3, 0.6]), (0.15, [0.9, 0.03, 0.07])])
def test_selective_preserves_high_departure_or_upward_correction(prediction, native):
    sc = scene()
    traj, prob = forecasts(sc)
    prob[:] = native
    new_traj, new_prob, metadata = correct_predictions(sc, [0, 1, 2], 49, traj, prob, calibrator(prediction),
                                                      selective=True, threshold=0.2)
    np.testing.assert_array_equal(new_traj[:, :3], traj)
    np.testing.assert_array_equal(new_prob[:, :3], prob)
    assert not new_prob[:, -1].any()
    assert metadata["n_corrected"] == 0
    assert metadata["n_selective_retained"] == 2


def test_selective_downward_low_departure_correction_applies():
    sc = scene()
    traj, prob = forecasts(sc)
    _, new_prob, metadata = correct_predictions(sc, [0, 1, 2], 49, traj, prob, calibrator(0.1),
                                              selective=True, threshold=0.2)
    assert new_prob[0, 1:].sum() == pytest.approx(0.1)
    assert metadata["n_corrected"] == 2


def test_missing_stationary_support_adds_cv_but_retains_moving_mass():
    sc = scene()
    traj, prob = forecasts(sc)
    traj = traj[:, 1:]
    prob = np.tile([0.25, 0.75], (3, 1))
    new_traj, new_prob, metadata = correct_predictions(sc, [0, 1, 2], 49, traj, prob, calibrator(1e-12))
    np.testing.assert_array_equal(new_traj[:, :2], traj)
    np.testing.assert_allclose(new_traj[0, -1], np.repeat(sc.pos[0, 49][None], 60, axis=0))
    assert new_prob[0, :2].sum() == pytest.approx(0.02)
    assert new_prob[0, -1] == pytest.approx(0.98)
    assert metadata["cv_admitted"] == [True, True, False]
    assert metadata["n_cv_admitted"] == 2


def test_zero_native_moving_support_cannot_invent_departure_prediction():
    sc = scene()
    traj, prob = forecasts(sc)
    prob[:] = [1.0, 0.0, 0.0]
    _, new_prob, metadata = correct_predictions(sc, [0, 1, 2], 49, traj, prob, calibrator(0.8))
    np.testing.assert_array_equal(new_prob[:, :3], prob)
    assert metadata["unrepresentable_departure_target"] == [True, True, False]
    assert not new_prob[:, -1].any()


def test_fit_is_deterministic_json_safe_and_predicts_useful_signal():
    rng = np.random.default_rng(19)
    features = rng.normal(size=(160, len(HISTORY_FEATURE_NAMES)))
    labels = (features[:, 0] + 0.2 * rng.normal(size=160) > 0).astype(float)
    native = np.full(160, 0.8)
    ids = [f"scene-{i // 4}" for i in range(160)]
    model = fit_calibrator(features, native, labels, ids)
    assert model == fit_calibrator(features, native, labels, ids)
    restored = json.loads(json.dumps(model, allow_nan=False))
    predicted = predict_departure(restored, features, native)
    np.testing.assert_array_equal(predicted, predict_departure(model, features, native))
    assert np.mean((predicted - labels) ** 2) < np.mean((native - labels) ** 2) / 2
    assert model["n_fit_scenarios"] == 40
    assert model["converged"]


def test_scenario_equal_weighting_prevents_repeated_agent_domination():
    # Constant features isolate the unpenalized intercept and weighting policy.
    features = np.zeros((12, len(HISTORY_FEATURE_NAMES)))
    mass = np.full(12, 0.5)
    labels = np.array([0] * 10 + [1, 1])
    model = fit_calibrator(features, mass, labels, ["many"] * 10 + ["one", "two"])
    assert predict_departure(model, features[:1], mass[:1])[0] == pytest.approx(2 / 3)


@pytest.mark.parametrize("mutation", ["missing", "nan", "zero_scale", "names", "event", "not_converged", "shape"])
def test_malformed_calibrator_rejected(mutation):
    model = calibrator()
    if mutation == "missing":
        del model["intercept"]
    elif mutation == "nan":
        model["coefficients"][0] = float("nan")
    elif mutation == "zero_scale":
        model["scale"][0] = 0
    elif mutation == "names":
        model["feature_names"][0] = "future_departure"
    elif mutation == "event":
        model["event"]["horizon_steps"] = 20
    elif mutation == "not_converged":
        model["converged"] = False
    else:
        model["coefficients"] = [0.0]
    with pytest.raises(ValueError):
        predict_departure(model, np.zeros((1, len(HISTORY_FEATURE_NAMES))), [0.5])


@pytest.mark.parametrize("mutation", ["not_observed", "nonfinite_current", "wrong_t", "wrong_agent", "duplicate_agent"])
def test_invalid_current_state_rejected(mutation):
    sc = scene()
    agents, t = [0], 49
    if mutation == "not_observed":
        sc.valid[0, 49] = False
    elif mutation == "nonfinite_current":
        sc.vel[0, 49, 0] = np.nan
    elif mutation == "wrong_t":
        t = 110
    elif mutation == "wrong_agent":
        agents = [4]
    else:
        agents = [0, 0]
    with pytest.raises(ValueError):
        history_features(sc, agents, t)


@pytest.mark.parametrize("case", ["single_class", "nonbinary", "nan", "ids", "ridge", "features", "mass"])
def test_degenerate_or_invalid_fit_rejected(case):
    features = np.zeros((4, len(HISTORY_FEATURE_NAMES)))
    mass = np.full(4, 0.5)
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    ids, ridge = ["a", "b", "c", "d"], 1.0
    if case == "single_class":
        labels[:] = 0
    elif case == "nonbinary":
        labels[0] = 0.2
    elif case == "nan":
        labels[0] = np.nan
    elif case == "ids":
        ids = ["a"]
    elif case == "ridge":
        ridge = 0
    elif case == "features":
        features = features[:, :-1]
    else:
        mass[0] = -0.1
    with pytest.raises(ValueError):
        fit_calibrator(features, mass, labels, ids, ridge=ridge)


@pytest.mark.parametrize("bad_prob", [[[0, 0, 0]] * 3, [[-0.1, 0.5, 0.6]] * 3, [[0.1, 0.2, np.nan]] * 3,
                                      [[0.1, 0.2, 0.3]] * 3])
def test_invalid_probability_rejected(bad_prob):
    sc = scene()
    traj, _ = forecasts(sc)
    with pytest.raises(ValueError):
        correct_predictions(sc, [0, 1, 2], 49, traj, bad_prob, calibrator())


def test_empty_agents_supported():
    sc = scene()
    traj = np.zeros((0, 3, 60, 2))
    prob = np.zeros((0, 3))
    output = correct_predictions(sc, [], 49, traj, prob, calibrator())
    assert output[0].shape == (0, 4, 60, 2)
    assert output[1].shape == (0, 4)
    assert output[2]["n_corrected"] == 0
