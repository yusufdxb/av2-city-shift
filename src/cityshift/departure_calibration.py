"""History-only departure calibration that retains native moving-mode support.

The event is displacement greater than 2 m at *any* point in the first 4 s.
Labels belong to the offline fitting caller. Prediction and correction never
read a scene after ``t``. This is a replay research intervention, not a safety
guarantee or evidence that an unobserved departure will be forecast correctly.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .scene import Scene

DT = 0.1
HORIZON = 40
DISTANCE_M = 2.0
STOPPED_MPS = 0.5
HISTORY_FEATURE_NAMES = (
    "speed_mps",
    "mean_speed_1s_mps",
    "max_speed_2s_mps",
    "displacement_1s_m",
    "displacement_2s_m",
    "stopped_fraction_2s",
    "observed_fraction_2s",
    "consecutive_stopped_s",
    "recent_acceleration_mps2",
    "history_span_s",
)
CALIBRATOR_FEATURE_NAMES = (*HISTORY_FEATURE_NAMES, "native_departure_logit")
SCHEMA_VERSION = 1


def _integer(value, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite_array(value, name: str) -> np.ndarray:
    try:
        out = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out


def _observations(scene: Scene, agents, t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and return observations only through the requested time step."""
    t = _integer(t, "t")
    raw_agents = np.asarray(agents)
    if raw_agents.ndim != 1 or (raw_agents.size and raw_agents.dtype.kind not in "iu"):
        raise ValueError("agents must be a one-dimensional sequence of integer indices")
    indices = raw_agents.astype(np.int64)
    if len(set(indices.tolist())) != len(indices):
        raise ValueError("agents must not contain duplicate indices")
    # Shape checks inspect metadata, not future observations or validity values.
    pos, vel, valid = np.asarray(scene.pos), np.asarray(scene.vel), np.asarray(scene.valid)
    if pos.ndim != 3 or pos.shape[-1] != 2 or vel.shape != pos.shape or valid.shape != pos.shape[:2]:
        raise ValueError("scene positions, velocities, and validity shapes disagree")
    if valid.dtype.kind != "b":
        raise ValueError("scene validity must be boolean")
    if t >= pos.shape[1] or (indices < 0).any() or (indices >= pos.shape[0]).any():
        raise ValueError("time or agent index outside scene")
    p, v, mask = pos[indices, : t + 1], vel[indices, : t + 1], valid[indices, : t + 1]
    if not mask[:, -1].all():
        raise ValueError("each selected agent must be observed at t")
    if not np.isfinite(p[mask]).all() or not np.isfinite(v[mask]).all():
        raise ValueError("observed positions and velocities must be finite")
    return indices, p, v, mask


def history_features(scene: Scene, agents, t: int) -> np.ndarray:
    """Return [N, 10] scalar features from observed positions/velocities <= t.

    Windows include the current observation. Missing earlier samples contribute
    to observation coverage but not to motion statistics. Consecutive stop time
    ends at the first missing or moving sample and is capped at 5 s.
    """
    _, pos, vel, valid = _observations(scene, agents, t)
    out = np.empty((len(pos), len(HISTORY_FEATURE_NAMES)), dtype=np.float64)
    for j in range(len(pos)):
        steps = np.flatnonzero(valid[j])
        speeds = np.linalg.norm(vel[j, steps], axis=-1)
        current_speed = float(speeds[-1])
        recent1 = steps >= max(0, t - 10)
        recent2 = steps >= max(0, t - 20)
        start1, start2 = steps[recent1][0], steps[recent2][0]
        consecutive = 0
        for step in range(t, max(-1, t - 50), -1):
            if not valid[j, step] or np.linalg.norm(vel[j, step]) >= STOPPED_MPS:
                break
            consecutive += 1
        acceleration = 0.0
        if len(steps) > 1:
            acceleration = (current_speed - speeds[-2]) / ((t - steps[-2]) * DT)
        out[j] = (
            current_speed,
            speeds[recent1].mean(),
            speeds[recent2].max(),
            np.linalg.norm(pos[j, t] - pos[j, start1]),
            np.linalg.norm(pos[j, t] - pos[j, start2]),
            np.mean(speeds[recent2] < STOPPED_MPS),
            recent2.sum() / (min(t, 20) + 1),
            consecutive * DT,
            acceleration,
            min((t - steps[0]) * DT, 5.0),
        )
    return out


def _forecasts(traj, prob, origins, horizon: int, distance_m: float):
    horizon = _integer(horizon, "horizon", minimum=1)
    if not np.isfinite(distance_m) or distance_m <= 0:
        raise ValueError("distance_m must be positive and finite")
    traj = _finite_array(traj, "traj")
    prob = _finite_array(prob, "prob")
    origins = _finite_array(origins, "origins")
    if traj.ndim != 4 or traj.shape[-1] != 2 or traj.shape[1] < 1 or traj.shape[2] < horizon:
        raise ValueError("traj must have shape [N, K>=1, T>=horizon, 2]")
    if prob.shape != traj.shape[:2] or origins.shape != (traj.shape[0], 2):
        raise ValueError("prob or origins shape does not match trajectories")
    totals = prob.sum(axis=1)
    if (prob < 0).any() or not np.allclose(totals, 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError("probabilities must be nonnegative and sum to one per agent")
    normalized = prob / totals[:, None]
    moving = (np.linalg.norm(traj[:, :, :horizon] - origins[:, None, None], axis=-1) > distance_m).any(-1)
    return traj, prob, normalized, moving


def native_departure_mass(traj, prob, origins, horizon: int = HORIZON, distance_m: float = DISTANCE_M) -> np.ndarray:
    """Probability of native modes whose maximum displacement exceeds the bar."""
    _, _, normalized, moving = _forecasts(traj, prob, origins, horizon, distance_m)
    return np.clip((normalized * moving).sum(axis=1), 0.0, 1.0)


def _design(features, native_mass) -> np.ndarray:
    features = _finite_array(features, "features")
    mass = _finite_array(native_mass, "native_mass")
    if features.ndim != 2 or features.shape[1] != len(HISTORY_FEATURE_NAMES):
        raise ValueError(f"features must have shape [N, {len(HISTORY_FEATURE_NAMES)}]")
    if mass.shape != (len(features),) or (mass < 0).any() or (mass > 1).any():
        raise ValueError("native_mass must have shape [N] with values in [0, 1]")
    bounded = np.clip(mass, 1e-6, 1 - 1e-6)
    return np.column_stack([features, np.log(bounded) - np.log1p(-bounded)])


def _sigmoid(logits) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -35.0, 35.0)))


def fit_calibrator(features, native_mass, labels, scenario_ids, ridge: float = 1.0) -> dict:
    """Fit deterministic ridge logistic calibration; return a JSON-safe model.

    Each scenario gets equal total fitting weight, even with repeated agents,
    replans, or predictor seeds. The caller must supply disjoint fit/tune/test
    scenes and complete 4 s binary event labels. The intercept is unpenalized.
    """
    raw = _design(features, native_mass)
    labels = _finite_array(labels, "labels")
    if len(raw) < 2 or labels.shape != (len(raw),) or not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("labels must be binary with shape [N], N >= 2")
    if len(np.unique(labels)) != 2:
        raise ValueError("fitting requires both departure and nondeparture labels")
    if not np.isscalar(ridge) or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be positive and finite")
    ids = np.asarray(scenario_ids)
    if ids.shape != (len(raw),) or any(not isinstance(s, str) or not s for s in ids.tolist()):
        raise ValueError("scenario_ids must be nonempty strings with shape [N]")
    counts = Counter(ids.tolist())
    weight = np.array([1.0 / counts[s] for s in ids], dtype=np.float64)
    weight *= len(raw) / weight.sum()
    mean = np.average(raw, axis=0, weights=weight)
    scale = np.sqrt(np.average((raw - mean) ** 2, axis=0, weights=weight))
    scale[scale < 1e-8] = 1.0
    design = np.column_stack([np.ones(len(raw)), (raw - mean) / scale])
    penalty = np.full(design.shape[1], float(ridge))
    penalty[0] = 0.0
    beta = np.zeros(design.shape[1])
    positive_rate = float(np.average(labels, weights=weight))
    beta[0] = np.log(positive_rate) - np.log1p(-positive_rate)

    def objective(coef):
        logits = design @ coef
        return float(np.dot(weight, np.logaddexp(0.0, logits) - labels * logits) + 0.5 * np.dot(penalty, coef**2))

    converged = False
    for iteration in range(1, 101):
        prediction = _sigmoid(design @ beta)
        gradient = design.T @ (weight * (prediction - labels)) + penalty * beta
        hessian = design.T @ (design * (weight * prediction * (1 - prediction))[:, None]) + np.diag(penalty)
        try:
            update = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise ValueError("calibration fit has singular curvature") from exc
        if np.max(np.abs(gradient)) < 1e-7 or np.max(np.abs(update)) < 1e-8:
            converged = True
            break
        before = objective(beta)
        step = 1.0
        for _ in range(40):
            candidate = beta - step * update
            if objective(candidate) <= before - 1e-4 * step * float(gradient @ update):
                beta = candidate
                break
            step *= 0.5
        else:
            if np.max(np.abs(gradient)) < 1e-5:
                converged = True
                break
            raise ValueError("calibration fit line search failed")
    if not converged:
        raise ValueError("calibration fit did not converge")
    prediction = _sigmoid(design @ beta)
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_names": list(CALIBRATOR_FEATURE_NAMES),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": beta[1:].tolist(),
        "intercept": float(beta[0]),
        "ridge": float(ridge),
        "n_fit_samples": len(raw),
        "n_fit_scenarios": len(counts),
        "n_fit_positive": int(labels.sum()),
        "fit_scenario_sha256": hashlib.sha256("\n".join(sorted(counts)).encode()).hexdigest(),
        "scenario_weighting": "inverse_records_per_scenario_scaled_to_N",
        "iterations": iteration,
        "converged": True,
        "weighted_fit_brier": float(np.average((prediction - labels) ** 2, weights=weight)),
        "event": {"horizon_steps": HORIZON, "dt_s": DT, "max_displacement_gt_m": DISTANCE_M},
    }


def _model_arrays(calibrator):
    if not isinstance(calibrator, dict) or calibrator.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported or malformed calibrator schema")
    if calibrator.get("feature_names") != list(CALIBRATOR_FEATURE_NAMES):
        raise ValueError("calibrator feature names do not match this implementation")
    expected_event = {"horizon_steps": HORIZON, "dt_s": DT, "max_displacement_gt_m": DISTANCE_M}
    if calibrator.get("event") != expected_event:
        raise ValueError("calibrator event definition does not match correction")
    try:
        mean = _finite_array(calibrator["mean"], "calibrator mean")
        scale = _finite_array(calibrator["scale"], "calibrator scale")
        coefficients = _finite_array(calibrator["coefficients"], "calibrator coefficients")
        intercept = _finite_array(calibrator["intercept"], "calibrator intercept")
    except KeyError as exc:
        raise ValueError(f"missing calibrator field: {exc.args[0]}") from exc
    shape = (len(CALIBRATOR_FEATURE_NAMES),)
    if mean.shape != shape or scale.shape != shape or coefficients.shape != shape or intercept.shape != ():
        raise ValueError("calibrator parameter shapes are invalid")
    if (scale <= 0).any() or calibrator.get("converged") is not True:
        raise ValueError("calibrator must be converged with positive scales")
    return mean, scale, coefficients, float(intercept)


def predict_departure(calibrator, features, mass) -> np.ndarray:
    """Predict event probabilities from a validated JSON-roundtrippable model."""
    mean, scale, coefficients, intercept = _model_arrays(calibrator)
    raw = _design(features, mass)
    with np.errstate(over="ignore", invalid="ignore"):
        logits = ((raw - mean) / scale) @ coefficients + intercept
    if not np.isfinite(logits).all():
        raise ValueError("calibrator produced nonfinite logits")
    return _sigmoid(logits)


def correct_predictions(scene: Scene, agents, t: int, traj, prob, calibrator, selective: bool = False,
                        threshold: float = 0.2, min_retained_fraction: float = 0.02):
    """Reweight stopped agents without deleting native trajectory geometry.

    Append one CV slot per agent (zero weight unless stationary support is
    missing). Native moving probabilities keep their conditional proportions
    and at least ``min_retained_fraction`` of their original total mass. With
    ``selective=True``, apply only a downward correction below ``threshold``.
    Native zero-probability moving groups are never given invented support.
    """
    if not isinstance(selective, (bool, np.bool_)):
        raise ValueError("selective must be boolean")
    if not np.isscalar(threshold) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if (not np.isscalar(min_retained_fraction) or not np.isfinite(min_retained_fraction)
            or not 0 < min_retained_fraction <= 1):
        raise ValueError("min_retained_fraction must be in (0, 1]")
    _, pos, vel, _ = _observations(scene, agents, t)
    origins = pos[:, -1]
    traj, original_prob, normalized, moving = _forecasts(traj, prob, origins, HORIZON, DISTANCE_M)
    if len(traj) != len(pos):
        raise ValueError("agent count does not match forecast count")
    native = np.clip((normalized * moving).sum(axis=1), 0.0, 1.0)
    calibrated = predict_departure(calibrator, history_features(scene, agents, t), native)
    stopped = np.linalg.norm(vel[:, -1], axis=1) < STOPPED_MPS
    retained = stopped & ((calibrated >= threshold) | (calibrated >= native)) if selective else np.zeros(len(traj), bool)
    changed = stopped & ~retained
    cv = origins[:, None] + vel[:, -1, None] * DT * np.arange(1, traj.shape[2] + 1)[None, :, None]
    out_traj = np.concatenate([traj.copy(), cv[:, None]], axis=1)
    out_prob = np.column_stack([original_prob.copy(), np.zeros(len(traj))])
    target = np.maximum(calibrated, min_retained_fraction * native)
    zero_moving_support = native <= 0
    target[zero_moving_support] = 0.0
    for j in np.flatnonzero(changed):
        q = target[j]
        out_prob[j] = 0.0
        if native[j] > 0:
            out_prob[j, :-1][moving[j]] = normalized[j, moving[j]] * (q / native[j])
        stationary_mass = normalized[j, ~moving[j]].sum()
        if stationary_mass > 0:
            out_prob[j, :-1][~moving[j]] = normalized[j, ~moving[j]] * ((1 - q) / stationary_mass)
        else:
            out_prob[j, -1] = 1 - q
        out_prob[j] /= out_prob[j].sum()
    realized = (out_prob[:, :-1] * moving).sum(axis=1) / out_prob.sum(axis=1)
    cv_admitted = out_prob[:, -1] > 0
    metadata = {
        "native_departure_mass": native.tolist(),
        "calibrated_departure_mass": calibrated.tolist(),
        "realized_departure_mass": realized.tolist(),
        "stopped": stopped.tolist(),
        "corrected": changed.tolist(),
        "selective_retained": retained.tolist(),
        "cv_admitted": cv_admitted.tolist(),
        "zero_native_moving_support": zero_moving_support.tolist(),
        "unrepresentable_departure_target": (changed & zero_moving_support & (calibrated > 0)).tolist(),
        "n_stopped": int(stopped.sum()),
        "n_corrected": int(changed.sum()),
        "n_cv_admitted": int(cv_admitted.sum()),
        "n_selective_retained": int(retained.sum()),
    }
    return out_traj, out_prob, metadata
