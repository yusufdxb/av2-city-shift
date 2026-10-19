"""Explicit planning objectives and independent replay geometry for transfer experiments.

The risk budget constrains a union-bound surrogate over forecast modes. It is
not a calibrated collision probability or a guarantee about physical safety.
No function installs a process-global planner or changes historical harnesses.
"""

from __future__ import annotations

import numpy as np

from . import closedloop as base
from . import closedloop_v2 as v2
from . import closedloop_v4 as v4
from .scene import Scene

OBJECTIVES = ("weighted", "risk_budget")
RISK_BUDGET = 0.05


def candidate_risks(
    sc: Scene, ego: base.EgoSim, t: int, agents: list[int], traj: np.ndarray, prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the frozen v4 forecast risk kernel and route-capped progress."""
    if traj.shape[:2] != prob.shape or traj.ndim != 4 or traj.shape[-1] != 2:
        raise ValueError("forecast/probability shape mismatch")
    if len(agents) != len(traj) or traj.shape[2] < base.CHECK_STEPS:
        raise ValueError("forecast agent count or horizon mismatch")
    if not np.isfinite(traj).all() or not np.isfinite(prob).all() or (prob < 0).any():
        raise ValueError("nonfinite or negative forecast probability")
    if len(agents) and not np.allclose(prob.sum(1), 1, atol=1e-5):
        raise ValueError("forecast probabilities must sum to one")
    _, distance = base.profiles(ego.v[t], ego.vmax, base.PLAN_STEPS)
    distance = np.minimum(distance, max(ego.path.length_logged - ego.s[t], 0.0))
    progress = distance[:, -1] / max(distance[:, -1].max(), 1e-6)
    risk = np.zeros(len(base.ACCELS))
    if agents:
        ep, eh = ego.path.at(ego.s[t] + distance[:, : base.CHECK_STEPS])
        ap = traj[:, :, : base.CHECK_STEPS]
        previous = np.concatenate([np.repeat(sc.pos[agents, t][:, None, None], ap.shape[1], axis=1), ap[:, :, :-1]], axis=2)
        delta = ap - previous
        heading = np.arctan2(delta[..., 1], delta[..., 0])
        heading = np.where(np.linalg.norm(delta, axis=-1) < 0.05, sc.head[agents, t][:, None, None], heading)
        dims = base.DIMS[sc.types[agents]]
        hit = base.boxes_overlap(
            ep[:, None, None],
            eh[:, None, None],
            base.EGO_DIMS[0] + 2 * base.MARGIN_M,
            base.EGO_DIMS[1] + 2 * base.MARGIN_M,
            ap[None],
            heading[None],
            dims[:, 0][None, :, None, None],
            dims[:, 1][None, :, None, None],
        )
        hit &= base.in_front(ep[:, None, None], eh[:, None, None], ap[None])
        risk = (hit.any(-1) * prob[None]).sum(-1).sum(-1)
    return risk, progress


def choose_candidate(
    risk: np.ndarray, progress: np.ndarray, feasible: np.ndarray, objective: str, risk_budget: float = RISK_BUDGET
) -> tuple[int, dict]:
    """Weighted v4 cost or feasibility-first risk-budget optimization."""
    if objective not in OBJECTIVES or not np.isfinite(risk_budget) or not 0 <= risk_budget <= 1:
        raise ValueError("invalid objective or risk budget")
    if any(np.shape(a) != base.ACCELS.shape for a in (risk, progress, feasible)):
        raise ValueError("candidate vector shape mismatch")
    if not np.isfinite(risk).all() or not np.isfinite(progress).all() or (risk < 0).any():
        raise ValueError("invalid candidate risk or progress")
    feasible = np.asarray(feasible, bool)
    if not feasible.any():
        return 0, {"stopline_infeasible": True, "risk_budget_fallback": True}
    if objective == "weighted":
        cost = base.W_RISK * risk + base.W_PROGRESS * (1 - progress) + base.W_ACCEL * np.abs(base.ACCELS)
        selected = int(np.argmin(np.where(feasible, cost, np.inf)))
        return selected, {"stopline_infeasible": False, "risk_budget_fallback": False}
    admissible = feasible & (risk <= risk_budget + 1e-12)
    fallback = not admissible.any()
    candidates = np.flatnonzero(feasible if fallback else admissible)
    # lexsort's last key is primary. Original candidate index resolves exact ties.
    if fallback:
        order = np.lexsort((candidates, np.abs(base.ACCELS[candidates]), -progress[candidates], risk[candidates]))
    else:
        order = np.lexsort((candidates, np.abs(base.ACCELS[candidates]), -progress[candidates]))
    return int(candidates[order[0]]), {"stopline_infeasible": False, "risk_budget_fallback": fallback}


def plan(
    sc: Scene,
    ego: base.EgoSim,
    t: int,
    agents: list[int],
    traj: np.ndarray,
    prob: np.ndarray,
    objective: str,
    risk_budget: float = RISK_BUDGET,
) -> tuple[int, dict]:
    risk, progress = candidate_risks(sc, ego, t, agents, traj, prob)
    candidate, diagnostic = choose_candidate(risk, progress, v4.feasible(ego, t), objective, risk_budget)
    return candidate, diagnostic | {"chosen_forecast_risk": float(risk[candidate])}


def box_corners(center, heading, length, width):
    """Broadcast oriented rectangle corners, counterclockwise."""
    center = np.asarray(center)
    heading, length, width = np.broadcast_arrays(heading, length, width)
    forward = np.stack([np.cos(heading), np.sin(heading)], -1)
    lateral = np.stack([-np.sin(heading), np.cos(heading)], -1)
    signs = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    return (
        center[..., None, :]
        + signs[:, 0, None] * forward[..., None, :] * length[..., None, None] / 2
        + signs[:, 1, None] * lateral[..., None, :] * width[..., None, None] / 2
    )


def _point_edge_distance(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    a, b = polygon, np.roll(polygon, -1, axis=-2)
    delta = b - a
    offset = points[..., :, None, :] - a[..., None, :, :]
    projection = np.clip(
        (offset * delta[..., None, :, :]).sum(-1) / np.maximum((delta * delta).sum(-1)[..., None, :], 1e-15), 0, 1
    )
    distance = np.linalg.norm(offset - projection[..., None] * delta[..., None, :, :], axis=-1)
    return distance.min(axis=(-2, -1))


def box_clearance(c1, h1, l1, w1, c2, h2, l2, w2) -> np.ndarray:
    """True Euclidean edge-to-edge OBB clearance; contact/overlap returns zero."""
    a, b = box_corners(c1, h1, l1, w1), box_corners(c2, h2, l2, w2)
    distance = np.minimum(_point_edge_distance(a, b), _point_edge_distance(b, a))
    return np.where(base.boxes_overlap(c1, h1, l1, w1, c2, h2, l2, w2), 0.0, distance)


def complete_future(sc: Scene, agents: list[int], t: int, horizon: int = base.CHECK_STEPS) -> bool:
    return bool(t + horizon <= base.END and sc.valid[agents, t : t + horizon + 1].all())


def departure_label(sc: Scene, agent: int, t: int, horizon: int = base.CHECK_STEPS) -> bool | None:
    """Offline-only label. Incomplete future is unknown, never extrapolated."""
    if not complete_future(sc, [agent], t, horizon):
        return None
    return bool((np.linalg.norm(sc.pos[agent, t + 1 : t + horizon + 1] - sc.pos[agent, t], axis=-1) > 2.0).any())


def true_candidate_contacts(sc: Scene, ego: base.EgoSim, t: int, agents: list[int]) -> np.ndarray:
    """All-direction, uninflated overlap with complete logged futures, scored offline."""
    if not complete_future(sc, agents, t):
        raise ValueError("true candidate contacts require complete logged futures")
    if not agents:
        return np.zeros(len(base.ACCELS), bool)
    _, distance = base.profiles(ego.v[t], ego.vmax, base.CHECK_STEPS)
    distance = np.minimum(distance, max(ego.path.length_logged - ego.s[t], 0))
    ep, eh = ego.path.at(ego.s[t] + distance)
    steps = np.arange(t + 1, t + base.CHECK_STEPS + 1)
    dims = base.DIMS[sc.types[agents]]
    hits = base.boxes_overlap(
        ep[:, None],
        eh[:, None],
        *base.EGO_DIMS,
        sc.pos[agents][:, steps][None],
        sc.head[agents][:, steps][None],
        dims[:, 0][None, :, None],
        dims[:, 1][None, :, None],
    )
    return hits.any(axis=(1, 2))


def replay_metrics(sc: Scene, ego: base.EgoSim) -> dict:
    """Observed replay contacts, geometric clearance, comfort and route progress.

    Missing agent samples remain unknown. Zero observed contacts is not a claim
    that no contact would occur with unobserved or reactive actors.
    """
    pos, _, head = ego.states(base.END)
    decel = [(ego.v[t + base.EXEC_STEPS] - ego.v[t]) for t in base.REPLANS]
    result = v2.score(sc, pos, head, ego.v, decel)
    steps = np.arange(base.HANDOFF + 1, base.END + 1)
    others = np.array([i for i in range(len(sc.track_ids)) if i != sc.av and sc.types[i] in base.DYN_IDX], int)
    contact, near, departing_contact = False, False, False
    clearance = np.nan
    observed = 0
    if len(others):
        dims = base.DIMS[sc.types[others]]
        distance = box_clearance(
            pos[steps][None],
            head[steps][None],
            *base.EGO_DIMS,
            sc.pos[others][:, steps],
            sc.head[others][:, steps],
            dims[:, 0, None],
            dims[:, 1, None],
        )
        valid = sc.valid[others][:, steps]
        observed = int(valid.sum())
        if observed:
            clearance = float(np.where(valid, distance, np.inf).min())
            contact = bool((valid & (distance <= 1e-9)).any())
            near = bool((valid & (distance < 1.0)).any())
            departing = np.array(
                [
                    np.linalg.norm(sc.vel[i, base.HANDOFF]) < 0.5 and departure_label(sc, int(i), base.HANDOFF) is True
                    for i in others
                ]
            )
            departing_contact = bool((valid & departing[:, None] & (distance <= 1e-9)).any())
    acceleration = np.diff(ego.v[base.HANDOFF : base.END + 1]) / base.DT
    jerk = np.diff(acceleration) / base.DT
    length = ego.path.length_logged
    return result | {
        "progress_m": float(min(ego.s[base.END], length)),
        "travelled_m": float(ego.s[base.END]),
        "route_length_m": length,
        "progress_fraction": float(min(ego.s[base.END], length) / length) if length > 1 else np.nan,
        "route_overshoot_m": float(max(0.0, ego.s[base.END] - length)),
        "observed_contact": contact,
        "observed_near_conflict": near,
        "observed_departing_contact": departing_contact,
        "min_clearance_m": clearance,
        "observed_actor_steps": observed,
        "mean_abs_jerk_mps3": float(np.abs(jerk).mean()),
        "accels": ego.chosen,
    }
