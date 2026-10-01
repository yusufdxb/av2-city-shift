"""Stop-line closed loop (EXPLORATORY study E; docs/preregistration/exploratory-followups.md).

The registered closed loop extends the human's logged route straight past its end, and study D showed that most at-fault
collisions, and the collision checker's positive control, come from driving on that extension. Here the end of the
logged route acts as a stop line: when feasible, the planner chooses an acceleration after which the ego can still stop at
or before the route end decelerating at STOP_DECEL (gentler than the -4 m/s^2 hard-brake threshold, so respecting the
line can be maintained without a hard brake from a feasible state), and its risk and progress terms treat the ego as stopping there.
Everything else (agent selection, forecasts, candidate set, weights, execution, scoring) is the closedloop_v2 harness.

From any state that satisfies the rule, braking at exactly STOP_DECEL keeps the stopping point fixed, so once a
feasible candidate has been executed one always exists at the next replan. The rule is NOT guaranteed feasible at the
handoff (the remaining route depends on how the human braked; e.g. 10 m/s with 5 m left cannot be met even at
-8 m/s^2), and if no candidate is feasible the hardest brake is chosen and execution can pass the line. Compliance is
therefore checked empirically: on all 122,100 study E drives there were 0 infeasible replans and no drive passed the
route end by more than 1 cm; maximum overshoot was below 1e-6 m (reports/followups/study_e_stopline_diagnostics.json).
"""

from __future__ import annotations

import numpy as np

from . import closedloop as base
from .closedloop import (ACCELS, CHECK_STEPS, DIMS, EGO_DIMS, EXEC_STEPS, MARGIN_M, PLAN_STEPS, W_ACCEL, W_PROGRESS,
                         EgoSim, boxes_overlap, in_front, profiles)
from .scene import Scene

STOP_DECEL = 3.0
INFEASIBLE = {"count": 0}  # per-process counter of replans with no feasible candidate


def feasible(ego: EgoSim, t: int) -> np.ndarray:
    """Candidates after whose 1 s segment the ego can still stop at the route end at <= STOP_DECEL."""
    v, dist = profiles(ego.v[t], ego.vmax, EXEC_STEPS)
    s1, v1 = ego.s[t] + dist[:, -1], v[:, -1]
    return s1 + v1 ** 2 / (2 * STOP_DECEL) <= ego.path.length_logged + 1e-6


def plan(sc: Scene, ego: EgoSim, t: int, agents: list[int], traj: np.ndarray, prob: np.ndarray) -> int:
    """closedloop.plan with the stop line: positions and progress clipped at the route end, infeasible candidates out."""
    v, dist = profiles(ego.v[t], ego.vmax, PLAN_STEPS)
    room = max(ego.path.length_logged - ego.s[t], 0.0)
    dist = np.minimum(dist, room)  # the ego stops at the line
    progress = dist[:, -1] / max(dist[:, -1].max(), 1e-6)
    risk = np.zeros(len(ACCELS))
    if agents:
        ep, eh = ego.path.at(ego.s[t] + dist[:, :CHECK_STEPS])
        ap = traj[:, :, :CHECK_STEPS]
        prev = np.concatenate([np.repeat(sc.pos[agents, t][:, None, None], ap.shape[1], 1), ap[:, :, :-1]], 2)
        dxy = ap - prev
        ah = np.arctan2(dxy[..., 1], dxy[..., 0])
        ah = np.where(np.linalg.norm(dxy, axis=-1) < 0.05, sc.head[agents, t][:, None, None], ah)
        dims = DIMS[sc.types[agents]]
        al, aw = dims[:, 0][:, None, None], dims[:, 1][:, None, None]
        c1, h1 = ep[:, None, None], eh[:, None, None]
        hit = boxes_overlap(c1, h1, EGO_DIMS[0] + 2 * MARGIN_M, EGO_DIMS[1] + 2 * MARGIN_M, ap[None], ah[None],
                            al[None], aw[None])
        hit &= in_front(c1, h1, ap[None])
        risk = ((hit.any(-1) * prob[None]).sum(-1)).sum(-1)
    cost = base.W_RISK * risk + W_PROGRESS * (1.0 - progress) + W_ACCEL * np.abs(ACCELS)
    ok = feasible(ego, t)
    if not ok.any():
        INFEASIBLE["count"] += 1
        return 0  # hardest brake
    return int(np.argmin(np.where(ok, cost, np.inf)))


def install() -> None:
    """Switch closedloop's planner to the stop-line planner (workers forked afterwards inherit it)."""
    base.plan = plan
