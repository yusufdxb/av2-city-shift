"""Route-end censored scoring: a closed-loop scoring mode added after Stage 4 (EXPLORATORY).

The registered closed-loop score (closedloop_v2.score) counts the whole 6 s drive, including the part after the ego
passes the end of the human's logged route and continues along its straight extension. This mode counts outcomes only
up to a cutoff step. At cutoff END it equals the registered score exactly; cityshift.replay checks that on every Stage 3
and Stage 4 drive. Registered as study D in docs/preregistration/exploratory-followups.md. Nothing registered uses it.

A drive is reduced to three primitives, from which any cutoff is scored:
  exec_decel [6]: executed speed change over each replan's 1 s segment (m/s^2), as score_sim computes it;
  log_decel [46]: the logged human's smoothed 1 s speed changes for windows starting at t=49..94, as the score does;
  fault_step: the first overlap step if it is an at-fault collision, else -1.
"""

from __future__ import annotations

import numpy as np

from .closedloop import DT, END, EXEC_STEPS, HANDOFF, HARD_BRAKE, REPLANS, EgoSim
from .scene import Scene

REPLAN_END = np.asarray(REPLANS) + EXEC_STEPS
N_LOG = END - 14 - HANDOFF  # 46 logged windows
LOG_END = HANDOFF + np.arange(N_LOG) + EXEC_STEPS


def executed_decel(ego: EgoSim) -> np.ndarray:
    return np.array([(ego.v[t + EXEC_STEPS] - ego.v[t]) / (EXEC_STEPS * DT) for t in REPLANS])


def logged_decel(sc: Scene) -> np.ndarray:
    sp = np.linalg.norm(np.diff(sc.pos[sc.av], axis=0), axis=1) / DT
    vs = np.convolve(sp, np.ones(5) / 5, mode="same")
    out = vs[HANDOFF + 10:END - 4] - vs[HANDOFF:END - 14]
    assert len(out) == N_LOG
    return out


def route_end_step(ego: EgoSim) -> int:
    """First step whose distance along the route exceeds the logged route length (END + 1 if never)."""
    beyond = np.flatnonzero(ego.s[HANDOFF:END + 1] > ego.path.length_logged)
    return HANDOFF + int(beyond[0]) if len(beyond) else END + 1


def window_cutoff(cross_steps: np.ndarray) -> np.ndarray:
    """Common cutoff for arms compared on the same scenario and seed: last step on which all are still on the route.

    cross_steps [..., arms] -> [...]
    """
    return np.minimum(END, np.min(cross_steps, axis=-1) - 1)


def score_on_route(exec_decel: np.ndarray, log_decel: np.ndarray, fault_step: np.ndarray,
                   cutoff: np.ndarray) -> dict[str, np.ndarray]:
    """Registered outcomes counted only up to `cutoff`, vectorised over drives.

    exec_decel [N, 6], log_decel [N, 46], fault_step [N], cutoff [N]. A replan counts if its 1 s segment ends at or
    before the cutoff, a logged window likewise, and a collision if its first at-fault overlap is at or before it.
    """
    exec_decel, log_decel = np.asarray(exec_decel, float), np.asarray(log_decel, float)
    fault_step, cutoff = np.asarray(fault_step), np.asarray(cutoff)
    planner = ((exec_decel <= HARD_BRAKE + 1e-9) & (REPLAN_END[None] <= cutoff[:, None])).any(1)
    logged = ((log_decel <= HARD_BRAKE + 1e-9) & (LOG_END[None] <= cutoff[:, None])).any(1)
    return {"planner_hard_brake": planner, "logged_hard_brake": logged,
            "unnecessary_hard_brake": planner & ~logged,
            "collision": (fault_step >= 0) & (fault_step <= cutoff),
            "scored_replans": (REPLAN_END[None] <= cutoff[:, None]).sum(1)}
