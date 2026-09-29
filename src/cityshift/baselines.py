"""Fixed, training-free six-mode forecasts in the center agent's local frame.

CV modes are (speed scale, yaw rate in rad/s): (1, 0), (0.75, 0),
(1.25, 0), (1, -0.15), (1, 0.15), (0, 0). Their probabilities are fixed
at (0.4, 0.1, 0.1, 0.15, 0.15, 0.1). No validation data set these values.
"""

from __future__ import annotations

import numpy as np

from .preprocess import FUT, POLYLINE_TYPES

DT = 0.1
CV_GRID = np.array([[1.0, 0.0], [0.75, 0.0], [1.25, 0.0], [1.0, -0.15], [1.0, 0.15], [0.0, 0.0]])
CV_PROB = np.array([0.4, 0.1, 0.1, 0.15, 0.15, 0.1], np.float32)
LANE_MAX_OFFSET_M = 3.0
LANE_MAX_HEADING_RAD = np.pi / 3
LANE_OFFSET_SCALE_M = 2.0
LANE_HEADING_SCALE_RAD = np.pi / 6


def constant_velocity(inp: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Predict six fixed constant-turn-rate continuations from the current velocity."""
    v = np.asarray(inp["agent_hist"][0, -1, 2:4], np.float64)
    time = DT * np.arange(1, FUT + 1, dtype=np.float64)
    traj = np.empty((6, FUT, 2), np.float64)
    for k, (scale, yaw) in enumerate(CV_GRID):
        if yaw == 0:
            traj[k] = scale * time[:, None] * v
        else:
            a = yaw * time
            # Integral of R(yaw * t) v dt; stable because fixed nonzero yaw rates.
            traj[k, :, 0] = scale * (v[0] * np.sin(a) + v[1] * (np.cos(a) - 1)) / yaw
            traj[k, :, 1] = scale * (v[0] * (1 - np.cos(a)) + v[1] * np.sin(a)) / yaw
    return traj.astype(np.float32), CV_PROB.copy()


def stationary(inp: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Deliberately weak positive control, with one occupied mode."""
    traj = np.zeros((6, FUT, 2), np.float32)
    prob = np.array([1, 0, 0, 0, 0, 0], np.float32)
    return traj, prob


def lane_following(inp: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Follow up to six nearby, forward-facing lane centerlines at current speed."""
    pts = np.asarray(inp["lane_pts"], np.float64)
    attr = np.asarray(inp["lane_attr"])
    v = np.asarray(inp["agent_hist"][0, -1, 2:4], np.float64)
    speed = float(np.linalg.norm(v))
    # The agent-centric x axis is the heading, independent of velocity direction.
    heading = np.array([1.0, 0.0])
    delta = np.diff(pts, axis=1)
    seg_len = np.linalg.norm(delta, axis=-1)
    safe = np.maximum(seg_len, 1e-9)
    frac = np.clip(-(pts[:, :-1] * delta).sum(-1) / safe**2, 0, 1)
    projection = pts[:, :-1] + frac[..., None] * delta
    d2 = (projection**2).sum(-1)
    d2[seg_len < 1e-6] = np.inf
    seg = np.argmin(d2, axis=1)
    row = np.arange(len(pts))
    offset = np.sqrt(d2[row, seg])
    direction = delta[row, seg] / safe[row, seg, None]
    cos_err = np.clip((direction * heading).sum(-1), -1, 1)
    angle = np.arccos(cos_err)
    eligible = (attr[:, 0] >= 0) & (attr[:, 0] < POLYLINE_TYPES.index("CROSSWALK"))
    eligible &= (offset <= LANE_MAX_OFFSET_M) & (angle <= LANE_MAX_HEADING_RAD)
    if not eligible.any():
        return constant_velocity(inp)
    logit = -0.5 * (offset / LANE_OFFSET_SCALE_M) ** 2 - 0.5 * (angle / LANE_HEADING_SCALE_RAD) ** 2
    choices = np.where(eligible)[0]
    choices = choices[np.argsort(-logit[choices], kind="stable")[:6]]
    traj = np.zeros((6, FUT, 2), np.float32)
    prob = np.zeros(6, np.float32)
    z = np.exp(logit[choices] - np.max(logit[choices]))
    prob[: len(choices)] = z / z.sum()
    for k, lane in enumerate(choices):
        lengths = seg_len[lane]
        arc = np.concatenate([[0.0], np.cumsum(lengths)])
        start = arc[seg[lane]] + frac[lane, seg[lane]] * lengths[seg[lane]]
        distance = start + speed * DT * np.arange(1, FUT + 1)
        clipped = np.minimum(distance, arc[-1])
        keep = np.r_[True, lengths > 1e-6]
        xy = np.column_stack([np.interp(clipped, arc[keep], pts[lane, keep, axis]) for axis in range(2)])
        last_nonzero = np.flatnonzero(lengths > 1e-6)[-1]
        last_dir = delta[lane, last_nonzero] / lengths[last_nonzero]
        xy += np.maximum(distance - arc[-1], 0)[:, None] * last_dir
        traj[k] = xy
    # Unused slots must not introduce artificial stationary modes into minFDE.
    traj[len(choices):] = traj[0]
    return traj, prob


BASELINES = {"CV": constant_velocity, "LANE": lane_following, "STATIC": stationary}
