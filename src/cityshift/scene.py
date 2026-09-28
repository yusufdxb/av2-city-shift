"""Raw scenario container and an agent-centric input builder usable at any time step.

``build_input(scene, center, t)`` reproduces ``preprocess.process_scenario`` for any
agent and any "now" index t (preprocessing is the special case center = focal,
t = 49). The closed-loop harness uses it to forecast arbitrary agents at replan
times, with the ego's logged states replaced by simulated ones.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq

from .preprocess import HIST, LANE_PTS, MAX_AGENTS, MAX_LANES, OBJECT_TYPES, load_polylines, resample_polyline

N_STEPS = 110
DYNAMIC = {"vehicle", "bus", "motorcyclist", "cyclist", "pedestrian"}


@dataclass
class Scene:
    scenario_id: str
    city: str
    track_ids: list[str]
    types: np.ndarray  # [n] int, index into OBJECT_TYPES
    pos: np.ndarray  # [n, 110, 2] world
    vel: np.ndarray  # [n, 110, 2]
    head: np.ndarray  # [n, 110]
    valid: np.ndarray  # [n, 110] bool
    focal: int
    av: int
    poly_res: np.ndarray  # [Lp, 20, 2] resampled in world frame
    poly_attr: np.ndarray  # [Lp, 2]
    poly_orig: np.ndarray  # [sum pts, 2] original points, concatenated
    poly_start: np.ndarray  # [Lp] start offset of each polyline in poly_orig

    def with_track(self, i: int, pos: np.ndarray, vel: np.ndarray, head: np.ndarray, upto: int) -> "Scene":
        """Copy with track i's states for steps [0, upto] replaced (used for the simulated ego)."""
        p, v, h = self.pos.copy(), self.vel.copy(), self.head.copy()
        p[i, : upto + 1], v[i, : upto + 1], h[i, : upto + 1] = pos[: upto + 1], vel[: upto + 1], head[: upto + 1]
        return Scene(**{**self.__dict__, "pos": p, "vel": v, "head": h})


def load_scene(scenario_dir: str) -> Scene:
    sid = os.path.basename(scenario_dir.rstrip("/"))
    df = pq.read_table(os.path.join(scenario_dir, f"scenario_{sid}.parquet")).to_pandas()
    ids = list(dict.fromkeys(df["track_id"].tolist()))
    index = {t: i for i, t in enumerate(ids)}
    n = len(ids)
    pos = np.zeros((n, N_STEPS, 2))
    vel = np.zeros((n, N_STEPS, 2))
    head = np.zeros((n, N_STEPS))
    valid = np.zeros((n, N_STEPS), bool)
    types = np.zeros(n, np.int64)
    ti = df["track_id"].map(index).to_numpy()
    ts = df["timestep"].to_numpy()
    pos[ti, ts] = df[["position_x", "position_y"]].to_numpy(np.float64)
    vel[ti, ts] = df[["velocity_x", "velocity_y"]].to_numpy(np.float64)
    head[ti, ts] = df["heading"].to_numpy(np.float64)
    valid[ti, ts] = True
    first = df.drop_duplicates("track_id")
    for tid, ot in zip(first.track_id, first.object_type):
        types[index[tid]] = OBJECT_TYPES.index(ot) if ot in OBJECT_TYPES else OBJECT_TYPES.index("unknown")
    map_path = glob.glob(os.path.join(scenario_dir, "log_map_archive_*.json"))[0]
    polys, attrs = load_polylines(map_path)
    starts = np.cumsum([0] + [len(p) for p in polys[:-1]]) if polys else np.zeros(0, np.int64)
    return Scene(
        scenario_id=sid,
        city=df["city"].iloc[0],
        track_ids=ids,
        types=types,
        pos=pos,
        vel=vel,
        head=head,
        valid=valid,
        focal=index[df["focal_track_id"].iloc[0]],
        av=index["AV"],
        poly_res=np.stack([resample_polyline(p, LANE_PTS) for p in polys]) if polys else np.zeros((0, LANE_PTS, 2)),
        poly_attr=np.array(attrs, np.int8).reshape(-1, 2),
        poly_orig=np.concatenate(polys) if polys else np.zeros((0, 2)),
        poly_start=np.asarray(starts, np.int64),
    )


def frame(scene: Scene, center: int, t: int) -> tuple[np.ndarray, float, np.ndarray]:
    origin = scene.pos[center, t].copy()
    theta = float(scene.head[center, t])
    c, s = np.cos(theta), np.sin(theta)
    return origin, theta, np.array([[c, -s], [s, c]])


def build_input(scene: Scene, center: int, t: int) -> dict[str, np.ndarray]:
    """Agent-centric model input with "now" = step t. Requires scene.valid[center, t]."""
    assert scene.valid[center, t], "center agent must be observed at t"
    origin, theta, rot = frame(scene, center, t)
    lo = t - HIST + 1  # window [lo, t]; steps before 0 stay invalid
    w0 = max(lo, 0)
    win_valid = scene.valid[:, w0 : t + 1]
    present = np.where(win_valid.any(1))[0]
    # distance at each track's last observed step within the window
    last = w0 + (win_valid.shape[1] - 1 - np.argmax(win_valid[:, ::-1], axis=1))
    d = np.linalg.norm(scene.pos[np.arange(len(last)), last] - origin, axis=1)
    others = [i for i in present[np.argsort(d[present], kind="stable")] if i != center]
    tracks = [center] + others[: MAX_AGENTS - 1]

    agent_hist = np.zeros((MAX_AGENTS, HIST, 6), np.float32)
    agent_valid = np.zeros((MAX_AGENTS, HIST), bool)
    agent_type = np.full((MAX_AGENTS,), -1, np.int8)
    off = w0 - lo
    for j, i in enumerate(tracks):
        v = scene.valid[i, w0 : t + 1]
        xy = (scene.pos[i, w0 : t + 1] - origin) @ rot
        vv = scene.vel[i, w0 : t + 1] @ rot
        h = scene.head[i, w0 : t + 1] - theta
        feats = np.column_stack([xy, vv, np.cos(h), np.sin(h)])
        agent_hist[j, off:][v] = feats[v]
        agent_valid[j, off:] = v
        agent_type[j] = scene.types[i]

    lane_pts = np.zeros((MAX_LANES, LANE_PTS, 2), np.float32)
    lane_attr = np.full((MAX_LANES, 2), -1, np.int8)
    if len(scene.poly_start):
        dist = np.linalg.norm(scene.poly_orig - origin, axis=1)
        mind = np.minimum.reduceat(dist, scene.poly_start)
        order = np.argsort(mind, kind="stable")[:MAX_LANES]
        lane_pts[: len(order)] = (scene.poly_res[order] - origin) @ rot
        lane_attr[: len(order)] = scene.poly_attr[order]
    return {
        "agent_hist": agent_hist,
        "agent_valid": agent_valid,
        "agent_type": agent_type,
        "lane_pts": lane_pts,
        "lane_attr": lane_attr,
        "origin": origin,
        "theta": np.float64(theta),
    }
