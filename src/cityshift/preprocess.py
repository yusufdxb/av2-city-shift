"""Convert raw Argoverse 2 motion-forecasting scenarios into fixed-shape arrays.

Every scenario becomes one sample in the focal agent's frame: origin at the focal
agent's position at the last observed step (t=49), x-axis along its heading.

Output per split (in ``<out>/<split>/``), all memory-mappable ``.npy``:
    agent_hist  float32 [N, A, 50, 6]   x, y, vx, vy, cos(h), sin(h)
    agent_valid bool    [N, A, 50]
    agent_type  int8    [N, A]          index into OBJECT_TYPES, -1 = padding
    lane_pts    float32 [N, L, P, 2]    resampled centerline points
    lane_attr   int8    [N, L, 2]       (polyline type, is_intersection), -1 = padding
    target      float32 [N, 60, 2]      focal future positions
    meta.parquet                        scenario_id, city, focal_type, speed, origin/heading
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HIST = 50
FUT = 60
MAX_AGENTS = 32
MAX_LANES = 128
LANE_PTS = 20

OBJECT_TYPES = (
    "vehicle",
    "pedestrian",
    "motorcyclist",
    "cyclist",
    "bus",
    "static",
    "background",
    "construction",
    "riderless_bicycle",
    "unknown",
)
POLYLINE_TYPES = ("VEHICLE", "BIKE", "BUS", "CROSSWALK")


def resample_polyline(xy: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to ``n`` points evenly spaced by arc length."""
    if len(xy) == 1:
        return np.repeat(xy, n, axis=0)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return np.repeat(xy[:1], n, axis=0)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])], axis=1)


def _xy(points: list[dict]) -> np.ndarray:
    return np.array([[p["x"], p["y"]] for p in points], dtype=np.float64)


def load_polylines(map_path: str) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    with open(map_path) as f:
        m = json.load(f)
    polys, attrs = [], []
    for lane in m["lane_segments"].values():
        polys.append(_xy(lane["centerline"]))
        attrs.append((POLYLINE_TYPES.index(lane["lane_type"]), int(lane["is_intersection"])))
    for pc in m["pedestrian_crossings"].values():
        # A crossing is two parallel edges; its axis runs between the edge midpoints.
        e1, e2 = _xy(pc["edge1"]), _xy(pc["edge2"])
        polys.append(np.stack([e1.mean(0), e2.mean(0)]))
        attrs.append((POLYLINE_TYPES.index("CROSSWALK"), 0))
    return polys, attrs


def process_scenario(scenario_dir: str) -> dict:
    sid = os.path.basename(scenario_dir.rstrip("/"))
    df = pq.read_table(os.path.join(scenario_dir, f"scenario_{sid}.parquet")).to_pandas()
    focal_id = df["focal_track_id"].iloc[0]
    city = df["city"].iloc[0]

    focal = df[df.track_id == focal_id].set_index("timestep")
    origin = focal.loc[HIST - 1, ["position_x", "position_y"]].to_numpy(np.float64)
    theta = float(focal.loc[HIST - 1, "heading"])
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])  # local = (world - origin) @ rot

    def to_local(xy: np.ndarray) -> np.ndarray:
        return (xy - origin) @ rot

    # --- agents: focal first, then nearest others observed in the history ---
    hist = df[df.timestep < HIST]
    last = hist.sort_values("timestep").groupby("track_id").tail(1)
    last = last.assign(d=np.hypot(last.position_x - origin[0], last.position_y - origin[1]))
    others = last[last.track_id != focal_id].sort_values("d")["track_id"].tolist()
    track_ids = [focal_id] + others[: MAX_AGENTS - 1]

    agent_hist = np.zeros((MAX_AGENTS, HIST, 6), np.float32)
    agent_valid = np.zeros((MAX_AGENTS, HIST), bool)
    agent_type = np.full((MAX_AGENTS,), -1, np.int8)
    groups = hist.groupby("track_id")
    for i, tid in enumerate(track_ids):
        g = groups.get_group(tid)
        t = g["timestep"].to_numpy()
        xy = to_local(g[["position_x", "position_y"]].to_numpy(np.float64))
        v = g[["velocity_x", "velocity_y"]].to_numpy(np.float64) @ rot
        h = g["heading"].to_numpy(np.float64) - theta
        agent_hist[i, t] = np.column_stack([xy, v, np.cos(h), np.sin(h)])
        agent_valid[i, t] = True
        otype = g["object_type"].iloc[0]
        agent_type[i] = OBJECT_TYPES.index(otype) if otype in OBJECT_TYPES else OBJECT_TYPES.index("unknown")

    fut = focal.loc[HIST : HIST + FUT - 1, ["position_x", "position_y"]].to_numpy(np.float64)
    if len(fut) != FUT:
        raise ValueError(f"{sid}: focal future has {len(fut)} steps, expected {FUT}")
    target = to_local(fut)

    # --- map: nearest polylines by closest point ---
    map_path = glob.glob(os.path.join(scenario_dir, "log_map_archive_*.json"))[0]
    polys, attrs = load_polylines(map_path)
    local = [to_local(p) for p in polys]
    dists = np.array([np.min(np.linalg.norm(p, axis=1)) for p in local]) if local else np.zeros(0)
    order = np.argsort(dists)[:MAX_LANES]
    lane_pts = np.zeros((MAX_LANES, LANE_PTS, 2), np.float32)
    lane_attr = np.full((MAX_LANES, 2), -1, np.int8)
    for j, k in enumerate(order):
        lane_pts[j] = resample_polyline(local[k], LANE_PTS)
        lane_attr[j] = attrs[k]

    speed = float(np.hypot(*focal.loc[HIST - 1, ["velocity_x", "velocity_y"]].to_numpy(np.float64)))
    return {
        "agent_hist": agent_hist,
        "agent_valid": agent_valid,
        "agent_type": agent_type,
        "lane_pts": lane_pts,
        "lane_attr": lane_attr,
        "target": target,
        "meta": {
            "scenario_id": sid,
            "city": city,
            "focal_type": focal["object_type"].iloc[0],
            "speed": speed,
            "origin_x": origin[0],
            "origin_y": origin[1],
            "theta": theta,
            "n_agents": len(track_ids),
            "n_lanes": len(order),
        },
    }


ARRAY_KEYS = ("agent_hist", "agent_valid", "agent_type", "lane_pts", "lane_attr", "target")


def _safe(d: str):
    try:
        return process_scenario(d)
    except Exception as e:  # recorded, not swallowed: failures are written to failures.txt
        return {"error": f"{d}: {type(e).__name__}: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="dir holding train/ and val/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    dirs = sorted(glob.glob(os.path.join(args.raw, args.split, "*")))
    if args.limit:
        dirs = dirs[: args.limit]
    out = os.path.join(args.out, args.split)
    os.makedirs(out, exist_ok=True)
    n = len(dirs)
    shapes = {
        "agent_hist": ((n, MAX_AGENTS, HIST, 6), np.float32),
        "agent_valid": ((n, MAX_AGENTS, HIST), bool),
        "agent_type": ((n, MAX_AGENTS), np.int8),
        "lane_pts": ((n, MAX_LANES, LANE_PTS, 2), np.float32),
        "lane_attr": ((n, MAX_LANES, 2), np.int8),
        "target": ((n, FUT, 2), np.float32),
    }
    arrs = {
        k: np.lib.format.open_memmap(os.path.join(out, f"{k}.npy"), mode="w+", dtype=dt, shape=sh)
        for k, (sh, dt) in shapes.items()
    }
    metas, failures, row = [], [], 0
    with ProcessPoolExecutor(args.workers) as ex:
        for i, res in enumerate(ex.map(_safe, dirs, chunksize=64)):
            if "error" in res:
                failures.append(res["error"])
                continue
            for k in ARRAY_KEYS:
                arrs[k][row] = res[k]
            metas.append(res["meta"])
            row += 1
            if i % 10000 == 0:
                print(f"{args.split}: {i}/{n}", flush=True)
    for a in arrs.values():
        a.flush()
    # Truncate if any failed: copy the valid prefix to a new file, then swap it in.
    if row < n:
        for k in ARRAY_KEYS:
            tmp = os.path.join(out, f"{k}.tmp.npy")
            dst = np.lib.format.open_memmap(tmp, mode="w+", dtype=arrs[k].dtype, shape=(row,) + arrs[k].shape[1:])
            for s in range(0, row, 10000):
                dst[s : s + 10000] = arrs[k][s : min(s + 10000, row)]
            dst.flush()
            del dst
            arrs[k] = None
            os.replace(tmp, os.path.join(out, f"{k}.npy"))
    pd.DataFrame(metas).to_parquet(os.path.join(out, "meta.parquet"))
    with open(os.path.join(out, "failures.txt"), "w") as f:
        f.write("\n".join(failures))
    print(f"{args.split}: wrote {row} samples, {len(failures)} failures")
    if failures:  # outputs are written, but a partial split must not pass silently
        raise SystemExit(f"{len(failures)} scenarios failed; see {os.path.join(out, 'failures.txt')}")


if __name__ == "__main__":
    main()
