"""Build agent-centred TRAIN samples for fully observed FOCAL and SCORED tracks."""

from __future__ import annotations

import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .data import Split, dev_indices
from .preprocess import ARRAY_KEYS, FUT, HIST, LANE_PTS, MAX_AGENTS, MAX_LANES, OBJECT_TYPES
from .scene import build_input, load_scene


def dev_scenario_ids(root: str, scenarios: str | None = None) -> set[str]:
    """Get the fixed focal TRAIN dev scenario IDs, checking an optional saved list."""
    meta = Split(root, "train").meta
    fixed = set(meta.scenario_id.iloc[dev_indices(meta, 0.02)].astype(str))
    if scenarios:
        saved = set(str(x) for x in np.load(scenarios, allow_pickle=True))
        if saved != fixed:
            raise ValueError("dev scenario IDs differ from cityshift.data.dev_indices(meta, 0.02)")
    return fixed


def eligible_ids(path: str) -> list[tuple[str, bool]]:
    """Select category 3 focal and category 2 scored tracks observed at all 110 steps."""
    sid = os.path.basename(path.rstrip("/"))
    df = pq.read_table(os.path.join(path, f"scenario_{sid}.parquet"), columns=[
        "track_id", "timestep", "object_category", "focal_track_id",
    ]).to_pandas()
    focal = str(df.focal_track_id.iloc[0])
    out = []
    for track_id, group in df.groupby("track_id", sort=False):
        category = int(group.object_category.iloc[0])
        is_focal = str(track_id) == focal and category == 3
        if not (is_focal or category == 2):
            continue
        ts = group.timestep.to_numpy()
        if len(ts) == HIST + FUT and np.array_equal(np.sort(ts), np.arange(HIST + FUT)):
            out.append((str(track_id), is_focal))
    return out


def _count(path: str) -> tuple[str, int]:
    return path, len(eligible_ids(path))


def _process(path: str) -> list[dict]:
    sc = load_scene(path)
    ids = eligible_ids(path)
    lookup = {str(tid): i for i, tid in enumerate(sc.track_ids)}
    rows = []
    for tid, focal in ids:
        i = lookup[tid]
        inp = build_input(sc, i, HIST - 1)
        origin, theta = inp["origin"], float(inp["theta"])
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        target = ((sc.pos[i, HIST:] - origin) @ rot).astype(np.float32)
        rows.append({**{k: inp[k] for k in ARRAY_KEYS if k != "target"}, "target": target, "meta": {
            "scenario_id": sc.scenario_id, "city": sc.city, "agent_id": tid,
            "agent_type": OBJECT_TYPES[sc.types[i]], "focal": focal,
            "speed": float(np.linalg.norm(sc.vel[i, HIST - 1])),
        }})
    return rows


_OPEN: dict[str, dict[str, np.memmap]] = {}


def _process_into(job: tuple[str, str, int, int]) -> list[dict]:
    """Write one scenario's samples into the split's memmaps at a fixed offset; return their metadata."""
    path, split_dir, offset, expected = job
    rows = _process(path)
    if len(rows) != expected:
        raise RuntimeError(f"sample count changed for {path}")
    if split_dir not in _OPEN:
        _OPEN[split_dir] = {k: np.load(os.path.join(split_dir, f"{k}.npy"), mmap_mode="r+") for k in ARRAY_KEYS}
    arrs = _OPEN[split_dir]
    for r, row in enumerate(rows):
        for name in ARRAY_KEYS:
            arrs[name][offset + r] = row[name]
    return [row["meta"] for row in rows]


def _arrays(out: str, n: int) -> dict[str, np.memmap]:
    os.makedirs(out, exist_ok=True)
    shapes = {
        "agent_hist": ((n, MAX_AGENTS, HIST, 6), np.float32),
        "agent_valid": ((n, MAX_AGENTS, HIST), bool),
        "agent_type": ((n, MAX_AGENTS), np.int8),
        "lane_pts": ((n, MAX_LANES, LANE_PTS, 2), np.float32),
        "lane_attr": ((n, MAX_LANES, 2), np.int8),
        "target": ((n, FUT, 2), np.float32),
    }
    return {k: np.lib.format.open_memmap(os.path.join(out, f"{k}.npy"), mode="w+", dtype=dt, shape=shape)
            for k, (shape, dt) in shapes.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw TRAIN split directory")
    ap.add_argument("--focal-root", required=True, help="Stage 1 preprocessed root, for fixed dev IDs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev-scenarios", default="runs/dev_scenarios.npy")
    ap.add_argument("--scenarios", help="optional scenario-ID file; for smoke tests")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    if args.limit < 0 or args.workers < 1:
        ap.error("--limit must be nonnegative and --workers positive")
    dev_ids = dev_scenario_ids(args.focal_root, args.dev_scenarios)
    dirs = sorted(glob.glob(os.path.join(args.raw, "*")))
    if args.scenarios:
        wanted = set(str(x) for x in np.load(args.scenarios, allow_pickle=True))
        dirs = [d for d in dirs if os.path.basename(d) in wanted]
    if args.limit:
        dirs = dirs[:args.limit]
    with ProcessPoolExecutor(args.workers) as ex:
        counts = list(ex.map(_count, dirs, chunksize=16))
    totals = {"train": 0, "dev": 0}
    for path, n in counts:
        totals["dev" if os.path.basename(path) in dev_ids else "train"] += n
    arrays = {split: _arrays(os.path.join(args.out, split), n) for split, n in totals.items() if n}
    for arrs in arrays.values():
        for arr in arrs.values():
            arr.flush()
    del arrays
    # Workers write their samples straight into the memory-mapped outputs at offsets fixed by the count pass and
    # return only metadata. Streaming ~1 MB results per scenario through ProcessPoolExecutor's result pipe, with
    # one task per scenario, deadlocked on the full 199,908-scenario split (Python 3.10: a worker blocked writing
    # its result while the parent's main thread was blocked on the executor's wakeup pipe).
    offsets, written = [], {"train": 0, "dev": 0}
    for path, n in counts:
        split = "dev" if os.path.basename(path) in dev_ids else "train"
        offsets.append((path, os.path.join(args.out, split), written[split], n))
        written[split] += n
    metas: dict[str, list[dict]] = {split: [] for split in ("train", "dev") if totals[split]}
    with ProcessPoolExecutor(args.workers) as ex:
        for j, (job, meta_rows) in enumerate(zip(offsets, ex.map(_process_into, offsets, chunksize=64)), 1):
            split = "dev" if job[1].endswith(os.sep + "dev") else "train"
            metas[split].extend(meta_rows)
            if j % 10000 == 0:
                print(f"{j}/{len(dirs)} scenarios", flush=True)
    for split in metas:
        pd.DataFrame(metas[split]).to_parquet(os.path.join(args.out, split, "meta.parquet"))
    report = {"scenarios": len(dirs), "samples": written,
              "city_counts": {s: pd.Series([m["city"] for m in rows]).value_counts().to_dict() for s, rows in metas.items()},
              "type_counts": {s: pd.Series([m["agent_type"] for m in rows]).value_counts().to_dict()
                              for s, rows in metas.items()},
              "focal": {s: sum(m["focal"] for m in rows) for s, rows in metas.items()},
              "non_focal": {s: sum(not m["focal"] for m in rows) for s, rows in metas.items()}}
    with open(os.path.join(args.out, "counts.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
