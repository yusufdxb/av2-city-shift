"""EXPLORATORY planner sensitivity sweep (not registered; on already-scored replication-pool scenes).

Does PATCH's braking advantage over the focal-only model survive other planner settings? Re-runs closed loop v2 with
the risk weight and speed cap changed, and records each drive's minimum executed and logged deceleration so any
hard-brake threshold can be scored afterwards. Arms: ALL, PATCH, constant velocity, oracle; model seed 0 only.
Nothing here changes a registered result.

    PYTHONPATH=src python -m cityshift.sensitivity --raw <train dir> --out runs/sensitivity/sweep.parquet
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
import torch

from . import closedloop as base
from . import closedloop_v2 as v2
from .closedloop import DT, END, EXEC_STEPS, HANDOFF, REPLANS
from .closedloop_v3 import POOL_SHA256, checked_ids, run_forecast

RISK_WEIGHTS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)  # 100 = registered; the learned model's few-percent modes trade off against progress (<= 1) in this range
CAPS = {"v2": lambda v0: max(v0 + 3.0, 15.0), "tight": lambda v0: max(v0 + 1.0, 8.0)}
ARMS = ("ALL", "PATCH", "cv", "oracle")
_score_sim = base.score_sim


def score_with_decel(sc, ego) -> dict:
    """Registered v2 score plus minimum executed / logged deceleration and a past-the-route flag."""
    r = _score_sim(sc, ego)
    r["min_exec_decel"] = float(min((ego.v[t + EXEC_STEPS] - ego.v[t]) / (EXEC_STEPS * DT) for t in REPLANS))
    sp = np.linalg.norm(np.diff(sc.pos[sc.av], axis=0), axis=1) / DT
    vs = np.convolve(sp, np.ones(5) / 5, mode="same")
    r["min_log_decel"] = float((vs[HANDOFF + 10 : END - 4] - vs[HANDOFF : END - 14]).min())
    r["beyond_logged_route"] = bool(ego.s[END] > ego.path.length_logged)
    r.pop("accels", None)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--pool", default="runs/replication_pool.npy")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=500)
    args = ap.parse_args()
    ids = checked_ids(args.pool, POOL_SHA256)
    subset = set(np.random.default_rng(2026).choice(sorted(ids), size=args.n, replace=False).tolist())
    dirs = [d for d in sorted(glob.glob(os.path.join(args.raw, "*"))) if os.path.basename(d) in subset]
    assert len(dirs) == args.n
    device = torch.device("cuda")
    models = base.load_models({"ALL_s0": "runs/ALL/seed0/model.pt"}, device)
    base.init_ego = v2.init_ego
    base.score = v2.score
    base.score_sim = score_with_decel
    rows, start = [], time.time()
    ctx = mp.get_context("fork")
    for offset in range(0, len(dirs), args.chunk):
        with ctx.Pool(args.workers) as pool:
            base._SCENES = pool.map(base._load, dirs[offset:offset + args.chunk], chunksize=8)
        n = len(base._SCENES)
        for cap_name, cap in CAPS.items():
            for weight in RISK_WEIGHTS:
                base.W_RISK = weight  # read by closedloop.plan in the parent and in the forked workers
                v2.speed_cap = cap  # read by closedloop_v2.init_ego
                with ctx.Pool(args.workers) as pool:
                    for arm in ARMS:
                        key = "ALL_s0" if arm in ("ALL", "PATCH") else None
                        scores, _, _ = run_forecast(pool, n, arm, key, models, device, 0)
                        for sc, s in zip(base._SCENES, scores):
                            rows.append({"scenario_id": sc.scenario_id, "city": sc.city, "cap": cap_name,
                                         "risk_weight": weight, "arm": arm,
                                         **{k: s[k] for k in ("collision", "unnecessary_hard_brake", "progress",
                                                              "min_exec_decel", "min_log_decel", "beyond_logged_route")}})
        print(json.dumps({"done": offset + n, "of": len(dirs), "sec": round(time.time() - start)}), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out)
    print(f"wrote {args.out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
