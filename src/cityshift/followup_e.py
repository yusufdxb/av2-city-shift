"""EXPLORATORY study E runner: H10-H12 arms in the stop-line harness (closedloop_v4) on the replication pool.

Registered in docs/preregistration/exploratory-followups.md (study E). Writes one parquet part per chunk (Stage 4 column
layout) so an interrupted run resumes where it stopped; built to share the GPU: --mem-fraction caps this process's
VRAM (an overrun fails this job, not the other GPU user).

    PYTHONPATH=src python -m cityshift.followup_e --raw <train dir> --out-dir runs/followups/study_e --mem-fraction 0.3
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import time

import pandas as pd
import torch

from . import closedloop as base
from . import closedloop_v2 as v2
from . import closedloop_v4 as v4
from .closedloop_v3 import POOL_SHA256, checked_ids, run_forecast
from .run_manifest import run_settings, validate_manifest, validate_parts

MODEL_ARMS = ("ALL", "PATCH", "SHAM2", "TRIM")  # PATCH before SHAM2: its per-replan dose is SHAM2's reference
ANALYTIC = ("cv", "oracle", "static")


def score_chunk(pool, seeds: list[int], models: dict, device) -> list[dict]:
    n = len(base._SCENES)
    rows = [{"scenario_id": sc.scenario_id, "city": sc.city} for sc in base._SCENES]
    for seed in seeds:
        dose = None
        for arm in MODEL_ARMS:
            scores, _, d = run_forecast(pool, n, arm, f"ALL_s{seed}", models, device, seed,
                                        dose if arm == "SHAM2" else None, calibrate=False)
            if arm == "PATCH":
                dose = d
            for row, s, dd in zip(rows, scores, d):
                row.update({f"{arm}_s{seed}_{k}": v for k, v in s.items()})
                row[f"{arm}_s{seed}_dose_by_replan"] = dd.tolist()
    for arm in ANALYTIC:
        scores, _, _ = run_forecast(pool, n, arm, None, models, device, 99, calibrate=False)
        for row, s in zip(rows, scores):
            row.update({f"{arm}_{k}": v for k, v in s.items()})
    for row, sc in zip(rows, base._SCENES):
        row.update({f"log_{k}": v for k, v in base.score_log(sc).items()})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--pool", default="runs/replication_pool.npy")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mem-fraction", type=float, default=0.0, help="cap this process's share of GPU memory")
    ap.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    if not seeds or not set(seeds) <= {0, 1, 2} or len(set(seeds)) != len(seeds):
        ap.error("seeds must be a unique subset of 0,1,2")
    if args.chunk <= 0 or args.workers <= 0 or args.limit < 0 or not 0 <= args.mem_fraction <= 1:
        ap.error("chunk and workers must be positive; limit nonnegative; mem-fraction between 0 and 1")
    ids = sorted(checked_ids(args.pool, POOL_SHA256))
    ids = ids[:args.limit] if args.limit else ids
    if not ids:
        ap.error("no scenarios selected")
    device = torch.device(args.device)
    checkpoints = {f"ALL_s{s}": f"{args.runs}/ALL/seed{s}/model.pt" for s in seeds}
    settings = run_settings("closedloop_v4", str(device), seeds, args.pool, ids, args.raw, checkpoints,
                            arms=list(MODEL_ARMS), analytic=list(ANALYTIC), stop_decel=v4.STOP_DECEL,
                            chunk=args.chunk, workers=args.workers, limit=args.limit, mem_fraction=args.mem_fraction,
                            calibrate=False)
    parts = sorted(glob.glob(os.path.join(args.out_dir, "part_*.parquet")))
    validate_manifest(os.path.join(args.out_dir, "manifest.json"), settings, parts)
    validate_parts(args.out_dir, ids, args.chunk, seeds)
    if device.type == "cuda" and args.mem_fraction:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    models = base.load_models(checkpoints, device)
    base.init_ego, base.score = v2.init_ego, v2.score
    v4.install()  # stop-line planner, inherited by every worker forked below
    ctx = mp.get_context("fork")
    start, done_drives = time.time(), 0
    for offset in range(0, len(ids), args.chunk):
        part = os.path.join(args.out_dir, f"part_{offset:05d}.parquet")
        if os.path.exists(part):
            continue
        dirs = [os.path.join(args.raw, i) for i in ids[offset:offset + args.chunk]]
        with ctx.Pool(args.workers) as pool:
            base._SCENES = pool.map(base._load, dirs, chunksize=8)
        with ctx.Pool(args.workers) as pool:
            rows = score_chunk(pool, seeds, models, device)
        pd.DataFrame(rows).to_parquet(part + ".tmp", index=False)
        os.replace(part + ".tmp", part)
        done_drives += len(rows) * (len(MODEL_ARMS) * len(seeds) + len(ANALYTIC))
        peak = torch.cuda.max_memory_reserved() / 2**20 if device.type == "cuda" else 0
        print(json.dumps({"done": offset + len(rows), "of": len(ids), "sec": round(time.time() - start),
                          "drives_per_sec": round(done_drives / (time.time() - start), 1),
                          "peak_reserved_mib": round(peak)}), flush=True)
    parts = sorted(glob.glob(os.path.join(args.out_dir, "part_*.parquet")))
    print(json.dumps({"parts": len(parts), "runtime_sec": round(time.time() - start)}))


if __name__ == "__main__":
    main()
