"""v2 studies R and G runner: fresh-reserve closed loop with newly trained focal-only checkpoints (ALLR).

Registered in docs/preregistration/v2-fresh-reserve.md. Scores the 10,000-scenario v2 reserve in the stop-line harness
(primary, closedloop_v4) or the Stage 4 full-drive harness (secondary, closedloop_v2) with the Stage 4 interventions
and the study G factorial arms. One parquet part per chunk (Stage 4 column layout) so an interrupted run resumes; the
run manifest refuses to mix settings, sources or checkpoints across parts.

    PYTHONPATH=src python -m cityshift.study_r --raw <train dir> --harness v4 --out-dir runs/v2/study_r_v4
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
from .closedloop_v3 import checked_ids, run_forecast
from .run_manifest import run_settings, validate_manifest, validate_parts

RESERVE_SHA256 = "286a9423ffd907eebf7d67a6c7f66b2edfc6d1c4309e2a7f03e2633f8aac52ed"
MODEL_ARMS = ("ALL", "PATCH", "SHAM2", "TRIM", "GEO", "PROB")  # PATCH before SHAM2: its dose is SHAM2's reference
ANALYTIC = ("cv", "oracle", "static")


def score_chunk(pool, seeds: list[int], models: dict, device, arms: tuple[str, ...] = MODEL_ARMS) -> list[dict]:
    n = len(base._SCENES)
    rows = [{"scenario_id": sc.scenario_id, "city": sc.city} for sc in base._SCENES]
    for seed in seeds:
        dose = None
        for arm in arms:
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
    ap.add_argument("--pool", default="runs/v2/reserve_ids.npy")
    ap.add_argument("--runs", default="runs/v2/ALLR", help="directory with seed{0,1,2}/model.pt")
    ap.add_argument("--harness", required=True, choices=("v4", "v2"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--arms", default=",".join(MODEL_ARMS), help="model arms; study S scores ALL only")
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke runs only; production scores the full reserve")
    ap.add_argument("--smoke", action="store_true", help="<= 100 development-slice scenarios (runs/dev_scenarios.npy)")
    ap.add_argument("--mem-fraction", type=float, default=0.0, help="cap this process's share of GPU memory")
    ap.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    arms = tuple(a for a in MODEL_ARMS if a in args.arms.split(","))
    if not arms or set(args.arms.split(",")) - set(MODEL_ARMS) or ("SHAM2" in arms and "PATCH" not in arms):
        ap.error("arms must be a subset of " + ",".join(MODEL_ARMS) + " (SHAM2 needs PATCH)")
    if not seeds or not set(seeds) <= {0, 1, 2} or len(set(seeds)) != len(seeds):
        ap.error("seeds must be a unique subset of 0,1,2")
    if args.chunk <= 0 or args.workers <= 0 or args.limit < 0 or not 0 <= args.mem_fraction <= 1:
        ap.error("chunk and workers must be positive; limit nonnegative; mem-fraction between 0 and 1")
    if args.smoke:  # the registration allows smoke runs on the development slice only, never on the reserve
        if not 0 < args.limit <= 100 or os.path.basename(args.pool) != "dev_scenarios.npy":
            ap.error("--smoke requires --pool .../dev_scenarios.npy and --limit 1..100")
        ids = sorted(checked_ids(args.pool))[:args.limit]
    else:
        if args.limit:
            ap.error("production scores the full reserve; use --smoke for limited runs")
        ids = sorted(checked_ids(args.pool, RESERVE_SHA256))
    device = torch.device(args.device)
    checkpoints = {f"ALL_s{s}": f"{args.runs}/seed{s}/model.pt" for s in seeds}
    harness = "closedloop_v4" if args.harness == "v4" else "closedloop_v2"
    settings = run_settings(harness, str(device), seeds, args.pool, ids, args.raw, checkpoints, runner="study_r",
                            arms=list(arms), analytic=list(ANALYTIC),
                            stop_decel=v4.STOP_DECEL if args.harness == "v4" else None, chunk=args.chunk,
                            workers=args.workers, limit=args.limit, mem_fraction=args.mem_fraction, calibrate=False)
    parts = sorted(glob.glob(os.path.join(args.out_dir, "part_*.parquet")))
    validate_manifest(os.path.join(args.out_dir, "manifest.json"), settings, parts)
    validate_parts(args.out_dir, ids, args.chunk, seeds, arms=arms)
    if device.type == "cuda" and args.mem_fraction:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    models = base.load_models(checkpoints, device)
    base.init_ego, base.score = v2.init_ego, v2.score
    if args.harness == "v4":
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
            rows = score_chunk(pool, seeds, models, device, arms)
        pd.DataFrame(rows).to_parquet(part + ".tmp", index=False)
        os.replace(part + ".tmp", part)
        done_drives += len(rows) * (len(arms) * len(seeds) + len(ANALYTIC))
        peak = torch.cuda.max_memory_reserved() / 2**20 if device.type == "cuda" else 0
        print(json.dumps({"done": offset + len(rows), "of": len(ids), "sec": round(time.time() - start),
                          "drives_per_sec": round(done_drives / (time.time() - start), 1),
                          "peak_reserved_mib": round(peak)}), flush=True)
    parts = sorted(glob.glob(os.path.join(args.out_dir, "part_*.parquet")))
    print(json.dumps({"parts": len(parts), "runtime_sec": round(time.time() - start)}))


if __name__ == "__main__":
    main()
