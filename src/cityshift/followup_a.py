"""EXPLORATORY study A runner: TRIM without the CV fallback (TRIMNF) vs CV on the same agents (PATCHSUB).

Registered in docs/preregistration/exploratory-followups.md. Runs the Stage 4 closed loop (closedloop_v3, v2 scoring) on
the replication pool for the requested arms and seeds and writes one row per scenario in the Stage 4 column layout.
An ALL rerun is compared with the stored Stage 4 rows (the registration requires exact agreement on 100 scenarios).

    PYTHONPATH=src python -m cityshift.followup_a --raw <train dir> --out runs/followups/study_a.parquet
    PYTHONPATH=src python -m cityshift.followup_a --raw <train dir> --out /tmp/a_smoke.parquet --arms ALL --limit 100
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
import torch

from . import closedloop as base
from . import closedloop_v2 as v2
from .closedloop_v3 import POOL_SHA256, checked_ids, run_forecast
from .run_manifest import file_sha256, run_settings, validate_manifest

ARMS = ("ALL", "TRIMNF", "PATCHSUB")
PARITY = ("collision", "first_collision_step", "planner_hard_brake", "logged_hard_brake", "unnecessary_hard_brake",
          "accels")


def parity(rows: pd.DataFrame, stage4: str, seeds: list[int]) -> dict:
    """ALL rerun vs the stored Stage 4 ALL rows on the scenarios run here."""
    s = pd.read_parquet(stage4).set_index("scenario_id").loc[rows.scenario_id]
    r = rows.set_index("scenario_id")
    bad = {}
    for seed in seeds:
        for k in PARITY:
            col = f"ALL_s{seed}_{k}"
            a, b = r[col].tolist(), s[col].tolist()
            mism = sum(not np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))
            if mism:
                bad[col] = mism
    return {"scenarios": int(len(r)), "seeds": seeds, "mismatches": bad, "passed": not bad}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--pool", default="runs/replication_pool.npy")
    ap.add_argument("--stage4", default="runs/stage4/closedloop_pool.parquet")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="TRIMNF,PATCHSUB")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0, help="first scenario (sorted pool order); for split runs")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = ap.parse_args()
    arms, seeds = args.arms.split(","), [int(x) for x in args.seeds.split(",")]
    if (not arms or not set(arms) <= set(ARMS) or len(set(arms)) != len(arms) or not seeds
            or not set(seeds) <= {0, 1, 2} or len(set(seeds)) != len(seeds)):
        ap.error("arms must be a subset of ALL,TRIMNF,PATCHSUB and seeds of 0,1,2")
    if args.chunk <= 0 or args.workers <= 0 or args.limit < 0 or args.offset < 0:
        ap.error("chunk and workers must be positive; limit and offset must be nonnegative")
    ids = sorted(checked_ids(args.pool, POOL_SHA256))
    ids = ids[args.offset:args.offset + args.limit] if args.limit else ids[args.offset:]
    if not ids:
        ap.error("no scenarios selected")
    dirs = [os.path.join(args.raw, i) for i in ids]
    if not all(os.path.isdir(d) for d in dirs):
        ap.error("missing raw scenario directories")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    checkpoints = {f"ALL_s{s}": f"{args.runs}/ALL/seed{s}/model.pt" for s in seeds}
    settings = run_settings("closedloop_v3", str(device), seeds, args.pool, ids, args.raw, checkpoints,
                            arms=arms, chunk=args.chunk, workers=args.workers, limit=args.limit, offset=args.offset,
                            calibrate=False, stage4_sha256=file_sha256(args.stage4) if "ALL" in arms else None)
    validate_manifest(args.out + ".manifest.json", settings, [args.out, args.out + ".report.json"])
    models = base.load_models(checkpoints, device)
    base.init_ego = v2.init_ego
    base.score = v2.score
    ctx = mp.get_context("fork")
    rows, start = [], time.time()
    for offset in range(0, len(dirs), args.chunk):
        with ctx.Pool(args.workers) as pool:
            base._SCENES = pool.map(base._load, dirs[offset:offset + args.chunk], chunksize=8)
        n = len(base._SCENES)
        chunk_rows = [{"scenario_id": sc.scenario_id, "city": sc.city} for sc in base._SCENES]
        with ctx.Pool(args.workers) as pool:
            for arm in arms:
                for seed in seeds:
                    scores, _, dose = run_forecast(pool, n, arm, f"ALL_s{seed}", models, device, seed, calibrate=False)
                    for row, score, d in zip(chunk_rows, scores, dose):
                        row.update({f"{arm}_s{seed}_{k}": v for k, v in score.items()})
                        row[f"{arm}_s{seed}_eligible_by_replan"] = d.tolist()
        rows.extend(chunk_rows)
        print(json.dumps({"done": offset + n, "of": len(dirs), "sec": round(time.time() - start),
                          "drives_per_sec": round((offset + n) * len(arms) * len(seeds) / (time.time() - start), 1)}),
              flush=True)
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_parquet(args.out + ".tmp", index=False)
    os.replace(args.out + ".tmp", args.out)
    report = {"device": str(device), "arms": arms, "seeds": seeds, "scenarios": len(out),
              "runtime_sec": round(time.time() - start)}
    if "ALL" in arms:
        report["ALL_parity_vs_stage4"] = parity(out, args.stage4, seeds)
    print(json.dumps(report))
    with open(args.out + ".report.json", "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
