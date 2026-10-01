"""CPU pilot of the stop-line harness (closedloop_v4) against the registered harness on the TRAIN development slice.

Checks feasibility and the controls before anything is registered or run on the pool: the stand-still forecast must
collide at least twice as often as the focal-only model, the replayed human drive must stay under 1% at-fault
collisions, and the stop line must not need hard brakes by itself. Model arms run on the CPU (pilot only; production
runs paired with GPU rows must run on the GPU). No forecast intervention arms are run here.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python scripts/pilot_stopline.py --raw <train dir>
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

from cityshift import closedloop as base
from cityshift import closedloop_v2 as v2
from cityshift import closedloop_v4 as v4
from cityshift.closedloop import REPLANS
from cityshift.closedloop_v3 import run_forecast

ARMS = (("cv", None), ("static", None), ("oracle", None), ("ALL", "ALL_s0"))


def infeasible_replans(sc, accels) -> int:
    """Replans at which no candidate satisfied the stop line (rebuilt from the executed accelerations)."""
    ego = v2.init_ego(sc)
    bad = 0
    for t, a in zip(REPLANS, accels):
        bad += int(not v4.feasible(ego, t).any())
        (c,) = np.flatnonzero(base.ACCELS == a)
        base.execute(ego, t, int(c))
    return bad


def summarize(df: pd.DataFrame) -> dict:
    out = {}
    for harness, h in df.groupby("harness"):
        arms = {}
        for arm, g in h.groupby("arm"):
            arms[arm] = {"collision": float(g.collision.mean()), "unnecessary_hard_brake": float(g.unnecessary_hard_brake.mean()),
                         "planner_hard_brake": float(g.planner_hard_brake.mean()),
                         "reached_route_end": float((g.progress >= 0.999).mean()),
                         "median_progress": float(g.progress.median()),
                         "infeasible_replans": int(g.infeasible.sum()) if "infeasible" in g else None,
                         "collision_route_gt_1m": float(g[g.route_m > 1].collision.mean())}
        all_c, st_c = arms["ALL"]["collision"], arms["static"]["collision"]
        all_c1, st_c1 = arms["ALL"]["collision_route_gt_1m"], arms["static"]["collision_route_gt_1m"]
        out[harness] = {"arms": arms, "STATIC_over_ALL": st_c / all_c if all_c else None,
                        "STATIC_over_ALL_route_gt_1m": st_c1 / all_c1 if all_c1 else None,
                        "LOG_collision": float(h[h.arm == "log"].collision.mean()),
                        "ALL_unnecessary_brake_events": int(h[h.arm == "ALL"].unnecessary_hard_brake.sum())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--scenarios", default="runs/dev_scenarios.npy")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out", default="runs/pilot_stopline/rows.parquet")
    ap.add_argument("--summary", default="reports/pilot/stopline_pilot.json")
    args = ap.parse_args()
    assert not torch.cuda.is_available(), "pilot runs on the CPU: set CUDA_VISIBLE_DEVICES=''"
    ids = list(np.load(args.scenarios, allow_pickle=True))
    ids = ids[:args.limit] if args.limit else ids
    dirs = [os.path.join(args.raw, str(i)) for i in ids]
    device = torch.device("cpu")
    models = base.load_models({"ALL_s0": "runs/ALL/seed0/model.pt"}, device)
    base.init_ego, base.score = v2.init_ego, v2.score
    registered_plan = base.plan
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        base._SCENES = pool.map(base._load, dirs, chunksize=8)
    n = len(base._SCENES)
    rows, start = [], time.time()
    for harness, planner in (("registered", registered_plan), ("stopline", v4.plan)):
        base.plan = planner  # set before forking so the workers plan with it
        with ctx.Pool(args.workers) as pool:
            for arm, key in ARMS:
                scores, _, _ = run_forecast(pool, n, arm, key, models, device, 0, calibrate=False)
                for sc, s in zip(base._SCENES, scores):
                    rows.append({"harness": harness, "arm": arm, "scenario_id": sc.scenario_id,
                                 "route_m": float(base.Path.from_scene(sc).length_logged),
                                 **{k: s[k] for k in ("collision", "unnecessary_hard_brake", "planner_hard_brake",
                                                      "progress", "accels")}})
                print(json.dumps({"harness": harness, "arm": arm, "sec": round(time.time() - start)}), flush=True)
        for sc in base._SCENES:
            s = base.score_log(sc)
            rows.append({"harness": harness, "arm": "log", "scenario_id": sc.scenario_id,
                         "route_m": float(base.Path.from_scene(sc).length_logged), "collision": s["collision"],
                         "unnecessary_hard_brake": s["unnecessary_hard_brake"], "planner_hard_brake": False,
                         "progress": 1.0, "accels": []})
    base.plan = registered_plan
    df = pd.DataFrame(rows)
    by_id = {sc.scenario_id: sc for sc in base._SCENES}
    stop = df.harness == "stopline"
    df.loc[stop & (df.arm != "log"), "infeasible"] = [infeasible_replans(by_id[s], a) for s, a in
                                                       zip(df[stop & (df.arm != "log")].scenario_id,
                                                           df[stop & (df.arm != "log")].accels)]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_parquet(args.out, index=False)
    result = {"scenarios": n, "runtime_sec": round(time.time() - start), "stop_decel": v4.STOP_DECEL,
              **summarize(df)}
    os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)
    with open(args.summary, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({h: {k: result[h][k] for k in ("STATIC_over_ALL", "STATIC_over_ALL_route_gt_1m", "LOG_collision")}
                      for h in ("registered", "stopline")}))


if __name__ == "__main__":
    main()
