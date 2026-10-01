"""Stop-line compliance diagnostic for study E: replay every stored drive, count replans with no feasible candidate
and measure how far any drive passed the end of the logged route.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python scripts/diagnose_stopline.py --raw <train dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

from cityshift import closedloop as base
from cityshift import closedloop_v2 as v2
from cityshift import closedloop_v4 as v4
from cityshift.closedloop import END, HANDOFF, REPLANS

COLS = [f"{a}_s{s}" for a in ("ALL", "PATCH", "SHAM2", "TRIM") for s in range(3)] + ["cv", "oracle", "static"]


def diag(job):
    d, acc = job
    sc = base.load_scene(d)
    out = []
    for col in COLS:
        ego = v2.init_ego(sc)
        infeasible = 0
        for t, a in zip(REPLANS, acc[col]):
            infeasible += int(not v4.feasible(ego, t).any())
            (c,) = np.flatnonzero(base.ACCELS == a)
            base.execute(ego, t, int(c))
        length = ego.path.length_logged
        out.append((sc.scenario_id, col, infeasible, float(ego.s[HANDOFF:END + 1].max() - length), length))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--parts", default="runs/followups/study_e")
    ap.add_argument("--out", default="reports/followups/study_e_stopline_diagnostics.json")
    args = ap.parse_args()
    parts = sorted(glob.glob(os.path.join(args.parts, "part_*.parquet")))
    d = pd.concat([pd.read_parquet(p, columns=["scenario_id"] + [f"{c}_accels" for c in COLS]) for p in parts])
    jobs = [(os.path.join(args.raw, r["scenario_id"]), {c: list(r[f"{c}_accels"]) for c in COLS})
            for r in d.to_dict("records")]
    with mp.get_context("fork").Pool(20) as pool:
        rows = [x for xs in pool.imap(diag, jobs, chunksize=16) for x in xs]
    r = pd.DataFrame(rows, columns=["scenario_id", "arm", "infeasible_replans", "max_overshoot_m", "route_m"])
    g = r.assign(arm=r.arm.str.replace(r"_s\d", "", regex=True)).groupby("arm")
    summary = {"drives": len(r), "drives_with_infeasible_replan": int((r.infeasible_replans > 0).sum()),
               "infeasible_replans": int(r.infeasible_replans.sum()),
               "drives_past_route_end_gt_1cm": int((r.max_overshoot_m > 0.01).sum()),
               "max_overshoot_m": float(r.max_overshoot_m.max()),
               "short_routes_le_1m_drives": int((r.route_m <= 1).sum()),
               "short_route_drives_past_end_gt_1cm": int(((r.route_m <= 1) & (r.max_overshoot_m > 0.01)).sum()),
               "by_arm": {a: {"drives": int(len(x)), "infeasible": int((x.infeasible_replans > 0).sum()),
                              "past_end_gt_1cm": int((x.max_overshoot_m > 0.01).sum()),
                              "max_overshoot_m": round(float(x.max_overshoot_m.max()), 3)} for a, x in g}}
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_arm"}))


if __name__ == "__main__":
    main()
