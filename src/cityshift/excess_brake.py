"""v2 study R secondary outcome: excess hard brakes, judged against the logged futures (CPU replay).

Registered in docs/preregistration/v2-fresh-reserve.md. A replan is an *excess hard brake* if the chosen acceleration is
-4 m/s^2 or harder, the logged human never decelerated that hard, every selected agent's logged future covers the 4 s
look-ahead (so only replans 49, 59 and 69 can qualify), and some candidate of -2 m/s^2 or gentler has zero planner hit
probability when the selected agents are given their logged futures (the ORACLE forecast) and, in the stop-line harness,
satisfies the stop-line rule. A drive counts if any replan qualifies. Drives are rebuilt from their stored accelerations
(the closed loop is deterministic given them), so no forecasts are rerun.

    PYTHONPATH=src python -m cityshift.excess_brake --raw <train dir> --parts runs/v2/study_r_v4 --harness v4 \
        --out reports/v2/excess_brake_v4.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

from . import closedloop as base
from . import closedloop_v4 as v4
from .analysis_v2 import R_ARMS, city_bootstrap, load_parts, summarise
from .closedloop import (ACCELS, CHECK_STEPS, DIMS, EGO_DIMS, END, MARGIN_M, PLAN_STEPS, REPLANS, boxes_overlap,
                         in_front, profiles)
from .replay import replay_ego

GENTLE = -2.0
HARD = -4.0
_ROWS: pd.DataFrame | None = None
_HARNESS = "v4"
_RAW = ""


def oracle_hits(sc, ego, t: int, agents: list[int], stopline: bool) -> np.ndarray:
    """Planner hit indicator per candidate with every selected agent given its logged future (closedloop.plan)."""
    _, dist = profiles(ego.v[t], ego.vmax, PLAN_STEPS)
    if stopline:
        dist = np.minimum(dist, max(ego.path.length_logged - ego.s[t], 0.0))
    if not agents:
        return np.zeros(len(ACCELS), bool)
    traj = np.stack([base.oracle_forecast(sc, i, t) for i in agents])[:, None]  # [M, 1, 60, 2]
    ep, eh = ego.path.at(ego.s[t] + dist[:, :CHECK_STEPS])
    ap = traj[:, :, :CHECK_STEPS]
    prev = np.concatenate([np.repeat(sc.pos[agents, t][:, None, None], 1, 1), ap[:, :, :-1]], 2)
    dxy = ap - prev
    ah = np.arctan2(dxy[..., 1], dxy[..., 0])
    ah = np.where(np.linalg.norm(dxy, axis=-1) < 0.05, sc.head[agents, t][:, None, None], ah)
    dims = DIMS[sc.types[agents]]
    c1, h1 = ep[:, None, None], eh[:, None, None]
    hit = boxes_overlap(c1, h1, EGO_DIMS[0] + 2 * MARGIN_M, EGO_DIMS[1] + 2 * MARGIN_M, ap[None], ah[None],
                        dims[:, 0][None, :, None, None], dims[:, 1][None, :, None, None])
    hit &= in_front(c1, h1, ap[None])
    return hit.any(-1).any(-1).any(-1)  # [C]: any agent, any step


def excess(sc, accels, logged_hard: bool, stopline: bool) -> bool:
    if logged_hard:
        return False
    ego = replay_ego(sc, accels)
    for t, a in zip(REPLANS, accels):
        if a > HARD or t + CHECK_STEPS > END:
            continue
        scw = base.scene_with_ego(sc, ego, t)
        agents = base.select_agents(scw, ego, t)
        if any(not sc.valid[i, t + 1:t + CHECK_STEPS + 1].all() for i in agents):
            continue  # some selected agent's logged future does not cover the look-ahead
        safe = ~oracle_hits(sc, ego, t, agents, stopline)
        if stopline:
            safe &= v4.feasible(ego, t)
        if (safe & (ACCELS >= GENTLE)).any():
            return True
    return False


def _scene(k: int) -> dict:
    row = _ROWS.iloc[k]
    sc = base._load(os.path.join(_RAW, row.scenario_id))
    out = {"scenario_id": row.scenario_id, "city": row.city}
    for a in R_ARMS:
        for s in range(3):
            p = f"{a}_s{s}"
            out[p] = excess(sc, list(row[f"{p}_accels"]), bool(row[f"{p}_logged_hard_brake"]), _HARNESS == "v4")
    out["oracle"] = excess(sc, list(row["oracle_accels"]), bool(row["oracle_logged_hard_brake"]), _HARNESS == "v4")
    return out


def main() -> None:
    global _ROWS, _HARNESS, _RAW
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--parts", required=True)
    ap.add_argument("--harness", required=True, choices=("v4", "v2"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    _ROWS, _HARNESS, _RAW = load_parts(args.parts), args.harness, args.raw
    with mp.get_context("fork").Pool(args.workers) as pool:
        flags = pd.DataFrame(pool.map(_scene, range(len(_ROWS)), chunksize=16))
    data = pd.DataFrame({"city": flags.city, **{a: flags[[f"{a}_s{s}" for s in range(3)]].to_numpy(float).mean(1)
                                                 for a in R_ARMS}})
    eff = city_bootstrap(data, R_ARMS, lambda m: {f"{a}_reduction": (m["ALL"] - m[a]) / m["ALL"] for a in R_ARMS[1:]}
                         | {"sham_gap": (m["SHAM2"] - m["PATCH"]) / m["ALL"]})
    result = {"_note": "v2 study R secondary outcome (docs/preregistration/v2-fresh-reserve.md): excess hard brakes "
                       "judged against logged futures; 95% intervals; seeds averaged, cities equal-weighted.",
              "harness": args.harness, "scenarios": len(flags),
              "rates": {a: float(data[a].mean()) for a in R_ARMS} | {"oracle": float(flags.oracle.mean())},
              "effects": {k: summarise(v, {"ci95": 0.95}) for k, v in eff.items()}}
    for e in result["effects"].values():
        e.pop("draws", None)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({k: v["point"] for k, v in result["effects"].items()}, indent=2))


if __name__ == "__main__":
    main()
