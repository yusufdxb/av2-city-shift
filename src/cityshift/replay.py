"""EXPLORATORY: replay the Stage 4 drives from their executed accelerations and rescore them.

Registered in docs/preregistration/exploratory-followups.md (studies B and C). The closed loop is deterministic given
the scenario and the six executed accelerations, so every Stage 4 drive is rebuilt here without forecasts or planning.
For each scenario, arm and seed it records the registered outcomes (for the parity gate), the identity class of the
at-fault collision partner (study B), and outcomes scored inside the ALL/PATCH common on-route window (study C).

    PYTHONPATH=src python -m cityshift.replay --raw <train dir> --out runs/followups/replay_rows.parquet
    PYTHONPATH=src python -m cityshift.replay --stage 3 --raw <val dir> --out runs/followups/replay_rows_stage3.parquet
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

from . import closedloop as base
from . import closedloop_v2 as v2
from .closedloop import ACCELS, DIMS, EGO_DIMS, END, HANDOFF, REPLANS, EgoSim, in_front
from .closedloop_v3 import POOL_SHA256, checked_ids
from .route_censor import executed_decel, logged_decel, route_end_step, score_on_route
from .scene import Scene

STAGES = {4: {"seeded": ("ALL", "PATCH", "TRIM", "SHAM2", "MIX"), "table": "runs/stage4/closedloop_pool.parquet",
              "parity": "reports/followups/replay_parity.json", "n": 8140},
          3: {"seeded": ("ALL", "MULTI", "PATCH", "SHAM"), "table": "runs/stage3/closedloop_val.parquet",
              "parity": "reports/followups/replay_parity_stage3.json", "n": 24988}}
SINGLE = ("cv", "oracle", "static")


def drives(stage: int) -> list[tuple[str, int | None]]:
    return [(a, s) for a in STAGES[stage]["seeded"] for s in range(3)] + [(a, None) for a in SINGLE]


DRIVES = drives(4)
STOPPED_MPS = 0.5  # PATCH's trigger
DEPART_M = 2.0
PARITY = ("collision", "first_collision_step", "planner_hard_brake", "logged_hard_brake", "unnecessary_hard_brake")


def column(arm: str, seed: int | None) -> str:
    return arm if seed is None else f"{arm}_s{seed}"


def replay_ego(sc: Scene, accels) -> EgoSim:
    """Rebuild the ego from its six executed accelerations with the Stage 3/4 dynamics."""
    ego = v2.init_ego(sc)
    assert len(accels) == len(REPLANS), "expected one acceleration per replan"
    for t, a in zip(REPLANS, accels):
        (c,) = np.flatnonzero(ACCELS == a)
        base.execute(ego, t, int(c))
    return ego


def departing(sc: Scene) -> tuple[set[int], set[int], set[int]]:
    """Log-defined agent sets, identical for every arm.

    primary: dynamic, observed and < 0.5 m/s at t=49, later more than 2 m from its t=49 position at an observed step;
    stopped49: observed and < 0.5 m/s at t=49 (primary is a subset);
    secondary: stopped at any replan and later more than 2 m from that replan's position.
    """
    primary, stopped49, secondary = set(), set(), set()
    for i in range(len(sc.track_ids)):
        if i == sc.av or sc.types[i] not in base.DYN_IDX:
            continue
        for t in REPLANS:
            if not sc.valid[i, t] or np.linalg.norm(sc.vel[i, t]) >= STOPPED_MPS:
                continue
            later = np.arange(t + 1, END + 1)
            later = later[sc.valid[i, later]]
            moved = bool(len(later)) and float(np.linalg.norm(sc.pos[i, later] - sc.pos[i, t], axis=1).max()) > DEPART_M
            if t == HANDOFF:
                stopped49.add(i)
                if moved:
                    primary.add(i)
            if moved:
                secondary.add(i)
    return primary, stopped49, secondary


def partners(sc: Scene, pos: np.ndarray, head: np.ndarray) -> tuple[int, list[int]]:
    """First overlap step and the agents there that pass the at-fault contact test (mirrors closedloop_v2.score)."""
    steps = np.arange(HANDOFF + 1, END + 1)
    others = [i for i in range(len(sc.track_ids)) if i != sc.av and sc.types[i] in base.DYN_IDX]
    if not others:
        return -1, []
    o = np.asarray(others)
    c2, h2, valid = sc.pos[o][:, steps], sc.head[o][:, steps], sc.valid[o][:, steps]
    dims = DIMS[sc.types[o]]
    overlap = base.boxes_overlap(pos[steps][None], head[steps][None], EGO_DIMS[0], EGO_DIMS[1], c2, h2,
                                 dims[:, 0][:, None], dims[:, 1][:, None]) & valid
    if not overlap.any():
        return -1, []
    q = int(np.flatnonzero(overlap.any(0))[0])
    first = int(steps[q])
    fault = []
    for a in np.flatnonzero(overlap[:, q]):
        point = v2.contact_point(pos[first], float(head[first]), c2[a, q], float(h2[a, q]), dims[a])
        if point is not None and bool(in_front(pos[first], head[first], point)):
            fault.append(int(o[a]))
    return first, fault


def crossing(ego: EgoSim) -> int:
    return route_end_step(ego)


def censored(sc: Scene, ego: EgoSim, first: int, fault: list[int], cutoff: int) -> dict:
    """Registered outcomes counted only up to step `cutoff` (cutoff = END reproduces the registered score)."""
    r = score_on_route(executed_decel(ego)[None], logged_decel(sc)[None],
                       np.array([first if fault else -1]), np.array([cutoff]))
    return {k: (int(v[0]) if k == "scored_replans" else bool(v[0])) for k, v in r.items()}


def replay_scene(job: tuple[str, dict, int]) -> list[dict]:
    """All drives of one scenario: parity outcomes, partner classes, route-censoring primitives, and the ALL/PATCH
    common-window scores."""
    d, stored, stage = job
    sc = base.load_scene(d)
    primary, stopped49, secondary = departing(sc)
    out, egos, hits = [], {}, {}
    log_decel = logged_decel(sc).tolist()
    for arm, seed in drives(stage):
        col = column(arm, seed)
        ego = replay_ego(sc, stored[f"{col}_accels"])
        r = base.score_sim(sc, ego)
        pos, _, head = ego.states(END)
        first, fault = partners(sc, pos, head)
        egos[col], hits[col] = ego, (first, fault)
        row = {"scenario_id": sc.scenario_id, "city": sc.city, "arm": arm, "model_seed": -1 if seed is None else seed,
               **{k: r[k] for k in PARITY}, "progress": r["progress"], "distance_m": float(ego.s[END]),
               "route_m": float(ego.path.length_logged), "cross_step": crossing(ego),
               "partner_first_step": first, "n_fault_partners": len(fault),
               "partner_departing": any(i in primary for i in fault),
               "partner_stopped_not_departing": any(i in stopped49 and i not in primary for i in fault),
               "partner_moving": any(i not in stopped49 for i in fault),
               "partner_departing_any_replan": any(i in secondary for i in fault),
               "n_departing": len(primary), "n_stopped49": len(stopped49),
               "exec_decel": executed_decel(ego).tolist(), "fault_step": first if fault else -1,
               "log_decel": log_decel}
        full = censored(sc, ego, first, fault, END)
        row |= {f"full_{k}": v for k, v in full.items()}
        out.append(row)
    for seed in range(3):
        a, p = column("ALL", seed), column("PATCH", seed)
        cutoff = min(END, min(crossing(egos[a]), crossing(egos[p])) - 1)
        for row in out:
            if row["model_seed"] == seed and row["arm"] in ("ALL", "PATCH"):
                c = censored(sc, egos[column(row["arm"], seed)], *hits[column(row["arm"], seed)], cutoff)
                row |= {"common_cutoff": cutoff, **{f"common_{k}": v for k, v in c.items()}}
                own = censored(sc, egos[column(row["arm"], seed)], *hits[column(row["arm"], seed)],
                               min(END, row["cross_step"] - 1))  # descriptive only: unequal exposure across arms
                row |= {f"own_{k}": v for k, v in own.items()}
    return out


def parity(rows: pd.DataFrame, stored: pd.DataFrame, stage: int = 4) -> dict:
    """Replay vs stored Stage 4 outcomes, every scenario, arm and seed; the gate for studies B and C."""
    s = stored.set_index("scenario_id")
    bad: dict[str, int] = {}
    for arm, seed in drives(stage):
        col = column(arm, seed)
        r = rows[(rows.arm == arm) & (rows.model_seed == (-1 if seed is None else seed))].set_index("scenario_id")
        r = r.loc[s.index]
        for k in PARITY:
            mism = int((r[k].to_numpy() != s[f"{col}_{k}"].to_numpy()).sum())
            full = k if k == "first_collision_step" else f"full_{k}"
            if full in r:
                mism += int((r[full].to_numpy() != s[f"{col}_{k}"].to_numpy()).sum())
            if mism:
                bad[f"{col}_{k}"] = mism
        # the partner attribution must agree with the registered collision and first-overlap step
        mism = int(((r["n_fault_partners"] > 0).to_numpy() != s[f"{col}_collision"].to_numpy()).sum())
        mism += int((r["partner_first_step"].to_numpy() != s[f"{col}_first_collision_step"].to_numpy()).sum())
        if mism:
            bad[f"{col}_partners"] = mism
        a, b = r["progress"].to_numpy(float), s[f"{col}_progress"].to_numpy(float)
        both_nan = np.isnan(a) & np.isnan(b)
        mism = int((~both_nan & ~(np.abs(a - b) <= 1e-9)).sum())
        if mism:
            bad[f"{col}_progress"] = mism
    return {"stage": stage, "scenarios": int(len(s)), "drives_per_scenario": len(drives(stage)), "mismatches": bad,
            "passed": not bad}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=(3, 4), default=4)
    ap.add_argument("--raw", required=True, help="Argoverse 2 TRAIN (Stage 4) or VAL (Stage 3) directory")
    ap.add_argument("--pool", default="runs/replication_pool.npy")
    ap.add_argument("--table", default=None, help="registered closed-loop table (default per stage)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parity-out", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="smoke test on the first N scenarios (no parity file)")
    args = ap.parse_args()
    base.score = v2.score  # the Stage 3/4 contact-based scorer, as closedloop_v3 sets it
    cfg = STAGES[args.stage]
    stored = pd.read_parquet(args.table or cfg["table"])
    assert len(stored) == cfg["n"] and not stored.scenario_id.duplicated().any()
    if args.stage == 4:
        assert set(stored.scenario_id.astype(str)) == set(checked_ids(args.pool, POOL_SHA256))
    stored = stored.sort_values("scenario_id").reset_index(drop=True)
    if args.limit:
        stored = stored.iloc[:args.limit]
    keep = [f"{column(a, s)}_accels" for a, s in drives(args.stage)]
    jobs = [(os.path.join(args.raw, sid), {k: list(row[k]) for k in keep}, args.stage)
            for sid, row in zip(stored.scenario_id, stored[keep].to_dict("records"))]
    assert all(glob.glob(j[0]) for j in jobs), "missing raw scenario directories"
    start = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        rows = [r for rs in pool.imap(replay_scene, jobs, chunksize=16) for r in rs]
    rows = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows.to_parquet(args.out, index=False)
    report = parity(rows, stored, args.stage) | {"runtime_sec": round(time.time() - start)}
    parity_out = args.parity_out or cfg["parity"]
    print(json.dumps(report))
    if not args.limit:
        os.makedirs(os.path.dirname(os.path.abspath(parity_out)), exist_ok=True)
        with open(parity_out, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
    if not report["passed"]:
        raise SystemExit("replay parity FAILED: studies B and C must not be analysed")


if __name__ == "__main__":
    main()
