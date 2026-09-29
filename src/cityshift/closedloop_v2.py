"""Stage 3 closed loop: observable speed cap, contact-based fault, and PATCH arm."""

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
from .closedloop import (DIMS, DT, EGO_DIMS, END, HANDOFF, HARD_BRAKE, REPLANS,
                         EgoSim, Path, boxes_overlap, in_front)
from .scene import Scene


def speed_cap(v0: float) -> float:
    """Stage 3 cap using only ego speed at the handoff, in metres per second."""
    return max(v0 + 3.0, 15.0)


def init_ego(sc: Scene) -> EgoSim:
    ego = EgoSim(path=Path.from_scene(sc), vmax=speed_cap(float(np.linalg.norm(sc.vel[sc.av, HANDOFF]))))
    ego.v[HANDOFF] = float(np.linalg.norm(sc.vel[sc.av, HANDOFF]))
    return ego


def _corners(center: np.ndarray, heading: float, length: float, width: float) -> np.ndarray:
    local = np.array([[length / 2, width / 2], [-length / 2, width / 2],
                      [-length / 2, -width / 2], [length / 2, -width / 2]])
    c, s = np.cos(heading), np.sin(heading)
    return local @ np.array([[c, s], [-s, c]]) + center


def _clip(subject: np.ndarray, edge_a: np.ndarray, edge_b: np.ndarray) -> np.ndarray:
    """Clip a convex polygon to the left of a directed edge."""
    if not len(subject):
        return subject
    direction = edge_b - edge_a

    def signed(p):
        delta = p - edge_a
        return float(direction[0] * delta[1] - direction[1] * delta[0])

    result = []
    prev = subject[-1]
    prev_d = signed(prev)
    for point in subject:
        d = signed(point)
        if (d >= -1e-9) != (prev_d >= -1e-9):
            result.append(prev + (point - prev) * (prev_d / (prev_d - d)))
        if d >= -1e-9:
            result.append(point)
        prev, prev_d = point, d
    return np.asarray(result).reshape(-1, 2)


def contact_point(c_ego: np.ndarray, h_ego: float, c_agent: np.ndarray, h_agent: float,
                  agent_dims: np.ndarray) -> np.ndarray | None:
    """Centroid of oriented-box overlap polygon, or None when boxes do not overlap."""
    ego_poly = _corners(c_ego, h_ego, *EGO_DIMS)
    poly = _corners(c_agent, h_agent, *agent_dims)
    for j in range(4):
        poly = _clip(poly, ego_poly[j], ego_poly[(j + 1) % 4])
        if not len(poly):
            return None
    if len(poly) < 3:
        return poly.mean(0)
    next_poly = np.roll(poly, -1, axis=0)
    cross = poly[:, 0] * next_poly[:, 1] - poly[:, 1] * next_poly[:, 0]
    area2 = cross.sum()
    if abs(area2) < 1e-9:
        return poly.mean(0)
    return ((poly + next_poly) * cross[:, None]).sum(0) / (3 * area2)


def score(sc: Scene, pos: np.ndarray, head: np.ndarray, speed: np.ndarray, replan_decel: list[float]) -> dict:
    """Stage 2 scoring with first-contact geometry for at-fault attribution."""
    steps = np.arange(HANDOFF + 1, END + 1)
    others = [i for i in range(len(sc.track_ids)) if i != sc.av and sc.types[i] in base.DYN_IDX]
    collision, first, gap = False, -1, np.inf
    if others:
        o = np.asarray(others)
        c2 = sc.pos[o][:, steps]
        h2 = sc.head[o][:, steps]
        valid = sc.valid[o][:, steps]
        dims = DIMS[sc.types[o]]
        c1, h1 = pos[steps][None], head[steps][None]
        overlap = boxes_overlap(c1, h1, EGO_DIMS[0], EGO_DIMS[1], c2, h2,
                                dims[:, 0][:, None], dims[:, 1][:, None]) & valid
        if overlap.any():
            step_index = int(np.flatnonzero(overlap.any(0))[0])
            first = int(steps[step_index])
            for agent_index in np.flatnonzero(overlap[:, step_index]):
                point = contact_point(pos[first], float(head[first]), c2[agent_index, step_index],
                                      float(h2[agent_index, step_index]), dims[agent_index])
                if point is not None and bool(in_front(pos[first], head[first], point)):
                    collision = True
                    break
        front = in_front(c1, h1, c2)
        d = np.linalg.norm(c2 - c1, axis=-1) - (EGO_DIMS[0] / 2 + dims[:, 0][:, None] / 2)
        gap = float(np.where(front & valid, d, np.inf).min())
    sp = np.linalg.norm(np.diff(sc.pos[sc.av], axis=0), axis=1) / DT
    vs = np.convolve(sp, np.ones(5) / 5, mode="same")
    log_decel = vs[HANDOFF + 10 : END - 4] - vs[HANDOFF : END - 14]
    logged_hard = bool((log_decel <= HARD_BRAKE + 1e-9).any())
    planner_hard = bool(any(dv <= HARD_BRAKE + 1e-9 for dv in replan_decel))
    unnecessary = planner_hard and not logged_hard
    return {"collision": collision, "first_collision_step": first, "front_gap_m": gap,
            "planner_hard_brake": planner_hard, "logged_hard_brake": logged_hard,
            "unnecessary_hard_brake": unnecessary, "failure": collision or unnecessary}


def patch_predictions(sc: Scene, agents: list[int], t: int, traj: np.ndarray, prob: np.ndarray,
                      sham: bool = False) -> tuple[np.ndarray, np.ndarray, int]:
    """PATCH: constant-velocity forecasts for the selected agents stopped at t (< 0.5 m/s).

    SHAM (dose-matched): the same NUMBER of agents per replan get constant-velocity forecasts, but drawn at random
    from the selected agents that are moving, so the dose of CV substitution matches PATCH while the stopped-agent
    information is removed. Returns the realised number of substituted agents.
    """
    stopped = [j for j, i in enumerate(agents) if np.linalg.norm(sc.vel[i, t]) < 0.5]
    if sham:
        moving = [j for j in range(len(agents)) if j not in stopped]
        seed = (sum(map(ord, sc.scenario_id)) * 1009 + t) % (2**32)
        rng = np.random.default_rng(seed)
        chosen = list(rng.choice(moving, size=min(len(stopped), len(moving)), replace=False)) if moving else []
    else:
        chosen = stopped
    for j in chosen:
        traj[j] = base.cv_forecast(sc, agents[j], t)[None]
        prob[j] = 0.0
        prob[j, 0] = 1.0
    return traj, prob, len(chosen)


def run_patch(pool, n: int, model_key: str, models: dict, device, sham: bool = False) -> list[dict]:
    egos = [init_ego(base._SCENES[k]) for k in range(n)]
    trigger_counts = np.zeros(n, np.int32)
    substituted = np.zeros(n, np.int32)
    for t in REPLANS:
        res = pool.map(base._inputs, [(k, egos[k], t) for k in range(n)], chunksize=8)
        flat = [(model_key, inp) for _, _, inputs in res for inp in inputs]
        preds = base.predict(models, flat, device)
        jobs, p = [], 0
        for k, agents, _ in res:
            m = len(agents)
            traj = np.stack([x[0] for x in preds[p:p + m]]) if m else np.zeros((0, 6, 60, 2))
            prob = np.stack([x[1] for x in preds[p:p + m]]) if m else np.zeros((0, 6))
            p += m
            trigger_counts[k] += sum(np.linalg.norm(base._SCENES[k].vel[i, t]) < 0.5 for i in agents)
            traj, prob, n_sub = patch_predictions(base._SCENES[k], agents, t, traj, prob, sham)
            substituted[k] += n_sub
            jobs.append((k, egos[k], t, agents, traj, prob))
        for k, ego in pool.map(base._plan, jobs, chunksize=8):
            egos[k] = ego
    return [base.score_sim(base._SCENES[k], egos[k]) | {"trigger_count": int(trigger_counts[k]), "substituted_count": int(substituted[k])}
            for k in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--multi-runs", default="runs/MULTI")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenarios", help="dev scenario IDs; required with --dev-model")
    ap.add_argument("--dev-model", help="smoke checkpoint used for ALL, MULTI and PATCH")
    ap.add_argument("--arms", default="log,oracle,cv,static,ALL,MULTI,PATCH,SHAM")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.dev_model and not args.scenarios:
        ap.error("--dev-model requires --scenarios")
    if args.chunk < 1 or args.workers < 1 or args.limit < 0:
        ap.error("invalid chunk, workers, or limit")
    arms_req = args.arms.split(",")
    if any(a not in {"log", "oracle", "cv", "static", "ALL", "MULTI", "PATCH", "SHAM"} for a in arms_req):
        ap.error("unknown arm")
    dirs = sorted(glob.glob(os.path.join(args.raw, "*")))
    if args.scenarios:
        keep = set(str(x) for x in np.load(args.scenarios, allow_pickle=True))
        dirs = [d for d in dirs if os.path.basename(d) in keep]
    if args.limit:
        dirs = dirs[:args.limit]
    if not dirs:
        ap.error("no scenarios selected")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]
    spec = {}
    if args.dev_model:
        spec["DEV"] = args.dev_model
    else:
        for s in seeds:
            if "ALL" in arms_req or "PATCH" in arms_req or "SHAM" in arms_req:
                spec[f"ALL_s{s}"] = f"{args.runs}/ALL/seed{s}/model.pt"
            if "MULTI" in arms_req:
                spec[f"MULTI_s{s}"] = f"{args.multi_runs}/seed{s}/model.pt"
    models = base.load_models(spec, device)
    base.init_ego = init_ego
    base.score = score
    rows = []
    start = time.time()
    ctx = mp.get_context("fork")
    for offset in range(0, len(dirs), args.chunk):
        chunk = dirs[offset:offset + args.chunk]
        with ctx.Pool(args.workers) as pool:
            base._SCENES = pool.map(base._load, chunk, chunksize=16)
        with ctx.Pool(args.workers) as pool:
            n = len(base._SCENES)
            base_rows = [{"scenario_id": sc.scenario_id, "city": sc.city} for sc in base._SCENES]
            for arm in arms_req:
                keys = [None] if arm in ("log", "oracle", "cv", "static") or args.dev_model else seeds
                for seed in keys:
                    name = arm if seed is None else f"{arm}_s{seed}"
                    if arm in ("PATCH", "SHAM"):
                        key = "DEV" if args.dev_model else f"ALL_s{seed}"
                        result = run_patch(pool, n, key, models, device, sham=arm == "SHAM")
                    elif arm in ("ALL", "MULTI"):
                        key = "DEV" if args.dev_model else f"{arm}_s{seed}"
                        result = base.run_arm(pool, n, {"kind": "model", "model_for": lambda k, key=key: key}, models, device)
                    else:
                        result = base.run_arm(pool, n, {"kind": arm}, models, device)
                    for row, values in zip(base_rows, result):
                        row.update({f"{name}_{k}": v for k, v in values.items()})
            rows.extend(base_rows)
        print(json.dumps({"done": offset + len(chunk), "of": len(dirs), "sec": round(time.time() - start)}), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out)
    print(f"wrote {args.out}: {len(rows)} scenarios")


if __name__ == "__main__":
    main()
