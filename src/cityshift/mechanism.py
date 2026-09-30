"""EXPLORATORY per-replan mechanism audit (not registered; on already-scored replication-pool scenes).

For the focal-only model (ALL seed 0) in closed loop v2, at every replan that chooses a hard brake (<= -4 m/s^2) instead
of the choice it would make with no risk term, find the agents whose predicted hit probability ruled that choice out
("blockers"), and record for each: stopped or moving, the share of its risk from modes whose endpoint moves more than
2 m, whether its TRUE logged future actually conflicts with the rejected plan (same box test and front rule), and
whether its future is fully observed. Also records whether PATCH (CV for stopped agents) would have removed that brake at
that replan. Every recomputed choice is asserted equal to closedloop.plan.

    PYTHONPATH=src python -m cityshift.mechanism --raw <train dir> --out reports/mechanism/mechanism_audit.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from . import closedloop as base
from . import closedloop_v2 as v2
from .closedloop import (ACCELS, CHECK_STEPS, DIMS, EGO_DIMS, END, MARGIN_M, PLAN_STEPS, REPLANS, W_ACCEL,
                         W_PROGRESS, boxes_overlap, in_front, profiles)
from .closedloop_v3 import POOL_SHA256, checked_ids
from .scene import build_input


def hit_tensor(sc, ego, t, agents, traj):
    """[C, M, K] mode hits over the 4 s look-ahead, exactly as closedloop.plan computes them."""
    v, dist = profiles(ego.v[t], ego.vmax, PLAN_STEPS)
    ep, eh = ego.path.at(ego.s[t] + dist[:, :CHECK_STEPS])
    ap = traj[:, :, :CHECK_STEPS]
    prev = np.concatenate([np.repeat(sc.pos[agents, t][:, None, None], ap.shape[1], 1), ap[:, :, :-1]], 2)
    dxy = ap - prev
    ah = np.where(np.linalg.norm(dxy, axis=-1) < 0.05, sc.head[agents, t][:, None, None], np.arctan2(dxy[..., 1], dxy[..., 0]))
    dims = DIMS[sc.types[agents]]
    c1, h1 = ep[:, None, None], eh[:, None, None]
    hit = boxes_overlap(c1, h1, EGO_DIMS[0] + 2 * MARGIN_M, EGO_DIMS[1] + 2 * MARGIN_M, ap[None], ah[None],
                        dims[:, 0][:, None, None][None], dims[:, 1][:, None, None][None])
    return (hit & in_front(c1, h1, ap[None])).any(-1), dist, ep, eh


def choose(p_hit, dist):
    progress = dist[:, -1] / max(dist[:, -1].max(), 1e-6)
    return int(np.argmin(base.W_RISK * p_hit.sum(-1) + W_PROGRESS * (1.0 - progress) + W_ACCEL * np.abs(ACCELS)))


def true_conflict(sc, t, i, ep, eh, c) -> tuple[bool, bool]:
    """Does agent i's logged future overlap the ego's candidate-c plan in the look-ahead? Returns (conflict, full)."""
    steps = np.arange(t + 1, min(t + 1 + CHECK_STEPS, END + 1))
    valid = sc.valid[i, steps]
    n = len(steps)
    if not valid.any():
        return False, False
    dims = DIMS[sc.types[i]]
    ov = boxes_overlap(ep[c, :n], eh[c, :n], EGO_DIMS[0] + 2 * MARGIN_M, EGO_DIMS[1] + 2 * MARGIN_M,
                       sc.pos[i, steps], sc.head[i, steps], dims[0], dims[1])
    ov &= in_front(ep[c, :n], eh[c, :n], sc.pos[i, steps]) & valid
    return bool(ov.any()), bool(valid.all() and n == CHECK_STEPS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--pool", default="runs/replication_pool.npy")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ids = sorted(checked_ids(args.pool, POOL_SHA256))
    chosen_ids = np.random.default_rng(7).choice(ids, size=args.n, replace=False)
    device = torch.device("cuda")
    models = base.load_models({"ALL_s0": "runs/ALL/seed0/model.pt"}, device)
    base.init_ego = v2.init_ego
    decisions, blockers, replans = [], [], 0
    for sid in chosen_ids:
        sc = base.load_scene(os.path.join(args.raw, sid))
        ego = v2.init_ego(sc)
        for t in REPLANS:
            replans += 1
            sce = base.scene_with_ego(sc, ego, t)
            agents = base.select_agents(sce, ego, t)
            if agents:
                preds = base.predict(models, [("ALL_s0", build_input(sce, i, t)) for i in agents], device)
                traj, prob = np.stack([p[0] for p in preds]), np.stack([p[1] for p in preds])
            else:
                traj, prob = np.zeros((0, 6, 60, 2)), np.zeros((0, 6))
            planned = base.plan(sc, ego, t, agents, traj, prob)
            if agents:
                hits, dist, ep, eh = hit_tensor(sc, ego, t, agents, traj)
                p_hit = (hits * prob[None]).sum(-1)  # [C, M]
                c = choose(p_hit, dist)
                assert c == planned, "mechanism recomputation diverged from closedloop.plan"
                c0 = choose(np.zeros_like(p_hit), dist)
                if ACCELS[c] <= -4 and c != c0:
                    stopped = np.array([np.linalg.norm(sc.vel[i, t]) < 0.5 for i in agents])
                    # PATCH at this replan: CV (p=1) for stopped agents
                    ptraj, pprob = traj.copy(), prob.copy()
                    for j in np.flatnonzero(stopped):
                        ptraj[j] = base.cv_forecast(sc, agents[j], t)[None]
                        pprob[j] = 0.0
                        pprob[j, 0] = 1.0
                    phits, _, _, _ = hit_tensor(sc, ego, t, agents, ptraj)
                    c_patch = choose((phits * pprob[None]).sum(-1), dist)
                    moving_mode = np.linalg.norm(traj[:, :, -1] - sc.pos[agents, t][:, None], axis=-1) > 2.0  # [M, K]
                    blk = np.flatnonzero(p_hit[c0] > 0)
                    rows = []
                    for j in blk:
                        conflict, full = true_conflict(sc, t, agents[j], ep, eh, c0)
                        mass = float(p_hit[c0, j])
                        mv = float((hits[c0, j] * prob[j] * moving_mode[j]).sum())
                        rows.append({"stopped": bool(stopped[j]), "risk": mass, "moving_mode_share": mv / mass if mass else 0.0,
                                     "true_conflict": conflict, "full_future": full})
                    blockers.extend(rows)
                    decisions.append({"any_stopped_blocker": any(r["stopped"] for r in rows),
                                      "all_blockers_stopped": bool(rows) and all(r["stopped"] for r in rows),
                                      "any_true_conflict": any(r["true_conflict"] for r in rows),
                                      "patch_removes_brake": bool(ACCELS[c_patch] > -4)})
            base.execute(ego, t, planned)

    def share(rows, key, cond=lambda r: True):
        sel = [r for r in rows if cond(r)]
        return {"n": len(sel), "share": float(np.mean([r[key] for r in sel])) if sel else None}

    stop = lambda r: r["stopped"]  # noqa: E731
    move = lambda r: not r["stopped"]  # noqa: E731
    summary = {
        "_note": "EXPLORATORY, not registered. ALL seed 0, closed loop v2 (registered settings), "
                 f"{args.n} replication-pool scenes (fixed random subset, seed 7), {replans} replans.",
        "hard_brake_decisions": len(decisions),
        "decisions": {k: share(decisions, k) for k in ("any_stopped_blocker", "all_blockers_stopped",
                                                       "any_true_conflict", "patch_removes_brake")},
        "patch_removes_brake_when_all_blockers_stopped": share(decisions, "patch_removes_brake", lambda r: r["all_blockers_stopped"]),
        "blockers": {
            "n": len(blockers),
            "stopped_share": share(blockers, "stopped"),
            "true_conflict_stopped": share(blockers, "true_conflict", stop),
            "true_conflict_moving": share(blockers, "true_conflict", move),
            "moving_mode_share_of_risk_stopped": float(np.mean([r["moving_mode_share"] for r in blockers if r["stopped"]]))
            if any(r["stopped"] for r in blockers) else None,
            "true_conflict_stopped_full_future": share(blockers, "true_conflict", lambda r: r["stopped"] and r["full_future"]),
            "true_conflict_stopped_partial_future": share(blockers, "true_conflict", lambda r: r["stopped"] and not r["full_future"]),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
