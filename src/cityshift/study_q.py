"""v2 study Q runner: the released QCNet Argoverse 2 checkpoint in the Stage 3a open-loop scoring and the closed loop.

Registered in docs/preregistration/v2-qcnet.md. ``--mode open`` scores QCNet at the t=49 handoff on every validation
scenario with Stage 3a's memberships and scoring (`multiagent_eval.memberships` / `score`), so its rows join the
released Stage 3a CV-6 and ALL rows agent for agent. ``--mode closed`` runs QBASE / QPATCH / QSHAM2 and the analytic
arms in the stop-line harness (closedloop_v4) on the registered validation subset, with QCNet re-run at every replan
on the 50 steps ending there and the simulated ego in its history. Selected agents that QCNet does not forecast (not
valid at both the current and previous step) get constant velocity; their count is recorded.

    PYTHONPATH=src python -m cityshift.study_q --mode open --raw <val dir> --ckpt QCNet_AV2.ckpt --out runs/v2/q_open
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from . import closedloop as base
from . import closedloop_v2 as v2
from . import closedloop_v4 as v4
from . import qcnet_adapter as qa
from .closedloop import REPLANS
from .closedloop_v3 import _plan_calibrate, intervention
from .multiagent_eval import memberships, score
from .preprocess import FUT, OBJECT_TYPES

ARMS = ("QBASE", "QPATCH", "QSHAM2")  # QPATCH before QSHAM2: its per-replan dose is the sham's reference
ANALYTIC = ("cv", "oracle", "static")
SUBSET_SEED, SUBSET_N = 20261002, 6000
_QC: list = []


def validation_subset(raw: str) -> list[str]:
    """The registered closed-loop subset: a fixed-seed uniform draw of SUBSET_N validation scenario IDs."""
    ids = sorted(os.path.basename(d) for d in glob.glob(os.path.join(raw, "*")) if os.path.isdir(d))
    return sorted(np.random.default_rng(SUBSET_SEED).choice(ids, size=SUBSET_N, replace=False).tolist())


def _load(d: str):
    return base._load(d), qa.QCNetScenario(d)


def _categories(qc) -> dict[str, int]:
    cat = qc.df.groupby("track_id").object_category.first()
    return {str(k): int(v) for k, v in cat.items()}


# ------------------------------------------------------------------ open loop (t=49, logged ego)
def _open_job(k: int):
    sc, qc = base._SCENES[k], _QC[k]
    return k, memberships(sc, _categories(qc)), qc.data_at(base.HANDOFF)


def open_rows(pool, model, device, batch: int) -> list[dict]:
    jobs = pool.map(_open_job, range(len(base._SCENES)), chunksize=4)
    rows = []
    for lo in range(0, len(jobs), batch):
        part = jobs[lo:lo + batch]
        preds = qa.predict(model, [d for _, _, d in part], device)
        for (k, members, _), pred in zip(part, preds):
            sc = base._SCENES[k]
            for i, (focal, scored, planner) in members.items():
                tid = sc.track_ids[i]
                if tid not in pred:
                    continue
                traj, prob = pred[tid]
                valid = np.ones(FUT, bool) if focal else sc.valid[i, base.HANDOFF + 1:]
                rows.append({"scenario_id": sc.scenario_id, "agent_id": tid, "city": sc.city,
                             "agent_type": OBJECT_TYPES[sc.types[i]], "focal": focal, "scored": scored,
                             "planner_relevant": planner, "n_valid": int(valid.sum()),
                             "speed": float(np.linalg.norm(sc.vel[i, base.HANDOFF])), "predictor": "QCNET",
                             "moving_mode_mass": float(prob[np.linalg.norm(traj[:, -1] - sc.pos[i, base.HANDOFF],
                                                                          axis=-1) > 2.0].sum())}
                            | score(traj, prob, sc.pos[i, base.HANDOFF + 1:], valid))
    return rows


# ------------------------------------------------------------------ closed loop
def _closed_job(job):
    k, ego, t = job
    sc = base.scene_with_ego(base._SCENES[k], ego, t)
    agents = base.select_agents(sc, ego, t)
    ego_track = (sc.pos[sc.av], sc.head[sc.av], np.linalg.norm(sc.vel[sc.av], axis=-1))
    return k, agents, _QC[k].data_at(t, ego_track)


def run_arm(pool, n: int, arm: str, model, device, batch: int, reference_dose: np.ndarray | None = None):
    egos = [v2.init_ego(base._SCENES[k]) for k in range(n)]
    trigger, substituted, missing = (np.zeros(n, np.int32) for _ in range(3))
    dose = np.zeros((n, len(REPLANS)), np.int32)
    policy = {"QBASE": "ALL", "QPATCH": "PATCH", "QSHAM2": "SHAM2"}[arm]
    for r, t in enumerate(REPLANS):
        built = pool.map(_closed_job, [(k, egos[k], t) for k in range(n)], chunksize=4)
        preds: list = []
        for lo in range(0, n, batch):
            preds.extend(qa.predict(model, [d for _, _, d in built[lo:lo + batch]], device))
        jobs = []
        for (k, agents, _), pred in zip(built, preds):
            sc = base._SCENES[k]
            traj = np.zeros((len(agents), 6, FUT, 2))
            prob = np.zeros((len(agents), 6))
            for j, i in enumerate(agents):
                tid = sc.track_ids[i]
                if tid in pred:
                    traj[j], prob[j] = pred[tid]
                else:  # not forecast by QCNet: constant velocity, counted
                    traj[j] = base.cv_forecast(sc, i, t)[None]
                    prob[j, 0] = 1.0
                    missing[k] += 1
            ref = int(reference_dose[k, r]) if reference_dose is not None else None
            traj, prob, trig, sub, _ = intervention(sc, agents, t, policy, 0, traj, prob, ref)
            trigger[k] += trig
            substituted[k] += sub
            dose[k, r] = sub if policy == "SHAM2" else trig
            jobs.append((k, egos[k], t, agents, traj, prob, arm, 0, False))
        for k, ego, _ in pool.map(_plan_calibrate, jobs, chunksize=8):
            egos[k] = ego
    scores = [base.score_sim(base._SCENES[k], egos[k]) | {"trigger_count": int(trigger[k]),
              "substituted_count": int(substituted[k]), "qcnet_missing_count": int(missing[k]),
              "dose_by_replan": dose[k].tolist()} for k in range(n)]  # fmt: skip
    return scores, dose


def closed_rows(pool, model, device, batch: int, small: dict | None = None) -> list[dict]:
    from .closedloop_v3 import run_forecast

    n = len(base._SCENES)
    rows = [{"scenario_id": sc.scenario_id, "city": sc.city} for sc in base._SCENES]
    ref = None
    for arm in ARMS:
        scores, dose = run_arm(pool, n, arm, model, device, batch, ref if arm == "QSHAM2" else None)
        if arm == "QPATCH":
            ref = dose
        for row, s in zip(rows, scores):
            row.update({f"{arm}_{k}": v for k, v in s.items()})
    for seed in range(3):  # the study's own focal-only model on the same scenes, for the paired comparison
        scores, _, _ = run_forecast(pool, n, "ALL", f"ALL_s{seed}", small, device, seed, calibrate=False)
        for row, s in zip(rows, scores):
            row.update({f"ALL_s{seed}_{k}": v for k, v in s.items()})
    for arm in ANALYTIC:
        scores, _, _ = run_forecast(pool, n, arm, None, {}, device, 99, calibrate=False)
        for row, s in zip(rows, scores):
            row.update({f"{arm}_{k}": v for k, v in s.items()})
    for row, sc in zip(rows, base._SCENES):
        row.update({f"log_{k}": v for k, v in base.score_log(sc).items()})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("open", "closed"))
    ap.add_argument("--raw", required=True, help="AV2 validation directory")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--qcnet-root", default=None)
    ap.add_argument("--runs", default="runs", help="the study's focal-only checkpoints, runs/ALL/seed{0,1,2}")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke runs only")
    ap.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = ap.parse_args()
    smoke = os.path.basename(os.path.normpath(args.raw)) != "val"
    if smoke and not 0 < args.limit <= 100:
        ap.error("study Q scores the validation split; other splits only for <= 100-scenario dev-slice smoke runs")
    # Workers forked below build QCNet inputs with torch CPU ops; a parent that has started torch's OpenMP pool
    # deadlocks such children, so the parent stays single-threaded (its heavy work is on the GPU).
    torch.set_num_threads(1)
    device = torch.device(args.device)
    model, shims = qa.load_qcnet(args.ckpt, device, args.qcnet_root)
    small = base.load_models({f"ALL_s{s}": f"{args.runs}/ALL/seed{s}/model.pt" for s in range(3)}, device) \
        if args.mode == "closed" else {}
    if smoke:
        ids = sorted(str(i) for i in np.load("runs/dev_scenarios.npy", allow_pickle=True))
    elif args.mode == "open":
        ids = sorted(os.path.basename(d) for d in glob.glob(os.path.join(args.raw, "*")) if os.path.isdir(d))
    else:
        ids = validation_subset(args.raw)
    if args.mode == "closed":
        base.init_ego, base.score = v2.init_ego, v2.score
        v4.install()  # stop-line planner, inherited by the workers forked below
    ids = ids[:args.limit] if args.limit else ids
    os.makedirs(args.out_dir, exist_ok=True)
    manifest = {"mode": args.mode, "scenarios": len(ids),
                "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                "checkpoint_sha256": hashlib.sha256(open(args.ckpt, "rb").read()).hexdigest(),
                "shims": shims, "chunk": args.chunk, "batch": args.batch, "limit": args.limit,
                "torch": torch.__version__}
    path = os.path.join(args.out_dir, "manifest.json")
    if os.path.exists(path) and json.load(open(path)) != manifest:
        raise SystemExit(f"refusing to resume: {path} has different settings")
    json.dump(manifest, open(path, "w"), indent=2)
    ctx = mp.get_context("fork")
    start = time.time()
    global _QC
    for offset in range(0, len(ids), args.chunk):
        part = os.path.join(args.out_dir, f"part_{offset:05d}.parquet")
        if os.path.exists(part):
            continue
        dirs = [os.path.join(args.raw, i) for i in ids[offset:offset + args.chunk]]
        with ctx.Pool(args.workers) as pool:
            loaded = pool.map(_load, dirs, chunksize=4)
        base._SCENES, _QC = [x[0] for x in loaded], [x[1] for x in loaded]
        with ctx.Pool(args.workers) as pool:
            rows = open_rows(pool, model, device, args.batch) if args.mode == "open" else \
                closed_rows(pool, model, device, args.batch, small)
        pd.DataFrame(rows).to_parquet(part + ".tmp", index=False)
        os.replace(part + ".tmp", part)
        print(json.dumps({"done": offset + len(dirs), "of": len(ids), "sec": round(time.time() - start)}), flush=True)
    parts = sorted(glob.glob(os.path.join(args.out_dir, "part_*.parquet")))
    print(json.dumps({"parts": len(parts), "rows": sum(pq.read_metadata(p).num_rows for p in parts),
                      "runtime_sec": round(time.time() - start)}))


if __name__ == "__main__":
    main()
