"""Stage 4 replication on fresh TRAIN scenarios with matched sham and risk logging."""

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
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from . import closedloop as base
from . import closedloop_v2 as v2
from .closedloop import (ACCELS, CHECK_STEPS, DIMS, EGO_DIMS, END, MARGIN_M, REPLANS,
                         boxes_overlap, in_front)
from .scene import Scene

POOL_SHA256 = "a7489d7bd7b95986807ee360e62a01c45a15f4c3297258ba4751f8520caee0c8"
ARMS = ("ALL", "PATCH", "TRIM", "SHAM2", "MIX", "cv", "oracle", "static", "log")
MODEL_ARMS = {"ALL", "PATCH", "TRIM", "SHAM2", "MIX"}


def checked_ids(path: str, expected_hash: str | None = None) -> list[str]:
    """Load unique IDs and optionally check the registered sorted-ID digest."""
    ids = [str(x) for x in np.load(path, allow_pickle=True)]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate scenario IDs")
    if expected_hash and hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest() != expected_hash:
        raise ValueError("scenario pool hash mismatch")
    return ids


def trim_predictions(sc: Scene, agents: list[int], t: int, traj: np.ndarray,
                     prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Remove moving modes of stopped agents; use CV when no stationary mode exists."""
    fallback = 0
    for j, i in enumerate(agents):
        if np.linalg.norm(sc.vel[i, t]) >= 0.5:
            continue
        keep = np.linalg.norm(traj[j, :, -1] - sc.pos[i, t], axis=-1) <= 2.0
        if not keep.any() or prob[j, keep].sum() <= 0:
            traj[j] = base.cv_forecast(sc, i, t)[None]
            prob[j] = 0
            prob[j, 0] = 1
            fallback += 1
        else:
            prob[j, ~keep] = 0
            prob[j] /= prob[j].sum()
    return traj, prob, fallback


def stationary_eligible(sc: Scene, agents: list[int], t: int, traj: np.ndarray, prob: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Study A eligibility: selected agents moving < 0.5 m/s whose own forecast has a mode ending <= 2 m from the
    current position with positive probability (TRIM's test without its fallback). Returns (index, keep mask)."""
    out = []
    for j, i in enumerate(agents):
        if np.linalg.norm(sc.vel[i, t]) >= 0.5:
            continue
        keep = np.linalg.norm(traj[j, :, -1] - sc.pos[i, t], axis=-1) <= 2.0
        if keep.any() and prob[j, keep].sum() > 0:
            out.append((j, keep))
    return out


def sham_indices(sc: Scene, agents: list[int], t: int, model_seed: int, dose: int) -> np.ndarray:
    """Select PATCH's per-replan dose from all selected agents, capped at the number available.

    The cap binds only when this arm's (diverged) ego selects fewer agents than PATCH's dose at that replan; the
    shortfall is recorded and the analysis requires the realised total dose to be >= 99% of PATCH's.
    """
    dose = min(dose, len(agents))
    token = f"{sc.scenario_id}:{t}:{model_seed}".encode()
    seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    chosen = np.random.default_rng(seed).choice(len(agents), size=dose, replace=False)
    assert len(chosen) == dose
    return chosen


def most_stationary_mode(sc: Scene, i: int, t: int, traj: np.ndarray) -> int:
    """v2 study G: the agent's own mode with the smallest maximum displacement over the planner look-ahead."""
    disp = np.linalg.norm(traj[:, :CHECK_STEPS] - sc.pos[i, t], axis=-1).max(-1)
    return int(np.argmin(disp))


def factorial(sc: Scene, agents: list[int], t: int, arm: str, traj: np.ndarray, prob: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, int]:
    """v2 study G (docs/preregistration/v2-fresh-reserve.md), stopped selected agents only (PATCH's trigger).

    GEO replaces the most stationary mode's trajectory with constant velocity and keeps every probability; PROB keeps
    every trajectory and puts probability one on the most stationary mode. Applying both is PATCH (one CV mode with
    probability one; zero-probability modes carry no planner risk).
    """
    n = 0
    for j, i in enumerate(agents):
        if np.linalg.norm(sc.vel[i, t]) >= 0.5:
            continue
        k = most_stationary_mode(sc, i, t, traj[j])
        if arm == "GEO":
            traj[j, k] = base.cv_forecast(sc, i, t)
        else:  # PROB
            prob[j] = 0
            prob[j, k] = 1
        n += 1
    return traj, prob, n


def intervention(sc: Scene, agents: list[int], t: int, arm: str, seed: int,
                 traj: np.ndarray, prob: np.ndarray, reference_dose: int | None = None,
                 ) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Apply a forecast intervention and return trigger, substitution and fallback counts."""
    trigger = sum(np.linalg.norm(sc.vel[i, t]) < 0.5 for i in agents)
    if arm in ("GEO", "PROB"):
        traj, prob, n = factorial(sc, agents, t, arm, traj, prob)
        return traj, prob, trigger, n, 0
    if arm in ("TRIMNF", "PATCHSUB"):  # exploratory study A (docs/preregistration/exploratory-followups.md)
        eligible = stationary_eligible(sc, agents, t, traj, prob)
        for j, keep in eligible:
            if arm == "TRIMNF":
                prob[j, ~keep] = 0
                prob[j] /= prob[j].sum()
            else:
                traj[j] = base.cv_forecast(sc, agents[j], t)[None]
                prob[j] = 0
                prob[j, 0] = 1
        return traj, prob, len(eligible), len(eligible), 0
    if arm == "TRIM":
        traj, prob, fallback = trim_predictions(sc, agents, t, traj, prob)
        return traj, prob, trigger, 0, fallback
    if arm == "PATCH":
        chosen = [j for j, i in enumerate(agents) if np.linalg.norm(sc.vel[i, t]) < 0.5]
    elif arm == "SHAM2":
        if reference_dose is None:
            raise ValueError("SHAM2 requires PATCH replan dose")
        chosen = sham_indices(sc, agents, t, seed, reference_dose)
        trigger = reference_dose
    else:
        return traj, prob, trigger, 0, 0
    for j in chosen:
        traj[j] = base.cv_forecast(sc, agents[j], t)[None]
        prob[j] = 0
        prob[j, 0] = 1
    if arm == "SHAM2":
        assert len(chosen) == min(trigger, len(agents)), "SHAM2 dose differs from its capped PATCH reference"
    return traj, prob, trigger, len(chosen), 0


def calibration_rows(sc: Scene, ego: base.EgoSim, t: int, agents: list[int], traj: np.ndarray,
                     prob: np.ndarray, candidate: int, arm: str, seed: int) -> list[dict]:
    """Predicted planner risk and true front-contact for the chosen four-second candidate."""
    if not agents:
        return []
    _, dist = base.profiles(ego.v[t], ego.vmax, base.PLAN_STEPS)
    ep, eh = ego.path.at(ego.s[t] + dist[candidate, :CHECK_STEPS])
    ap = traj[:, :, :CHECK_STEPS]
    prev = np.concatenate([np.repeat(sc.pos[agents, t][:, None, None], ap.shape[1], 1), ap[:, :, :-1]], 2)
    dxy = ap - prev
    ah = np.arctan2(dxy[..., 1], dxy[..., 0])
    ah = np.where(np.linalg.norm(dxy, axis=-1) < 0.05, sc.head[agents, t][:, None, None], ah)
    dims = DIMS[sc.types[agents]]
    hit = boxes_overlap(ep[None, None], eh[None, None], EGO_DIMS[0] + 2 * MARGIN_M,
                        EGO_DIMS[1] + 2 * MARGIN_M, ap, ah, dims[:, 0, None, None], dims[:, 1, None, None])
    hit &= in_front(ep[None, None], eh[None, None], ap)
    p_hit = (hit.any(-1) * prob).sum(-1)
    observed_steps = min(CHECK_STEPS, END - t)
    steps = t + 1 + np.arange(observed_steps)
    rows = []
    for j, i in enumerate(agents):
        valid = sc.valid[i, steps]
        overlap = boxes_overlap(ep[:observed_steps], eh[:observed_steps], EGO_DIMS[0], EGO_DIMS[1],
                                sc.pos[i, steps], sc.head[i, steps], dims[j, 0], dims[j, 1]) & valid
        contact = False
        for q in np.flatnonzero(overlap):
            point = v2.contact_point(ep[q], float(eh[q]), sc.pos[i, steps[q]],
                                     float(sc.head[i, steps[q]]), dims[j])
            if point is not None and bool(in_front(ep[q], eh[q], point)):
                contact = True
                break
        rows.append({"scenario_id": sc.scenario_id, "city": sc.city, "replan": t, "arm": arm,
                     "seed": seed, "agent_id": sc.track_ids[i], "stopped": bool(np.linalg.norm(sc.vel[i, t]) < 0.5),
                     "predicted_hit_probability": float(p_hit[j]), "contact": contact,
                     "acceleration": float(ACCELS[candidate]), "observed_steps": observed_steps,
                     "complete_future": observed_steps == CHECK_STEPS})
    return rows


def _plan_calibrate(job):
    """Worker: plan once, record calibration rows for that same candidate, then execute it.

    Before 2026-09-30 the parent planned every scenario serially for the calibration rows and the workers planned it
    again; this does the identical computation once, in parallel (outputs verified identical on CPU).
    """
    k, ego, t, agents, traj, prob, arm, seed, calibrate = job
    sc = base._SCENES[k]
    candidate = base.plan(sc, ego, t, agents, traj, prob)
    rows = calibration_rows(sc, ego, t, agents, traj, prob, candidate, arm, seed) if calibrate else []
    base.execute(ego, t, candidate)
    return k, ego, rows


def run_forecast(pool, n: int, arm: str, key: str | None, models: dict, device: torch.device,
                 seed: int, reference_dose: np.ndarray | None = None,
                 calibrate: bool = True) -> tuple[list[dict], list[dict], np.ndarray]:
    """Run one v2 planner arm with forecast changes and calibration recording."""
    egos = [v2.init_ego(base._SCENES[k]) for k in range(n)]
    trigger_count, substituted_count, fallback_count = (np.zeros(n, np.int32) for _ in range(3))
    replan_dose = np.zeros((n, len(REPLANS)), np.int32)
    risk_rows: list[dict] = []
    for replan_index, t in enumerate(REPLANS):
        if key is None:
            results = pool.map(base._analytic, [(k, egos[k], t, arm) for k in range(n)], chunksize=8)
        else:
            inputs = pool.map(base._inputs, [(k, egos[k], t) for k in range(n)], chunksize=8)
            flat = [(key, inp) for _, _, values in inputs for inp in values]
            predictions = base.predict(models, flat, device)
            results = []
            offset = 0
            for k, agents, _ in inputs:
                count = len(agents)
                traj = np.stack([x[0] for x in predictions[offset:offset + count]]) if count else np.zeros((0, 6, 60, 2))
                prob = np.stack([x[1] for x in predictions[offset:offset + count]]) if count else np.zeros((0, 6))
                results.append((k, agents, traj, prob))
                offset += count
        jobs = []
        for k, agents, traj, prob in results:
            sc = base._SCENES[k]
            reference = int(reference_dose[k, replan_index]) if reference_dose is not None else None
            traj, prob, trigger, substituted, fallback = intervention(sc, agents, t, arm, seed, traj, prob, reference)
            trigger_count[k] += trigger
            substituted_count[k] += substituted
            fallback_count[k] += fallback
            replan_dose[k, replan_index] = substituted if arm in ("SHAM2", "TRIMNF", "PATCHSUB") else trigger
            if arm == "SHAM2":
                assert substituted == min(reference, len(agents)), "SHAM2 dose differs from capped PATCH reference"
            jobs.append((k, egos[k], t, agents, traj, prob, arm, seed, calibrate))
        for k, ego, rows in pool.map(_plan_calibrate, jobs, chunksize=8):
            egos[k] = ego
            risk_rows.extend(rows)
    scores = [base.score_sim(base._SCENES[k], egos[k]) |
              {"trigger_count": int(trigger_count[k]), "substituted_count": int(substituted_count[k]),
               "trim_fallback_count": int(fallback_count[k])} |
              {f"dose_t{t}": int(replan_dose[k, j]) for j, t in enumerate(REPLANS)} for k in range(n)]
    return scores, risk_rows, replan_dose


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--risk-out", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--mix-runs", default="runs/stage4/MIX")
    ap.add_argument("--dev-model")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    arms = args.arms.split(",")
    seeds = [int(x) for x in args.seeds.split(",")]
    if not set(arms) <= set(ARMS) or len(arms) != len(set(arms)) or not arms:
        ap.error("invalid arms")
    if "SHAM2" in arms and ("PATCH" not in arms or arms.index("PATCH") > arms.index("SHAM2")):
        ap.error("PATCH must precede SHAM2 to establish the per-replan reference dose")
    if args.chunk < 1 or args.workers < 1 or args.limit < 0 or not seeds:
        ap.error("invalid chunk, workers, limit or seeds")
    if args.dev_model:
        if args.limit > 100 or args.limit == 0 or os.path.basename(args.scenarios) != "dev_scenarios.npy":
            ap.error("smoke requires --limit 1..100 and runs/dev_scenarios.npy")
        ids = checked_ids(args.scenarios)
    else:
        if os.path.basename(args.scenarios) != "replication_pool.npy" or args.limit:
            ap.error("full scoring requires replication_pool.npy without a limit")
        ids = checked_ids(args.scenarios, POOL_SHA256)
        if len(ids) != 8140 or seeds != [0, 1, 2] or set(arms) != set(ARMS):
            ap.error("full scoring requires 8,140 pool scenarios, seeds 0-2 and all arms")
    dirs = [d for d in sorted(glob.glob(os.path.join(args.raw, "*"))) if os.path.basename(d) in set(ids)]
    if len(dirs) != len(ids):
        ap.error("missing raw scenario directories")
    if args.limit:
        dirs = dirs[:args.limit]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = {}
    if args.dev_model:
        spec["DEV"] = args.dev_model
    else:
        for seed in seeds:
            if set(arms) & {"ALL", "PATCH", "TRIM", "SHAM2"}:
                spec[f"ALL_s{seed}"] = f"{args.runs}/ALL/seed{seed}/model.pt"
            if "MIX" in arms:
                spec[f"MIX_s{seed}"] = f"{args.mix_runs}/seed{seed}/model.pt"
    models = base.load_models(spec, device)
    base.init_ego = v2.init_ego
    base.score = v2.score
    rows: list[dict] = []
    writer = None
    start = time.time()
    ctx = mp.get_context("fork")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.risk_out)), exist_ok=True)
    try:
        for offset in range(0, len(dirs), args.chunk):
            with ctx.Pool(args.workers) as pool:
                base._SCENES = pool.map(base._load, dirs[offset:offset + args.chunk], chunksize=8)
            with ctx.Pool(args.workers) as pool:
                n = len(base._SCENES)
                base_rows = [{"scenario_id": sc.scenario_id, "city": sc.city} for sc in base._SCENES]
                risk_chunk: list[dict] = []
                patch_dose: dict[int, np.ndarray] = {}
                for arm in arms:
                    for seed in ([None] if arm not in MODEL_ARMS or args.dev_model else seeds):
                        name = arm if seed is None else f"{arm}_s{seed}"
                        if arm == "log":
                            scores = [base.score_log(sc) for sc in base._SCENES]
                            calibration = []
                        else:
                            model_seed = seed if seed is not None else 99
                            key = ("DEV" if args.dev_model else f"{'MIX' if arm == 'MIX' else 'ALL'}_s{seed}") \
                                if arm in MODEL_ARMS else None
                            reference = patch_dose.get(model_seed) if arm == "SHAM2" else None
                            scores, calibration, dose = run_forecast(pool, n, arm, key, models, device, model_seed,
                                                                     reference)
                            if arm == "PATCH":
                                patch_dose[model_seed] = dose
                        for row, score in zip(base_rows, scores):
                            row.update({f"{name}_{k}": v for k, v in score.items()})
                        risk_chunk.extend(calibration)
                rows.extend(base_rows)
                if risk_chunk:
                    table = pa.Table.from_pandas(pd.DataFrame(risk_chunk), preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(args.risk_out, table.schema, compression="zstd")
                    writer.write_table(table)
            print(json.dumps({"done": offset + len(base._SCENES), "of": len(dirs),
                              "sec": round(time.time() - start)}), flush=True)
    finally:
        if writer is not None:
            writer.close()
    pd.DataFrame(rows).to_parquet(args.out)
    print(f"wrote {args.out}: {len(rows)} scenarios; risk: {args.risk_out}")


if __name__ == "__main__":
    main()
