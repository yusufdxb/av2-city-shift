"""EXPLORATORY planner sensitivity sweep (not registered; on already-scored replication-pool scenes).

Does PATCH's braking advantage over the focal-only model survive other planner settings? Re-runs closed loop v2 with
the risk weight and speed cap changed, and records each drive's minimum executed and logged deceleration so any
hard-brake threshold can be scored afterwards. Arms: ALL, PATCH, constant velocity, oracle. Model seeds 0, 1 and 2
(the forecast-free cv and oracle arms do not depend on the model seed and are run with seed 0 only).
Nothing here changes a registered result.

    PYTHONPATH=src python -m cityshift.sensitivity --raw <train dir> --seed 0 --out runs/sensitivity/sweep_seed0.parquet
    PYTHONPATH=src python -m cityshift.sensitivity --raw <train dir> --seed 1 --arms ALL,PATCH \
        --out runs/sensitivity/sweep_seed1.parquet            # likewise seed 2
    python scripts/summarize_sensitivity.py            # summary.json with scenario-bootstrap intervals, SHA256SUMS
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
import torch

from . import closedloop as base
from . import closedloop_v2 as v2
from .closedloop import DT, END, EXEC_STEPS, HANDOFF, REPLANS
from .closedloop_v3 import POOL_SHA256, checked_ids, run_forecast

RISK_WEIGHTS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)  # 100 = registered; the learned model's few-percent modes trade off against progress (<= 1) in this range
CAPS = {"v2": lambda v0: max(v0 + 3.0, 15.0), "tight": lambda v0: max(v0 + 1.0, 8.0)}
ARMS = ("ALL", "PATCH", "cv", "oracle")
_score_sim = base.score_sim


def score_with_decel(sc, ego) -> dict:
    """Registered v2 score plus minimum executed / logged deceleration and a past-the-route flag."""
    r = _score_sim(sc, ego)
    r["min_exec_decel"] = float(min((ego.v[t + EXEC_STEPS] - ego.v[t]) / (EXEC_STEPS * DT) for t in REPLANS))
    sp = np.linalg.norm(np.diff(sc.pos[sc.av], axis=0), axis=1) / DT
    vs = np.convolve(sp, np.ones(5) / 5, mode="same")
    r["min_log_decel"] = float((vs[HANDOFF + 10 : END - 4] - vs[HANDOFF : END - 14]).min())
    r["beyond_logged_route"] = bool(ego.s[END] > ego.path.length_logged)
    r.pop("accels", None)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--pool", default="runs/replication_pool.npy")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0, help="model seed: runs/ALL/seed<k>/model.pt")
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    arms = args.arms.split(",")
    if not arms or not set(arms) <= set(ARMS) or args.seed not in (0, 1, 2):
        ap.error("arms must be a subset of ALL,PATCH,cv,oracle and seed one of 0, 1, 2")
    ids = checked_ids(args.pool, POOL_SHA256)
    subset = set(np.random.default_rng(2026).choice(sorted(ids), size=args.n, replace=False).tolist())
    dirs = [d for d in sorted(glob.glob(os.path.join(args.raw, "*"))) if os.path.basename(d) in subset]
    assert len(dirs) == args.n
    device = torch.device("cuda")
    key_name = f"ALL_s{args.seed}"
    models = base.load_models({key_name: f"runs/ALL/seed{args.seed}/model.pt"}, device)
    base.init_ego = v2.init_ego
    base.score = v2.score
    base.score_sim = score_with_decel
    rows, start = [], time.time()
    ctx = mp.get_context("fork")
    for offset in range(0, len(dirs), args.chunk):
        with ctx.Pool(args.workers) as pool:
            base._SCENES = pool.map(base._load, dirs[offset:offset + args.chunk], chunksize=8)
        n = len(base._SCENES)
        for cap_name, cap in CAPS.items():
            for weight in RISK_WEIGHTS:
                base.W_RISK = weight  # read by closedloop.plan in the parent and in the forked workers
                v2.speed_cap = cap  # read by closedloop_v2.init_ego
                with ctx.Pool(args.workers) as pool:
                    for arm in arms:
                        key = key_name if arm in ("ALL", "PATCH") else None
                        # the seed argument only feeds the SHAM arms' agent choice, unused by these four arms
                        scores, _, _ = run_forecast(pool, n, arm, key, models, device, args.seed)
                        for sc, s in zip(base._SCENES, scores):
                            rows.append({"scenario_id": sc.scenario_id, "city": sc.city, "model_seed": args.seed,
                                         "cap": cap_name, "risk_weight": weight, "arm": arm,
                                         **{k: s[k] for k in ("collision", "unnecessary_hard_brake", "progress",
                                                              "min_exec_decel", "min_log_decel", "beyond_logged_route")}})
        print(json.dumps({"done": offset + n, "of": len(dirs), "sec": round(time.time() - start)}), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out)
    print(f"wrote {args.out}: {len(rows)} rows")


if __name__ == "__main__":
    main()


BRAKE_THRESHOLDS = (-3.0, -4.0, -5.0)
RATE_COLUMNS = ("unnecessary_hard_brake", "collision", "progress", "beyond_logged_route")
NOTE = ("EXPLORATORY, not registered. 3,000 replication-pool scenes (fixed random subset, seed 2026), closed loop v2 "
        "with the risk weight and speed cap varied. hard brake = executed decel <= threshold while the human never "
        "decelerated that hard. ALL and PATCH use model seeds {seeds}; cv and oracle use no model and come from the "
        "seed-0 run. Intervals are unadjusted 95% percentile intervals from a scenario-cluster bootstrap "
        "({boots} draws: resample scenes with replacement, average the seeds within each resampled scene); no "
        "multiplicity correction across the {n_configs} configurations.")


def with_brake(d: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Score the hard-brake outcome at any threshold from the recorded minimum decelerations."""
    return d.assign(unnecessary_hard_brake=(d.min_exec_decel <= threshold + 1e-9) & ~(d.min_log_decel <= threshold + 1e-9))


def legacy_configs(d: pd.DataFrame) -> list[dict]:
    """The original single-seed summary format (pooled rates, no intervals), for one model seed's rows."""
    out = []
    for cap in sorted(d.cap.unique()):
        for weight in sorted(d.risk_weight.unique()):
            for threshold in BRAKE_THRESHOLDS:
                g = with_brake(d[(d.cap == cap) & (d.risk_weight == weight)], threshold)
                c: dict = {"cap": cap, "risk_weight": float(weight), "brake_threshold": threshold}
                for arm in ARMS:
                    x = g[g.arm == arm]
                    c[arm] = {k: float(x[k].astype(float).mean()) for k in RATE_COLUMNS}
                all_rate = c["ALL"]["unnecessary_hard_brake"]
                c["PATCH_pooled_brake_reduction"] = 1 - c["PATCH"]["unnecessary_hard_brake"] / all_rate if all_rate > 0 else None
                out.append(c)
    return out


def _interval(draws: np.ndarray) -> list[float | None]:
    draws = draws[np.isfinite(draws)]
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))] if len(draws) else [None, None]


def summarize(d: pd.DataFrame, n_boot: int = 10_000, boot_seed: int = 20260930) -> dict:
    """Multi-seed PATCH vs ALL effects per configuration with scenario-cluster bootstrap intervals."""
    seeds = sorted(d.model_seed.unique().tolist())
    scenes = sorted(d.scenario_id.unique())
    n = len(scenes)
    idx = np.random.default_rng(boot_seed).integers(0, n, size=(n_boot, n), dtype=np.int32)
    configs = []
    for cap in sorted(d.cap.unique()):
        for weight in sorted(d.risk_weight.unique()):
            for threshold in BRAKE_THRESHOLDS:
                g = with_brake(d[(d.cap == cap) & (d.risk_weight == weight)], threshold)
                c: dict = {"cap": cap, "risk_weight": float(weight), "brake_threshold": threshold, "rates": {}}
                per_scene = {}
                for arm in ARMS:
                    x = g[g.arm == arm]
                    arm_seeds = sorted(x.model_seed.unique().tolist())
                    wide = {k: x.pivot(index="scenario_id", columns="model_seed", values=k).astype(float).reindex(scenes)
                            for k in RATE_COLUMNS}
                    # progress is NaN for some drives by design (averaged over the rest, as in the legacy summary)
                    assert len(x) == n * len(arm_seeds) and not wide["collision"].isna().any().any(), f"missing {arm} rows"
                    c["rates"][arm] = {"model_seeds": arm_seeds,
                                       **{k: float(np.nanmean(w.to_numpy())) for k, w in wide.items()},
                                       "per_seed": {str(s): {k: float(w[s].mean()) for k, w in wide.items()}
                                                    for s in arm_seeds}}
                    per_scene[arm] = {k: wide[k].to_numpy().mean(1) for k in ("unnecessary_hard_brake", "collision")}
                    if arm in ("ALL", "PATCH"):
                        assert arm_seeds == seeds, f"{arm} lacks some model seeds"
                a_b, p_b = per_scene["ALL"]["unnecessary_hard_brake"], per_scene["PATCH"]["unnecessary_hard_brake"]
                a_c, p_c = per_scene["ALL"]["collision"], per_scene["PATCH"]["collision"]
                boot_red, boot_coll = np.empty(n_boot), np.empty(n_boot)
                for lo in range(0, n_boot, 1000):
                    block = idx[lo:lo + 1000]
                    den = a_b[block].mean(1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        boot_red[lo:lo + 1000] = np.where(den > 0, 1 - p_b[block].mean(1) / den, np.nan)
                    boot_coll[lo:lo + 1000] = 100 * (p_c[block].mean(1) - a_c[block].mean(1))
                per_seed = c["rates"]
                c["PATCH_vs_ALL"] = {
                    "brake_reduction": {
                        "point": float(1 - p_b.mean() / a_b.mean()) if a_b.mean() > 0 else None,
                        "ci95": _interval(boot_red),
                        "undefined_bootstrap_share": float(np.isnan(boot_red).mean()),
                        "per_seed": {str(s): (1 - per_seed["PATCH"]["per_seed"][str(s)]["unnecessary_hard_brake"]
                                              / per_seed["ALL"]["per_seed"][str(s)]["unnecessary_hard_brake"])
                                     if per_seed["ALL"]["per_seed"][str(s)]["unnecessary_hard_brake"] > 0 else None
                                     for s in seeds}},
                    "collision_diff_pp": {
                        "point": float(100 * (p_c.mean() - a_c.mean())),
                        "ci95": _interval(boot_coll),
                        "per_seed": {str(s): 100 * (per_seed["PATCH"]["per_seed"][str(s)]["collision"]
                                                    - per_seed["ALL"]["per_seed"][str(s)]["collision"]) for s in seeds}}}
                configs.append(c)
    return {"_note": NOTE.format(seeds=", ".join(map(str, seeds)), boots=f"{n_boot:,}", n_configs=len(configs)),
            "n_scenes": n, "model_seeds": seeds, "bootstraps": n_boot, "bootstrap_seed": boot_seed, "ci_level": 0.95,
            "configs": configs,
            "seed0_legacy": {"_note": "Original single-seed summary (model seed 0, pooled over scenes, no intervals), "
                                      "recomputed from the seed-0 rows; kept for comparison with release v1.3.",
                             "configs": legacy_configs(d[d.model_seed == 0])}}
