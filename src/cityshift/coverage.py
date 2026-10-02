"""Study F: fixed-horizon planner-agent coverage with the logged ego on AV2 val.

Score: python -m cityshift.coverage --raw <val> --runs <runs> --out <rows.parquet>
CPU smoke: add --limit 20 --device cpu. Analyze: --parquet <rows.parquet> --out <summary.json>.
Miss uses the fixed 2 m threshold at both 2 s and 4 s, with no horizon scaling.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .baselines import constant_velocity
from .closedloop import DT, END, EgoSim, Path as EgoPath, load_models, predict, select_agents
from .data import CITIES
from .preprocess import OBJECT_TYPES
from .scene import Scene, build_input, load_scene

REPLANS = (49, 59, 69, 79, 89)
N_BOOT = 10000
SEED = 20261002
MISS_THRESHOLD_M = 2.0
KEYS = ["scenario_id", "replan", "agent_id"]
META = ["city", "agent_type", "focal", "speed", "complete_2s", "complete_4s"]


def horizons(t: int) -> tuple[int, ...]:
    """Return the registered horizons in seconds for a replan."""
    if t not in REPLANS:
        raise ValueError(f"unregistered replan: {t}")
    return (2, 4) if t <= 69 else (2,)


def complete_future(valid: np.ndarray, t: int, horizon: int) -> bool:
    """Require every step t+1 through t+horizon/DT, including the endpoint."""
    n = int(round(horizon / DT))
    if horizon not in horizons(t):
        return False
    return bool(t + n < len(valid) and valid[t] and valid[t + 1 : t + n + 1].all())


def score_endpoint(traj: np.ndarray, target: np.ndarray, horizon: int) -> dict[str, float]:
    """Best-of-six endpoint error at exactly the requested complete horizon."""
    step = int(round(horizon / DT)) - 1
    if traj.shape[0] != 6 or traj.shape[-1] != 2 or step >= traj.shape[1] or step >= len(target):
        raise ValueError("six trajectories and a full target horizon are required")
    fde = float(np.linalg.norm(traj[:, step] - target[step], axis=-1).min())
    if not np.isfinite(fde):
        raise ValueError("nonfinite endpoint error")
    return {f"min_fde_{horizon}s": fde, f"miss_{horizon}s": float(fde > MISS_THRESHOLD_M)}


def logged_ego(sc: Scene, t: int) -> EgoSim:
    """Anchor the planner route at the exact logged ego position at this replan.

    Build the remaining logged route with the planner's 0.05 m vertex filter and
    straight extension. No acceleration is chosen and no ego state is simulated.
    At t=49 this is the same path as closedloop.Path.from_scene.
    """
    p = sc.pos[sc.av, t : END + 1]
    if not sc.valid[sc.av, t : END + 1].all():
        raise ValueError("logged ego route must be fully observed")
    keep = [0]
    for i in range(1, len(p)):
        if np.linalg.norm(p[i] - p[keep[-1]]) > 0.05:
            keep.append(i)
    pts = p[keep]
    direction = pts[-1] - pts[-2] if len(pts) > 1 else np.array([np.cos(sc.head[sc.av, t]), np.sin(sc.head[sc.av, t])])
    direction /= np.linalg.norm(direction)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    path = EgoPath(s=np.r_[arc, arc[-1] + 300.0], xy=np.vstack([pts, pts[-1] + 300.0 * direction]), length_logged=float(arc[-1]))
    ego = EgoSim(path=path, vmax=float(np.linalg.norm(sc.vel[sc.av, t])))
    ego.v[t] = ego.vmax
    return ego


def score_scene(sc: Scene, models: dict[str, torch.nn.Module], device: torch.device) -> list[dict]:
    """Score each seed separately, then average metrics per selected agent."""
    if set(models) != {f"ALL_s{s}" for s in range(3)}:
        raise ValueError("ALL seeds 0, 1 and 2 are required")
    rows = []
    for t in REPLANS:
        agents = select_agents(sc, logged_ego(sc, t), t)
        inputs = [build_input(sc, i, t) for i in agents]
        eligible = [j for j, i in enumerate(agents) if any(complete_future(sc.valid[i], t, h) for h in horizons(t))]
        preds = {key: dict(zip(eligible, predict(models, [(key, inputs[j]) for j in eligible], device))) for key in models}
        for j, i in enumerate(agents):
            inp = inputs[j]
            th = float(inp["theta"])
            rot = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
            cv, _ = constant_velocity(inp)
            cv = cv @ rot + inp["origin"]
            meta = {
                "scenario_id": sc.scenario_id,
                "city": sc.city,
                "replan": t,
                "agent_id": sc.track_ids[i],
                "agent_type": OBJECT_TYPES[sc.types[i]],
                "focal": i == sc.focal,
                "speed": float(np.linalg.norm(sc.vel[i, t])),
                "complete_2s": complete_future(sc.valid[i], t, 2),
                "complete_4s": complete_future(sc.valid[i], t, 4),
            }
            metrics = {
                name: {f"{metric}_{h}s": np.nan for h in (2, 4) for metric in ("min_fde", "miss")} for name in ("ALL", "CV-6")
            }
            for h in horizons(t):
                if not meta[f"complete_{h}s"]:
                    continue
                target = sc.pos[i, t + 1 :]
                per_seed = [score_endpoint(preds[key][j][0], target, h) for key in sorted(models)]
                metrics["ALL"].update({k: float(np.mean([m[k] for m in per_seed])) for k in per_seed[0]})
                metrics["CV-6"].update(score_endpoint(cv, target, h))
            rows.extend(meta | {"predictor": name} | values for name, values in metrics.items())
    return rows


def _paired(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    needed = set(KEYS + META + ["predictor", *[f"{m}_{h}s" for h in (2, 4) for m in ("miss", "min_fde")]])
    if not needed <= set(df):
        raise ValueError(f"missing columns: {sorted(needed - set(df))}")
    if df.duplicated(KEYS + ["predictor"]).any() or set(df.predictor) != {"ALL", "CV-6"}:
        raise ValueError("exactly one ALL and CV-6 row per scenario, replan and agent is required")
    a = df.loc[df.predictor == "ALL"].set_index(KEYS).sort_index()
    c = df.loc[df.predictor == "CV-6"].set_index(KEYS).sort_index()
    if not a.index.equals(c.index) or not a[META].equals(c[META]):
        raise ValueError("predictors must share agents and metadata")
    if not set(a.city) <= set(CITIES) or not set(a.index.get_level_values("replan")) <= set(REPLANS):
        raise ValueError("unknown city or replan")
    for h in (2, 4):
        complete = a[f"complete_{h}s"].to_numpy(bool)
        for arm in (a, c):
            values = arm[[f"miss_{h}s", f"min_fde_{h}s"]].to_numpy(float)
            if not np.isfinite(values[complete]).all() or not np.isnan(values[~complete]).all():
                raise ValueError("metrics must be finite exactly when the horizon is complete")
    return a.reset_index(), c.reset_index()


def analyze(df: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Agent-weighted city rates, equal city weights, paired scenario-cluster CIs.

    Resample scenarios containing eligible group agents in each city. Carry every
    eligible agent of a sampled scenario together, with both predictors paired.
    Missing cities yield inconclusive cells, never a renormalized city average.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    a, c = _paired(df)
    rng = np.random.default_rng(seed)
    selected_coverage = []
    for t in REPLANS:
        for h in horizons(t):
            selected = a.replan == t
            n_selected = int(selected.sum())
            n_excluded = int((selected & ~a[f"complete_{h}s"]).sum())
            selected_coverage.append(
                {
                    "replan": t,
                    "horizon_s": h,
                    "n_selected": n_selected,
                    "n_excluded": n_excluded,
                    "excluded_share": n_excluded / n_selected if n_selected else np.nan,
                }
            )
    cells = []
    for t in REPLANS:
        for h in horizons(t):
            for group in ("stopped", "moving"):
                base = (a.replan == t) & ~a.focal & ((a.speed < 0.5) if group == "stopped" else (a.speed >= 2.0))
                complete = a[f"complete_{h}s"]
                per_city, draws = [], []
                for city in CITIES:
                    selected = base & (a.city == city)
                    mask = selected & complete
                    all_m = a.loc[mask, f"miss_{h}s"].to_numpy(float)
                    cv = c.loc[mask, f"miss_{h}s"].to_numpy(float)
                    ids = a.loc[mask, "scenario_id"].to_numpy()
                    row = {
                        "city": city,
                        "n_selected": int(selected.sum()),
                        "n_agents": int(mask.sum()),
                        "n_scenarios": int(np.unique(ids).size),
                        "n_excluded": int((selected & ~complete).sum()),
                        "excluded_share": float((selected & ~complete).sum() / selected.sum()) if selected.any() else np.nan,
                        "miss_ALL": float(all_m.mean()) if len(all_m) else np.nan,
                        "miss_CV6": float(cv.mean()) if len(cv) else np.nan,
                        "min_fde_ALL": float(a.loc[mask, f"min_fde_{h}s"].mean()),
                        "min_fde_CV6": float(c.loc[mask, f"min_fde_{h}s"].mean()),
                    }
                    per_city.append(row)
                    boot = np.full(n_boot, np.nan)
                    if len(ids):
                        _, inverse = np.unique(ids, return_inverse=True)
                        cnt = np.bincount(inverse)
                        sums = np.bincount(inverse, weights=all_m - cv)
                        for start in range(0, n_boot, 128):
                            stop = min(start + 128, n_boot)
                            ii = rng.integers(0, len(cnt), size=(stop - start, len(cnt)))
                            boot[start:stop] = sums[ii].sum(1) / cnt[ii].sum(1)
                    draws.append(boot)
                pooled = np.mean(draws, axis=0)
                defined = np.isfinite(pooled).all()
                mr_all = float(np.mean([r["miss_ALL"] for r in per_city]))
                mr_cv = float(np.mean([r["miss_CV6"] for r in per_city]))
                point = mr_all - mr_cv
                ci = np.percentile(pooled, [2.5, 97.5]).tolist() if defined else [np.nan, np.nan]
                primary = group == "stopped" and h == 2 and t in (49, 89)
                status = "inconclusive" if not defined else "descriptive"
                if primary and defined:
                    status = "supported" if point >= 0.15 and ci[0] > 0 else "killed"
                n_selected = sum(r["n_selected"] for r in per_city)
                n_excluded = sum(r["n_excluded"] for r in per_city)
                cells.append(
                    {
                        "replan": t,
                        "horizon_s": h,
                        "group": group,
                        "primary": primary,
                        "miss_ALL": mr_all,
                        "miss_CV6": mr_cv,
                        "difference_ALL_minus_CV6": point,
                        "ci95": ci,
                        "status": status,
                        "per_city": per_city,
                        "n_selected": n_selected,
                        "n_agents": n_selected - n_excluded,
                        "n_scenarios": sum(r["n_scenarios"] for r in per_city),
                        "n_excluded": n_excluded,
                        "excluded_share": n_excluded / n_selected if n_selected else np.nan,
                    }
                )
    statuses = [cell["status"] for cell in cells if cell["primary"]]
    decision = "killed" if "killed" in statuses else "supported" if all(s == "supported" for s in statuses) else "inconclusive"
    return {
        "study": "F",
        "exploratory": True,
        "n_boot": n_boot,
        "seed": seed,
        "miss_threshold_m": {"2s": MISS_THRESHOLD_M, "4s": MISS_THRESHOLD_M},
        "decision": decision,
        "selected_coverage": selected_coverage,
        "cells": cells,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw", help="AV2 validation directory")
    source.add_argument("--parquet", help="analyze existing coverage rows")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="CPU smoke: 20 scenarios")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0, help="score every num-shards-th scenario starting here")
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    if args.limit < 0 or args.threads < 1 or not 0 <= args.shard < args.num_shards:
        ap.error("--limit must be nonnegative, --threads positive and 0 <= --shard < --num-shards")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.parquet:
        paths = sorted(glob.glob(args.parquet))  # one file or every shard
        result = analyze(pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True))
        metadata = pq.read_schema(paths[0]).metadata or {}
        if metadata.get(b"cityshift.coverage.smoke") == b"true":
            result["inferential"] = False
            result["decision"] = "smoke only"
            for cell in result["cells"]:
                cell["status"] = "smoke only"
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {out}")
        return
    if Path(args.raw).name != "val":
        ap.error("--raw must be the AV2 val split")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    models = load_models({f"ALL_s{s}": str(Path(args.runs) / "ALL" / f"seed{s}" / "model.pt") for s in range(3)}, device)
    dirs = sorted(p for p in Path(args.raw).iterdir() if p.is_dir())
    if args.limit:
        dirs = dirs[: args.limit]
    dirs = dirs[args.shard :: args.num_shards]
    if not dirs:
        ap.error("no scenarios selected")
    writer = None
    count = 0
    try:
        for j, directory in enumerate(dirs, 1):
            rows = score_scene(load_scene(str(directory)), models, device)
            if rows:
                table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
                table = table.replace_schema_metadata(
                    (table.schema.metadata or {})
                    | {
                        b"cityshift.coverage.smoke": b"true" if args.limit else b"false",
                        b"cityshift.coverage.scenarios": str(len(dirs)).encode("ascii"),
                    }
                )
                if writer is None:
                    writer = pq.ParquetWriter(out, table.schema, compression="zstd")
                writer.write_table(table)
                count += len(rows)
            if j % 20 == 0 or j == len(dirs):
                print(f"{j}/{len(dirs)} scenarios, {count} rows", flush=True)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("no planner-selected agents in the requested scenarios")
    print(f"wrote {out}: {count} rows")


if __name__ == "__main__":
    main()
