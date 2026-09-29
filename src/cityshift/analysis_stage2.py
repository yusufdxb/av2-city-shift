"""Stage 2 pre-registered analysis (docs/preregistration/2026-09-28-stage2-closed-loop.md).

Input: the closed-loop parquet from ``closedloop.py`` run with arms
log,oracle,cv,static,ALL,LOCO on the val split. Output: <out>.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .analysis import CI_BONF, sign_flip_p
from .data import CITIES

N_BOOT = 10000
MIN_EVENTS = 100


def seed_avg(df: pd.DataFrame, arm: str, metric: str) -> np.ndarray:
    cols = sorted(c for c in df.columns if c.startswith(f"{arm}_s") and c.endswith(f"_{metric}"))
    assert cols, f"no columns for {arm} {metric}"
    return df[cols].to_numpy(float).mean(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    df = pd.read_parquet(args.parquet)
    city = df.city.to_numpy()
    f_all, f_loco = seed_avg(df, "ALL", "failure"), seed_avg(df, "LOCO", "failure")
    res: dict = {"per_city": {}}
    rel, boots = [], []
    for c in CITIES:
        m = city == c
        a, lo = f_all[m], f_loco[m]
        rel.append((lo.mean() - a.mean()) / a.mean() if a.mean() > 0 else np.nan)
        ii = rng.integers(0, m.sum(), size=(N_BOOT, m.sum()))
        am = a[ii].mean(1)
        boots.append(np.where(am > 0, (lo[ii].mean(1) - am) / np.where(am > 0, am, 1), np.nan))
        row = {"n": int(m.sum()), "failure_ALL": float(a.mean()), "failure_LOCO": float(lo.mean()), "rel_change": float(rel[-1])}
        for comp in ("collision", "unnecessary_hard_brake", "progress"):
            row[f"{comp}_ALL"] = float(np.nanmean(seed_avg(df, "ALL", comp)[m]))
            row[f"{comp}_LOCO"] = float(np.nanmean(seed_avg(df, "LOCO", comp)[m]))
        for arm in ("oracle", "cv", "static", "log"):
            row[f"failure_{arm}"] = float(df[f"{arm}_failure"].to_numpy(float)[m].mean())
            row[f"collision_{arm}"] = float(df[f"{arm}_collision"].to_numpy(float)[m].mean())
        res["per_city"][c] = row
    rel = np.array(rel, float)
    # A fold's relative change is undefined when its ALL failure rate is zero. Exclude a fold
    # if that happens at the point estimate or in more than 1% of its bootstrap resamples, so
    # the pooled interval is not silently conditioned on non-zero-event draws.
    undefined_share = np.array([float(np.isnan(b).mean()) for b in boots])
    ok = np.isfinite(rel) & (undefined_share <= 0.01)
    for c, u in zip(CITIES, undefined_share):
        res["per_city"][c]["bootstrap_undefined_share"] = float(u)
    pooled = np.nanmean(np.array(boots)[ok], 0) if ok.any() else np.full(N_BOOT, np.nan)
    events_all = float(f_all.sum())
    res["H4"] = {
        "pooled_rel_change": float(rel[ok].mean()) if ok.any() else float("nan"),
        "ci95": [float(np.nanpercentile(pooled, 2.5)), float(np.nanpercentile(pooled, 97.5))],
        "ci_bonferroni": [float(np.nanpercentile(pooled, CI_BONF[0])), float(np.nanpercentile(pooled, CI_BONF[1]))],
        "fold_consistency_p_two_sided": sign_flip_p(rel[ok]) if ok.sum() >= 2 else float("nan"),
        "folds_undefined": [c for c, g in zip(CITIES, ok) if not g],
        "events_ALL_seed_avg": events_all,
        "underpowered": events_all < MIN_EVENTS,
    }
    col_all = float(seed_avg(df, "ALL", "collision").mean())
    col_static = float(df["static_collision"].to_numpy(float).mean())
    res["PC3"] = {"collision_ALL": col_all, "collision_STATIC": col_static, "pass": col_static >= 2 * col_all}
    col_log = float(df["log_collision"].to_numpy(float).mean())
    res["checker_calibration"] = {"collision_LOG": col_log, "pass": col_log < 0.01}
    lo = res["H4"]["ci_bonferroni"][0]
    if res["H4"]["underpowered"]:
        res["H4"]["decision"] = "underpowered"
    elif not (res["PC3"]["pass"] and res["checker_calibration"]["pass"]):
        res["H4"]["decision"] = "uninterpretable (PC3 or checker calibration failed)"
    elif lo > 0 and res["H4"]["pooled_rel_change"] >= 0.10:
        res["H4"]["decision"] = "supported"
    else:
        res["H4"]["decision"] = "dead"
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "per_city"}, indent=2))


if __name__ == "__main__":
    main()
