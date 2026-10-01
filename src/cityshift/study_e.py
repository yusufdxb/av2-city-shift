"""EXPLORATORY study E analysis: H10-H12 braking components in the stop-line harness (braking only).

Registered in docs/preregistration/exploratory-followups.md (study E). Collisions are descriptive: the CPU pilot showed
the collision checker has no positive control in this harness.

    PYTHONPATH=src python -m cityshift.study_e --parts runs/followups/study_e
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from .analysis_stage3 import CITIES
from .closedloop_v3 import POOL_SHA256, checked_ids

LEVEL, LEVEL_BONF, BOOT_SEED = 0.95, 1 - 0.05 / 3, 20260930
ARMS = ("ALL", "PATCH", "SHAM2", "TRIM")


def interval(draws: np.ndarray, level: float) -> list[float]:
    return [float(x) for x in np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])]


def seed_mean(df: pd.DataFrame, arm: str, metric: str) -> np.ndarray:
    return df[[f"{arm}_s{s}_{metric}" for s in range(3)]].to_numpy(float).mean(1)


def effects(m: dict) -> dict:
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"PATCH_reduction": (m["ALL"] - m["PATCH"]) / m["ALL"],
                "TRIM_reduction": (m["ALL"] - m["TRIM"]) / m["ALL"],
                "sham_gap": (m["SHAM2"] - m["PATCH"]) / m["ALL"]}


def analyze(data: pd.DataFrame, n_boot: int = 10_000, seed: int = BOOT_SEED) -> dict:
    """data: scenario rows with city and seed-averaged unnecessary-brake columns ALL, PATCH, SHAM2, TRIM."""
    rng = np.random.default_rng(seed)
    points, draws = {}, {}
    for city in CITIES:
        m = data.loc[data.city.astype(str) == city, list(ARMS)].to_numpy(float)
        assert len(m), f"missing city {city}"
        for k, v in effects(dict(zip(ARMS, m.mean(0)))).items():
            points.setdefault(k, []).append(float(v))
        idx = rng.integers(0, len(m), size=(n_boot, len(m)))
        for k, v in effects(dict(zip(ARMS, m[idx].mean(1).T))).items():
            draws.setdefault(k, []).append(v)
    out = {}
    for k, v in points.items():
        boot = np.mean(np.stack(draws[k]), 0)
        bad = ~np.isfinite(boot)
        ok = boot[~bad]
        out[k] = {"point": float(np.mean(v)), "undefined_share": float(bad.mean()),
                  "ci95": interval(ok, LEVEL) if len(ok) else [None, None],
                  "ci9833": interval(ok, LEVEL_BONF) if len(ok) else [None, None]}
    return out


def decide(e: dict, bar: float, powered: bool, interpretable: bool) -> str:
    if not interpretable:
        return "uninterpretable"
    if not powered or e["undefined_share"] > 0.01 or e["ci95"][0] is None:
        return "inconclusive (A8)"
    return "supported" if e["point"] >= bar and e["ci95"][0] > 0 else "killed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True)
    ap.add_argument("--pool", default="runs/replication_pool.npy")
    ap.add_argument("--out", default="reports/followups/study_e.json")
    ap.add_argument("--bootstraps", type=int, default=10_000)
    args = ap.parse_args()
    d = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(args.parts, "part_*.parquet")))],
                  ignore_index=True)
    assert len(d) == 8140 and not d.scenario_id.duplicated().any()
    assert set(d.scenario_id) == set(checked_ids(args.pool, POOL_SHA256)), "study E must cover the whole pool"
    data = pd.DataFrame({"city": d.city, **{a: seed_mean(d, a, "unnecessary_hard_brake") for a in ARMS}})
    eff = analyze(data, args.bootstraps)
    rates = {a: float(data[a].mean()) for a in ARMS} | {
        a: float(d[f"{a}_unnecessary_hard_brake"].astype(float).mean()) for a in ("cv", "oracle", "static")}
    collisions = {a: float(seed_mean(d, a, "collision").mean()) for a in ARMS} | {
        a: float(d[f"{a}_collision"].astype(float).mean()) for a in ("cv", "oracle", "static", "log")}
    dose = {a: int(sum(np.sum(x) for s in range(3) for x in d[f"{a}_s{s}_dose_by_replan"])) for a in ("PATCH", "SHAM2")}
    dose_ratio = dose["SHAM2"] / dose["PATCH"]
    control = rates["ALL"] >= 2 * rates["oracle"]
    events = float(data.ALL.sum())
    powered = events >= 100 and all(data.loc[data.city.astype(str) == c, "ALL"].sum() > 0 for c in CITIES)
    verdicts = {"H10e PATCH": decide(eff["PATCH_reduction"], 0.30, powered, control),
                "H11e sham gap": decide(eff["sham_gap"], 0.20, powered, control and dose_ratio >= 0.99),
                "H12e TRIM": decide(eff["TRIM_reduction"], 0.30, powered, control)}
    result = {"_note": "EXPLORATORY study E (docs/preregistration/exploratory-followups.md): braking only, stop-line "
                       "harness, 8,140 pool scenarios, seeds averaged per scenario, cities equal-weighted, 10,000 "
                       "within-city bootstraps, 95% (Bonferroni 98.33% alongside). Collisions descriptive only.",
              "verdicts": verdicts, "effects": eff, "brake_rates": rates, "brake_control_ALL_over_oracle":
              rates["ALL"] / rates["oracle"] if rates["oracle"] else None, "brake_control_pass": bool(control),
              "dose": dose, "dose_ratio": dose_ratio, "seed_averaged_ALL_brake_events": events,
              "collisions_descriptive": collisions,
              "median_progress": {a: float(np.nanmedian(seed_mean(d, a, "progress"))) for a in ARMS}}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(verdicts))


if __name__ == "__main__":
    main()
