"""EXPLORATORY study A analysis: does dropping moving-mode probability (TRIMNF) give at least half of the brake
reduction of CV on the same agents (PATCHSUB)? Registered in docs/preregistration/exploratory-followups.md.

    PYTHONPATH=src python -m cityshift.study_a --rows runs/followups/study_a.parquet \
        --parity runs/followups/study_a_parity.parquet.report.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .analysis_stage3 import CITIES

LEVEL, LEVEL_BONF, MARGIN, BOOT_SEED = 0.95, 1 - 0.05 / 3, 0.003, 20260930


def interval(draws: np.ndarray, level: float) -> list[float]:
    return [float(x) for x in np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])]


def seed_mean(df: pd.DataFrame, arm: str, metric: str) -> np.ndarray:
    return df[[f"{arm}_s{s}_{metric}" for s in range(3)]].to_numpy(float).mean(1)


def effects(m: dict) -> dict:
    """City-level means -> effects. Reductions relative to ALL; R is formed after equal-weighting cities."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"TRIMNF_reduction": (m["ALL_b"] - m["TRIMNF_b"]) / m["ALL_b"],
                "PATCHSUB_reduction": (m["ALL_b"] - m["PATCHSUB_b"]) / m["ALL_b"],
                "TRIMNF_collision_diff": m["TRIMNF_c"] - m["ALL_c"],
                "PATCHSUB_collision_diff": m["PATCHSUB_c"] - m["ALL_c"]}


def analyze(data: pd.DataFrame, n_boot: int = 10_000, seed: int = BOOT_SEED) -> dict:
    """data: scenario rows with city and seed-averaged ALL_b, TRIMNF_b, PATCHSUB_b, ALL_c, TRIMNF_c, PATCHSUB_c."""
    cols = ["ALL_b", "TRIMNF_b", "PATCHSUB_b", "ALL_c", "TRIMNF_c", "PATCHSUB_c"]
    rng = np.random.default_rng(seed)
    points, draws = {}, {}
    for city in CITIES:
        m = data.loc[data.city.astype(str) == city, cols].to_numpy(float)
        assert len(m), f"missing city {city}"
        for k, v in effects(dict(zip(cols, m.mean(0)))).items():
            points.setdefault(k, []).append(float(v))
        idx = rng.integers(0, len(m), size=(n_boot, len(m)))
        for k, v in effects(dict(zip(cols, m[idx].mean(1).T))).items():
            draws.setdefault(k, []).append(v)
    point = {k: float(np.mean(v)) for k, v in points.items()}
    boot = {k: np.mean(np.stack(v), 0) for k, v in draws.items()}
    with np.errstate(divide="ignore", invalid="ignore"):
        den = point["PATCHSUB_reduction"]
        point["R"] = point["TRIMNF_reduction"] / den if den != 0 else float("nan")  # failed control: undefined, not a crash
        boot["R"] = boot["TRIMNF_reduction"] / boot["PATCHSUB_reduction"]
    out = {}
    for k in point:
        bad = ~np.isfinite(boot[k])
        ok = boot[k][~bad]
        out[k] = {"point": point[k], "undefined_share": float(bad.mean()),
                  "ci95": interval(ok, LEVEL) if len(ok) else [None, None],
                  "ci9833": interval(ok, LEVEL_BONF) if len(ok) else [None, None]}
    return out


def decide(eff: dict, all_events: float, zero_cities: list[str], dose_ratio: float) -> str:
    """The registered decision rules for study A."""
    if not np.isfinite(dose_ratio) or not 0.95 <= dose_ratio <= 1.05:
        return "uninterpretable (eligible-count ratio outside 0.95 to 1.05)"
    if all_events < 100 or zero_cities or max(eff[k]["undefined_share"] for k in eff) > 0.01:
        return "inconclusive (A8)"
    if not eff["PATCHSUB_reduction"]["ci95"][0] > 0:
        return "uninterpretable (PATCHSUB positive control failed)"
    lo, hi = eff["R"]["ci95"]
    return "supported" if lo >= 0.5 else "killed" if hi < 0.5 else "inconclusive"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="followup_a output with TRIMNF and PATCHSUB, seeds 0-2")
    ap.add_argument("--parity", required=True, help="report of the ALL rerun (must pass, >= 100 scenarios, on CUDA)")
    ap.add_argument("--stage4", default="runs/stage4/closedloop_pool.parquet")
    ap.add_argument("--out", default="reports/followups/study_a.json")
    ap.add_argument("--bootstraps", type=int, default=10_000)
    args = ap.parse_args()
    par = json.load(open(args.parity))
    p = par.get("ALL_parity_vs_stage4", {})
    if not (p.get("passed") and p.get("scenarios", 0) >= 100 and par.get("device", "").startswith("cuda")):
        raise SystemExit("ALL rerun parity (>= 100 scenarios, CUDA, exact) is required before study A is analysed")
    a = pd.read_parquet(args.rows)
    s4 = pd.read_parquet(args.stage4)
    assert len(a) == 8140 and set(a.scenario_id) == set(s4.scenario_id), "study A must cover the whole pool"
    d = s4[["scenario_id", "city"]].merge(a, on=["scenario_id", "city"], validate="one_to_one")
    s4 = s4.set_index("scenario_id").loc[d.scenario_id]
    data = pd.DataFrame({"city": d.city.to_numpy(),
                         "ALL_b": seed_mean(s4, "ALL", "unnecessary_hard_brake"),
                         "ALL_c": seed_mean(s4, "ALL", "collision"),
                         **{f"{arm}_{x}": seed_mean(d, arm, m) for arm in ("TRIMNF", "PATCHSUB")
                            for x, m in (("b", "unnecessary_hard_brake"), ("c", "collision"))}})
    eff = analyze(data, args.bootstraps)
    elig = {arm: float(sum(np.sum(x) for s in range(3) for x in d[f"{arm}_s{s}_eligible_by_replan"]))
            for arm in ("TRIMNF", "PATCHSUB")}
    dose_ratio = elig["TRIMNF"] / elig["PATCHSUB"] if elig["PATCHSUB"] > 0 else float("nan")  # nan fails the gate
    zero = [c for c in CITIES if data.loc[data.city.astype(str) == c, "ALL_b"].sum() == 0]
    events = float(data.ALL_b.sum())
    result = {"_note": "EXPLORATORY study A, docs/preregistration/exploratory-followups.md. 95% within-city scenario "
                       "bootstrap (Bonferroni 98.33% alongside), seeds averaged per scenario, cities equal-weighted.",
              "verdict": decide(eff, events, zero, dose_ratio), "effects": eff, "eligible_totals": elig,
              "eligible_ratio": dose_ratio, "seed_averaged_ALL_brake_events": events, "parity": par,
              "TRIMNF_collision_noninferior": bool(eff["TRIMNF_collision_diff"]["ci95"][1] < MARGIN)}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({"verdict": result["verdict"], "R": eff["R"]["point"], "R_ci95": eff["R"]["ci95"]}))


if __name__ == "__main__":
    main()
