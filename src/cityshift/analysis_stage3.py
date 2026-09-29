"""Pre-registered Stage 3 paired scenario-cluster analysis within city."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

BOOTSTRAPS = 10000
CI_LEVEL = 1 - 0.05 / 6  # six directional inequalities in H7, H8, H9
CITIES = ("austin", "dearborn", "miami", "palo-alto", "pittsburgh", "washington-dc")


def interval(draws: np.ndarray, level: float = CI_LEVEL) -> list[float]:
    return [float(x) for x in np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])]


def decide_reduction(effect: float | None, ci: list[float | None], minimum: float) -> bool:
    return bool(effect is not None and ci[0] is not None and np.isfinite(effect) and ci[0] > 0 and effect >= minimum)


def decide_noninferiority(effect: float | None, ci: list[float | None], margin: float = 0.003) -> bool:
    return bool(effect is not None and ci[1] is not None and np.isfinite(effect) and ci[1] < margin)


def decide_improvement(effect: float | None, ci: list[float | None], minimum: float) -> bool:
    return bool(effect is not None and ci[1] is not None and np.isfinite(effect) and ci[1] < 0 and effect <= -minimum)


def _seed_mean(df: pd.DataFrame, arm: str, outcome: str) -> np.ndarray:
    cols = [f"{arm}_s{s}_{outcome}" for s in range(3)]
    dev = f"{arm}_dev_{outcome}"
    if all(c in df for c in cols):
        return df[cols].to_numpy(float).mean(1)
    if dev in df:
        return df[dev].to_numpy(float)
    single = f"{arm}_{outcome}"
    if single in df:
        return df[single].to_numpy(float)
    raise ValueError(f"missing {arm} {outcome} columns")


def _arm_mean(df: pd.DataFrame, arm: str, outcome: str) -> np.ndarray:
    name = f"{arm}_{outcome}"
    if name not in df:
        raise ValueError(f"missing {name}")
    return df[name].to_numpy(float)


def _closed_arrays(df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    arms = ("ALL", "MULTI", "PATCH", "SHAM", "cv", "oracle", "static", "log")
    brake, collision = {}, {}
    for arm in arms:
        if arm != "log":
            brake[arm] = _seed_mean(df, arm, "unnecessary_hard_brake") if arm.isupper() and arm != "STATIC" else _arm_mean(df, arm, "unnecessary_hard_brake")
        collision[arm] = _seed_mean(df, arm, "collision") if arm in ("ALL", "MULTI", "PATCH", "SHAM") else _arm_mean(df, arm, "collision")
    return brake, collision


def _city_draws(df: pd.DataFrame, arrays: dict[str, np.ndarray], rng: np.random.Generator,
                n_boot: int) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    cities = df.city.astype(str).to_numpy()
    point, boot = {}, {}
    for city in CITIES:
        idx = np.flatnonzero(cities == city)
        if not len(idx):
            raise ValueError(f"missing city {city}")
        point[city] = {arm: float(val[idx].mean()) for arm, val in arrays.items()}
        sampled = rng.integers(0, len(idx), size=(n_boot, len(idx)))
        boot[city] = {arm: val[idx][sampled].mean(1) for arm, val in arrays.items()}
    return point, boot


def _aggregate(point: dict, boot: dict, fn) -> tuple[float, np.ndarray]:
    value = float(np.mean([fn(point[c]) for c in CITIES]))
    draws = np.mean(np.stack([fn(boot[c]) for c in CITIES]), axis=0)
    return value, draws


def closedloop_analysis(df: pd.DataFrame, rng: np.random.Generator, n_boot: int = BOOTSTRAPS) -> dict:
    if df.scenario_id.duplicated().any():
        raise ValueError("duplicate closed-loop scenario IDs")
    brake, collision = _closed_arrays(df)
    bp, bb = _city_draws(df, brake, rng, n_boot)
    cp, cb = _city_draws(df, collision, rng, n_boot)
    result = {"rates": {"brake": {a: float(v.mean()) for a, v in brake.items()},
                        "collision": {a: float(v.mean()) for a, v in collision.items()}},
              "per_city_rates": {c: {"brake": bp[c], "collision": cp[c]} for c in CITIES},
              "mean_trigger_count": {a: float(_seed_mean(df, a, "trigger_count").mean()) for a in ("PATCH", "SHAM")
                                     if f"{a}_s0_trigger_count" in df or f"{a}_trigger_count" in df}}
    for name, arm in (("H7", "MULTI"), ("H8", "PATCH")):
        undefined_cities = [city for city in CITIES if bp[city]["ALL"] == 0]
        if undefined_cities:
            effect, ci, undefined_draws = None, [None, None], 1.0
        else:
            effect = float(np.mean([(bp[c]["ALL"] - bp[c][arm]) / bp[c]["ALL"] for c in CITIES]))
            city_draws = []
            for city in CITIES:
                denominator = bb[city]["ALL"]
                with np.errstate(divide="ignore", invalid="ignore"):
                    city_draws.append((denominator - bb[city][arm]) / denominator)
            draws = np.mean(np.stack(city_draws), axis=0)
            valid = np.isfinite(draws)
            undefined_draws = float((~valid).mean())
            ci = interval(draws[valid]) if valid.any() and undefined_draws <= 0.01 else [None, None]
        diff, diff_draws = _aggregate(cp, cb, lambda x: x[arm] - x["ALL"])
        diff_ci = interval(diff_draws)
        underpowered = bool(sum(brake["ALL"]) < 100)
        result[name] = {"brake_reduction": effect, "brake_ci": ci, "collision_difference": diff,
                        "collision_ci": diff_ci, "underpowered": underpowered,
                        "undefined_cities": undefined_cities, "undefined_bootstrap_share": undefined_draws,
                        "per_city": {c: {"brake_reduction": (bp[c]["ALL"] - bp[c][arm]) / bp[c]["ALL"]
                                          if bp[c]["ALL"] > 0 else None,
                                          "collision_difference": cp[c][arm] - cp[c]["ALL"]} for c in CITIES},
                        "supported": not underpowered and not undefined_cities and undefined_draws <= 0.01
                        and decide_reduction(effect, ci, 0.30) and decide_noninferiority(diff, diff_ci)}
    # Dose-matched sham (deviation 1): CV substitution on an equal number of MOVING agents per replan.
    sham_reduction = float(np.mean([(bp[c]["ALL"] - bp[c]["SHAM"]) / bp[c]["ALL"] for c in CITIES if bp[c]["ALL"] > 0]))
    patch_reduction = result["H8"]["brake_reduction"]
    all_col = float(collision["ALL"].mean())
    static_col = float(collision["static"].mean())
    log_col = float(collision["log"].mean())
    control = {"STATIC_over_ALL": static_col / all_col if all_col else None,
               "STATIC_pass": static_col >= 2 * all_col and all_col > 0,
               "LOG_collision": log_col, "LOG_pass": log_col < 0.01,
               "SHAM_brake_reduction": sham_reduction,
               # H8 is stopped-specific only if the dose-matched sham buys less than half of PATCH's reduction.
               "SHAM_pass": patch_reduction is not None and sham_reduction < 0.5 * patch_reduction}
    result["controls"] = control
    for name in ("H7", "H8"):
        claim = result[name]
        sham_ok = control["SHAM_pass"] if name == "H8" else True
        claim["supported"] = claim["supported"] and control["STATIC_pass"] and control["LOG_pass"] and sham_ok
        if claim["underpowered"]:
            claim["status"] = "underpowered"
        elif claim["undefined_cities"] or claim["undefined_bootstrap_share"] > 0.01:
            claim["status"] = "inconclusive"
        elif not (control["STATIC_pass"] and control["LOG_pass"]):
            claim["status"] = "uninterpretable"
        elif name == "H8" and not sham_ok and claim["supported"] is False and decide_reduction(
                claim["brake_reduction"], claim["brake_ci"], 0.30) and decide_noninferiority(
                claim["collision_difference"], claim["collision_ci"]):
            claim["status"] = "reduction real but not stopped-specific (dose-matched sham matched it)"
        else:
            claim["status"] = "supported" if claim["supported"] else "killed"
    return result


def _open_group(df: pd.DataFrame, mask: pd.Series, rng: np.random.Generator,
                n_boot: int) -> tuple[float, np.ndarray, dict[str, float]]:
    subset = df[mask & df.predictor.isin([f"ALL_s{s}" for s in range(3)] + [f"MULTI_s{s}" for s in range(3)])]
    if subset.empty:
        raise ValueError("open-loop group is empty")
    wide = subset.pivot(index=["city", "scenario_id", "agent_id"], columns="predictor", values="miss")
    all_cols = [f"ALL_s{s}" for s in range(3)]
    multi_cols = [f"MULTI_s{s}" for s in range(3)]
    if wide[all_cols + multi_cols].isna().any().any():
        raise ValueError("unpaired or missing open-loop predictions")
    wide = wide.assign(ALL=wide[all_cols].mean(1), MULTI=wide[multi_cols].mean(1)).reset_index()
    grouped = wide.groupby(["city", "scenario_id"], sort=False)[["ALL", "MULTI"]].agg(["sum", "count"])
    grouped.columns = ["all_sum", "all_n", "multi_sum", "multi_n"]
    grouped = grouped.reset_index()
    points, draws = [], []
    per_city = {}
    for city in CITIES:
        x = grouped[grouped.city == city]
        if x.empty:
            raise ValueError(f"missing open-loop group in {city}")
        a, m = x.all_sum.to_numpy(), x.multi_sum.to_numpy()
        counts = x.all_n.to_numpy()
        per_city[city] = float((m.sum() - a.sum()) / counts.sum())
        points.append(per_city[city])
        sampled = rng.integers(0, len(x), size=(n_boot, len(x)))
        draws.append((m[sampled].sum(1) - a[sampled].sum(1)) / counts[sampled].sum(1))
    return float(np.mean(points)), np.mean(np.stack(draws), axis=0), per_city


def openloop_analysis(df: pd.DataFrame, rng: np.random.Generator, n_boot: int = BOOTSTRAPS) -> dict:
    stopped = df.planner_relevant.astype(bool) & ~df.focal.astype(bool) & (df.speed < 0.5)
    focal = df.focal.astype(bool)
    stopped_diff, stopped_draws, stopped_cities = _open_group(df, stopped, rng, n_boot)
    focal_diff, focal_draws, focal_cities = _open_group(df, focal, rng, n_boot)
    stopped_ci, focal_ci = interval(stopped_draws), interval(focal_draws)
    unique_stopped = df[stopped].drop_duplicates(["scenario_id", "agent_id"])
    unique_focal = df[focal].drop_duplicates(["scenario_id", "agent_id"])
    return {"n_stopped_agents": len(unique_stopped), "n_stopped_scenarios": unique_stopped.scenario_id.nunique(),
            "n_focal_agents": len(unique_focal),
            "stopped_valid_steps": unique_stopped.n_valid.describe()[["min", "50%", "max"]].to_dict()
            if "n_valid" in unique_stopped else None,
            "per_city": {c: {"stopped_nonfocal_difference": stopped_cities[c],
                             "focal_difference": focal_cities[c]} for c in CITIES},
            "stopped_nonfocal_difference": stopped_diff, "stopped_nonfocal_ci": stopped_ci,
            "focal_difference": focal_diff, "focal_ci": focal_ci,
            "supported": decide_improvement(stopped_diff, stopped_ci, 0.15)
            and decide_noninferiority(focal_diff, focal_ci, 0.02),
            "status": "supported" if decide_improvement(stopped_diff, stopped_ci, 0.15)
            and decide_noninferiority(focal_diff, focal_ci, 0.02) else "killed"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--closedloop", required=True)
    ap.add_argument("--openloop", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    args = ap.parse_args()
    rng = np.random.default_rng(3407)
    result = {"ci_level": CI_LEVEL, "bootstraps": args.bootstraps,
              "closedloop": closedloop_analysis(pd.read_parquet(args.closedloop), rng, args.bootstraps),
              "H9": openloop_analysis(pd.read_parquet(args.openloop), rng, args.bootstraps)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
