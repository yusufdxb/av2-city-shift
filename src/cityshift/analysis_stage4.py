"""Stage 4 paired within-city scenario bootstrap and descriptive risk calibration."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .analysis_stage3 import CITIES
from .closedloop_v3 import POOL_SHA256, checked_ids
from .closedloop import REPLANS

BOOTSTRAPS = 10000
CI_LEVEL = 1 - 0.05 / 8
DESCRIPTIVE_CI_LEVEL = 0.9875
TREATMENTS = ("PATCH", "TRIM", "SHAM2", "MIX")
MODEL_ARMS = ("ALL", *TREATMENTS)


def interval(draws: np.ndarray, level: float = CI_LEVEL) -> list[float]:
    """Two-sided percentile interval."""
    return [float(x) for x in np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])]


def decide_reduction(point: float, ci: list[float], minimum: float = 0.30) -> bool:
    return bool(np.isfinite(point) and point >= minimum and ci[0] > 0)


def decide_noninferiority(ci: list[float], margin: float) -> bool:
    return bool(ci[1] < margin)


def decide_gap(point: float, ci: list[float]) -> bool:
    return bool(np.isfinite(point) and point >= 0.20 and ci[0] > 0)


def _seed_mean(df: pd.DataFrame, arm: str, metric: str) -> np.ndarray:
    names = [f"{arm}_s{seed}_{metric}" for seed in range(3)]
    if not all(name in df for name in names):
        raise ValueError(f"missing seed-level {arm} {metric}")
    return df[names].to_numpy(float).mean(1)


def _focal_miss(df: pd.DataFrame) -> pd.DataFrame:
    if not df.focal.astype(bool).all():
        raise ValueError("open-loop file must contain focal agents only")
    if df.duplicated(["scenario_id", "predictor"]).any():
        raise ValueError("duplicate focal predictions")
    wide = df.pivot(index=["city", "scenario_id"], columns="predictor", values="miss")
    names = [f"{arm}_s{seed}" for arm in ("ALL", "MIX") for seed in range(3)]
    if not set(names) <= set(wide) or wide[names].isna().any().any():
        raise ValueError("incomplete focal seed pairing")
    return pd.DataFrame({"scenario_id": wide.index.get_level_values("scenario_id"),
                         "focal_all": wide[[f"ALL_s{s}" for s in range(3)]].mean(1).to_numpy(),
                         "focal_mix": wide[[f"MIX_s{s}" for s in range(3)]].mean(1).to_numpy()})


def _effects(means: dict[str, np.ndarray | float]) -> dict[str, np.ndarray | float]:
    den = means["ALL_brake"]
    with np.errstate(divide="ignore", invalid="ignore"):
        effects = {f"{arm}_reduction": (den - means[f"{arm}_brake"]) / den for arm in TREATMENTS}
        effects["sham_gap"] = (means["SHAM2_brake"] - means["PATCH_brake"]) / den
    for arm in ("PATCH", "TRIM", "MIX"):
        effects[f"{arm}_collision_diff"] = means[f"{arm}_collision"] - means["ALL_collision"]
    effects["focal_miss_diff"] = means["focal_mix"] - means["focal_all"]
    return effects


def analyze(closed: pd.DataFrame, focal: pd.DataFrame, n_boot: int = BOOTSTRAPS) -> dict:
    """Analyze paired scenario rows, averaging model seeds before resampling."""
    if closed.scenario_id.duplicated().any():
        raise ValueError("duplicate closed-loop scenarios")
    data = closed[["scenario_id", "city"]].copy()
    for arm in MODEL_ARMS:
        for metric in ("unnecessary_hard_brake", "collision", "trigger_count", "substituted_count"):
            data[f"{arm}_{'brake' if metric == 'unnecessary_hard_brake' else metric}"] = _seed_mean(closed, arm, metric)
    data = data.merge(_focal_miss(focal), on="scenario_id", validate="one_to_one")
    if len(data) != len(closed):
        raise ValueError("missing focal scenarios")
    numeric = [name for name in data if name not in ("scenario_id", "city")]
    rng = np.random.default_rng(4404)
    points = {}
    draws: dict[str, list[np.ndarray]] = defaultdict(list)
    per_city = {}
    undefined_share = 0.0
    for city in CITIES:
        part = data[data.city.astype(str) == city]
        if part.empty:
            raise ValueError(f"missing city {city}")
        matrix = part[numeric].to_numpy(float)
        mean = dict(zip(numeric, matrix.mean(0)))
        city_effect = _effects(mean)
        per_city[city] = {key: float(value) for key, value in city_effect.items()}
        for key, value in city_effect.items():
            points.setdefault(key, []).append(float(value))
        city_draws: dict[str, list[np.ndarray]] = defaultdict(list)
        for start in range(0, n_boot, 128):
            count = min(128, n_boot - start)
            index = rng.integers(0, len(matrix), size=(count, len(matrix)))
            averages = matrix[index].mean(1)
            effect = _effects(dict(zip(numeric, averages.T)))
            for key, value in effect.items():
                city_draws[key].append(np.asarray(value))
        for key, pieces in city_draws.items():
            draws[key].append(np.concatenate(pieces))
    aggregated = {}
    for key, values in points.items():
        boot = np.mean(np.stack(draws[key]), axis=0)
        invalid = ~np.isfinite(boot)
        undefined_share = max(undefined_share, float(invalid.mean()))
        valid = boot[~invalid]
        aggregated[key] = {"point": float(np.mean(values)),
                           "ci": interval(valid) if len(valid) and invalid.mean() <= 0.01 else [None, None],
                           "ci_98_75": interval(valid, DESCRIPTIVE_CI_LEVEL)
                           if len(valid) and invalid.mean() <= 0.01 else [None, None]}
    # Matched sham: SHAM2 copies PATCH's dose at every scenario, seed and replan, capped only where its own
    # (diverged) ego selects fewer agents. H11 is interpretable only if the realised total dose is >= 99% of PATCH's.
    patch_total = sham_total = short_replans = total_replans = 0
    for seed in range(3):
        for t in REPLANS:
            patch_replan = closed[f"PATCH_s{seed}_dose_t{t}"].to_numpy(int)
            sham_replan = closed[f"SHAM2_s{seed}_dose_t{t}"].to_numpy(int)
            if (patch_replan < 0).any() or (sham_replan > patch_replan).any():
                raise ValueError(f"invalid SHAM2 dose at seed {seed}, replan {t}")
            patch_total += int(patch_replan.sum())
            sham_total += int(sham_replan.sum())
            short_replans += int((sham_replan < patch_replan).sum())
            total_replans += len(patch_replan)
    dose_ratio = sham_total / patch_total if patch_total else float("nan")
    dose_report = {"PATCH_total": patch_total, "SHAM2_total": sham_total, "SHAM2_over_PATCH": dose_ratio,
                   "short_replan_share": short_replans / total_replans, "dose_pass": bool(dose_ratio >= 0.99)}
    all_collision = float(data.ALL_collision.mean())
    static_collision = float(closed.static_collision.mean())
    log_collision = float(closed.log_collision.mean())
    controls = {"STATIC_over_ALL": static_collision / all_collision if all_collision > 0 else None,
                "STATIC_pass": all_collision > 0 and static_collision >= 2 * all_collision,
                "LOG_collision": log_collision, "LOG_pass": log_collision < 0.01,
                "ALL_brake_events": float(data.ALL_brake.sum()),
                "power_pass": data.ALL_brake.sum() >= 100,
                **{f"dose_{k}": v for k, v in dose_report.items()}}
    interpretable = controls["STATIC_pass"] and controls["LOG_pass"]
    powered = controls["power_pass"]
    finite = undefined_share <= 0.01 and all(np.isfinite(x) for x in points["PATCH_reduction"])

    def status(passed: bool) -> str:
        if not powered:
            return "underpowered"
        if not finite:
            return "inconclusive"
        if not interpretable:
            return "uninterpretable"
        return "supported" if passed else "killed"

    claims = {}
    for label, arm in (("H10", "PATCH"), ("H12", "TRIM"), ("H13", "MIX")):
        reduction = aggregated[f"{arm}_reduction"]
        collision = aggregated[f"{arm}_collision_diff"]
        passed = decide_reduction(reduction["point"], reduction["ci"]) and \
            decide_noninferiority(collision["ci"], 0.003)
        if label == "H13":
            passed = passed and decide_noninferiority(aggregated["focal_miss_diff"]["ci"], 0.02)
        claims[label] = {"status": status(passed), "supported": status(passed) == "supported"}
    gap = aggregated["sham_gap"]
    h11 = status(decide_gap(gap["point"], gap["ci"]))
    if h11 in ("supported", "killed") and not controls["dose_dose_pass"]:
        h11 = "uninterpretable (SHAM2 realised dose below 99% of PATCH)"
    claims["H11"] = {"status": h11, "supported": h11 == "supported"}
    return {"ci_level": CI_LEVEL, "descriptive_ci_level": DESCRIPTIVE_CI_LEVEL, "bootstraps": n_boot,
            "n_scenarios": len(data), "controls": controls, "undefined_bootstrap_share": undefined_share,
            "effects": aggregated, "per_city": per_city, "claims": claims,
            "dose": {arm: {"trigger_mean": float(data[f"{arm}_trigger_count"].mean()),
                           "substituted_mean": float(data[f"{arm}_substituted_count"].mean())}
                     for arm in MODEL_ARMS},
            "trim_fallback_mean": float(_seed_mean(closed, "TRIM", "trim_fallback_count").mean())}


def _auc(prob: np.ndarray, contact: np.ndarray) -> float | None:
    """Tie-aware binary AUROC by rank sum."""
    positive = int(contact.sum())
    negative = len(contact) - positive
    if not positive or not negative:
        return None
    ranks = pd.Series(prob).rank(method="average").to_numpy()
    return float((ranks[contact].sum() - positive * (positive + 1) / 2) / (positive * negative))


def summarize_calibration(path: str) -> dict:
    """Descriptive Brier, decile reliability and AUROC for complete four-second futures."""
    groups: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    censored: dict[str, int] = defaultdict(int)
    table = pq.ParquetFile(path)
    columns = ["arm", "stopped", "predicted_hit_probability", "contact", "complete_future"]
    for batch in table.iter_batches(batch_size=100000, columns=columns):
        df = batch.to_pandas()
        for arm, frame in df.groupby("arm"):
            censored[str(arm)] += int((~frame.complete_future.astype(bool)).sum())
            complete = frame[frame.complete_future.astype(bool)]
            for name, subset in (("all", complete), ("stopped", complete[complete.stopped.astype(bool)]),
                                 ("moving", complete[~complete.stopped.astype(bool)])):
                if not subset.empty:
                    p = subset.predicted_hit_probability.to_numpy(float)
                    y = subset.contact.to_numpy(bool)
                    # probabilities are float32 sums, so allow float32 rounding (observed max excess 1.2e-7)
                    if ((p < -1e-6) | (p > 1 + 1e-6)).any():
                        raise ValueError("predicted risk outside [0, 1]")
                    p = np.clip(p, 0.0, 1.0)
                    groups[str(arm), name].append((p, y))
    result = {}
    for (arm, name), parts in groups.items():
        p = np.concatenate([x[0] for x in parts])
        y = np.concatenate([x[1] for x in parts])
        bins = np.minimum((p * 10).astype(int), 9)
        reliability = []
        for bin_id in range(10):
            mask = bins == bin_id
            reliability.append({"lower": bin_id / 10, "upper": (bin_id + 1) / 10,
                                "n": int(mask.sum()), "predicted_mean": float(p[mask].mean()) if mask.any() else None,
                                "contact_rate": float(y[mask].mean()) if mask.any() else None})
        result.setdefault(arm, {})[name] = {"n": len(p), "brier": float(np.mean((p - y) ** 2)),
                                             "contact_rate": float(y.mean()), "auroc": _auc(p, y),
                                             "reliability": reliability}
    return {"by_arm": result, "censored_rows_excluded": dict(censored)}


def _plain(value):
    """JSON fallback for numpy scalars (for example numpy.bool_ from comparisons)."""
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--closedloop", required=True)
    ap.add_argument("--openloop", required=True)
    ap.add_argument("--risk", required=True)
    ap.add_argument("--pool", default="runs/replication_pool.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    args = ap.parse_args()
    ids = checked_ids(args.pool, POOL_SHA256)
    closed = pd.read_parquet(args.closedloop)
    if len(ids) != 8140 or set(closed.scenario_id.astype(str)) != set(ids):
        ap.error("closed-loop input does not match the registered replication pool")
    focal = pd.read_parquet(args.openloop)
    if set(focal.scenario_id.astype(str)) != set(ids):
        ap.error("focal input does not match the registered replication pool")
    result = analyze(closed, focal, args.bootstraps)
    result["calibration"] = summarize_calibration(args.risk)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False, default=_plain)
    print(json.dumps({"claims": result["claims"], "controls": result["controls"]}, indent=2, default=_plain))


if __name__ == "__main__":
    main()
