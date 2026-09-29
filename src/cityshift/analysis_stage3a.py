"""Pre-registered Stage 3a analysis of per-agent parquet rows."""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from .data import CITIES

N_BOOT = 10000
CI = (0.625, 99.375)  # 98.75% interval
KEYS = ["scenario_id", "agent_id"]
METRICS = ["min_ade", "min_fde", "miss", "brier_min_fde"]
SETS = {"focal": "focal", "scored": "scored", "planner_relevant": "planner_relevant"}


def cluster_relative(treatment: np.ndarray, control: np.ndarray, clusters: np.ndarray,
                     rng: np.random.Generator, n_boot: int = N_BOOT) -> tuple[float, np.ndarray]:
    """Paired relative change, resampling whole clusters with their agents."""
    _, inverse = np.unique(clusters, return_inverse=True)
    n = int(inverse.max()) + 1
    t_sum = np.bincount(inverse, weights=treatment, minlength=n)
    c_sum = np.bincount(inverse, weights=control, minlength=n)
    point = float((t_sum.sum() - c_sum.sum()) / c_sum.sum()) if c_sum.sum() else float("nan")
    boots = np.empty(n_boot)
    for start in range(0, n_boot, 128):
        stop = min(start + 128, n_boot)
        draw = rng.integers(0, n, size=(stop - start, n))
        t, c = t_sum[draw].sum(1), c_sum[draw].sum(1)
        boots[start:stop] = np.divide(t - c, c, out=np.full(len(c), np.nan), where=c > 0)
    return point, boots


def _arm(name: str) -> str:
    if re.fullmatch(r"ALL_s[0-2]", name):
        return "ALL"
    if re.fullmatch(r"LOCO-[a-z-]+_s[0-2]", name):
        return "LOCO"
    return name


def analyze(df: pd.DataFrame, n_boot: int = N_BOOT) -> dict:
    """Compute H5, descriptive H1 replication, and stratified metric tables."""
    needed = set(KEYS + ["city", "agent_type", "speed", *SETS.values(), "predictor", *METRICS])
    if not needed <= set(df):
        raise ValueError(f"missing columns: {sorted(needed - set(df))}")
    if df.duplicated(KEYS + ["predictor"]).any():
        raise ValueError("duplicate scenario, agent, predictor rows")
    meta = df.loc[df.predictor == "CV", KEYS + ["city", "agent_type", "speed", *SETS.values()]].set_index(KEYS)
    if meta.empty:
        raise ValueError("CV rows are required as the membership reference")
    data = df.copy()
    data["arm"] = data.predictor.map(_arm)
    loco = data.arm == "LOCO"
    loco_city = data.predictor.str.extract(r"^LOCO-(.*)_s[0-2]$", expand=False)
    data = data.loc[~loco | (loco_city == data.city)]
    counts = data.groupby(KEYS + ["arm"]).size().unstack(fill_value=0).reindex(meta.index)
    for arm, want in (("ALL", 3), ("LOCO", 3), ("CV", 1), ("LANE", 1), ("STATIC", 1)):
        if arm not in counts or not (counts[arm] == want).all():
            raise ValueError(f"every agent requires {want} {arm} row(s)")
    means = data.groupby(KEYS + ["arm"])[METRICS].mean().unstack("arm").reindex(meta.index)
    if not np.isfinite(means.to_numpy()).all():
        raise ValueError("missing or nonfinite per-agent outcomes")
    rng = np.random.default_rng(0)
    planner = meta.planner_relevant.to_numpy(bool)
    cities = meta.city.to_numpy()
    ids = meta.index.get_level_values("scenario_id").to_numpy()
    miss = means["miss"]
    primary, replication, primary_boots, replication_boots, city_rows = [], [], [], [], []
    for city in CITIES:
        mask = planner & (cities == city)
        if not mask.any():
            raise ValueError(f"no planner-relevant agents in {city}")
        lane, all_m, loco = (miss[arm].to_numpy()[mask] for arm in ("LANE", "ALL", "LOCO"))
        # cluster_relative is treatment relative to control; negative is better for H5.
        change, boots = cluster_relative(all_m, lane, ids[mask], rng, n_boot)
        h1, h1_boots = cluster_relative(loco, all_m, ids[mask], rng, n_boot)
        primary.append(-change)
        primary_boots.append(-boots)
        replication.append(h1)
        replication_boots.append(h1_boots)
        city_rows.append({"city": city, "agents": int(mask.sum()), "scenarios": int(np.unique(ids[mask]).size),
                          "miss_LANE": float(lane.mean()), "miss_ALL": float(all_m.mean()),
                          "miss_LOCO": float(loco.mean()), "reduction_ALL_vs_LANE": -change,
                          "relative_LOCO_vs_ALL": h1})
    p_boot = np.mean(primary_boots, axis=0)
    h_boot = np.mean(replication_boots, axis=0)
    pc_mask = planner & (meta.speed.to_numpy(float) > 2)
    cv_rate = float(miss.CV.to_numpy()[pc_mask].mean())
    static_rate = float(miss.STATIC.to_numpy()[pc_mask].mean())
    pc_pass = bool(cv_rate > 0 and static_rate >= 1.1 * cv_rate)
    lane_misses = float(miss.LANE.to_numpy()[planner].sum())
    defined = bool(np.isfinite(primary).all() and np.isfinite(p_boot).mean() >= 0.99)
    ci = np.nanpercentile(p_boot, CI).tolist() if defined else [float("nan")] * 2
    point = float(np.mean(primary)) if defined else float("nan")
    if not defined or lane_misses < 100:
        decision = "inconclusive"
    elif ci[0] > 0 and point >= 0.10 and pc_pass:
        decision = "supported"
    else:
        decision = "killed" if pc_pass else "uninterpretable (positive control failed)"
    result = {
        "H5": {"pooled_reduction": point, "ci_98_75": ci, "decision": decision,
               "lane_misses": lane_misses, "bootstrap_defined_share": float(np.isfinite(p_boot).mean())},
        "H1_planner_descriptive": {"pooled_rel_change": float(np.mean(replication)),
                                   "ci_98_75": np.nanpercentile(h_boot, CI).tolist(), "inferential": False},
        "positive_control": {"cv_miss_moving": cv_rate, "static_miss_moving": static_rate, "pass": pc_pass},
        "per_city": city_rows,
    }
    flat = means.stack("arm", future_stack=True).reset_index().merge(meta.reset_index(), on=KEYS, validate="many_to_one")
    tables = []
    for set_name, flag in SETS.items():
        subset = flat.loc[flat[flag]]
        for by in (("arm",), ("city", "arm"), ("agent_type", "arm"), ("city", "agent_type", "arm")):
            grouped = subset.groupby(list(by), as_index=False)[METRICS].agg("mean")
            counts_group = subset.groupby(list(by)).size().rename("n_agents").reset_index()
            grouped = grouped.merge(counts_group, on=list(by))
            for row in grouped.to_dict("records"):
                tables.append({"set": set_name, "stratum": "+".join(by), **row})
    result["descriptive_tables"] = tables
    result.update(h6(meta, miss, ids, cities, rng, n_boot))
    return result


def cluster_difference(treatment: np.ndarray, control: np.ndarray, clusters: np.ndarray,
                       rng: np.random.Generator, n_boot: int = N_BOOT) -> tuple[float, np.ndarray]:
    """Paired difference in means (treatment minus control), resampling whole clusters."""
    _, inverse = np.unique(clusters, return_inverse=True)
    n = int(inverse.max()) + 1
    d_sum = np.bincount(inverse, weights=treatment - control, minlength=n)
    cnt = np.bincount(inverse, minlength=n).astype(float)
    point = float(d_sum.sum() / cnt.sum())
    boots = np.empty(n_boot)
    for start in range(0, n_boot, 128):
        stop = min(start + 128, n_boot)
        draw = rng.integers(0, n, size=(stop - start, n))
        boots[start:stop] = d_sum[draw].sum(1) / cnt[draw].sum(1)
    return point, boots


def h6(meta: pd.DataFrame, miss: pd.DataFrame, ids: np.ndarray, cities: np.ndarray,
       rng: np.random.Generator, n_boot: int = N_BOOT) -> dict:
    """H6a/H6b (registration amendment): ALL vs CV on stopped vs moving non-focal planner agents."""
    speed = meta.speed.to_numpy(float)
    base = meta.planner_relevant.to_numpy(bool) & ~meta.focal.to_numpy(bool)
    out = {}
    for key, sel, bar in (("H6a", base & (speed < 0.5), 0.15), ("H6b", base & (speed >= 2.0), 0.30)):
        points, boots, rows = [], [], []
        for city in CITIES:
            mask = sel & (cities == city)
            if not mask.any():
                continue
            all_m, cv = miss["ALL"].to_numpy()[mask], miss["CV"].to_numpy()[mask]
            if key == "H6a":  # absolute miss-rate difference ALL - CV (the ratio is fragile when CV rarely misses)
                point, b = cluster_difference(all_m, cv, ids[mask], rng, n_boot)
            else:  # relative reduction of ALL vs CV
                change, b = cluster_relative(all_m, cv, ids[mask], rng, n_boot)
                point, b = -change, -b
            points.append(point)
            boots.append(b)
            rows.append({"city": city, "agents": int(mask.sum()), "miss_ALL": float(all_m.mean()), "miss_CV": float(cv.mean()),
                         "ratio_ALL_over_CV_descriptive": float(all_m.mean() / cv.mean()) if cv.mean() > 0 else float("nan")})
        pts, bt = np.array(points), np.array(boots)
        defined = len(pts) == len(CITIES) and np.isfinite(pts).all() and np.isfinite(bt.mean(0)).mean() >= 0.99
        pooled = float(pts.mean()) if defined else float("nan")
        ci = np.nanpercentile(bt.mean(0), CI).tolist() if defined else [float("nan")] * 2
        null = 0.0
        if not defined:
            decision = "inconclusive"
        elif ci[0] > null and pooled >= bar:
            decision = "supported"
        else:
            decision = "killed"
        name = "difference_ALL_minus_CV" if key == "H6a" else "reduction_ALL_vs_CV"
        out[key] = {f"pooled_{name}": pooled, "ci_98_75": ci, "bar": bar, "decision": decision, "per_city": rows}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--development", action="store_true", help="label train-dev output as descriptive only")
    args = ap.parse_args()
    result = analyze(pd.read_parquet(args.parquet))
    if args.development:
        for key in ("H5", "H6a", "H6b"):
            result[key]["decision"] = "development only"
            result[key]["inferential"] = False
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "descriptive_tables"}, indent=2))


if __name__ == "__main__":
    main()
