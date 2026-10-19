"""v2 analyses for studies R, G (docs/preregistration/v2-fresh-reserve.md) and Q (docs/preregistration/v2-qcnet.md).

Shared estimator, as registered for Stage 4 and study E: per scenario, average model seeds; within each city compute
the effect; equal city weights; bootstrap scenarios with replacement within city (arms and seeds paired), 10,000 draws,
generator seed 20261002. Ratios of pooled effects (study G) are formed per draw from the city-equal pooled values.

    PYTHONPATH=src python -m cityshift.analysis_v2 r --parts runs/v2/study_r_v4 --harness v4 --out reports/v2/study_r_v4.json
    PYTHONPATH=src python -m cityshift.analysis_v2 q-open --rows runs/v2/q_open --stage3a evals/stage3a/per_agent.parquet
    PYTHONPATH=src python -m cityshift.analysis_v2 q-closed --parts runs/v2/q_closed
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from .analysis_stage3 import CITIES
from .closedloop import REPLANS
from .v2_validation import AGENT_KEYS, agent_cohort, matched_scenarios, registered_cohort, scenario_cohort

SEED, N_BOOT = 20261002, 10_000


def interval(draws: np.ndarray, level: float) -> list[float | None]:
    ok = draws[np.isfinite(draws)]
    if not len(ok):
        return [None, None]
    return [float(x) for x in np.quantile(ok, [(1 - level) / 2, 1 - (1 - level) / 2])]


def load_parts(directory: str, expected_ids: list[str] | None = None) -> pd.DataFrame:
    parts = sorted(glob.glob(os.path.join(directory, "part_*.parquet")))
    if not parts:
        raise SystemExit(f"no parts in {directory}")
    d = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    scenario_cohort(d, expected_ids)
    return d


def arm_column(d: pd.DataFrame, arm: str, metric: str) -> np.ndarray:
    """Seed-averaged per-scenario outcome (or the single run for unseeded arms)."""
    seeded = [f"{arm}_s{s}_{metric}" for s in range(3)]
    if any(c in d for c in seeded):
        if not set(seeded) <= set(d):
            raise ValueError(f"all three seeds are required for {arm}/{metric}")
        values = d[seeded].to_numpy(float)
    else:
        values = d[[f"{arm}_{metric}"]].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite {arm}/{metric}")
    if metric in ("collision", "unnecessary_hard_brake") and not np.isin(values, [0, 1]).all():
        raise ValueError(f"non-binary {arm}/{metric}")
    return values.mean(1)


def city_bootstrap(data: pd.DataFrame, arms: list[str], effects, n_boot: int = N_BOOT, seed: int = SEED,
                   derived=None) -> dict[str, dict]:
    """City-equal effects with paired within-city scenario bootstrap; ``derived`` maps pooled effects to more."""
    rng = np.random.default_rng(seed)
    points, draws = {}, {}
    for city in CITIES:
        m = data.loc[data.city.astype(str) == city, arms].to_numpy(float)
        if not len(m):
            raise SystemExit(f"missing city {city}")
        with np.errstate(divide="ignore", invalid="ignore"):
            for k, v in effects(dict(zip(arms, m.mean(0)))).items():
                points.setdefault(k, []).append(float(v))
            idx = rng.integers(0, len(m), size=(n_boot, len(m)))
            for k, v in effects(dict(zip(arms, m[idx].mean(1).T))).items():
                draws.setdefault(k, []).append(v)
    pooled_point = {k: float(np.mean(v)) for k, v in points.items()}
    pooled_draws = {k: np.mean(np.stack(v), 0) for k, v in draws.items()}
    if derived:
        with np.errstate(divide="ignore", invalid="ignore"):
            pooled_point |= {k: float(v) for k, v in derived(pooled_point).items()}
            pooled_draws |= derived(pooled_draws)
    return {k: {"point": pooled_point[k], "draws": pooled_draws[k]} for k in pooled_point}


def summarise(e: dict, levels: dict[str, float]) -> dict:
    out = {"point": e["point"], "undefined_share": float(np.mean(~np.isfinite(e["draws"])))}
    out |= {name: interval(e["draws"], level) for name, level in levels.items()}
    return out


def json_values(value):
    """Represent undefined estimates as JSON null, including nested result cells."""
    if isinstance(value, dict):
        return {k: json_values(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_values(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def decide_bar(e: dict, key: str, bar: float, powered: bool, interpretable: bool) -> str:
    if not interpretable:
        return "uninterpretable"
    lo = e[key][0]
    if not powered or e["undefined_share"] > 0.01 or lo is None:
        return "inconclusive (A8)"
    return "supported" if e["point"] >= bar and lo > 0 else "killed"


def validate_dose(d: pd.DataFrame, patch: str, sham: str, seeded: bool = True) -> float:
    total = {patch: 0, sham: 0}
    for s in (range(3) if seeded else [None]):
        p = np.stack(d[f"{patch}_s{s}_dose_by_replan" if seeded else f"{patch}_dose_by_replan"].to_numpy())
        q = np.stack(d[f"{sham}_s{s}_dose_by_replan" if seeded else f"{sham}_dose_by_replan"].to_numpy())
        if p.shape[1] != len(REPLANS) or (q > p).any() or (q < 0).any():
            raise SystemExit("invalid sham dose: must be nonnegative and never above the treatment's at any replan")
        total[patch] += int(p.sum())
        total[sham] += int(q.sum())
    return total[sham] / total[patch] if total[patch] else float("nan")


def powered(data: pd.DataFrame, control: str) -> tuple[bool, float]:
    events = float(data[control].sum())
    ok = events >= 100 and all(data.loc[data.city.astype(str) == c, control].sum() > 0 for c in CITIES)
    return ok, events


# ------------------------------------------------------------------ studies R and G
R_ARMS = ["ALL", "PATCH", "SHAM2", "TRIM", "GEO", "PROB"]


def r_effects(m: dict) -> dict:
    red = {f"{a}_reduction": (m["ALL"] - m[a]) / m["ALL"] for a in R_ARMS[1:]}
    return red | {"sham_gap": (m["SHAM2"] - m["PATCH"]) / m["ALL"]}


def r_derived(p: dict) -> dict:
    return {"G_ratio_PROB_over_PATCH": np.divide(p["PROB_reduction"], p["PATCH_reduction"]),
            "interaction": p["PATCH_reduction"] - p["PROB_reduction"] - p["GEO_reduction"]}


def analyze_r(d: pd.DataFrame, harness: str, n_boot: int = N_BOOT) -> dict:
    scenario_cohort(d)
    primary = harness == "v4"
    levels = {"ci9875": 1 - 0.05 / 4, "ci95": 0.95} if primary else {"ci95": 0.95}
    decision_key = "ci9875" if primary else "ci95"
    data = pd.DataFrame({"city": d.city, **{a: arm_column(d, a, "unnecessary_hard_brake") for a in R_ARMS}})
    eff = {k: summarise(v, levels) for k, v in city_bootstrap(data, R_ARMS, r_effects, n_boot,
                                                               derived=r_derived).items()}
    rates = {a: float(data[a].mean()) for a in R_ARMS} | {
        a: float(d[f"{a}_unnecessary_hard_brake"].astype(float).mean()) for a in ("cv", "oracle", "static")}
    collisions = {a: float(arm_column(d, a, "collision").mean()) for a in R_ARMS} | {
        a: float(d[f"{a}_collision"].astype(float).mean()) for a in ("cv", "oracle", "static", "log")}
    control = rates["ALL"] >= 2 * rates["oracle"]
    dose_ratio = validate_dose(d, "PATCH", "SHAM2")
    is_powered, events = powered(data, "ALL")
    g = eff["G_ratio_PROB_over_PATCH"]
    lo, hi = g[decision_key]
    if not control:
        g_verdict = "uninterpretable"
    elif not is_powered or g["undefined_share"] > 0.01 or lo is None:
        g_verdict = "inconclusive (A8)"
    else:
        g_verdict = ("probability sufficient" if lo >= 0.5 else "geometry needed" if hi < 0.5 else "inconclusive")
    verdicts = {"R1 PATCH": decide_bar(eff["PATCH_reduction"], decision_key, 0.30, is_powered, control),
                "R2 sham gap": decide_bar(eff["sham_gap"], decision_key, 0.20, is_powered,
                                          control and dose_ratio >= 0.99),
                "R3 TRIM": decide_bar(eff["TRIM_reduction"], decision_key, 0.30, is_powered, control),
                "G1 PROB/PATCH": g_verdict}
    out = {"harness": harness, "role": "primary" if primary else "secondary", "decision_level": decision_key,
           "scenarios": len(d), "verdicts": verdicts, "effects": eff, "brake_rates": rates,
           "brake_control_ALL_over_oracle": rates["ALL"] / rates["oracle"] if rates["oracle"] else None,
           "brake_control_pass": bool(control), "sham_dose_ratio": dose_ratio, "seed_averaged_ALL_brake_events": events,
           "powered": is_powered, "collisions": collisions,
           "log_collision_calibration_pass": bool(collisions["log"] < 0.01),
           "trim_fallback_share": float(sum(d[f"TRIM_s{s}_trim_fallback_count"].sum() for s in range(3))
                                        / max(1, sum(d[f"TRIM_s{s}_trigger_count"].sum() for s in range(3)))),
           "factorial_dose": {a: int(sum(d[f"{a}_s{s}_substituted_count"].sum() for s in range(3)))
                              for a in ("GEO", "PROB")} | {"PATCH": int(sum(d[f"PATCH_s{s}_trigger_count"].sum()
                                                                            for s in range(3)))},
           "per_seed_reduction": {f"{a}_s{s}": float(1 - d[f"{a}_s{s}_unnecessary_hard_brake"].astype(float).mean()
                                                     / d[f"ALL_s{s}_unnecessary_hard_brake"].astype(float).mean())
                                  for a in R_ARMS[1:] for s in range(3)}}
    if not primary:  # full-drive harness: collision non-inferiority, gated on the STATIC control
        cdata = pd.DataFrame({"city": d.city, **{a: arm_column(d, a, "collision") for a in R_ARMS}})
        cdiff = city_bootstrap(cdata, R_ARMS, lambda m: {f"{a}_collision_diff": m[a] - m["ALL"] for a in R_ARMS[1:]},
                               n_boot)
        out["collision_effects"] = {k: summarise(v, levels) for k, v in cdiff.items()}
        out["static_over_ALL_collisions"] = collisions["static"] / collisions["ALL"] if collisions["ALL"] else None
        collision_control = collisions["ALL"] > 0 and collisions["static"] >= 2 * collisions["ALL"]
        out["collision_control_pass"] = bool(collision_control)
        for effect in out["collision_effects"].values():
            effect["margin"] = 0.003
            effect["status"] = ("uninterpretable" if not collision_control else
                                "within harness margin" if effect["ci95"][1] < 0.003 else "not established")
        out["collision_scope"] = "secondary harness only; does not establish collision safety"
    for e in out["effects"].values():
        e.pop("draws", None)
    return out


# ------------------------------------------------------------------ study Q
def q_open(rows: pd.DataFrame, stage3a: pd.DataFrame, n_boot: int = N_BOOT, native_only: bool = False) -> dict:
    """QCNet versus CV-6 and ALL on the Stage 3a agent sets (joined agent for agent)."""
    q = rows[rows.predictor == "QCNET"]
    focal = q[q.focal]
    base = stage3a[stage3a.predictor.isin(["CV", "ALL_s0", "ALL_s1", "ALL_s2"])]
    agent_cohort(base, ["CV", "ALL_s0", "ALL_s1", "ALL_s2"])
    if q.duplicated(AGENT_KEYS).any():
        raise ValueError("duplicate QCNet agent rows")
    reference = base[base.predictor == "CV"]
    expected = pd.MultiIndex.from_frame(reference[AGENT_KEYS])
    actual = pd.MultiIndex.from_frame(q[AGENT_KEYS])
    if len(actual.difference(expected)) or (not native_only and len(expected.difference(actual))):
        raise ValueError("QCNet cohort differs from registered agents; use documented native-only analysis or full fallback")
    if set(focal.scenario_id) != set(reference[reference.focal].scenario_id):
        raise ValueError("QCNet focal positive-control cohort is incomplete")
    matched_reference = reference.merge(q[AGENT_KEYS], on=AGENT_KEYS, validate="one_to_one")
    agent_cohort(pd.concat([q, matched_reference], ignore_index=True), ["QCNET", "CV"])
    base = base.assign(pred=np.where(base.predictor == "CV", "CV", "ALL"))
    keys = ["scenario_id", "agent_id"]
    ref = base.groupby(keys + ["pred"]).miss.mean().unstack("pred").reset_index()
    joined = q.merge(ref, on=keys, how="inner", validate="one_to_one")
    out = {"focal": {"n": int(len(focal)), "minFDE6": float(focal.min_fde.mean()), "minADE6": float(focal.min_ade.mean()),
                     "MR6": float(focal.miss.mean()), "brier_minFDE6": float(focal.brier_min_fde.mean())},
           "joined_agents": int(len(joined)), "qcnet_rows": int(len(q)),
           "cohort": {"reference_agents": len(reference), "omitted_agents": len(expected.difference(actual)),
                      "native_only": native_only, "fallback_agents": int(q.get("qcnet_missing", pd.Series(False, index=q.index)).sum())}}
    f = out["focal"]
    out["Q0_positive_control"] = {"pass": bool(len(focal) == 24988 and focal.scenario_id.nunique() == 24988
                                              and f["minFDE6"] <= 1.25 * 1.05 and abs(f["MR6"] - 0.16) <= 0.015),
                                  "published_val": {"minFDE6": 1.25, "MR6": 0.16, "minADE6": 0.72, "brier": 1.87}}
    rng = np.random.default_rng(SEED)
    groups = {"stopped": (joined.speed < 0.5), "moving": (joined.speed >= 2.0)}
    for name, gmask in groups.items():
        sel = joined[gmask & joined.planner_relevant & ~joined.focal]
        for subset, smask in (("all", np.ones(len(sel), bool)), ("complete_future", sel.n_valid.to_numpy() >= 60)):
            s = sel[smask]
            cell = {}
            for other in ("CV", "ALL"):
                city_pts, city_draws = [], []
                for c in CITIES:
                    sc = s[s.city == c]
                    if not len(sc):
                        raise SystemExit(f"missing city {c} in {name}/{subset}")
                    g = sc.groupby("scenario_id")[["miss", other]].agg(["sum", "count"])
                    qs, qn = g[("miss", "sum")].to_numpy(), g[("miss", "count")].to_numpy()
                    os_, on = g[(other, "sum")].to_numpy(), g[(other, "count")].to_numpy()
                    city_pts.append(qs.sum() / qn.sum() - os_.sum() / on.sum())
                    idx = rng.integers(0, len(qs), size=(n_boot, len(qs)))
                    city_draws.append(qs[idx].sum(1) / qn[idx].sum(1) - os_[idx].sum(1) / on[idx].sum(1))
                draws = np.mean(np.stack(city_draws), 0)
                cell[f"QCNET_minus_{other}"] = {"point": float(np.mean(city_pts)), "ci9833": interval(draws, 1 - 0.05 / 3),
                                                "ci95": interval(draws, 0.95)}
            cell["miss_rates_city_equal"] = {p: float(np.mean([s.loc[s.city == c, col].mean() for c in CITIES]))
                                             for p, col in (("QCNET", "miss"), ("CV", "CV"), ("ALL", "ALL"))}
            cell["agents"] = int(len(s))
            if name == "stopped":
                cell["mean_moving_mode_mass"] = float(s.moving_mode_mass.mean())
                cell["share_moving_mass_over_5pct"] = float((s.moving_mode_mass > 0.05).mean())
            out[f"{name}_{subset}"] = cell
    q1 = out["stopped_all"]["QCNET_minus_CV"]
    if not out["Q0_positive_control"]["pass"]:
        out["Q1_verdict"] = "uninterpretable (Q0 failed)"
    else:
        out["Q1_verdict"] = "supported" if q1["point"] >= 0.15 and q1["ci9833"][0] > 0 else "killed"
        if q1["ci9833"][1] < 0:
            out["Q1_verdict"] += "; QCNet better than CV-6 on stopped agents"
    return out


Q_ARMS = ["QBASE", "QPATCH", "QSHAM2"]


def q_closed(d: pd.DataFrame, q0_pass: bool, n_boot: int = N_BOOT) -> dict:
    scenario_cohort(d)
    levels = {"ci9833": 1 - 0.05 / 3, "ci95": 0.95}
    data = pd.DataFrame({"city": d.city, **{a: d[f"{a}_unnecessary_hard_brake"].to_numpy(float) for a in Q_ARMS},
                         "ALL": arm_column(d, "ALL", "unnecessary_hard_brake")})
    eff = city_bootstrap(data, Q_ARMS + ["ALL"], lambda m: {
        "QPATCH_reduction": (m["QBASE"] - m["QPATCH"]) / m["QBASE"],
        "QSHAM2_reduction": (m["QBASE"] - m["QSHAM2"]) / m["QBASE"],
        "sham_gap": (m["QSHAM2"] - m["QPATCH"]) / m["QBASE"],
        "QBASE_vs_ALL_relative": (m["QBASE"] - m["ALL"]) / m["ALL"]}, n_boot)
    eff = {k: summarise(v, levels) for k, v in eff.items()}
    rates = {a: float(data[a].mean()) for a in Q_ARMS + ["ALL"]} | {
        a: float(d[f"{a}_unnecessary_hard_brake"].astype(float).mean()) for a in ("cv", "oracle", "static")}
    control = rates["QBASE"] >= 2 * rates["oracle"]
    dose_ratio = validate_dose(d, "QPATCH", "QSHAM2", seeded=False)
    is_powered, events = powered(data, "QBASE")
    interpretable = control and q0_pass
    verdicts = {"Q2 QPATCH": decide_bar(eff["QPATCH_reduction"], "ci9833", 0.30, is_powered, interpretable),
                "Q3 sham gap": decide_bar(eff["sham_gap"], "ci9833", 0.20, is_powered,
                                          interpretable and dose_ratio >= 0.99)}
    for e in eff.values():
        e.pop("draws", None)
    return {"scenarios": len(d), "verdicts": verdicts, "effects": eff, "brake_rates": rates,
            "brake_control_QBASE_over_oracle": rates["QBASE"] / rates["oracle"] if rates["oracle"] else None,
            "brake_control_pass": bool(control), "q0_pass": q0_pass, "sham_dose_ratio": dose_ratio,
            "QBASE_brake_events": events, "powered": is_powered,
            "qcnet_missing_counts": {a: int(d[f"{a}_qcnet_missing_count"].sum()) for a in Q_ARMS},
            "collisions_descriptive": {a: float(d[f"{a}_collision"].astype(float).mean()) for a in Q_ARMS}
            | {"ALL": float(arm_column(d, "ALL", "collision").mean())}
            | {a: float(d[f"{a}_collision"].astype(float).mean()) for a in ("cv", "oracle", "static", "log")}}


# ------------------------------------------------------------------ study S
S_FAMILIES = ("ALLR", "SEL", "SELSHAM")


def s_open(rows: pd.DataFrame, n_boot: int = N_BOOT) -> dict:
    """Stopped non-focal planner agents and focal agents on the reserve: SEL versus ALLR and SELSHAM (Stage 3a scoring)."""
    agent_cohort(rows, [f"{f}_s{s}" for f in S_FAMILIES for s in range(3)] + ["CV"])
    rows = rows.assign(family=rows.predictor.str.replace(r"_s[0-2]$", "", regex=True))
    keys = ["scenario_id", "agent_id", "city", "focal", "planner_relevant", "speed", "n_valid"]
    wide = rows.groupby(keys + ["family"]).miss.mean().unstack("family").reset_index()
    out = {}
    rng = np.random.default_rng(SEED)
    cells = {"stopped_nonfocal": wide.planner_relevant & ~wide.focal & (wide.speed < 0.5),
             "moving_nonfocal": wide.planner_relevant & ~wide.focal & (wide.speed >= 2.0),
             "focal": wide.focal, "focal_stopped": wide.focal & (wide.speed < 0.5)}
    for name, mask in cells.items():
        sel = wide[mask]
        cell = {"agents": int(len(sel)),
                "miss_city_equal": {f: float(np.mean([sel.loc[sel.city == c, f].mean() for c in CITIES]))
                                    for f in S_FAMILIES + ("CV",)}}
        for a, b in (("SEL", "ALLR"), ("SEL", "SELSHAM"), ("SELSHAM", "ALLR")):
            pts, dr = [], []
            for c in CITIES:
                sc = sel[sel.city == c]
                g = sc.groupby("scenario_id")[[a, b]].agg(["sum", "count"])
                xa, na = g[(a, "sum")].to_numpy(), g[(a, "count")].to_numpy()
                xb, nb = g[(b, "sum")].to_numpy(), g[(b, "count")].to_numpy()
                if not len(xa):
                    pts.append(float("nan"))
                    dr.append(np.full(n_boot, np.nan))
                    continue
                pts.append(xa.sum() / na.sum() - xb.sum() / nb.sum())
                idx = rng.integers(0, len(xa), size=(n_boot, len(xa)))
                dr.append(xa[idx].sum(1) / na[idx].sum(1) - xb[idx].sum(1) / nb[idx].sum(1))
            draws = np.mean(np.stack(dr), 0)
            cell[f"{a}_minus_{b}"] = {"point": float(np.mean(pts)), "ci9833": interval(draws, 1 - 0.05 / 3),
                                     "ci95": interval(draws, 0.95),
                                     "undefined_share": float(np.mean(~np.isfinite(draws)))}
        out[name] = cell
    st = out["stopped_nonfocal"]
    for claim, key in (("S1 SEL vs ALLR", "SEL_minus_ALLR"), ("S2 SEL vs SELSHAM", "SEL_minus_SELSHAM")):
        e = st[key]
        out.setdefault("verdicts", {})[claim] = ("inconclusive (missing city/group)" if e["ci9833"][1] is None else
                                                "supported" if e["point"] <= -0.15 and e["ci9833"][1] < 0 else "killed")
    return out


def s_closed(allr: pd.DataFrame, sel: pd.DataFrame, sham: pd.DataFrame, n_boot: int = N_BOOT) -> dict:
    """Stop-line braking on the reserve: SEL (and SELSHAM) focal-only models versus ALLR, paired by scenario."""
    matched_scenarios(allr, sel)
    matched_scenarios(allr, sham)
    m = allr[["scenario_id", "city"]].copy()
    m["ALLR"] = arm_column(allr, "ALL", "unnecessary_hard_brake")
    for name, d in (("SEL", sel), ("SELSHAM", sham)):
        x = pd.DataFrame({"scenario_id": d.scenario_id, name: arm_column(d, "ALL", "unnecessary_hard_brake")})
        m = m.merge(x, on="scenario_id", validate="one_to_one")
    eff = city_bootstrap(m, ["ALLR", "SEL", "SELSHAM"], lambda r: {
        "SEL_reduction": (r["ALLR"] - r["SEL"]) / r["ALLR"],
        "SELSHAM_reduction": (r["ALLR"] - r["SELSHAM"]) / r["ALLR"],
        "SEL_minus_SELSHAM": (r["SELSHAM"] - r["SEL"]) / r["ALLR"]}, n_boot)
    eff = {k: summarise(v, {"ci9833": 1 - 0.05 / 3, "ci95": 0.95}) for k, v in eff.items()}
    rates = {f: float(m[f].mean()) for f in ("ALLR", "SEL", "SELSHAM")}
    oracle = float(allr["oracle_unnecessary_hard_brake"].astype(float).mean())
    control = rates["ALLR"] >= 2 * oracle
    is_powered, events = powered(m, "ALLR")
    for e in eff.values():
        e.pop("draws", None)
    return {"scenarios": len(m), "brake_rates": rates | {"oracle": oracle}, "brake_control_pass": bool(control),
            "powered": is_powered, "ALLR_brake_events": events, "effects": eff,
            "verdicts": {"S3 SEL braking": decide_bar(eff["SEL_reduction"], "ci9833", 0.30, is_powered, control)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("r")
    r.add_argument("--parts", required=True)
    r.add_argument("--harness", required=True, choices=("v4", "v2"))
    r.add_argument("--out", required=True)
    qo = sub.add_parser("q-open")
    qo.add_argument("--rows", required=True)
    qo.add_argument("--stage3a", default="evals/stage3a/per_agent.parquet")
    qo.add_argument("--out", default="reports/v2/study_q_open_corrected.json")
    qo.add_argument("--fallback", help="single-CV fallback rows for omitted native forecasts")
    qo.add_argument("--native-only", action="store_true", help="explicit historical-cohort sensitivity")
    qc = sub.add_parser("q-closed")
    qc.add_argument("--parts", required=True)
    qc.add_argument("--q-open", default="reports/v2/study_q_open_corrected.json")
    qc.add_argument("--out", default="reports/v2/study_q_closed.json")
    so = sub.add_parser("s-open")
    so.add_argument("--rows", required=True)
    so.add_argument("--out", default="reports/v2/study_s_open.json")
    sc = sub.add_parser("s-closed")
    sc.add_argument("--allr", default="runs/v2/study_r_v4")
    sc.add_argument("--sel", default="runs/v2/study_s_SEL_v4")
    sc.add_argument("--sham", default="runs/v2/study_s_SELSHAM_v4")
    sc.add_argument("--out", default="reports/v2/study_s_closed.json")
    args = ap.parse_args()
    if args.cmd == "s-open":
        paths = sorted(glob.glob(os.path.join(args.rows, "part_*.parquet"))) if os.path.isdir(args.rows) else [args.rows]
        rows = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        registered_cohort(rows, "reserve")
        result = s_open(rows)
    elif args.cmd == "s-closed":
        allr, sel, sham = load_parts(args.allr), load_parts(args.sel), load_parts(args.sham)
        for rows in (allr, sel, sham):
            registered_cohort(rows, "reserve")
        result = s_closed(allr, sel, sham)
    elif args.cmd == "r":
        rows = load_parts(args.parts)
        registered_cohort(rows, "reserve")
        result = analyze_r(rows, args.harness)
    elif args.cmd == "q-open":
        # per-agent rows (several per scenario), so not load_parts, which requires one row per scenario
        paths = sorted(glob.glob(os.path.join(args.rows, "part_*.parquet"))) if os.path.isdir(args.rows) else [args.rows]
        rows = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        if rows.duplicated(["scenario_id", "agent_id", "predictor"]).any():
            raise SystemExit("duplicate agent rows")
        if args.fallback:
            rows["qcnet_missing"] = False
            rows = pd.concat([rows, pd.read_parquet(args.fallback)], ignore_index=True)
        result = q_open(rows, pd.read_parquet(args.stage3a), native_only=args.native_only)
    else:
        with open(args.q_open) as f:
            q0 = json.load(f)["Q0_positive_control"]["pass"]
        rows = load_parts(args.parts)
        registered_cohort(rows, "q_closed")
        result = q_closed(rows, q0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(json_values(result), f, indent=2, default=float, allow_nan=False)
        f.write("\n")
    print(json.dumps(result.get("verdicts", {k: v for k, v in result.items() if k.endswith("verdict")}), indent=2))


if __name__ == "__main__":
    main()
