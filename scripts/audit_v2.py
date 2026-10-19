"""Independently audit v2 R/G, Q and S decisions from released per-row tables.

Imports nothing from cityshift. Requires NumPy, pandas and a parquet engine. Run
from an unpacked release, or set --root and the configurable input paths::

    python scripts/audit_v2.py --study r-v4
    python scripts/audit_v2.py --study q-open q-native q-closed --draws 10000
    python scripts/audit_v2.py --out /tmp/v2-audit.json

The bootstrap uses a fresh seed, resamples whole scenarios within each city, and
pairs all agents, arms and fixed model seeds. Memory scales with --chunk rather
than --draws. Points must agree within 1e-9; interval endpoints must agree within
15% of the reported interval width (configurable). Independent verdicts must also
agree. A decision disagreement near a bound fails: increase --draws to investigate
Monte Carlo uncertainty rather than treating endpoint tolerance as a verdict pass.

This checks released outcomes, cohorts and inference, not raw forecast correctness,
training execution, checkpoint provenance or physical driving safety. The excess
brake check starts from released flags; it does not independently replay the logs.
H/F are explicitly outside this independent rederivation's scope.
Missing requested inputs fail; selecting a subset does not certify the full v2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

CITIES = ("austin", "dearborn", "miami", "palo-alto", "pittsburgh", "washington-dc")
RESERVE = (10000, "286a9423ffd907eebf7d67a6c7f66b2edfc6d1c4309e2a7f03e2633f8aac52ed")
VALIDATION = (24988, "58b1110a8ffa2eead56e52f583a5655988b0a5e6c97031d4b3c750bc6e1fa625")
Q_SUBSET = (6000, "b35794e0ed5c14b08f5420e800bbc29ec7addfeea5fe6a349059b73d1b495084")
RESERVE_CITIES = dict(zip(CITIES, (2218, 1203, 2644, 612, 2103, 1220)))
KEYS = ["scenario_id", "agent_id"]
META = ["city", "focal", "scored", "planner_relevant", "n_valid", "speed"]
R_ARMS = ("ALL", "PATCH", "SHAM2", "TRIM", "GEO", "PROB")
Q_ARMS = ("QBASE", "QPATCH", "QSHAM2")
FAMILIES = ("ALLR", "SEL", "SELSHAM")
STUDIES = ("r-v4", "r-v2", "q-open", "q-native", "q-closed", "s-open", "s-closed", "excess-v4", "excess-v2")
# Fixed from the preregistrations, never inferred from a reported interval.
LEVEL_R = {"ci9875": 1 - .05 / 4, "ci95": .95}
LEVEL_QS = {"ci9833": 1 - .05 / 3, "ci95": .95}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(ids):
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def identities(ids, registered, label):
    ids = list(map(str, ids))
    require(len(ids) == len(set(ids)), f"{label}: duplicate scenario IDs")
    require((len(ids), digest(ids)) == registered, f"{label}: incomplete or unexpected registered cohort")
    return sorted(ids)


def table(path):
    path = Path(path)
    files = sorted(path.glob("part_*.parquet")) if path.is_dir() else [path]
    require(bool(files), f"no parquet parts in {path}")
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def scenario_rows(rows, expected, label, reference=None):
    require({"scenario_id", "city"} <= set(rows), f"{label}: missing identity columns")
    require(not rows[["scenario_id", "city"]].isna().any().any(), f"{label}: missing identity values")
    identities(rows.scenario_id, expected, label)
    require(set(rows.city) == set(CITIES), f"{label}: unexpected or missing cities")
    if expected == RESERVE:
        require(rows.city.value_counts().to_dict() == RESERVE_CITIES, f"{label}: reserve city counts differ")
    if reference is not None:
        cities = rows.set_index("scenario_id").city.sort_index()
        ref = reference.set_index("scenario_id").city.reindex(cities.index)
        require(cities.equals(ref), f"{label}: inconsistent paired city metadata")


def numeric_column(rows, name, binary=False, integer=False):
    require(name in rows, f"missing column {name}")
    a = rows[name].to_numpy(float)
    require(np.isfinite(a).all(), f"non-finite {name}")
    if binary:
        require(np.isin(a, [0, 1]).all(), f"non-binary {name}")
    if integer:
        require(((a >= 0) & (a == np.floor(a))).all(), f"invalid count {name}")
    return a


def seed_average(rows, arm, metric):
    columns = [f"{arm}_s{s}_{metric}" for s in range(3)]
    actual = {c for c in rows if c.startswith(f"{arm}_s") and c.endswith(f"_{metric}")}
    require(actual == set(columns), f"{arm}/{metric}: require exactly seeds 0, 1, 2")
    return np.column_stack([numeric_column(rows, c, binary=True) for c in columns]).mean(axis=1)


def agent_rows(rows, predictors, label):
    required = KEYS + META + ["predictor", "miss"]
    require(set(required) <= set(rows), f"{label}: missing agent columns")
    require(not rows[required].isna().any().any(), f"{label}: missing agent values")
    require(set(rows.predictor) == set(predictors), f"{label}: incomplete predictor/seed set")
    require(not rows.duplicated(KEYS + ["predictor"]).any(), f"{label}: duplicate agent/predictor rows")
    numeric_column(rows, "miss", binary=True)
    speed = numeric_column(rows, "speed")
    n = numeric_column(rows, "n_valid", integer=True)
    require((speed >= 0).all() and ((n >= 1) & (n <= 60)).all(), f"{label}: invalid speed/future coverage")
    for name in ("focal", "scored", "planner_relevant"):
        numeric_column(rows, name, binary=True)
        require(pd.api.types.is_bool_dtype(rows[name]), f"{label}: {name} must have boolean dtype")
    reference = rows.loc[rows.predictor == predictors[0], KEYS + META].set_index(KEYS).sort_index()
    for predictor in predictors[1:]:
        other = rows.loc[rows.predictor == predictor, KEYS + META].set_index(KEYS).sort_index()
        require(reference.equals(other), f"{label}: agent cohort or metadata differs for {predictor}")
    require(set(reference.city) == set(CITIES), f"{label}: missing cities")
    require((reference.reset_index().groupby("scenario_id").city.nunique() == 1).all(),
            f"{label}: city differs within scenario")
    wide = rows.pivot(index=KEYS, columns="predictor", values="miss")
    return reference.join(wide).reset_index()


def dose(rows, patch, sham, seeded):
    totals = np.zeros(2, dtype=np.int64)
    for seed in range(3) if seeded else [None]:
        suffix = f"_s{seed}" if seeded else ""
        names = [patch + suffix, sham + suffix]
        arrays = [np.stack(rows[f"{name}_dose_by_replan"].to_numpy()).astype(float) for name in names]
        p, s = arrays
        require(p.shape == s.shape == (len(rows), 6), "dose must contain six replans per scenario")
        require(all(np.isfinite(a).all() and ((a >= 0) & (a == np.floor(a))).all() for a in arrays),
                "dose must be finite nonnegative integers")
        require((s <= p).all(), "sham exceeds PATCH dose at a replan")
        for name, values, metric in zip(names, arrays, ("trigger_count", "substituted_count")):
            require(np.array_equal(values.sum(1), numeric_column(rows, f"{name}_{metric}", integer=True)),
                    f"{name}: inconsistent dose total")
        totals += [int(p.sum()), int(s.sum())]
    return float(totals[1] / totals[0]) if totals[0] else float("nan")


def bootstrap(frame, names, effect, draws, rng, chunk, derived=None):
    """Bootstrap paired scenario totals/counts; never weight crowded scenes equally with sparse scenes."""
    points, samples = {}, {}
    for city in CITIES:
        selected = frame[frame.city == city]
        values = selected[names].to_numpy(float)
        counts = selected["count"].to_numpy(float)
        if len(selected):
            with np.errstate(divide="ignore", invalid="ignore"):
                city_points = effect(dict(zip(names, values.sum(0) / counts.sum())))
            city_samples = {name: np.empty(draws) for name in city_points}
            for start in range(0, draws, chunk):
                size = min(chunk, draws - start)
                # Multinomial weights are scenario resampling counts shared by all columns.
                weights = rng.multinomial(len(selected), np.full(len(selected), 1 / len(selected)), size=size)
                with np.errstate(divide="ignore", invalid="ignore"):
                    means = (weights @ values) / (weights @ counts)[:, None]
                    result = effect(dict(zip(names, means.T)))
                for name, value in result.items():
                    city_samples[name][start:start + size] = value
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                city_points = effect({name: np.float64(np.nan) for name in names})
            city_samples = {name: np.full(draws, np.nan) for name in city_points}
        for name, value in city_points.items():
            points.setdefault(name, []).append(value)
            samples.setdefault(name, []).append(city_samples[name])
    result_points = {name: np.mean(value) for name, value in points.items()}
    result_samples = {name: np.mean(value, axis=0) for name, value in samples.items()}
    if derived:
        with np.errstate(divide="ignore", invalid="ignore"):
            result_points.update(derived(result_points))
            result_samples.update(derived(result_samples))
    return {name: (float(result_points[name]), result_samples[name]) for name in result_points}


def summaries(estimates, levels):
    result = {}
    for name, (point, draws) in estimates.items():
        valid = draws[np.isfinite(draws)]
        result[name] = {"point": point, "undefined_share": float(1 - len(valid) / len(draws))}
        for key, level in levels.items():
            result[name][key] = list(np.quantile(valid, [(1 - level) / 2, (1 + level) / 2])) if len(valid) else [None, None]
    return result


class Audit:
    def __init__(self, args, study):
        self.args, self.study = args, study
        self.failures, self.comparisons = [], 0
        offset = int.from_bytes(hashlib.sha256(study.encode()).digest()[:4], "little")
        self.rng = np.random.default_rng(args.seed + offset)
        self.independent = {}

    def equal(self, label, actual, reported, tolerance=1e-9):
        self.comparisons += 1
        if isinstance(actual, (bool, str)):
            same = type(actual) is type(reported) and actual == reported
        elif actual is None or not np.isfinite(actual):
            same = reported is None or (isinstance(reported, (float, int)) and not np.isfinite(reported))
        else:
            same = reported is not None and np.isfinite(reported) and abs(float(actual) - float(reported)) <= tolerance
        if not same:
            self.failures.append(f"{label}: independent {actual!r}, reported {reported!r}")

    def fields(self, actual, reported, prefix=""):
        for key, value in actual.items():
            label = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                self.fields(value, reported.get(key, {}), label)
            else:
                self.equal(label, value, reported.get(key))

    def effects(self, actual, reported, levels):
        for name, result in actual.items():
            old = reported.get(name, {})
            self.equal(f"{name}.point", result["point"], old.get("point"))
            if "undefined_share" in old:
                p = old["undefined_share"]
                # Independent bootstrap frequencies vary. Points and deterministic gates do not get this tolerance.
                tolerance = 4 * np.sqrt(max(p * (1 - p), 1 / self.args.draws)
                                        * (1 / self.args.draws + 1 / 10000))
                self.equal(f"{name}.undefined_share", result["undefined_share"], p, tolerance)
            for key in levels:
                previous = old.get(key)
                require(isinstance(previous, list) and len(previous) == 2, f"{name}: missing registered {key} interval")
                finite = all(v is not None and np.isfinite(v) for v in previous)
                tolerance = max(1e-9, self.args.mc_tolerance * abs(previous[1] - previous[0])) if finite else 1e-9
                for endpoint, (mine, theirs) in enumerate(zip(result[key], previous)):
                    self.equal(f"{name}.{key}[{endpoint}]", mine, theirs, tolerance)

    def estimate(self, frame, names, effect, levels, derived=None):
        result = bootstrap(frame, names, effect, self.args.draws, self.rng, self.args.chunk, derived)
        return summaries(result, levels)


def positive_bar(effect, key, bar, power, control):
    if not control:
        return "uninterpretable"
    if not power or effect["undefined_share"] > .01 or effect[key][0] is None:
        return "inconclusive (A8)"
    return "supported" if effect["point"] >= bar and effect[key][0] > 0 else "killed"


def power(frame, name):
    events = float(frame[name].sum())
    return events >= 100 and all(frame.loc[frame.city == city, name].sum() > 0 for city in CITIES), events


def closed_frame(rows, arms, seeded):
    frame = rows[["scenario_id", "city"]].copy()
    frame["count"] = 1
    for arm in arms:
        frame[arm] = seed_average(rows, arm, "unnecessary_hard_brake") if arm in seeded else numeric_column(
            rows, f"{arm}_unnecessary_hard_brake", binary=True)
    return frame


def r_audit(audit, rows, report, primary):
    scenario_rows(rows, RESERVE, audit.study)
    frame = closed_frame(rows, R_ARMS, R_ARMS)
    levels = LEVEL_R if primary else {"ci95": .95}
    decision_key = "ci9875" if primary else "ci95"

    def effect(rates):
        result = {f"{arm}_reduction": 1 - rates[arm] / rates["ALL"] for arm in R_ARMS[1:]}
        result["sham_gap"] = (rates["SHAM2"] - rates["PATCH"]) / rates["ALL"]
        return result

    def derived(results):
        return {"G_ratio_PROB_over_PATCH": np.divide(results["PROB_reduction"], results["PATCH_reduction"]),
                "interaction": results["PATCH_reduction"] - results["PROB_reduction"] - results["GEO_reduction"]}

    effects = audit.estimate(frame, list(R_ARMS), effect, levels, derived)
    rates = {arm: float(frame[arm].mean()) for arm in R_ARMS}
    rates.update({arm: float(numeric_column(rows, f"{arm}_unnecessary_hard_brake", binary=True).mean())
                  for arm in ("cv", "oracle", "static")})
    collision = {arm: float(seed_average(rows, arm, "collision").mean()) for arm in R_ARMS}
    collision.update({arm: float(numeric_column(rows, f"{arm}_collision", binary=True).mean())
                      for arm in ("cv", "oracle", "static", "log")})
    control = rates["ALL"] >= 2 * rates["oracle"]
    powered, events = power(frame, "ALL")
    matched_dose = dose(rows, "PATCH", "SHAM2", True)
    g = effects["G_ratio_PROB_over_PATCH"]
    if not control:
        g_verdict = "uninterpretable"
    elif not powered or g["undefined_share"] > .01 or g[decision_key][0] is None:
        g_verdict = "inconclusive (A8)"
    else:
        g_verdict = ("probability sufficient" if g[decision_key][0] >= .5 else
                     "geometry needed" if g[decision_key][1] < .5 else "inconclusive")
    verdicts = {"R1 PATCH": positive_bar(effects["PATCH_reduction"], decision_key, .30, powered, control),
                "R2 sham gap": positive_bar(effects["sham_gap"], decision_key, .20, powered, control and matched_dose >= .99),
                "R3 TRIM": positive_bar(effects["TRIM_reduction"], decision_key, .30, powered, control),
                "G1 PROB/PATCH": g_verdict}
    actual = {"scenarios": len(rows), "harness": "v4" if primary else "v2", "decision_level": decision_key,
              "role": "primary" if primary else "secondary", "brake_rates": rates, "collisions": collision,
              "brake_control_ALL_over_oracle": rates["ALL"] / rates["oracle"] if rates["oracle"] else None,
              "brake_control_pass": bool(control), "sham_dose_ratio": matched_dose,
              "seed_averaged_ALL_brake_events": events, "powered": bool(powered), "verdicts": verdicts,
              "log_collision_calibration_pass": bool(collision["log"] < .01)}
    actual["factorial_dose"] = {arm: int(sum(numeric_column(rows, f"{arm}_s{s}_substituted_count", integer=True).sum()
                                              for s in range(3))) for arm in ("GEO", "PROB")}
    actual["factorial_dose"]["PATCH"] = int(sum(numeric_column(rows, f"PATCH_s{s}_trigger_count", integer=True).sum()
                                               for s in range(3)))
    audit.fields(actual, report)
    audit.effects(effects, report.get("effects", {}), levels)
    audit.independent = actual | {"effects": effects}
    if not primary:
        cframe = frame[["scenario_id", "city", "count"]].copy()
        for arm in R_ARMS:
            cframe[arm] = seed_average(rows, arm, "collision")
        ceffects = audit.estimate(cframe, list(R_ARMS), lambda r: {
            f"{arm}_collision_diff": r[arm] - r["ALL"] for arm in R_ARMS[1:]}, {"ci95": .95})
        audit.effects(ceffects, report.get("collision_effects", {}), {"ci95": .95})
        control = collision["ALL"] > 0 and collision["static"] >= 2 * collision["ALL"]
        audit.fields({"collision_control_pass": bool(control),
                      "static_over_ALL_collisions": collision["static"] / collision["ALL"] if collision["ALL"] else None}, report)
        for name, value in ceffects.items():
            status = ("uninterpretable" if not control else "within harness margin" if value["ci95"][1] < .003
                      else "not established")
            audit.fields({"margin": .003, "status": status}, report.get("collision_effects", {}).get(name, {}), name)
        audit.independent["collision_effects"] = ceffects


def grouped_frame(rows, names):
    totals = rows.groupby(["city", "scenario_id"], sort=True)[names].sum()
    totals["count"] = rows.groupby(["city", "scenario_id"], sort=True).size()
    return totals.reset_index()


def validation_base(path):
    names = ["CV", "ALL_s0", "ALL_s1", "ALL_s2"]
    rows = pd.read_parquet(path, columns=KEYS + META + ["predictor", "miss"], filters=[("predictor", "in", names)])
    wide = agent_rows(rows, names, "Stage 3a reference")
    identities(wide.loc[wide.focal, "scenario_id"], VALIDATION, "validation focal cohort")
    wide["ALL"] = wide[[f"ALL_s{s}" for s in range(3)]].mean(axis=1)
    return wide


def q_focal(rows):
    require(set(rows.predictor) == {"QCNET"}, "unexpected Q-open predictor")
    require(pd.api.types.is_bool_dtype(rows.focal), "Q0 focal flag must have boolean dtype")
    focal = rows[rows.focal]
    identities(focal.scenario_id, VALIDATION, "Q0 focal cohort")
    metrics = {"n": len(focal)} | {output: float(numeric_column(focal, metric).mean()) for output, metric in (
        ("minFDE6", "min_fde"), ("minADE6", "min_ade"), ("MR6", "miss"), ("brier_minFDE6", "brier_min_fde"))}
    passed = metrics["minFDE6"] <= 1.25 * 1.05 and abs(metrics["MR6"] - .16) <= .015
    return metrics, bool(passed)


def q_open_audit(audit, qrows, base, report, native):
    require(set(qrows.predictor) == {"QCNET"}, "unexpected Q-open predictor")
    agent_rows(qrows, ["QCNET"], "Q-open")
    qmeta = qrows.set_index(KEYS)[META].sort_index()
    reference = base.set_index(KEYS)[META].sort_index()
    require(qmeta.index.isin(reference.index).all(), "Q-open contains unexpected agents")
    require(qmeta.equals(reference.reindex(qmeta.index)), "Q-open metadata differs from Stage 3a")
    require(native or qmeta.index.equals(reference.index), "full Q-open cohort is incomplete")
    f, q0 = q_focal(qrows)
    actual = {"focal": f, "joined_agents": len(qrows), "qcnet_rows": len(qrows),
              "Q0_positive_control": {"pass": bool(q0)},
              "cohort": {"reference_agents": len(base), "omitted_agents": len(base) - len(qrows), "native_only": native,
                         "fallback_agents": int(qrows.get("qcnet_missing", pd.Series(False, index=qrows.index)).fillna(False).sum())}}
    joined = qrows.merge(base[KEYS + ["CV", "ALL"]], on=KEYS, validate="one_to_one")
    for group, mask in (("stopped", joined.speed < .5), ("moving", joined.speed >= 2)):
        selected = joined[mask & joined.planner_relevant & ~joined.focal]
        for subset, cellrows in (("all", selected), ("complete_future", selected[selected.n_valid == 60])):
            label = f"{group}_{subset}"
            frame = grouped_frame(cellrows, ["miss", "CV", "ALL"])
            effects = audit.estimate(frame, ["miss", "CV", "ALL"], lambda r: {
                "QCNET_minus_CV": r["miss"] - r["CV"], "QCNET_minus_ALL": r["miss"] - r["ALL"]}, LEVEL_QS)
            audit.effects(effects, report.get(label, {}), LEVEL_QS)
            cell = {"agents": len(cellrows), "miss_rates_city_equal": {
                output: float(np.mean([cellrows.loc[cellrows.city == c, column].mean() for c in CITIES]))
                for output, column in (("QCNET", "miss"), ("CV", "CV"), ("ALL", "ALL"))}}
            if group == "stopped":
                cell.update(mean_moving_mode_mass=float(cellrows.moving_mode_mass.mean()),
                            share_moving_mass_over_5pct=float((cellrows.moving_mode_mass > .05).mean()))
            audit.fields(cell, report.get(label, {}), label)
            actual[label] = cell | effects
    q1 = actual["stopped_all"]["QCNET_minus_CV"]
    verdict = "uninterpretable (Q0 failed)" if not q0 else (
        "supported" if q1["point"] >= .15 and q1["ci9833"][0] > 0 else "killed")
    if q0 and q1["ci9833"][1] < 0:
        verdict += "; QCNet better than CV-6 on stopped agents"
    actual["Q1_verdict"] = verdict
    audit.fields({k: v for k, v in actual.items() if k not in [f"{g}_{s}" for g in ("stopped", "moving")
                                                            for s in ("all", "complete_future")]}, report)
    audit.independent = actual


def q_closed_audit(audit, rows, base, qrows, q_open_report, report):
    scenario_rows(rows, Q_SUBSET, "Q-closed", reference=base.drop_duplicates("scenario_id"))
    val_ids = sorted(base.scenario_id.unique())
    expected = sorted(np.random.default_rng(20261002).choice(val_ids, size=6000, replace=False).tolist())
    require(set(rows.scenario_id) == set(expected), "Q-closed differs from independently sampled validation subset")
    focal, q0 = q_focal(qrows)
    audit.fields({"focal": focal, "Q0_positive_control": {"pass": q0}}, q_open_report, "Q0 source")
    frame = closed_frame(rows, list(Q_ARMS) + ["ALL"], ["ALL"])
    effects = audit.estimate(frame, list(Q_ARMS) + ["ALL"], lambda r: {
        "QPATCH_reduction": 1 - r["QPATCH"] / r["QBASE"], "QSHAM2_reduction": 1 - r["QSHAM2"] / r["QBASE"],
        "sham_gap": (r["QSHAM2"] - r["QPATCH"]) / r["QBASE"],
        "QBASE_vs_ALL_relative": r["QBASE"] / r["ALL"] - 1}, LEVEL_QS)
    rates = {arm: float(frame[arm].mean()) for arm in list(Q_ARMS) + ["ALL"]}
    rates.update({arm: float(numeric_column(rows, f"{arm}_unnecessary_hard_brake", binary=True).mean())
                  for arm in ("cv", "oracle", "static")})
    powered, events = power(frame, "QBASE")
    matched_dose = dose(rows, "QPATCH", "QSHAM2", False)
    control = rates["QBASE"] >= 2 * rates["oracle"]
    actual = {"scenarios": len(rows), "brake_rates": rates, "brake_control_pass": bool(control), "q0_pass": q0,
              "brake_control_QBASE_over_oracle": rates["QBASE"] / rates["oracle"] if rates["oracle"] else None,
              "sham_dose_ratio": matched_dose, "QBASE_brake_events": events, "powered": bool(powered),
              "qcnet_missing_counts": {arm: int(numeric_column(rows, f"{arm}_qcnet_missing_count", integer=True).sum())
                                       for arm in Q_ARMS},
              "verdicts": {"Q2 QPATCH": positive_bar(effects["QPATCH_reduction"], "ci9833", .30, powered, control and q0),
                           "Q3 sham gap": positive_bar(effects["sham_gap"], "ci9833", .20, powered,
                                                       control and q0 and matched_dose >= .99)}}
    audit.fields(actual, report)
    audit.effects(effects, report.get("effects", {}), LEVEL_QS)
    audit.independent = actual | {"effects": effects}


def s_open_audit(audit, rows, reserve, report):
    names = [f"{family}_s{seed}" for family in FAMILIES for seed in range(3)] + ["CV"]
    wide = agent_rows(rows, names, "S-open")
    scenario_rows(wide.loc[wide.focal, ["scenario_id", "city"]], RESERVE, "S-open focal reserve")
    require(set(wide.scenario_id) == set(reserve), "S-open differs from reserve IDs")
    for family in FAMILIES:
        wide[family] = wide[[f"{family}_s{seed}" for seed in range(3)]].mean(axis=1)
    actual = {}
    cells = {"stopped_nonfocal": wide.planner_relevant & ~wide.focal & (wide.speed < .5),
             "moving_nonfocal": wide.planner_relevant & ~wide.focal & (wide.speed >= 2),
             "focal": wide.focal, "focal_stopped": wide.focal & (wide.speed < .5)}
    for label, mask in cells.items():
        selected = wide[mask]
        effects = audit.estimate(grouped_frame(selected, list(FAMILIES)), list(FAMILIES), lambda r: {
            "SEL_minus_ALLR": r["SEL"] - r["ALLR"], "SEL_minus_SELSHAM": r["SEL"] - r["SELSHAM"],
            "SELSHAM_minus_ALLR": r["SELSHAM"] - r["ALLR"]}, LEVEL_QS)
        cell = {"agents": len(selected), "miss_city_equal": {
            family: float(np.mean([selected.loc[selected.city == city, family].mean() for city in CITIES]))
            for family in FAMILIES + ("CV",)}}
        audit.fields(cell, report.get(label, {}), label)
        audit.effects(effects, report.get(label, {}), LEVEL_QS)
        actual[label] = cell | effects
    verdicts = {}
    for claim, key in (("S1 SEL vs ALLR", "SEL_minus_ALLR"), ("S2 SEL vs SELSHAM", "SEL_minus_SELSHAM")):
        result = actual["stopped_nonfocal"][key]
        verdicts[claim] = ("inconclusive (missing city/group)" if result["ci9833"][1] is None else
                           "supported" if result["point"] <= -.15 and result["ci9833"][1] < 0 else "killed")
    audit.fields({"verdicts": verdicts}, report)
    audit.independent = actual | {"verdicts": verdicts}


def s_closed_audit(audit, allr, sel, sham, report):
    scenario_rows(allr, RESERVE, "ALLR")
    frame = closed_frame(allr, ["ALL"], ["ALL"]).rename(columns={"ALL": "ALLR"})
    for family, rows in (("SEL", sel), ("SELSHAM", sham)):
        scenario_rows(rows, RESERVE, family, reference=allr)
        outcome = pd.Series(seed_average(rows, "ALL", "unnecessary_hard_brake"), index=rows.scenario_id)
        frame[family] = frame.scenario_id.map(outcome)
    effects = audit.estimate(frame, list(FAMILIES), lambda r: {
        "SEL_reduction": 1 - r["SEL"] / r["ALLR"], "SELSHAM_reduction": 1 - r["SELSHAM"] / r["ALLR"],
        "SEL_minus_SELSHAM": (r["SELSHAM"] - r["SEL"]) / r["ALLR"]}, LEVEL_QS)
    rates = {family: float(frame[family].mean()) for family in FAMILIES}
    rates["oracle"] = float(numeric_column(allr, "oracle_unnecessary_hard_brake", binary=True).mean())
    control = rates["ALLR"] >= 2 * rates["oracle"]
    powered, events = power(frame, "ALLR")
    actual = {"scenarios": len(frame), "brake_rates": rates, "brake_control_pass": bool(control),
              "powered": bool(powered), "ALLR_brake_events": events,
              "verdicts": {"S3 SEL braking": positive_bar(effects["SEL_reduction"], "ci9833", .30, powered, control)}}
    audit.fields(actual, report)
    audit.effects(effects, report.get("effects", {}), LEVEL_QS)
    audit.independent = actual | {"effects": effects}


def excess_audit(audit, rows, report, harness="v4"):
    scenario_rows(rows, RESERVE, "excess-brake flags")
    expected = {f"{arm}_s{seed}" for arm in R_ARMS for seed in range(3)}
    require(set(rows) == expected | {"scenario_id", "city", "oracle"}, "unexpected excess flag columns/seed set")
    frame = rows[["scenario_id", "city"]].assign(count=1)
    for arm in R_ARMS:
        frame[arm] = np.column_stack([numeric_column(rows, f"{arm}_s{seed}", binary=True) for seed in range(3)]).mean(1)
    effects = audit.estimate(frame, list(R_ARMS), lambda r: {
        **{f"{arm}_reduction": 1 - r[arm] / r["ALL"] for arm in R_ARMS[1:]},
        "sham_gap": (r["SHAM2"] - r["PATCH"]) / r["ALL"]}, {"ci95": .95})
    actual = {"harness": harness, "scenarios": len(rows),
              "rates": {arm: float(frame[arm].mean()) for arm in R_ARMS}
              | {"oracle": float(numeric_column(rows, "oracle", binary=True).mean())}}
    audit.fields(actual, report)
    audit.effects(effects, report.get("effects", {}), {"ci95": .95})
    audit.independent = actual | {"effects": effects, "scope": "secondary descriptive flags; no verdict rederived"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("."), help="unpacked release root")
    parser.add_argument("--study", nargs="+", choices=STUDIES, default=list(STUDIES))
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--chunk", type=int, default=64, help="bootstrap draws held in memory per city")
    parser.add_argument("--seed", type=int, default=20261019, help="fresh audit seed; must differ from registered seed")
    parser.add_argument("--mc-tolerance", type=float, default=.15, help="endpoint tolerance as fraction of reported CI width")
    parser.add_argument("--runs-dir", default="runs/v2")
    parser.add_argument("--reports-dir", default="reports/v2")
    parser.add_argument("--reserve", default=None, help="reserve_ids.npy; default RUNS/reserve_ids.npy")
    parser.add_argument("--stage3a", default="evals/stage3a/per_agent.parquet")
    parser.add_argument("--q-rows", default=None, help="native Q parquet or part directory; default RUNS/q_open")
    parser.add_argument("--q-fallback", default=None, help="default RUNS/q_open_fallback.parquet")
    parser.add_argument("--s-rows", default=None, help="default RUNS/study_s_open")
    parser.add_argument("--excess-flags", default=None, help="override flags for a single requested excess study")
    for study, stem in (("r-v4", "study_r_v4"), ("r-v2", "study_r_v2"), ("q-closed", "q_closed"),
                        ("s-sel", "study_s_SEL_v4"), ("s-sham", "study_s_SELSHAM_v4")):
        parser.add_argument(f"--{study}-parts", default=None, help=f"default RUNS/{stem}")
    for study in STUDIES:
        parser.add_argument(f"--{study}-report", default=None, help="override the corresponding report JSON")
    parser.add_argument("--out", type=Path, help="write JSON audit; otherwise print JSON to stdout")
    args = parser.parse_args()
    if args.draws < 1 or args.chunk < 1 or not 0 < args.mc_tolerance < 1 or args.seed < 0 or args.seed == 20261002:
        parser.error("positive draws/chunk, fresh nonnegative seed and 0 < mc-tolerance < 1 are required")

    def path(value):
        p = Path(value)
        return p if p.is_absolute() else args.root / p

    runs, reports = path(args.runs_dir), path(args.reports_dir)
    report_names = dict(zip(STUDIES, ("study_r_v4", "study_r_v2", "study_q_open_corrected", "study_q_open_native",
                                     "study_q_closed", "study_s_open", "study_s_closed", "excess_brake_v4", "excess_brake_v2")))

    def report(study):
        p = getattr(args, study.replace("-", "_") + "_report")
        return json.loads((path(p) if p else reports / f"{report_names[study]}.json").read_text())

    def parts(key, stem):
        override = getattr(args, key.replace("-", "_") + "_parts")
        return table(path(override) if override else runs / stem)

    base = None
    results = {}
    for study in dict.fromkeys(args.study):
        audit = Audit(args, study)
        try:
            reserve = None
            if study.startswith(("r-", "s-", "excess-")):
                reserve = np.load(path(args.reserve) if args.reserve else runs / "reserve_ids.npy", allow_pickle=True).astype(str)
                identities(reserve, RESERVE, "reserve ID file")
            saved = report(study)
            if study in ("r-v4", "r-v2"):
                r_audit(audit, parts(study, "study_r_" + study[-2:]), saved, study == "r-v4")
            elif study in ("q-open", "q-native", "q-closed"):
                if base is None:
                    base = validation_base(path(args.stage3a))
                if study == "q-closed":
                    qrows = table(path(args.q_rows) if args.q_rows else runs / "q_open")
                    q_closed_audit(audit, parts("q-closed", "q_closed"), base, qrows, report("q-open"), saved)
                else:
                    rows = table(path(args.q_rows) if args.q_rows else runs / "q_open")
                    if study == "q-open":
                        rows["qcnet_missing"] = False
                        fallback = table(path(args.q_fallback) if args.q_fallback else runs / "q_open_fallback.parquet")
                        require(fallback.qcnet_missing.eq(True).all(), "fallback rows must be explicitly marked")
                        rows = pd.concat([rows, fallback], ignore_index=True)
                    q_open_audit(audit, rows, base, saved, study == "q-native")
            elif study == "s-open":
                s_open_audit(audit, table(path(args.s_rows) if args.s_rows else runs / "study_s_open"), reserve, saved)
            elif study == "s-closed":
                s_closed_audit(audit, parts("r-v4", "study_r_v4"), parts("s-sel", "study_s_SEL_v4"),
                               parts("s-sham", "study_s_SELSHAM_v4"), saved)
            else:
                harness = study[-2:]
                flags = path(args.excess_flags) if args.excess_flags else runs / f"excess_brake_{harness}.parquet"
                excess_audit(audit, table(flags), saved, harness)
        except (ValueError, OSError, KeyError, TypeError) as error:
            audit.failures.append(f"input or analysis failure: {error}")
        results[study] = {"pass": not audit.failures, "comparisons": audit.comparisons,
                          "failures": audit.failures, "independent": audit.independent}
        print(f"{study}: {'PASS' if not audit.failures else 'FAIL'} ({audit.comparisons} comparisons)", file=sys.stderr)
    result = {"pass": all(r["pass"] for r in results.values()), "requested_studies": list(results),
              "all_supported_studies_requested": set(results) == set(STUDIES), "draws": args.draws, "chunk": args.chunk,
              "seed": args.seed, "point_tolerance": 1e-9, "interval_width_tolerance": args.mc_tolerance,
              "unsupported_checks": ["H/F independent rederivation", "excess-brake flag generation/replay correctness", "raw forecast correctness",
                                     "training execution and historical checkpoint/source provenance", "physical driving safety"],
              "studies": results}

    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, np.integer):
            return int(value)
        return value

    payload = json.dumps(clean(result), indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(payload)
    else:
        print(payload, end="")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
