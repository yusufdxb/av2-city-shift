"""Independent row-based audit of v2 studies H and F. Imports no cityshift code.

Requires the v1.0 confirmatory evaluation tables and the v2 fixed-horizon rows.
Run: python scripts/audit_v2_amendments.py --draws 10000 --out reports/v2/audit_amendments.json
Confidence intervals use a fresh random seed and a 15%-of-interval-width Monte Carlo tolerance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CITIES = ("austin", "dearborn", "miami", "palo-alto", "pittsburgh", "washington-dc")


class Audit:
    def __init__(self):
        self.checks, self.failures = 0, []

    def equal(self, name, actual, expected, tol=1e-9):
        self.checks += 1
        undefined_actual = actual is None or isinstance(actual, (float, np.floating)) and not np.isfinite(actual)
        undefined_expected = expected is None or isinstance(expected, (float, np.floating)) and not np.isfinite(expected)
        if undefined_actual or undefined_expected:
            ok = undefined_actual and undefined_expected
        elif isinstance(actual, (float, int, np.number)) and isinstance(expected, (float, int, np.number)):
            ok = bool(np.isfinite([actual, expected]).all() and abs(actual - expected) <= tol)
        else:
            ok = actual == expected
        if not ok:
            self.failures.append(f"{name}: {actual!r} != {expected!r}")

    def ci(self, name, draws, expected, level):
        finite = draws[np.isfinite(draws)]
        actual = np.quantile(finite, [(1-level)/2, 1-(1-level)/2]) if len(finite) else np.array([np.nan, np.nan])
        width = expected[1] - expected[0] if all(v is not None and np.isfinite(v) for v in expected) else 0
        for side in range(2):
            self.equal(f"{name}[{side}]", float(actual[side]), expected[side], max(1e-9, .15 * width))
        return actual


def capture(errors, scores, oracle):
    n = len(errors)
    retain = round(.8 * n)
    if not n or not retain:
        return np.nan
    # Lexicographic index breaks score ties independently of the production stable-sort implementation.
    index = np.arange(n)
    oracle_mean = errors[np.lexsort((index, oracle))[:retain]].mean()
    denom = errors.mean() - oracle_mean
    if denom <= 0:
        return np.nan
    return (errors.mean() - errors[np.lexsort((index, scores))[:retain]].mean()) / denom


def audit_h(args, audit):
    root = Path(args.evals)
    original = pd.read_parquet(root / "ALL.parquet")
    city = original.city.to_numpy()
    audit.equal("H scenarios", len(original), 24988)
    assert not original.val_index.duplicated().any()
    assert set(original.city) == set(CITIES)
    assert hashlib.sha256("\n".join(sorted(original.scenario_id)).encode()).hexdigest() == (
        "58b1110a8ffa2eead56e52f583a5655988b0a5e6c97031d4b3c750bc6e1fa625")
    tables = {}
    for c in CITIES:
        d = pd.read_parquet(root / f"LOCO-{c}.parquet")
        assert d.val_index.equals(original.val_index) and d.city.equals(original.city)
        assert d.scenario_id.equals(original.scenario_id)
        assert np.isfinite(d[[f"s{s}_{m}" for s in range(3) for m in ("miss", "min_fde")]].to_numpy(float)).all()
        assert np.isin(d[[f"s{s}_miss" for s in range(3)]], [0, 1]).all()
        assert np.isfinite(d.disagree.to_numpy(float)).all()
        errors = d[[f"s{s}_miss" for s in range(3)]].to_numpy(float).mean(axis=1)
        oracle = d[[f"s{s}_min_fde" for s in range(3)]].to_numpy(float).mean(axis=1)
        tables[c] = errors, d.disagree.to_numpy(), oracle
    report = json.loads(Path(args.reports, "h3_shared_bootstrap.json").read_text())
    masks = {c: city == c for c in CITIES}
    indices = {c: np.flatnonzero(masks[c]) for c in CITIES}

    def effects(sample):
        h2, h3 = [], []
        for c in CITIES:
            miss, uncertainty, oracle = tables[c]
            held, seen = sample[masks[c]], sample[~masks[c]]
            left = capture(miss[held], uncertainty[held], oracle[held])
            right = capture(miss[seen], uncertainty[seen], oracle[seen])
            h2.append(left)
            h3.append(left-right)
        return np.nanmean(h2), np.nanmean(h3)

    points = effects(np.arange(len(city)))
    audit.equal("H2 point", points[0], report["H2"]["pooled_CF_U1_heldout"], 1e-12)
    audit.equal("H3 point", points[1], report["H3"]["pooled_CF_diff_heldout_minus_indist"], 1e-12)
    rng = np.random.default_rng(args.seed)
    draws = np.empty((args.draws, 2))
    for b in range(args.draws):
        sample = np.arange(len(city))
        for c in CITIES:
            sample[masks[c]] = rng.choice(indices[c], size=len(indices[c]), replace=True)
        draws[b] = effects(sample)
    for j, name in enumerate(("H2", "H3")):
        audit.ci(name + " 95%", draws[:, j], report[name]["ci95"], .95)
        ci = audit.ci(name + " 98.75%", draws[:, j], report[name]["ci_bonferroni"], .9875)
        audit.equal(name + " undefined", float((~np.isfinite(draws[:, j])).mean()),
                    report[name]["bootstrap_undefined_share"], .005)
        if name == "H3":
            verdict = ("inconclusive" if not np.isfinite(ci).all() else
                       "differs" if ci[0] > 0 or ci[1] < 0 else "no detectable difference")
            audit.equal("H3 amended verdict", verdict, report[name]["shared_draw_verdict"])


def audit_f(args, audit):
    files = sorted(Path(args.coverage).glob("*.parquet"))
    d = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    keys = ["scenario_id", "replan", "agent_id"]
    assert not d.duplicated(keys + ["predictor"]).any()
    a = d[d.predictor == "ALL"].set_index(keys).sort_index()
    c = d[d.predictor == "CV-6"].set_index(keys).sort_index()
    assert a.index.equals(c.index)
    meta = ["city", "agent_type", "focal", "speed", "complete_2s", "complete_4s"]
    assert a[meta].equals(c[meta])
    expected_ids = set(pd.read_parquet(Path(args.evals) / "ALL.parquet", columns=["scenario_id"]).scenario_id)
    observed_ids = set(a.index.get_level_values("scenario_id"))
    assert observed_ids <= expected_ids
    empty = pd.read_parquet(args.empty_coverage)
    assert set(empty.scenario_id) == expected_ids - observed_ids
    assert not empty.scenario_id.duplicated().any()
    assert (empty[[f"selected_{t}" for t in (49, 59, 69, 79, 89)]] == 0).all().all()
    if args.raw_val:
        for sid in sorted(expected_ids-observed_ids):
            raw = pd.read_parquet(Path(args.raw_val) / sid / f"scenario_{sid}.parquet")
            for t in (49, 59, 69, 79, 89):
                frame = raw[raw.timestep == t]
                origin = frame[frame.track_id == "AV"][["position_x", "position_y"]].to_numpy()[0]
                candidates = frame[(frame.track_id != "AV") & frame.object_type.isin(
                    ["vehicle", "bus", "motorcyclist", "cyclist", "pedestrian"])]
                count = int((np.linalg.norm(candidates[["position_x", "position_y"]].to_numpy()-origin, axis=1) < 60).sum())
                audit.equal(f"F raw empty-scene {sid}/{t}", count, 0)
    audit.equal("F processed scenarios incl. verified empty scenes", len(observed_ids) + len(empty), 24988)
    a, c = a.reset_index(), c.reset_index()
    report = json.loads(Path(args.reports, "study_f_coverage.json").read_text())
    schedule = [(t, h) for t in (49, 59, 69, 79, 89) for h in ((2, 4) if t <= 69 else (2,))]
    required_cells = {(t, h, group) for t, h in schedule for group in ("stopped", "moving")}
    actual_cells = [(c["replan"], c["horizon_s"], c["group"]) for c in report["cells"]]
    assert len(actual_cells) == 16 and set(actual_cells) == required_cells
    coverage_pairs = [(c["replan"], c["horizon_s"]) for c in report["selected_coverage"]]
    assert len(coverage_pairs) == 8 and set(coverage_pairs) == set(schedule)
    rng = np.random.default_rng(args.seed)
    for cell in report["selected_coverage"]:
        mask = a.replan == cell["replan"]
        audit.equal(f"F coverage {cell['replan']}/{cell['horizon_s']}", int(mask.sum()), cell["n_selected"])
        audit.equal("F exclusions", int((mask & ~a[f"complete_{cell['horizon_s']}s"]).sum()), cell["n_excluded"])
    decisions = []
    for cell in report["cells"]:
        t, horizon, group = cell["replan"], cell["horizon_s"], cell["group"]
        selected = (a.replan == t) & ~a.focal
        selected &= (a.speed < .5) if group == "stopped" else (a.speed >= 2)
        complete = a[f"complete_{horizon}s"]
        eligible = selected & complete
        metric = f"miss_{horizon}s"
        rates, draws, individual = [], [], {"ALL": [], "CV6": []}
        assert {x["city"] for x in cell["per_city"]} == set(CITIES) and len(cell["per_city"]) == 6
        for city in CITIES:
            mask = eligible & (a.city == city)
            saved = next(x for x in cell["per_city"] if x["city"] == city)
            chosen = selected & (a.city == city)
            label = f"F {t}/{horizon}/{group}/{city}"
            counts = {"n_selected": int(chosen.sum()), "n_agents": int(mask.sum()),
                      "n_scenarios": int(a.loc[mask, "scenario_id"].nunique()),
                      "n_excluded": int((chosen & ~complete).sum())}
            counts["excluded_share"] = counts["n_excluded"]/counts["n_selected"] if counts["n_selected"] else np.nan
            for key, value in counts.items():
                audit.equal(label+" "+key, value, saved[key])
            for name, source in (("ALL", a), ("CV6", c)):
                misses = source.loc[mask, metric].to_numpy(float)
                fde = source.loc[mask, f"min_fde_{horizon}s"].to_numpy(float)
                assert np.isfinite(misses).all() and ((misses >= 0) & (misses <= 1)).all()
                assert np.isfinite(fde).all() and (fde >= 0).all()
                mean = float(misses.mean()) if len(misses) else np.nan
                individual[name].append(mean)
                audit.equal(label+" miss_"+name, mean, saved["miss_"+name])
                audit.equal(label+" min_fde_"+name, float(fde.mean()) if len(fde) else np.nan, saved["min_fde_"+name])
            frame = pd.DataFrame({"scenario_id": a.loc[mask, "scenario_id"],
                                  "difference": a.loc[mask, metric] - c.loc[mask, metric]})
            grouped = frame.groupby("scenario_id").difference.agg(["sum", "count"])
            sums, counts = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
            if not len(counts):
                rates.append(np.nan)
                draws.append(np.full(args.draws, np.nan))
                continue
            rates.append(sums.sum()/counts.sum())
            boot = np.empty(args.draws)
            for start in range(0, args.draws, 128):
                stop = min(start+128, args.draws)
                sample = rng.integers(len(counts), size=(stop-start, len(counts)))
                boot[start:stop] = sums[sample].sum(axis=1)/counts[sample].sum(axis=1)
            draws.append(boot)
        point = float(np.mean(rates))
        label = f"F {t}/{horizon}/{group}"
        for name in individual:
            audit.equal(label+" city-equal miss_"+name, float(np.mean(individual[name])), cell["miss_"+name])
        audit.equal(label + " point", point, cell["difference_ALL_minus_CV6"])
        audit.equal(label + " eligible agents", int(eligible.sum()), cell["n_agents"])
        ci = audit.ci(label, np.mean(draws, axis=0), cell["ci95"], .95)
        primary = group == "stopped" and horizon == 2 and t in (49, 89)
        audit.equal(label + " primary flag", cell["primary"], primary)
        if primary:
            verdict = ("inconclusive" if not np.isfinite(ci).all() else
                       "supported" if point >= .15 and ci[0] > 0 else "killed")
            audit.equal(label + " verdict", verdict, cell["status"])
            decisions.append(verdict)
    assert len(decisions) == 2
    verdict = "killed" if "killed" in decisions else "supported" if all(s == "supported" for s in decisions) else "inconclusive"
    audit.equal("F decision", verdict, report["decision"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evals", default="evals")
    ap.add_argument("--coverage", default="runs/v2/coverage")
    ap.add_argument("--empty-coverage", default="runs/v2/coverage_empty.parquet")
    ap.add_argument("--raw-val", help="also independently verify the empty-scene witness from raw records")
    ap.add_argument("--reports", default="reports/v2")
    ap.add_argument("--studies", nargs="+", choices=["H", "F"], default=["H", "F"])
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--out", default="reports/v2/audit_amendments.json")
    args = ap.parse_args()
    if args.draws < 1 or args.seed < 0 or args.seed == 20261002:
        ap.error("positive draw count and a fresh nonnegative seed are required")
    audit = Audit()
    for study in args.studies:
        {"H": audit_h, "F": audit_f}[study](args, audit)
        print(f"{study}: {audit.checks} checks, {len(audit.failures)} failures", flush=True)
    result = {"independent": True, "studies": args.studies, "draws": args.draws, "seed": args.seed,
              "checks": audit.checks, "failures": audit.failures, "pass": not audit.failures}
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
