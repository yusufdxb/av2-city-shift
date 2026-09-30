"""EXPLORATORY studies B and C from docs/preregistration/exploratory-followups.md, on the replayed Stage 4 drives.

Refuses to run unless reports/followups/replay_parity.json passed and the uncensored common-window scorer reproduces
the registered H10 point estimates. Estimator as registered for Stage 4: seeds averaged within scenario, effects within
each city, six cities equal-weighted, 10,000 within-city scenario bootstraps; 95% intervals with Bonferroni 98.33%
(three primaries) alongside.

    PYTHONPATH=src python scripts/analyze_followups.py
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from cityshift.analysis_stage3 import CITIES

LEVEL, LEVEL_BONF = 0.95, 1 - 0.05 / 3
MARGIN = 0.003


def interval(draws: np.ndarray, level: float) -> list[float]:
    return [float(x) for x in np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])]


def per_scene(rows: pd.DataFrame, arm: str, metric: str) -> pd.Series:
    """Scenario-level value: mean over model seeds 0-2 for model arms, the single run otherwise."""
    x = rows[rows.arm == arm]
    seeds = sorted(x.model_seed.unique())
    assert seeds in ([0, 1, 2], [-1]), f"{arm} seeds {seeds}"
    return x.groupby("scenario_id")[metric].mean().astype(float)


def city_bootstrap(data: pd.DataFrame, effects, n_boot: int, seed: int) -> dict:
    """Point = equal-weighted city effects; draws resample scenarios within city (same rng layout as Stage 4)."""
    cols = [c for c in data if c not in ("scenario_id", "city")]
    rng = np.random.default_rng(seed)
    points, draws = {}, {}
    for city in CITIES:
        part = data[data.city.astype(str) == city]
        assert len(part), f"missing city {city}"
        m = part[cols].to_numpy(float)
        for k, v in effects(dict(zip(cols, m.mean(0)))).items():
            points.setdefault(k, []).append(float(v))
        pieces: dict[str, list] = {}
        for start in range(0, n_boot, 128):
            idx = rng.integers(0, len(m), size=(min(128, n_boot - start), len(m)))
            for k, v in effects(dict(zip(cols, m[idx].mean(1).T))).items():
                pieces.setdefault(k, []).append(np.asarray(v))
        for k, p in pieces.items():
            draws.setdefault(k, []).append(np.concatenate(p))
    out = {}
    for k, vals in points.items():
        boot = np.mean(np.stack(draws[k]), axis=0)
        bad = ~np.isfinite(boot)
        ok = boot[~bad]
        out[k] = {"point": float(np.mean(vals)), "per_city": dict(zip(CITIES, vals)),
                  "undefined_share": float(bad.mean()),
                  "ci95": interval(ok, LEVEL) if len(ok) else [None, None],
                  "ci9833": interval(ok, LEVEL_BONF) if len(ok) else [None, None]}
    return out


def frame(rows: pd.DataFrame, spec: dict[str, tuple[str, str]]) -> pd.DataFrame:
    city = rows.drop_duplicates("scenario_id").set_index("scenario_id").city
    d = pd.DataFrame({name: per_scene(rows, arm, metric) for name, (arm, metric) in spec.items()})
    return d.assign(city=city.loc[d.index]).reset_index(names="scenario_id")


def reduction(num: str, den: str):
    def f(m):
        with np.errstate(divide="ignore", invalid="ignore"):
            return (m[den] - m[num]) / m[den]
    return f


def study_b(rows: pd.DataFrame, n_boot: int, stage4: pd.DataFrame) -> dict:
    subs = {f"seed{k}": float(stage4[f"PATCH_s{k}_substituted_count"].mean()) for k in range(3)}
    arms = ("ALL", "PATCH", "cv", "oracle", "static")
    spec = {f"{a}_dep": (a, "partner_departing") for a in arms}
    spec |= {f"{a}_dep2": (a, "partner_departing_any_replan") for a in arms}
    d = frame(rows, spec)
    eff = lambda m: {f"{a}_minus_ALL": m[f"{a}_dep"] - m["ALL_dep"] for a in arms[1:]} | {  # noqa: E731
        f"{a}_minus_ALL_any_replan": m[f"{a}_dep2"] - m["ALL_dep2"] for a in arms[1:]}
    res = city_bootstrap(d, eff, n_boot, 20260930)
    events = {a: float(per_scene(rows, a, "partner_departing").sum()) for a in arms}
    primary, control = res["PATCH_minus_ALL"], res["static_minus_ALL"]
    powered = events["ALL"] + events["PATCH"] + events["static"] >= 20
    split = {}
    for a in arms:
        x = rows[rows.arm == a]
        n = len(x)
        split[a] = {"at_fault_collision": float(x.collision.mean()),
                    "departing_stopped": float(x.partner_departing.mean()),
                    "stopped_not_departing": float(x.partner_stopped_not_departing.mean()),
                    "moving": float(x.partner_moving.mean()),
                    "departing_per_km": float(x.partner_departing.sum() / x.distance_m.sum() * 1000),
                    "drives": n}
    control_pass = control["ci95"][0] is not None and control["ci95"][0] > 0
    upper = primary["ci95"][1]
    verdict = ("uninterpretable (positive control failed)" if not control_pass else
               "underpowered" if not powered else
               "non-inferiority holds" if upper < MARGIN else "not established (killed)")
    return {"effects": res, "seed_averaged_events": events, "powered": powered, "positive_control_pass": control_pass,
            "increase_detected": bool(primary["ci95"][0] is not None and primary["ci95"][0] > 0),
            "verdict": verdict, "by_arm_pooled": split,
            "patch_substitutions_per_drive": subs}


def study_c(rows: pd.DataFrame, n_boot: int, registered: dict) -> dict:
    spec = {"ALL_b": ("ALL", "common_unnecessary_hard_brake"), "PATCH_b": ("PATCH", "common_unnecessary_hard_brake"),
            "ALL_c": ("ALL", "common_collision"), "PATCH_c": ("PATCH", "common_collision")}
    full = {"ALL_b": ("ALL", "full_unnecessary_hard_brake"), "PATCH_b": ("PATCH", "full_unnecessary_hard_brake"),
            "ALL_c": ("ALL", "full_collision"), "PATCH_c": ("PATCH", "full_collision")}
    eff = lambda m: {"PATCH_reduction": reduction("PATCH_b", "ALL_b")(m),  # noqa: E731
                     "PATCH_collision_diff": m["PATCH_c"] - m["ALL_c"]}
    # positive control: the uncensored scorer through this estimator must give the registered H10 points
    ref = city_bootstrap(frame(rows, full), eff, 2, 0)
    for k in ("PATCH_reduction", "PATCH_collision_diff"):
        if abs(ref[k]["point"] - registered[k]["point"]) > 1e-12:
            raise SystemExit(f"uncensored reference {k} {ref[k]['point']} != registered {registered[k]['point']}")
    d = frame(rows, spec)
    res = city_bootstrap(d, eff, n_boot, 20260930)
    all_events = float(d.ALL_b.sum())
    zero_city = [c for c in CITIES if d[d.city.astype(str) == c].ALL_b.sum() == 0]
    red, coll = res["PATCH_reduction"], res["PATCH_collision_diff"]
    powered = all_events >= 100 and not zero_city and red["undefined_share"] <= 0.01
    supported = (red["point"] >= 0.30 and red["ci95"][0] > 0 and coll["ci95"][1] < MARGIN)
    verdict = "inconclusive (A8)" if not powered else "supported" if supported else "not established (killed)"
    pairs = rows[rows.arm == "ALL"]
    win = pairs.common_cutoff - 49
    own = {a: {"unnecessary_hard_brake": float(rows[rows.arm == a].own_unnecessary_hard_brake.mean()),
               "collision": float(rows[rows.arm == a].own_collision.mean()),
               "mean_window_steps": float((rows[rows.arm == a].cross_step.clip(upper=110) - 1 - 49).mean())}
           for a in ("ALL", "PATCH")}
    after = {a: {"brakes_counted_only_after_cutoff": float((rows[rows.arm == a].full_unnecessary_hard_brake
                                                          & ~rows[rows.arm == a].common_unnecessary_hard_brake).mean()),
                 "collisions_counted_only_after_cutoff": float((rows[rows.arm == a].full_collision
                                                               & ~rows[rows.arm == a].common_collision).mean())}
             for a in ("ALL", "PATCH")}
    short = pairs.route_m <= 1.0
    return {"uncensored_reference_matches_registered_H10": True, "effects": res,
            "seed_averaged_ALL_brake_events": all_events, "cities_with_zero_ALL_brakes": zero_city,
            "powered": powered, "verdict": verdict,
            "window": {"pairs": int(len(pairs)), "share_no_scored_segment": float((pairs.common_cutoff < 59).mean()),
                       "mean_steps_after_handoff": float(win.mean()), "median_steps_after_handoff": float(win.median()),
                       "share_full_window": float((pairs.common_cutoff == 109).mean()),
                       "short_routes_le_1m": {"pairs": int(short.sum()),
                                              "mean_steps_after_handoff": float(win[short].mean()) if short.any() else None}},
            "own_crossing_descriptive": own, "counted_after_cutoff": after}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="runs/followups/replay_rows.parquet")
    ap.add_argument("--parity", default="reports/followups/replay_parity.json")
    ap.add_argument("--stage4", default="reports/stage4/results.json")
    ap.add_argument("--stage4-rows", default="runs/stage4/closedloop_pool.parquet")
    ap.add_argument("--out", default="reports/followups/summary.json")
    ap.add_argument("--bootstraps", type=int, default=10_000)
    args = ap.parse_args()
    parity = json.load(open(args.parity))
    if not parity["passed"] or parity["scenarios"] != 8140:
        raise SystemExit("replay parity did not pass: studies B and C must not be analysed")
    rows = pd.read_parquet(args.rows)
    # common_* / own_* exist only for ALL and PATCH rows, so they load as object dtype; make them real booleans
    # there (on object dtype, ~True is -2 and truthy)
    flags = [c for c in rows if c.startswith(("common_", "own_")) and c not in ("common_cutoff",) and "replans" not in c]
    ap_rows = rows.arm.isin(["ALL", "PATCH"])
    assert rows.loc[ap_rows, flags].notna().all().all() and rows.loc[~ap_rows, flags].isna().all().all()
    for c in flags:
        rows[c] = rows[c].astype("boolean")
    registered = json.load(open(args.stage4))["effects"]
    result = {"_note": "EXPLORATORY, pre-registered in docs/preregistration/exploratory-followups.md; cannot change any "
                       "registered Stage 1-4 verdict. 95% within-city scenario-bootstrap intervals (Bonferroni 98.33% "
                       "for three primaries alongside), 10,000 draws, seeds averaged within scenario, cities "
                       "equal-weighted.",
              "replay_parity": parity, "B_departing_stopped_collisions": study_b(rows, args.bootstraps, pd.read_parquet(args.stage4_rows)),
              "C_route_end_censoring": study_c(rows, args.bootstraps, registered)}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({k: v["verdict"] for k, v in result.items() if isinstance(v, dict) and "verdict" in v}))


if __name__ == "__main__":
    main()
