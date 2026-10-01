"""EXPLORATORY study D: every registered Stage 3 and Stage 4 closed-loop component under route-end censored scoring.

Registered in docs/preregistration/exploratory-followups.md (study D). For each registered comparison, every arm in it
is scored only up to the last step on which all its drives (same scenario and seed) are still on the human's logged
route (cityshift.route_censor). The censored outcomes replace that comparison's columns in the registered table, the
stage's own registered analysis code runs unchanged, and the verdict is recomputed with the registered rules, with
controls (STATIC vs ALL) in their own common window. Positive controls: with every cutoff at t=109 the pipeline must
reproduce the registered results exactly, and the H10 point must equal study C's.

    PYTHONPATH=src python scripts/analyze_censored.py
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

from cityshift import analysis_stage3 as a3
from cityshift import analysis_stage4 as a4
from cityshift.closedloop import END
from cityshift.route_censor import score_on_route, window_cutoff

STAGE4 = {"H10": ("PATCH",), "H11": ("PATCH", "SHAM2"), "H12": ("TRIM",), "H13": ("MIX",)}
STAGE3 = {"H7": ("MULTI",), "H8": ("PATCH",), "SHAM": ("SHAM",)}


class Drives:
    """Replayed primitives for one stage, aligned to the registered table's scenario order."""

    def __init__(self, rows: pd.DataFrame, order: pd.Series):
        self.order = order.astype(str).to_numpy()
        self.by = {}
        for (arm, seed), g in rows.groupby(["arm", "model_seed"]):
            g = g.set_index("scenario_id").loc[self.order]
            self.by[(arm, seed)] = {"cross": g.cross_step.to_numpy(int), "fault": g.fault_step.to_numpy(int),
                                    "exec": np.stack(g.exec_decel.to_numpy()), "log": np.stack(g.log_decel.to_numpy())}

    def score(self, arm: str, seed: int, cutoff: np.ndarray) -> dict[str, np.ndarray]:
        d = self.by[(arm, seed)]
        return score_on_route(d["exec"], d["log"], d["fault"], cutoff)

    def cross(self, arm: str, seed: int) -> np.ndarray:
        return self.by[(arm, seed)]["cross"]


def censor(table: pd.DataFrame, drives: Drives, arms: tuple[str, ...], full: bool = False) -> tuple[pd.DataFrame, dict]:
    """Replace ALL's and `arms`' seed-level outcome columns with scores in their common per-seed window."""
    out = table.copy()
    lengths = []
    for seed in range(3):
        group = ("ALL", *arms)
        cutoff = np.full(len(table), END) if full else window_cutoff(
            np.stack([drives.cross(a, seed) for a in group], -1))
        lengths.append(cutoff)
        for a in group:
            r = drives.score(a, seed, cutoff)
            for k in ("unnecessary_hard_brake", "collision", "planner_hard_brake", "logged_hard_brake"):
                out[f"{a}_s{seed}_{k}"] = r[k]
    c = np.stack(lengths)
    return out, {"mean_steps_after_handoff": float((c - 49).mean()), "share_no_scored_segment": float((c < 59).mean()),
                 "share_full_window": float((c == END).mean())}


def control_window(table: pd.DataFrame, drives: Drives, full: bool = False) -> pd.DataFrame:
    """STATIC and ALL seeds 0-2 in one window per scenario (the minimum over all four drives)."""
    out = table.copy()
    cross = np.stack([drives.cross("ALL", s) for s in range(3)] + [drives.cross("static", -1)], -1)
    cutoff = np.full(len(table), END) if full else window_cutoff(cross)
    for s in range(3):
        out[f"ALL_s{s}_collision"] = drives.score("ALL", s, cutoff)["collision"]
    out["static_collision"] = drives.score("static", -1, cutoff)["collision"]
    return out


def close(a, b, tol=1e-12) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= tol
    return a == b


# ------------------------------------------------------------------ Stage 4

def stage4(rows: pd.DataFrame, table: pd.DataFrame, focal: pd.DataFrame, registered: dict, n_boot: int) -> dict:
    drives = Drives(rows, table.scenario_id)
    # positive control 1: cutoff t=109 everywhere reproduces the registered analysis exactly
    full = table.copy()
    for arm in ("ALL", "PATCH", "TRIM", "SHAM2", "MIX"):
        full, _ = censor(full, drives, (arm,), full=True)
    full = control_window(full, drives, full=True)
    ref = a4.analyze(full, focal, n_boot)
    for k in ("effects", "claims", "controls"):
        if not close(json.loads(json.dumps(ref[k], default=a4._plain)), registered[k]):
            raise SystemExit(f"Stage 4 uncensored reference differs from the registered {k}")
    ctrl_table = control_window(table, drives)
    all_col = float(ctrl_table[[f"ALL_s{s}_collision" for s in range(3)]].to_numpy(float).mean(1).mean())
    static_col = float(ctrl_table.static_collision.mean())
    log_col = float(table.log_collision.mean())
    controls = {"STATIC_over_ALL": static_col / all_col if all_col > 0 else None,
                "STATIC_pass": all_col > 0 and static_col >= 2 * all_col, "LOG_collision": log_col,
                "LOG_pass": log_col < 0.01}
    interpretable = controls["STATIC_pass"] and controls["LOG_pass"]
    out = {}
    for label, arms in STAGE4.items():
        t, window = censor(table, drives, arms)
        res = a4.analyze(t, focal, n_boot)
        eff = res["effects"]
        events = float(t[[f"ALL_s{s}_unnecessary_hard_brake" for s in range(3)]].to_numpy(float).mean(1).sum())
        if label == "H11":
            comp = {"sham_gap": eff["sham_gap"]}
            ok = comp["sham_gap"]["ci"][0] is not None
            passed = ok and a4.decide_gap(comp["sham_gap"]["point"], comp["sham_gap"]["ci"])
        else:
            arm = arms[0]
            comp = {f"{arm}_reduction": eff[f"{arm}_reduction"], f"{arm}_collision_diff": eff[f"{arm}_collision_diff"]}
            red, col = comp[f"{arm}_reduction"], comp[f"{arm}_collision_diff"]
            ok = red["ci"][0] is not None and np.isfinite(red["point"])
            passed = ok and a4.decide_reduction(red["point"], red["ci"]) and a4.decide_noninferiority(col["ci"], 0.003)
            if label == "H13":  # open-loop focal miss is unaffected by censoring: carried over
                comp["focal_miss_diff"] = registered["effects"]["focal_miss_diff"]
                passed = passed and a4.decide_noninferiority(comp["focal_miss_diff"]["ci"], 0.02)
        status = ("underpowered" if events < 100 else "inconclusive" if not ok else
                  "uninterpretable" if not interpretable else "supported" if passed else "killed")
        if label == "H11" and status in ("supported", "killed") and not registered["controls"]["dose_dose_pass"]:
            status = "uninterpretable (SHAM2 realised dose below 99% of PATCH)"
        reg_status = registered["claims"][label]["status"]
        out[label] = {"censored": comp, "censored_status": status, "registered_status": reg_status,
                      "flip": status != reg_status, "seed_averaged_ALL_brake_events": events, "window": window,
                      "registered": {k: registered["effects"][k] for k in comp}}
    return {"uncensored_reference_reproduces_registered": True, "controls_censored": controls, "components": out}


# ------------------------------------------------------------------ Stage 3

def stage3_status(c: dict, controls: dict, sham_ok: bool, h8: bool) -> str:
    """The registered Stage 3 status rules (analysis_stage3.closedloop_analysis), on censored inputs."""
    ci_ok = not c["undefined_cities"] and c["undefined_bootstrap_share"] <= 0.01
    passed = (not c["underpowered"] and ci_ok and a3.decide_reduction(c["brake_reduction"], c["brake_ci"], 0.30)
              and a3.decide_noninferiority(c["collision_difference"], c["collision_ci"]))
    if c["underpowered"]:
        return "underpowered"
    if not ci_ok:
        return "inconclusive"
    if not (controls["STATIC_pass"] and controls["LOG_pass"]):
        return "uninterpretable"
    if h8 and not sham_ok and passed:
        return "reduction real but not stopped-specific (dose-matched sham matched it)"
    return "supported" if passed and (sham_ok or not h8) else "killed"


def sham_reduction(t: pd.DataFrame) -> float:
    brake, _ = a3._closed_arrays(t)
    cities = t.city.astype(str).to_numpy()
    vals = []
    for c in a3.CITIES:
        idx = cities == c
        all_b, sham_b = brake["ALL"][idx].mean(), brake["SHAM"][idx].mean()
        if all_b > 0:
            vals.append((all_b - sham_b) / all_b)
    return float(np.mean(vals))


def stage3(rows: pd.DataFrame, table: pd.DataFrame, registered: dict, n_boot: int) -> dict:
    drives = Drives(rows, table.scenario_id)
    full = table.copy()
    for arm in ("ALL", "MULTI", "PATCH", "SHAM"):
        full, _ = censor(full, drives, (arm,), full=True)
    full = control_window(full, drives, full=True)
    ref = a3.closedloop_analysis(full, np.random.default_rng(3407), n_boot)
    if not close(json.loads(json.dumps(ref, default=a4._plain)), registered):
        raise SystemExit("Stage 3 uncensored reference differs from the registered closed-loop results")
    ctrl = control_window(table, drives)
    _, col = a3._closed_arrays(ctrl)
    all_col, static_col = float(col["ALL"].mean()), float(col["static"].mean())
    log_col = float(table.log_collision.mean())
    controls = {"STATIC_over_ALL": static_col / all_col if all_col else None,
                "STATIC_pass": static_col >= 2 * all_col and all_col > 0, "LOG_collision": log_col,
                "LOG_pass": log_col < 0.01}
    tables = {label: censor(table, drives, arms) for label, arms in STAGE3.items()}
    results = {label: a3.closedloop_analysis(t, np.random.default_rng(3407), n_boot) for label, (t, _) in tables.items()}
    sham_red = sham_reduction(tables["SHAM"][0])
    patch_red = results["H8"]["H8"]["brake_reduction"]
    sham_ok = patch_red is not None and sham_red < 0.5 * patch_red
    out = {}
    for label in ("H7", "H8"):
        c = results[label][label]
        keep = ("brake_reduction", "brake_ci", "collision_difference", "collision_ci", "underpowered",
                "undefined_cities", "undefined_bootstrap_share")
        status = stage3_status(c, controls, sham_ok, label == "H8")
        out[label] = {"censored": {k: c[k] for k in keep}, "censored_status": status,
                      "registered_status": registered[label]["status"],
                      "flip": status != registered[label]["status"],
                      "registered": {k: registered[label][k] for k in keep}, "window": tables[label][1]}
    out["SHAM (descriptive)"] = {"censored_brake_reduction": sham_red, "window": tables["SHAM"][1],
                                 "registered_brake_reduction": registered["controls"]["SHAM_brake_reduction"],
                                 "censored_SHAM_pass": sham_ok}
    return {"uncensored_reference_reproduces_registered": True, "controls_censored": controls, "components": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows4", default="runs/followups/replay_rows.parquet")
    ap.add_argument("--rows3", default="runs/followups/replay_rows_stage3.parquet")
    ap.add_argument("--out", default="reports/censored/summary.json")
    ap.add_argument("--bootstraps", type=int, default=10_000)
    args = ap.parse_args()
    for p in ("reports/followups/replay_parity.json", "reports/followups/replay_parity_stage3.json"):
        if not json.load(open(p))["passed"]:
            raise SystemExit(f"{p} did not pass: study D must not be analysed")
    s4 = stage4(pd.read_parquet(args.rows4), pd.read_parquet("runs/stage4/closedloop_pool.parquet"),
                pd.read_parquet("runs/stage4/openloop_pool.parquet"), json.load(open("reports/stage4/results.json")),
                args.bootstraps)
    c_point = json.load(open("reports/followups/summary.json"))["C_route_end_censoring"]["effects"]["PATCH_reduction"]["point"]
    if abs(s4["components"]["H10"]["censored"]["PATCH_reduction"]["point"] - c_point) > 1e-12:
        raise SystemExit("study D's H10 point differs from study C's primary point")
    s3 = stage3(pd.read_parquet(args.rows3), pd.read_parquet("runs/stage3/closedloop_val.parquet"),
                json.load(open("reports/stage3/results.json"))["closedloop"], args.bootstraps)
    result = {"_note": "EXPLORATORY study D (docs/preregistration/exploratory-followups.md): registered closed-loop "
                       "components rescored only while every compared drive is on the human's logged route. Registered "
                       "estimators, CI levels, decision rules and seeds; verdicts stay as registered.",
              "H10_point_equals_study_C": True, "stage4": s4, "stage3": s3}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=a4._plain)
        f.write("\n")
    flips = {f"S{n}:{k}": (v["registered_status"], v["censored_status"]) for n, s in ((4, s4), (3, s3))
             for k, v in s["components"].items() if "flip" in v}
    print(json.dumps(flips))


if __name__ == "__main__":
    main()
