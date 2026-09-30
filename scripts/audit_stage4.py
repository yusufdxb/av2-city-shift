"""Independent recomputation of the Stage 4 point estimates (H10 to H13) and controls.

Plain pandas/numpy, imports nothing from cityshift. Usage (after unpacking release v1.2 in the repo root):
    python scripts/audit_stage4.py
Checks reports/stage4/SHA256SUMS first; exits non-zero on any mismatch larger than 1e-9.
"""

import hashlib
import json
import sys

import numpy as np
import pandas as pd

TOL = 1e-9
ok = True
for line in open("reports/stage4/SHA256SUMS"):
    digest, path = line.split()
    actual = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(("sha256 OK   " if actual == digest else "sha256 FAIL ") + path)
    ok &= actual == digest
if not ok:
    sys.exit(1)

rep = json.load(open("reports/stage4/results.json"))
D = pd.read_parquet("runs/stage4/closedloop_pool.parquet")
cities = sorted(D.city.unique())


def seed_mean(arm: str, metric: str) -> pd.Series:
    return D[[f"{arm}_s{i}_{metric}" for i in range(3)]].astype(float).mean(axis=1)


brake = {a: seed_mean(a, "unnecessary_hard_brake") for a in ("ALL", "PATCH", "TRIM", "SHAM2", "MIX")}
coll = {a: seed_mean(a, "collision") for a in brake}


def reduction(arm: str) -> float:
    return float(np.mean([1 - brake[arm][D.city == c].mean() / brake["ALL"][D.city == c].mean() for c in cities]))


def coll_diff(arm: str) -> float:
    return float(np.mean([coll[arm][D.city == c].mean() - coll["ALL"][D.city == c].mean() for c in cities]))


openloop = pd.read_parquet("runs/stage4/openloop_pool.parquet")
openloop = openloop[openloop.focal].assign(arm=lambda f: f.predictor.str.split("_").str[0])
focal = openloop.groupby(["city", "scenario_id", "agent_id", "arm"]).miss.mean().unstack()
checks = {
    "H10 PATCH brake reduction": (reduction("PATCH"), rep["effects"]["PATCH_reduction"]["point"]),
    "H10 PATCH collision diff": (coll_diff("PATCH"), rep["effects"]["PATCH_collision_diff"]["point"]),
    "H11 SHAM2 brake reduction": (reduction("SHAM2"), rep["effects"]["SHAM2_reduction"]["point"]),
    "H11 PATCH minus SHAM2 gap": (reduction("PATCH") - reduction("SHAM2"), rep["effects"]["sham_gap"]["point"]),
    "H12 TRIM brake reduction": (reduction("TRIM"), rep["effects"]["TRIM_reduction"]["point"]),
    "H12 TRIM collision diff": (coll_diff("TRIM"), rep["effects"]["TRIM_collision_diff"]["point"]),
    "H13 MIX brake reduction": (reduction("MIX"), rep["effects"]["MIX_reduction"]["point"]),
    "H13 MIX collision diff": (coll_diff("MIX"), rep["effects"]["MIX_collision_diff"]["point"]),
    "H13 MIX - ALL focal miss": (float(np.mean([(focal.MIX - focal.ALL)[focal.index.get_level_values("city") == c].mean()
                                                for c in cities])), rep["effects"]["focal_miss_diff"]["point"]),
    "ALL brake events": (float(brake["ALL"].sum()), rep["controls"]["ALL_brake_events"]),
    "STATIC / ALL collision": (float(D.static_collision.mean() / coll["ALL"].mean()), rep["controls"]["STATIC_over_ALL"]),
    "LOG collision": (float(D.log_collision.mean()), rep["controls"]["LOG_collision"]),
}
dose = [(D[f"PATCH_s{s}_dose_t{t}"].sum(), D[f"SHAM2_s{s}_dose_t{t}"].sum()) for s in range(3) for t in (49, 59, 69, 79, 89, 99)]
checks["SHAM2 / PATCH dose"] = (sum(b for _, b in dose) / sum(a for a, _ in dose), rep["controls"]["dose_SHAM2_over_PATCH"])
for name, (mine, theirs) in checks.items():
    diff = abs(mine - theirs)
    print(f"{name:32s} recomputed {mine: .12f} reported {theirs: .12f} diff {diff:.1e}")
    ok &= diff <= TOL
print("ALL MATCH (tolerance 1e-9)" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
