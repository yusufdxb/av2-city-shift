"""Independent recomputation of the Stage 3a (H5, H6a, H6b) and Stage 3 (H7, H8, H9) point estimates: plain
pandas/numpy, no imports from cityshift. Usage, from the repository root after unpacking the per-row tables
attached to GitHub release v1.1: python scripts/audit_stage3.py [root]
Exits non-zero if a checksum fails or any recomputed value differs from the committed report by more than 1e-9."""

import hashlib
import json
import sys

import numpy as np
import pandas as pd

R = (sys.argv[1] if len(sys.argv) > 1 else ".").rstrip("/") + "/"
CITIES = ["austin", "dearborn", "miami", "palo-alto", "pittsburgh", "washington-dc"]
TOL = 1e-9
bad = []

# checksums first: a mismatched table would make every comparison below meaningless
for line in open(R + "reports/stage3/SHA256SUMS"):
    want, path = line.split()
    h = hashlib.sha256()
    with open(R + path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    ok = h.hexdigest() == want
    print(f"sha256 {'OK  ' if ok else 'FAIL'} {path}")
    if not ok:
        sys.exit(1)


def check(name, recomputed, reported):
    diff = abs(recomputed - reported)
    if not diff <= TOL:
        bad.append(name)
    print(f"{name:36s} recomputed {recomputed: .12f} reported {reported: .12f} diff {diff:.1e}")


# Stage 3a: one row per agent and predictor
s3a = json.load(open(R + "reports/stage3a/results.json"))
P = pd.read_parquet(R + "evals/stage3a/per_agent.parquet",
                    columns=["scenario_id", "agent_id", "city", "focal", "planner_relevant", "speed", "predictor", "miss"])
keys = ["scenario_id", "agent_id"]
agents = P[P.predictor == "CV"].drop(columns=["predictor", "miss"]).set_index(keys).sort_index()
wide = P[P.predictor.isin(["CV", "LANE", "ALL_s0", "ALL_s1", "ALL_s2"])].pivot(index=keys, columns="predictor", values="miss")
wide = wide.reindex(agents.index)
assert not wide.isna().any().any(), "every agent needs CV, LANE and three ALL rows"
all_miss = wide[["ALL_s0", "ALL_s1", "ALL_s2"]].mean(1)
planner = agents.planner_relevant
nonfocal = planner & ~agents.focal
h5, h6a, h6b = [], [], []
for c in CITIES:
    city = agents.city == c
    m = planner & city
    h5.append(1 - all_miss[m].mean() / wide.LANE[m].mean())
    m = nonfocal & city & (agents.speed < 0.5)
    h6a.append(all_miss[m].mean() - wide.CV[m].mean())
    m = nonfocal & city & (agents.speed >= 2.0)
    h6b.append(1 - all_miss[m].mean() / wide.CV[m].mean())
print(f"Stage 3a: {len(agents)} agents, {int(planner.sum())} planner-relevant")
check("H5 pooled reduction ALL vs LANE", np.mean(h5), s3a["H5"]["pooled_reduction"])
check("H5 LANE misses (count)", float(wide.LANE[planner].sum()), s3a["H5"]["lane_misses"])
check("H6a pooled ALL - CV (stopped)", np.mean(h6a), s3a["H6a"]["pooled_difference_ALL_minus_CV"])
check("H6b pooled reduction vs CV (moving)", np.mean(h6b), s3a["H6b"]["pooled_reduction_ALL_vs_CV"])
del P, wide

# Stage 3 closed loop: one row per scenario, seed-averaged per arm
s3 = json.load(open(R + "reports/stage3/results.json"))
cl = s3["closedloop"]
D = pd.read_parquet(R + "runs/stage3/closedloop_val.parquet")


def seeds(arm, field):
    return D[[f"{arm}_s{s}_{field}" for s in range(3)]].astype(float).mean(1)


brake = pd.DataFrame({a: seeds(a, "unnecessary_hard_brake") for a in ("ALL", "MULTI", "PATCH", "SHAM")}).groupby(D.city).mean()
coll = pd.DataFrame({a: seeds(a, "collision") for a in ("ALL", "MULTI", "PATCH")}).groupby(D.city).mean()
brake, coll = brake.loc[CITIES], coll.loc[CITIES]
print(f"Stage 3 closed loop: {len(D)} scenarios")
for h, arm in (("H7", "MULTI"), ("H8", "PATCH")):
    check(f"{h} brake reduction {arm} vs ALL", ((brake.ALL - brake[arm]) / brake.ALL).mean(), cl[h]["brake_reduction"])
    check(f"{h} collision diff {arm} - ALL", (coll[arm] - coll.ALL).mean(), cl[h]["collision_difference"])
check("SHAM brake reduction vs ALL", ((brake.ALL - brake.SHAM) / brake.ALL).mean(), cl["controls"]["SHAM_brake_reduction"])
check("STATIC / ALL collision", D.static_collision.mean() / seeds("ALL", "collision").mean(), cl["controls"]["STATIC_over_ALL"])
check("LOG collision", D.log_collision.mean(), cl["controls"]["LOG_collision"])

# Stage 3 open loop (H9): one row per agent and predictor
OL = pd.read_parquet(R + "runs/stage3/openloop_val.parquet",
                    columns=["scenario_id", "agent_id", "city", "focal", "planner_relevant", "speed", "predictor", "miss"])
meta = OL[OL.predictor == "ALL_s0"].drop(columns=["predictor", "miss"]).set_index(keys).sort_index()
w = OL.pivot(index=keys, columns="predictor", values="miss").reindex(meta.index)
assert not w.isna().any().any(), "every agent needs three ALL and three MULTI rows"
gap = w[[f"MULTI_s{s}" for s in range(3)]].mean(1) - w[[f"ALL_s{s}" for s in range(3)]].mean(1)
stopped = meta.planner_relevant & ~meta.focal & (meta.speed < 0.5)
print(f"Stage 3 open loop: {len(meta)} agents")
check("H9 stopped agents (count)", float(stopped.sum()), s3["H9"]["n_stopped_agents"])
check("H9 MULTI - ALL miss, stopped", np.mean([gap[stopped & (meta.city == c)].mean() for c in CITIES]),
      s3["H9"]["stopped_nonfocal_difference"])
check("H9 MULTI - ALL miss, focal", np.mean([gap[meta.focal & (meta.city == c)].mean() for c in CITIES]),
      s3["H9"]["focal_difference"])

if bad:
    print("MISMATCH:", ", ".join(bad))
    sys.exit(1)
print("ALL MATCH (tolerance 1e-9)")
