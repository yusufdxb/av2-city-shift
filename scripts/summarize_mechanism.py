"""Rebuild the EXPLORATORY mechanism-audit summary from its row CSVs and write SHA256SUMS.

Reads reports/mechanism/{decisions,blockers,scenes}_seed<S>.csv (written by cityshift.mechanism), recomputes every
summary number with cityshift.mechanism.summarize, checks it against each per-seed run JSON when present, and writes
reports/mechanism/mechanism_audit.json plus reports/mechanism/SHA256SUMS over the rows and the summary.

    PYTHONPATH=src python scripts/summarize_mechanism.py --seeds 0,1,2
"""

import argparse
import hashlib
import json
import math
import os

import pandas as pd

from cityshift.mechanism import LIMITATIONS, NOTE, rows_paths, summarize


def close(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="reports/mechanism")
ap.add_argument("--seeds", default="0")
ap.add_argument("--registered", default="runs/stage4/closedloop_pool.parquet",
                help="registered Stage 4 closed-loop rows, for a scene-level agreement check (skipped if absent)")
args = ap.parse_args()
per_seed, hashed = {}, []
for seed in [int(s) for s in args.seeds.split(",")]:
    paths = rows_paths(args.dir, seed)
    dec, blk, scn = (pd.read_csv(paths[k], dtype={"scenario_id": str, "agent_id": str, "decision_id": str})
                     for k in ("decisions", "blockers", "scenes"))
    summary = summarize(dec, blk, scn)
    run_json = os.path.join(args.dir, f"mechanism_audit_seed{seed}.json")
    if os.path.exists(run_json):
        ran = json.load(open(run_json))["per_seed"][f"seed{seed}"]
        runtime = ran.pop("runtime_sec", None)
        assert close(summary, ran), f"seed {seed}: summary from CSV differs from the run's in-memory summary"
        summary["runtime_sec"] = runtime
        hashed.append(run_json)
    if os.path.exists(args.registered):
        reg = pd.read_parquet(args.registered, columns=["scenario_id", f"ALL_s{seed}_unnecessary_hard_brake",
                                                        f"ALL_s{seed}_collision"])
        m = scn.merge(reg, on="scenario_id", how="left")
        summary["agreement_with_registered_stage4"] = {
            "scenes": int(len(m)),
            "unnecessary_hard_brake_same": int((m["unnecessary_hard_brake"] == m[f"ALL_s{seed}_unnecessary_hard_brake"]).sum()),
            "collision_same": int((m["collision"] == m[f"ALL_s{seed}_collision"]).sum()),
            "registered_unnecessary_hard_brake_scenes": int(m[f"ALL_s{seed}_unnecessary_hard_brake"].fillna(False).astype(bool).sum()),
        }
    per_seed[f"seed{seed}"] = summary
    hashed.extend(paths.values())
n = next(iter(per_seed.values()))["scenes"]
out = {"_note": NOTE.format(n=n, replans=n * 6), "_limitations": LIMITATIONS, "per_seed": per_seed}
summary_path = os.path.join(args.dir, "mechanism_audit.json")
json.dump(out, open(summary_path, "w"), indent=2)
hashed.append(summary_path)
with open(os.path.join(args.dir, "SHA256SUMS"), "w") as f:
    for p in hashed:
        f.write(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {os.path.basename(p)}\n")
print(json.dumps(out, indent=2))
