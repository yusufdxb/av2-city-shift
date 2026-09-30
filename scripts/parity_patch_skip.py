"""Real-scene parity: PATCH vs PATCH with stopped agents skipping inference must plan identically.

Skipping is an optimisation (stopped agents get constant-velocity forecasts either way), so every chosen acceleration
and every outcome should match exactly. Runs the serving path on dev-slice TRAIN scenes with the PyTorch FP32 backend.

    PYTHONPATH=src python scripts/parity_patch_skip.py --raw <train dir> --n 500
"""

import argparse
import json
import os
import sys

import numpy as np

from cityshift.closedloop import load_scene
from cityshift.serving import Backend, StageTimer
from cityshift.serving_policies import run_scenario

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--out", default="reports/serving/parity_patch_skip.json")
args = ap.parse_args()
ids = sorted(np.load("runs/dev_scenarios.npy", allow_pickle=True).tolist())[: args.n]
backend = Backend("pytorch-fp32", "runs/ALL/seed0/model.pt", None)
replans = accel_diff = outcome_diff = inferred_full = inferred_skip = nan_progress = 0
for sid in ids:
    sc = load_scene(os.path.join(args.raw, sid))
    full_score, full_trace = run_scenario(sc, backend, StageTimer(True), "PATCH")
    skip_score, skip_trace = run_scenario(sc, backend, StageTimer(True), "PATCH-skip-inference")
    for a, b in zip(full_trace, skip_trace):
        replans += 1
        accel_diff += int(a["accel"] != b["accel"])
        inferred_full += a.get("inferred_agents", 0)
        inferred_skip += b.get("inferred_agents", 0)
    same = lambda a, b: (a == b) or (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b))  # noqa: E731
    nan_progress += int(isinstance(full_score["progress"], float) and bool(np.isnan(full_score["progress"])))
    outcome_diff += int(not all(same(full_score[k], skip_score[k]) for k in ("collision", "unnecessary_hard_brake", "progress")))
result = {"scenarios": len(ids), "replans": replans, "acceleration_differs": accel_diff, "outcome_differs": outcome_diff,
          "inferred_agents_patch": inferred_full, "inferred_agents_skip": inferred_skip,
          "scenarios_with_undefined_progress": nan_progress}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(result, open(args.out, "w"), indent=2)
print(json.dumps(result, indent=2))
sys.exit(0 if accel_diff == 0 and outcome_diff == 0 else 1)
