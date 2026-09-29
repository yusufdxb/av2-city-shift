#!/usr/bin/env bash
# Stage 4 full compute. Usage: scripts/run_stage4.sh RAW_TRAIN FOCAL_ROOT MULTI_ROOT
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "usage: scripts/run_stage4.sh RAW_TRAIN FOCAL_ROOT MULTI_ROOT" >&2
  exit 2
fi
RAW_TRAIN=$1
FOCAL_ROOT=$2
MULTI_ROOT=$3
POOL=runs/replication_pool.npy
for path in "$RAW_TRAIN" "$FOCAL_ROOT/train/meta.parquet" "$MULTI_ROOT/train/meta.parquet" \
  "$MULTI_ROOT/dev/meta.parquet" runs/dev_scenarios.npy "$POOL"; do
  if [ ! -e "$path" ]; then echo "missing required input: $path" >&2; exit 1; fi
done
for seed in 0 1 2; do
  if [ ! -s "runs/ALL/seed${seed}/model.pt" ]; then
    echo "missing ALL checkpoint seed $seed" >&2
    exit 1
  fi
done
PYTHONPATH=src python3 - "$FOCAL_ROOT" "$POOL" <<'PY_CHECK'
import sys
import numpy as np
from cityshift.closedloop_v3 import POOL_SHA256, checked_ids
from cityshift.data import Split, dev_indices

root, pool_path = sys.argv[1:]
ids = set(checked_ids(pool_path, POOL_SHA256))
meta = Split(root, "train").meta
fixed_dev = set(meta.scenario_id.iloc[dev_indices(meta, 0.02)].astype(str))
if ids & fixed_dev:
    raise SystemExit("replication pool overlaps TRAIN-dev scenarios")
for seed in range(3):
    index = np.load(f"runs/ALL/seed{seed}/train_indices.npy")
    trained = set(meta.scenario_id.iloc[index].astype(str))
    if ids & trained:
        raise SystemExit(f"replication pool overlaps ALL seed {seed} training")
print(f"verified {len(ids)} unique pool IDs, dev exclusion and ALL training exclusion")
PY_CHECK
mkdir -p runs/stage4/MIX
for seed in 0 1 2; do
  out="runs/stage4/MIX/seed${seed}"
  if [ -e "$out/model.pt" ]; then echo "checkpoint already exists: $out/model.pt" >&2; exit 1; fi
  mkdir -p "$out"
  PYTHONPATH=src python3 -m cityshift.train_mix --focal-root "$FOCAL_ROOT" --multi-root "$MULTI_ROOT" \
    --root "$FOCAL_ROOT" --out "$out" --n-train 128000 --steps 60000 --batch 128 \
    --lr 0.001 --seed "$seed" > "$out/train_stdout.log" 2>&1 || {
      echo "MIX training failed: seed $seed; inspect $out/train_stdout.log" >&2
      exit 1
    }
  test -s "$out/model.pt" || { echo "missing MIX checkpoint: $out/model.pt" >&2; exit 1; }
  PYTHONPATH=src python3 - "$out/model.pt" <<'PY_FINITE'
import sys
import torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["model"]
if any(value.is_floating_point() and not torch.isfinite(value).all() for value in state.values()):
    raise SystemExit(f"non-finite MIX checkpoint: {sys.argv[1]}")
PY_FINITE
done
for path in runs/stage4/openloop_pool.parquet runs/stage4/closedloop_pool.parquet \
  runs/stage4/risk_pool.parquet runs/stage4/results.json; do
  if [ -e "$path" ]; then echo "refusing to overwrite existing Stage 4 result: $path" >&2; exit 1; fi
done
PYTHONPATH=src python3 -m cityshift.stage4_focal_eval --root "$FOCAL_ROOT" --raw "$RAW_TRAIN" \
  --scenarios "$POOL" --out runs/stage4/openloop_pool.parquet
PYTHONPATH=src python3 -m cityshift.closedloop_v3 --raw "$RAW_TRAIN" --scenarios "$POOL" \
  --out runs/stage4/closedloop_pool.parquet --risk-out runs/stage4/risk_pool.parquet
PYTHONPATH=src python3 -m cityshift.analysis_stage4 --closedloop runs/stage4/closedloop_pool.parquet \
  --openloop runs/stage4/openloop_pool.parquet --risk runs/stage4/risk_pool.parquet \
  --pool "$POOL" --out runs/stage4/results.json
