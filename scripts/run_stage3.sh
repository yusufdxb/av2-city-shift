#!/usr/bin/env bash
# Stage 3 full compute. Usage: scripts/run_stage3.sh RAW_TRAIN RAW_VAL FOCAL_ROOT MULTI_ROOT
set -euo pipefail
if [ "$#" -ne 4 ]; then
  echo "usage: scripts/run_stage3.sh RAW_TRAIN RAW_VAL FOCAL_ROOT MULTI_ROOT" >&2
  exit 2
fi
RAW_TRAIN=$1
RAW_VAL=$2
FOCAL_ROOT=$3
MULTI_ROOT=$4
for path in "$RAW_TRAIN" "$RAW_VAL" "$FOCAL_ROOT/train/meta.parquet" runs/dev_scenarios.npy; do
  if [ ! -e "$path" ]; then echo "missing required input: $path" >&2; exit 1; fi
done
mkdir -p runs/MULTI
PYTHONPATH=src python3 -m cityshift.preprocess_multi --raw "$RAW_TRAIN" --focal-root "$FOCAL_ROOT" --out "$MULTI_ROOT"
for seed in 0 1 2; do
  out="runs/MULTI/seed${seed}"
  mkdir -p "$out"
  PYTHONPATH=src python3 -m cityshift.train_multi --root "$MULTI_ROOT" --focal-root "$FOCAL_ROOT" \
    --out "$out" --n-train 128000 --steps 60000 --batch 128 --lr 0.001 --seed "$seed" \
    > "$out/train_stdout.log" 2>&1 || { echo "MULTI training failed: seed $seed" >&2; exit 1; }
  test -s "$out/model.pt" || { echo "missing checkpoint: $out/model.pt" >&2; exit 1; }
  PYTHONPATH=src python3 - "$out/model.pt" <<'PY_CHECK'
import sys
import torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["model"]
if any(value.is_floating_point() and not torch.isfinite(value).all() for value in state.values()):
    raise SystemExit(f"non-finite MULTI checkpoint: {sys.argv[1]}")
PY_CHECK
done
mkdir -p runs/stage3
PYTHONPATH=src python3 -m cityshift.multiagent_eval --root "$FOCAL_ROOT" --raw "$RAW_VAL" --split val \
  --checkpoint ALL_s0=runs/ALL/seed0/model.pt --checkpoint ALL_s1=runs/ALL/seed1/model.pt \
  --checkpoint ALL_s2=runs/ALL/seed2/model.pt --checkpoint MULTI_s0=runs/MULTI/seed0/model.pt \
  --checkpoint MULTI_s1=runs/MULTI/seed1/model.pt --checkpoint MULTI_s2=runs/MULTI/seed2/model.pt \
  --out runs/stage3/openloop_val.parquet
PYTHONPATH=src python3 -m cityshift.closedloop_v2 --raw "$RAW_VAL" --out runs/stage3/closedloop_val.parquet \
  --arms log,oracle,cv,static,ALL,MULTI,PATCH,SHAM
PYTHONPATH=src python3 -m cityshift.analysis_stage3 --closedloop runs/stage3/closedloop_val.parquet \
  --openloop runs/stage3/openloop_val.parquet --out runs/stage3/results.json

python3 - <<'PY_CONTROLS'
import json
result = json.load(open("runs/stage3/results.json"))
controls = result["closedloop"]["controls"]
failed = [name for name in ("STATIC_pass", "LOG_pass", "SHAM_pass") if not controls[name]]
if failed:
    raise SystemExit(f"Stage 3 controls failed: {', '.join(failed)}; inspect runs/stage3/results.json")
PY_CONTROLS
