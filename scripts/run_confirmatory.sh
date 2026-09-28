#!/usr/bin/env bash
# Confirmatory runs, exactly as pre-registered. Usage: scripts/run_confirmatory.sh <N_TRAIN>
set -euo pipefail
ROOT=${ROOT:-$HOME/datasets/av2/cityshift_pp}
RUNS=${RUNS:-runs}
N=$1
STEPS=${STEPS:-60000}
CITIES="austin dearborn miami palo-alto pittsburgh washington-dc"
train() {  # name seed extra-args...
  local name=$1 seed=$2; shift 2
  local out=$RUNS/$name/seed$seed
  if [ -f "$out/model.pt" ]; then echo "skip $out"; return; fi
  PYTHONPATH=src python3 -m cityshift.train --root "$ROOT" --out "$out" --n-train "$N" --steps "$STEPS" --seed "$seed" "$@" \
    > "$out.log" 2>&1 || { mkdir -p "$out"; echo "FAILED $out" | tee -a "$RUNS/failures.txt"; }
  echo "finished $out $(tail -1 "$out.log")"
}
mkdir -p "$RUNS"
for seed in 0 1 2; do
  mkdir -p "$RUNS/ALL"; train ALL $seed
  for c in $CITIES; do mkdir -p "$RUNS/LOCO-$c"; train "LOCO-$c" $seed --exclude-city "$c"; done
done
mkdir -p "$RUNS/NOMAP"; train NOMAP 0 --no-map
echo ALL_RUNS_DONE
