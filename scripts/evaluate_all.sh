#!/usr/bin/env bash
# Score every arm on val, then run the pre-registered analysis.
set -euo pipefail
ROOT=${ROOT:-$HOME/datasets/av2/cityshift_pp}
RUNS=${RUNS:-runs}
EVALS=${EVALS:-evals}
mkdir -p "$EVALS"
ev() { PYTHONPATH=src python3 -m cityshift.evaluate --root "$ROOT" "$@"; }
ev --ckpts $RUNS/ALL/seed{0,1,2}/model.pt --out $EVALS/ALL.parquet
ev --ckpts $RUNS/ALL/seed{0,1,2}/model.pt --out $EVALS/ALL_mapswap.parquet --map-swap
for c in austin dearborn miami palo-alto pittsburgh washington-dc; do
  ev --ckpts $RUNS/LOCO-$c/seed{0,1,2}/model.pt --out $EVALS/LOCO-$c.parquet
done
ev --ckpts $RUNS/NOMAP/seed0/model.pt --out $EVALS/NOMAP.parquet
PYTHONPATH=src python3 -m cityshift.analysis --evals "$EVALS"
