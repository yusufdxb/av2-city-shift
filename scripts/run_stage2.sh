#!/usr/bin/env bash
# Stage 2 confirmatory closed-loop run on val, then the pre-registered analysis.
set -euo pipefail
RAW=${RAW:-$HOME/datasets/av2/motion-forecasting/val}
EVALS=${EVALS:-evals}
mkdir -p "$EVALS"
PYTHONPATH=src python3 -m cityshift.closedloop --raw "$RAW" --arms log,oracle,cv,static,ALL,LOCO --out "$EVALS/stage2_closedloop.parquet" --chunk 1000
PYTHONPATH=src python3 -m cityshift.analysis_stage2 --parquet "$EVALS/stage2_closedloop.parquet" --out "$EVALS/stage2_results.json"
