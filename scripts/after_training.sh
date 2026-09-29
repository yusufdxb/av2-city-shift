#!/usr/bin/env bash
# Waits for the confirmatory training batch, then runs every pre-registered evaluation once.
set -uo pipefail
until grep -q ALL_RUNS_DONE runs/confirmatory.log; do sleep 60; done
if [ -s runs/failures.txt ]; then echo "TRAINING FAILURES PRESENT, stopping"; cat runs/failures.txt; exit 1; fi
PYTHONPATH=src python3 -m cityshift.validate_runs runs || { echo "RUN VALIDATION FAILED, stopping (rerun the bad seeds once, per the registration)"; exit 1; }
echo "== stage 1 evaluation"; scripts/evaluate_all.sh && echo STAGE1_DONE || echo STAGE1_FAILED
echo "== stage 2 closed loop"; scripts/run_stage2.sh && echo STAGE2_DONE || echo STAGE2_FAILED
echo "== deployment"; PYTHONPATH=src python3 -m cityshift.export_trt --root "$HOME/datasets/av2/cityshift_pp" --ckpt runs/ALL/seed0/model.pt --out evals/deploy --n 2000 && echo DEPLOY_DONE || echo DEPLOY_FAILED
echo PIPELINE_DONE
