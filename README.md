# City Shift

[![CI](https://github.com/yusufdxb/av2-city-shift/actions/workflows/ci.yml/badge.svg)](https://github.com/yusufdxb/av2-city-shift/actions/workflows/ci.yml)

**Does a learned trajectory predictor get worse in a city it has never seen, does it know when it is wrong there, and does any of that reach the car's driving?**

A pre-registered study on the [Argoverse 2 motion forecasting dataset](https://www.argoverse.org/av2.html) (this study uses its 224,896 train and validation scenarios from six US cities; the unlabelled test split is not used), with a closed-loop planning test and a TensorRT deployment path.

> **Status: complete.** 22 training runs, both stages, and the deployment gate ran as pre-registered; the post-run audit recomputed every headline number independently ([`scripts/audit_recompute.py`](scripts/audit_recompute.py)). The hypotheses, arms, and kill criteria were committed before any confirmatory model was scored on the validation split. (Throwaway smoke runs used validation data for pipeline debugging before registration; that is disclosed as deviation 1 and no design choice came from it.)

**Answers, in one line each.** Accuracy drops in an unseen city, but only by about 5%. The model's own uncertainty still flags the predictions that fail there (with no detectable difference from how it does at home), yet it cannot tell that it is in a new city. In closed loop the extra error shows up as a small rise in planning failures (+4%, mostly extra phantom braking), below the registered +10% bar, so H4 is dead as registered.

![Two Palo Alto validation scenarios: the model trained with Palo Alto vs the model that never saw it](docs/figures/example_scenarios.png)

## The questions

| | Question | Primary measure | Pre-set bar |
|---|---|---|---|
| **H1** | Is the predictor less accurate in an unseen city? | Miss rate (best of 6 endpoints > 2 m) on city *c*, model trained without *c* vs an equal-size model trained with it | at least +5% relative, pooled over 6 folds |
| **H2** | Does the model know which predictions are wrong there? | Share of oracle-removable misses caught by rejecting the 20% of scenarios where 3 seeds disagree most | capture fraction > 0 (useful at 0.25) |
| **H3** | Does that self-knowledge survive the shift? | Capture fraction, unseen city minus seen cities | two-sided, no direction registered |
| **H4** | Does the accuracy loss reach driving outcomes? | Planner failure rate (at-fault collision or unnecessary hard brake) using the unseen-city vs all-city predictor | at least +10% relative |

Pre-registrations, including every deviation and the reason for it:
[Stage 1](docs/preregistration/stage1-city-shift.md) (H1 to H3) and
[Stage 2](docs/preregistration/stage2-closed-loop.md) (H4).

## Design

**Leave one city out.** For each of the six cities (Austin, Dearborn, Miami, Palo Alto, Pittsburgh, Washington DC), a model is trained on the other five. Its comparator is a model trained on all six. Both see the same number of training scenarios (128,000) for the same number of steps, and both are scored on the *same* validation scenarios from the held-out city, so they differ only in whether that city was in training and in the random draw of training scenarios (which the three seeds vary). Three seeds per arm; the seeds vary both initialisation and the training-sample draw.

| Arm | Role | Runs |
|---|---|---|
| LOCO-*c* | treatment: trained without city *c* | 6 cities x 3 seeds |
| ALL | control: trained on all six cities, same size | 3 seeds |
| NOMAP | positive control: map input removed, must be clearly worse | 1 |
| Random rejection | sham for H2, rejects the same 20% dose | analysis only |
| Oracle rejection | reference for H2: rejects the 20% with the largest true error (seed-averaged minFDE) | analysis only |

**Statistics.** Pooled effects are means over the six city folds. Each hypothesis is decided by a scenario-level paired bootstrap CI (10,000 draws, resampling within each city) at the Bonferroni level for the four-hypothesis family, 98.75%, together with its magnitude bar. An exact sign-flip test over folds is reported as fold consistency only: with six folds its two-sided floor is 2/64 = 0.031, which no multiplicity correction over four tests can pass. The registration originally used that test with Holm correction; the error was caught by an external code review before any confirmatory scoring and is recorded as deviation 3.

## Model

A 1.44M-parameter query-based transformer, small enough to train in about 35 minutes on a single desktop GPU:

- **Agents:** up to 32 agents, 5 s of history each, encoded by a temporal 1D convolution. Everything is expressed in the focal agent's frame.
- **Map:** up to 128 lane centerlines and crosswalks, each resampled to 20 points and encoded with a PointNet.
- **Scene encoder:** 3 pre-LN transformer layers over all agent and map tokens.
- **Decoder:** 6 learned mode queries cross-attend to the scene, and each emits a 6 s trajectory and a probability.
- **Training:** winner-takes-all regression plus mode classification, AdamW with a cosine schedule, bf16 autocast.

Pilot model on a held-out development slice of the training split (not the validation split, not a study result; source: [`reports/pilot/dev_metrics.json`](reports/pilot/dev_metrics.json)):

| minADE (m) | minFDE (m) | Miss rate | brier-minFDE |
|---|---|---|---|
| 0.929 | 1.555 | 0.223 | 2.179 |

For context only: the Argoverse 2 paper's strongest baseline, WIMP, reports minFDE 2.90 and miss rate 0.42 at K=6 on vehicles, and 2022 leaderboard entries report brier-minFDE 1.92 to 2.16 on the test set. These are different splits and agent mixes, so the comparison is indicative, not a benchmark claim.

## Closed-loop test (Stage 2)

Open-loop accuracy cannot say whether an error matters: a 3 m miss on a car 80 m away changes nothing, a 1 m miss on a car cutting in changes everything. Stage 2 puts the predictor in a loop:

1. At 5 s the self-driving car's own logged track is handed to a planner. The other agents replay their logs and do not react.
   **The planner is privileged by design:** it follows the human driver's *future* path and its speed is capped at the human's future maximum speed plus 1 m/s. It only chooses how fast to go along that path. This makes Stage 2 a paired sensitivity test of one planner to its forecasts, not a general driving benchmark.
2. Every second, the planner forecasts up to 16 nearby agents, ranked by how close they come to the car's route. It picks one of 11 constant-acceleration speed profiles along the car's logged path and executes it for 1 s.
3. The drive is scored against where every agent actually went. The outcomes are an at-fault collision (exact oriented-box overlap, agent ahead of the car) or an unnecessary hard brake (at or below -4 m/s² when the human driver never braked that hard).

Development-slice numbers for every control are in [`reports/pilot/stage2_dev_summary.json`](reports/pilot/stage2_dev_summary.json). Controls: an oracle planner given the true futures (ceiling), constant-velocity and "everyone stands still" forecasts (floors, and a positive control), and a replay of the human's own drive (collision-checker calibration, must be below 1%).

## Deployment

The model exports to ONNX and TensorRT. The gate compares engines against a true FP32 reference on 2,000 validation scenarios, using the confirmatory all-city seed-0 model on an idle desktop GPU (source: [`reports/confirmatory/deploy_report.json`](reports/confirmatory/deploy_report.json)):

| | minFDE change | Miss rate change | Batch 1 (p50) | Batch 32 (p50) |
|---|---|---|---|---|
| PyTorch FP32 | reference | reference | 1.32 ms | 3.35 ms |
| TensorRT FP32 | 0.0% (paths within 0.14 mm) | 0.0% | 0.59 ms | 1.97 ms |
| TensorRT FP16 | +0.13% | -0.64% | 0.39 ms | 0.97 ms |

Gate: pass (FP32 within 1 cm; FP16 within 1% on minFDE and miss rate). The FP16 aggregate barely moves, but the largest single-coordinate difference across the 2,000 scenes is 1.45 m, on one mode of one scene. The latency figures time the predictor alone, not the closed-loop serving path (scene building and planning run on the CPU). FP16 is 3.4x faster than PyTorch at batch 1 and 3.5x at batch 32.

## Results

All numbers are on the 24,988 validation scenarios; per-arm outcomes average the three seeds. Source files: [`reports/confirmatory/`](reports/confirmatory/).

| | Result | 98.75% CI | Fold consistency | Registered verdict |
|---|---|---|---|---|
| **H1** accuracy gap (miss rate, unseen vs seen city) | **+5.3%** relative | [+3.3%, +7.2%] | 6/6 folds positive (p = 0.031, floor) | supported: CI excludes 0 and point estimate clears the +5% bar, but the CI's lower end does not |
| **H2** capture fraction of ensemble disagreement, unseen city (vs error-ranked oracle) | **0.333** (random rejection: 0.000) | [0.311, 0.356] | 6/6 folds (p = 0.016, floor) | supported, above the 0.25 useful bar |
| **H3** capture fraction, unseen minus seen cities | +0.005 | [-0.018, +0.030] | p = 0.56 | no detectable difference |
| **H4** planning-failure rate, unseen-city vs all-city predictor | **+4.3%** relative | [+0.1%, +9.1%] | 4/6 folds positive (p = 0.22) | **dead**: CI excludes 0, but the point estimate is below the +10% bar |
| PC1 no-map model | miss rate 0.355 vs 0.228 (+56%) | | | pass (needs +10%) |
| PC2 map-swap val set | miss rate 0.635 vs 0.228; disagreement AUROC 0.63 | | | pass (needs +10%, AUROC > 0.6) |
| PC3 "everyone stands still" planner | collisions 3.2% vs 1.3% | | | pass (needs 2x) |
| Collision-checker calibration (human drive replayed) | 0.42% at-fault collisions | | | pass (needs < 1%) |

The fold-consistency p-values sit at the 6-fold floor and are descriptive only; decisions use the Bonferroni-level bootstrap CI (see Statistics).

![H1 per city](docs/figures/h1_per_city.png)

**H1 by city** (miss rate, all-city model vs the model that never saw the city):

| City | Scenarios | Seen | Unseen | Relative change |
|---|---|---|---|---|
| Austin | 5,324 | 0.239 | 0.243 | +1.6% |
| Dearborn | 3,066 | 0.265 | 0.275 | +4.1% |
| Miami | 6,654 | 0.224 | 0.234 | +4.2% |
| Palo Alto | 1,413 | 0.220 | 0.237 | +8.1% |
| Pittsburgh | 5,329 | 0.204 | 0.217 | +6.5% |
| Washington DC | 3,202 | 0.229 | 0.245 | +7.1% |

![H2 risk-coverage curve](docs/figures/h2_risk_coverage.png)

**Closed loop, pooled over all validation scenarios:**

![Closed-loop planning failures by forecast source](docs/figures/closed_loop_failures.png)

| Planner forecasts from | Failure | At-fault collision | Unnecessary hard brake | Distance vs human |
|---|---|---|---|---|
| Oracle (true futures) | 1.1% | 0.4% | 0.7% | 1.76x |
| All-city model | 6.7% | 1.3% | 6.0% | 1.45x |
| Unseen-city model | 7.0% | 1.2% | 6.2% | 1.46x |
| Constant velocity | 4.3% | 2.1% | 2.4% | 1.71x |
| Everyone stands still | 5.1% | 3.2% | 2.4% | 1.57x |
| Human drive replayed | 0.4% | 0.4% | n/a | 1.00x |

### Exploratory findings (not registered, labelled permanently)

- **Knowing it is wrong is not knowing it is somewhere new.** No uncertainty signal separates unseen-city scenarios from seen-city ones (mean AUROC 0.505 to 0.518 across the four signals, chance is 0.5), even though ensemble disagreement ranks that city's errors with no detectable difference from the seen cities (H3; not a proof of equivalence).
- **A single model's mode spread beats the three-seed ensemble** at catching misses in the unseen city (capture fraction 0.40 vs 0.33), at a third of the inference cost. Mode entropy (0.24) and Mahalanobis distance of the scene embedding (0.17) are weaker.
- **The learned predictor phantom-brakes.** It collides less than constant velocity (1.3% vs 2.1%) but fails more overall (6.7% vs 4.3%), because the registered planner brakes for any predicted mode that crosses its path, even at a few percent probability. The unseen-city predictor's extra failures are extra hard brakes (6.2% vs 6.0%), not collisions (1.2% vs 1.3%). The planner was deliberately not retuned after this was first seen on development data.

## What broke along the way

Every one of these was found by a diagnostic before any change was made, and each is recorded in the pre-registration's deviations log. The last one was found after the results were in.

1. **Training diverged at about step 6,000.** The loss jumped from 4.0 to 10.5 and never recovered. Logging showed the gradient norm climbing from about 20 to 5e7, and the focal token's RMS growing from 1.7 to 17.5 over 5,600 steps. Cause: pre-LN transformer stacks were built without a final LayerNorm, so the residual stream grew without bound. After adding one, the RMS stays at 1.0.
2. **The six modes were interchangeable.** The mode-probability loss sat at log 6 even when memorising 512 scenarios. The mode queries were initialised at std 0.02 against scene tokens of RMS about 1, so dropout noise swamped mode identity. A single-seed ablation on the overfit test isolated it. Raising the query init to std 1.0 took development miss rate from 0.68 to 0.43 at 10k steps.
3. **The "FP32" deployment reference was not FP32.** PyTorch lets cuDNN convolutions use TF32 by default, which moved the GPU reference 14 cm from a CPU FP32 reference. TensorRT FP32 is within 0.2 mm of true FP32.
4. **The FP16 engine returned NaN on 94% of scenes.**
   - Not an overflow: pure FP16 in PyTorch was clean, and pinning LayerNorm or softmax to FP32 did not help.
   - Exposing intermediate tensors (which prevents layer fusion) made the NaN vanish.
   - The clean scenes were exactly the ones with no padded lanes. A padded lane is all zeros, and its direction is computed as 0 / (0 + 1e-6). The fused FP16 kernel flushes 1e-6 to zero, the resulting 0/0 NaN enters attention as a value vector, and zero attention weight times NaN is still NaN.
   - Fix: padded lanes get a dummy 1 m polyline before inference. Outputs are unchanged in FP32 (0.0 m difference).
5. **The oracle planner crashed.** Given perfect futures, it still hit 3 of 200 scenarios. A trace showed a stopped 12 m bus outside the 12 agents nearest the car (ranked by centre distance) until it was 12 m away at 13 m/s. Ranking agents by distance to the car's route instead brought oracle collisions from 1.5% to 0.2%.
6. **The human driver "hard braked" in 41% of drives.** This was noise in the logged velocity channel. Speed derived from smoothed positions flags 0.2%.
7. **The post-run audit caught an analysis bug.** An independent recomputation matched every headline number except H2 and H3 (0.3338 vs 0.3330). The per-seed column matcher selected columns by suffix, so asking for `min_fde` also picked up `brier_min_fde`, and the oracle ranking was a blend of the two. With an exact match and a regression test, H2 is 0.3330 and H3 is +0.0052; no decision changed. Both outputs are kept (deviation 4).

## Limitations

- **Stage 1 and Stage 2 score different forecasts.** Stage 1 measures the focal agent; the planner consumes forecasts of up to 16 surrounding agents at six replan times, whose accuracy is not separately reported.
- **At-fault attribution is centre-based:** a collision counts when the other agent's centre is ahead of the ego's centre, which can miss some side-swipes with long vehicles.
- **Stage 2 agents do not react to the simulated car.** The planner is longitudinal only and has no traffic-light or stop-sign awareness, so it drives about 1.5 to 1.7x the human's distance. There is no perception noise, and the window is 6 s. These limits apply equally to every arm, which is what the comparison needs, but they bound what the absolute failure rates mean.
- **One dataset, six US cities.** No left-hand traffic and no weather split.
- **The model is small.** City effects could differ at leaderboard scale.
- **Before registration, throwaway smoke runs used the validation split as training data** to debug the pipeline. No design choice was made from their scores; this is disclosed as deviation 1.

## Reproduce

```bash
pip install -e ".[dev,deploy,figures]"   # plus s5cmd (https://github.com/peak/s5cmd) for the download
# data: ~53 GB (train + val), public bucket
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/motion-forecasting/train/*" data/raw/train/
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/motion-forecasting/val/*"   data/raw/val/
python -m cityshift.preprocess --raw data/raw --out data/pp --split train
python -m cityshift.preprocess --raw data/raw --out data/pp --split val

ROOT=data/pp scripts/run_confirmatory.sh 128000     # 22 training runs
ROOT=data/pp scripts/evaluate_all.sh                # Stage 1 scoring + analysis
RAW=data/raw/val scripts/run_stage2.sh              # Stage 2 closed loop + analysis
python -m cityshift.export_trt --root data/pp --ckpt runs/ALL/seed0/model.pt --out evals/deploy
# check the published numbers without retraining: per-scenario tables from release v1.0
gh release download v1.0 -R yusufdxb/av2-city-shift -p per-scenario-results.tar.gz && tar xzf per-scenario-results.tar.gz
python scripts/audit_recompute.py evals             # independent recomputation of the headline numbers
python scripts/plot_results.py --root data/pp        # the figures in docs/figures
pytest -q                                           # 19 tests; 40 more are generated when raw data is present
```

| Path | What it is |
|---|---|
| `src/cityshift/preprocess.py` | raw scenarios to fixed-shape agent-centric arrays |
| `src/cityshift/scene.py` | the same input builder for any agent at any time step (used in closed loop; tested equal to preprocessing) |
| `src/cityshift/model.py`, `train.py` | predictor and training loop |
| `src/cityshift/evaluate.py`, `analysis.py` | Stage 1 scoring, uncertainty signals, pre-registered statistics |
| `src/cityshift/closedloop.py`, `analysis_stage2.py` | Stage 2 harness and statistics |
| `src/cityshift/export_trt.py` | ONNX and TensorRT export with the parity gate |
| `docs/preregistration/` | both registrations and their deviation logs |

## Data and license

Argoverse 2 data is © 2022 Argo AI, LLC, provided under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/): non-commercial use only, with attribution and share-alike. This repository contains no dataset files, only code that downloads and processes them. The code in this repository is MIT licensed (see `LICENSE`); that license does not extend to the data or to anything derived from it. This project is not affiliated with or endorsed by Argo AI.

```bibtex
@inproceedings{wilson2021argoverse2,
  title     = {Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting},
  author    = {Wilson, Benjamin and Qi, William and Agarwal, Tanmay and Lambert, John and Singh, Jagjeet and
               Khandelwal, Siddhesh and Pan, Bowen and Kumar, Ratnesh and Hartnett, Andrew and
               Kaesemodel Pontes, Jhony and Ramanan, Deva and Carr, Peter and Hays, James},
  booktitle = {Neural Information Processing Systems Datasets and Benchmarks Track},
  year      = {2021}
}
```
