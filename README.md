# City Shift

**Does a learned trajectory predictor get worse in a city it has never seen, does it know when it is wrong there, and does any of that reach the car's driving?**

A pre-registered study on the [Argoverse 2 motion forecasting dataset](https://www.argoverse.org/av2.html) (about 225,000 real driving scenarios from six US cities), with a closed-loop planning test and a TensorRT deployment path.

> **Status: confirmatory runs in progress.** The hypotheses, arms, and kill criteria were committed before any model was scored on the validation split. The results tables below stay empty until every run finishes and the post-run audit recomputes each number from the saved artifacts. Negative results will be reported with the same prominence as positive ones.

## The questions

| | Question | Primary measure | Pre-set bar |
|---|---|---|---|
| **H1** | Is the predictor less accurate in an unseen city? | Miss rate (best of 6 endpoints > 2 m) on city *c*, model trained without *c* vs an equal-size model trained with it | at least +5% relative, pooled over 6 folds |
| **H2** | Does the model know which predictions are wrong there? | Share of oracle-removable misses caught by rejecting the 20% of scenarios where 3 seeds disagree most | capture fraction > 0 (useful at 0.25) |
| **H3** | Does that self-knowledge survive the shift? | Capture fraction, unseen city minus seen cities | two-sided, no direction registered |
| **H4** | Does the accuracy loss reach driving outcomes? | Planner failure rate (at-fault collision or unnecessary hard brake) using the unseen-city vs all-city predictor | at least +10% relative |

Pre-registrations, including every deviation and the reason for it:
[Stage 1](docs/preregistration/2026-09-28-city-shift.md) (H1 to H3) and
[Stage 2](docs/preregistration/2026-09-29-stage2-closed-loop.md) (H4).

## Design

**Leave one city out.** For each of the six cities (Austin, Dearborn, Miami, Palo Alto, Pittsburgh, Washington DC), a model is trained on the other five. Its comparator is a model trained on all six. Both see the same number of training scenarios (128,000) for the same number of steps, and both are scored on the *same* validation scenarios from the held-out city, so the only difference between them is whether that city was in training. Three seeds per arm; the seeds vary both initialisation and the training-sample draw.

| Arm | Role | Runs |
|---|---|---|
| LOCO-*c* | treatment: trained without city *c* | 6 cities x 3 seeds |
| ALL | control: trained on all six cities, same size | 3 seeds |
| NOMAP | positive control: map input removed, must be clearly worse | 1 |
| Random rejection | sham for H2, rejects the same 20% dose | analysis only |
| Oracle rejection | ceiling for H2, rejects the truly worst 20% | analysis only |

**Statistics.** The city fold is the unit of inference (n = 6). Pooled effects are means over folds with 95% CIs from a scenario-level paired bootstrap within each city; p-values come from an exact sign-flip test over folds (two-sided floor 2/64 = 0.031). Holm correction across H1 to H4.

## Model

A 1.44M-parameter query-based transformer, small enough to train in about 35 minutes on a single desktop GPU:

- **Agents:** up to 32 agents, 5 s of history each, encoded by a temporal 1D convolution. Everything is expressed in the focal agent's frame.
- **Map:** up to 128 lane centerlines and crosswalks, each resampled to 20 points and encoded with a PointNet.
- **Scene encoder:** 3 pre-LN transformer layers over all agent and map tokens.
- **Decoder:** 6 learned mode queries cross-attend to the scene, and each emits a 6 s trajectory and a probability.
- **Training:** winner-takes-all regression plus mode classification, AdamW with a cosine schedule, bf16 autocast.

Pilot model on a held-out development slice of the training split (not the validation split, not a study result):

| minADE (m) | minFDE (m) | Miss rate | brier-minFDE |
|---|---|---|---|
| 0.929 | 1.555 | 0.223 | 2.179 |

For context only: the Argoverse 2 paper's strongest baseline, WIMP, reports minFDE 2.90 and miss rate 0.42 at K=6 on vehicles, and 2022 leaderboard entries report brier-minFDE 1.92 to 2.16 on the test set. These are different splits and agent mixes, so the comparison is indicative, not a benchmark claim.

## Closed-loop test (Stage 2)

Open-loop accuracy cannot say whether an error matters: a 3 m miss on a car 80 m away changes nothing, a 1 m miss on a car cutting in changes everything. Stage 2 puts the predictor in a loop:

1. At 5 s the self-driving car's own logged track is handed to a planner. The other agents replay their logs and do not react.
2. Every second, the planner forecasts up to 16 nearby agents, ranked by how close they come to the car's route. It picks one of 11 constant-acceleration speed profiles along the car's logged path and executes it for 1 s.
3. The drive is scored against where every agent actually went. The outcomes are an at-fault collision (exact oriented-box overlap, agent ahead of the car) or an unnecessary hard brake (at or below -4 m/s² when the human driver never braked that hard).

Controls: an oracle planner given the true futures (ceiling), constant-velocity and "everyone stands still" forecasts (floors, and a positive control), and a replay of the human's own drive (collision-checker calibration, must be below 1%).

## Deployment

The model exports to ONNX and TensorRT. The check compares engines against a true FP32 reference on 2,000 scenarios. Pilot model, development data, desktop GPU shared with training:

| | minFDE change | Miss rate change | Batch-32 latency (p50) |
|---|---|---|---|
| PyTorch FP32 | reference | reference | 7.8 ms |
| TensorRT FP32 | 0.0% (paths within 0.18 mm) | 0.0% | 4.1 ms |
| TensorRT FP16 | +0.18% | -1.0% | 3.2 ms |

The pre-registered gate (FP32 within 1 cm; FP16 within 1% on minFDE and miss rate) is re-run on the confirmatory model as part of the pipeline.

## Results

*Pending. Filled in only after the post-run audit.*

| | Result | 95% CI | p (Holm) | Verdict |
|---|---|---|---|---|
| H1 accuracy gap | | | | |
| H2 capture fraction | | | | |
| H3 capture difference | | | | |
| H4 planning failures | | | | |
| PC1 no-map control | | | | |
| PC2 map-swap control | | | | |
| PC3 static-forecast control | | | | |

## What broke along the way

Every one of these was found by a diagnostic before any change was made, and each is recorded in the pre-registration's deviations log.

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

## Limitations

- **Stage 2 agents do not react to the simulated car.** The planner is longitudinal only and has no traffic-light or stop-sign awareness, so it drives about 1.5 to 1.7x the human's distance. There is no perception noise, and the window is 6 s. These limits apply equally to every arm, which is what the comparison needs, but they bound what the absolute failure rates mean.
- **One dataset, six US cities.** No left-hand traffic and no weather split.
- **The model is small.** City effects could differ at leaderboard scale.
- **Before registration, throwaway smoke runs used the validation split as training data** to debug the pipeline. No design choice was made from their scores; this is disclosed as deviation 1.

## Reproduce

```bash
pip install -e ".[dev,deploy]"
# data: ~53 GB (train + val), public bucket
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/motion-forecasting/train/*" data/raw/train/
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/motion-forecasting/val/*"   data/raw/val/
python -m cityshift.preprocess --raw data/raw --out data/pp --split train
python -m cityshift.preprocess --raw data/raw --out data/pp --split val

ROOT=data/pp scripts/run_confirmatory.sh 128000     # 22 training runs
ROOT=data/pp scripts/evaluate_all.sh                # Stage 1 scoring + analysis
RAW=data/raw/val scripts/run_stage2.sh              # Stage 2 closed loop + analysis
python -m cityshift.export_trt --root data/pp --ckpt runs/ALL/seed0/model.pt --out evals/deploy
pytest -q                                           # 57 tests
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

Argoverse 2 data is provided under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). This repository contains no dataset files, only code that downloads and processes them. Code in this repository is MIT licensed (see `LICENSE`).

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
