# Pre-registration: city shift in a learned trajectory predictor

Registered 2026-09-28, before any confirmatory run and before any model has been
scored on the Argoverse 2 validation split. Everything below is fixed. Changes
after this commit are logged in "Deviations" at the bottom with a reason and date.

## Step 1. Claims

**H1 (accuracy).** On Argoverse 2 validation scenarios from city *c*, a predictor
trained without any city-*c* data has a higher miss rate (MR, K=6, 2 m) than an
equal-size predictor whose training set included city *c*, by at least **+5%
relative**, pooled over the six leave-one-city-out folds.

**H2 (knowing).** On held-out-city scenarios, rejecting the 20% of scenarios with
the highest ensemble disagreement (U1) removes a positive share of the misses that
a perfect-knowledge oracle would remove: capture fraction CF > 0, with a minimum
effect of interest of **CF >= 0.25**. Comparator: dose-matched random rejection
(CF = 0 in expectation).

**H3 (does knowing survive the shift).** CF on held-out-city scenarios differs
from CF on in-distribution scenarios for the same models (two-sided; no direction
registered).

Definitions:
- MR: fraction of scenarios where the best of 6 predicted endpoints is > 2.0 m from
  ground truth (`metrics.py`). Per-scenario outcome for an arm = mean miss over its
  3 seeds (values in {0, 1/3, 2/3, 1}).
- CF at 80% coverage: `(MR_all - MR_keep(U)) / (MR_all - MR_keep(oracle))`, where
  `MR_keep(U)` is the MR over the 80% of scenarios with the lowest score U, and the
  oracle ranks by the true seed-averaged minFDE.

## Step 2. Arms

| Arm | Purpose | Runs |
|---|---|---|
| **LOCO-c** (treatment, H1) | trained on N scenarios from the 5 cities other than c | 6 folds x 3 seeds |
| **ALL** (no-treatment control, H1) | trained on N scenarios from all 6 cities, same N, same steps | 3 seeds |
| **U1 rejection** (treatment, H2/H3) | reject top 20% by ensemble disagreement | analysis arm, no training |
| **Random rejection** (sham, dose-matched) | reject 20% uniformly at random | analysis arm |
| **Oracle rejection** (ceiling) | reject 20% with the largest true error | analysis arm |
| **NOMAP** (positive control PC1) | ALL recipe with the map input masked, 1 seed | 1 |
| **Map-swap val** (positive control PC2) | ALL models scored on val with each scenario's map replaced by another same-city scenario's map | analysis arm |

Training sample size N is set by this rule, applied to the train-split counts
before any run: N = 90% of the smallest LOCO eligible pool (train split minus the
2% dev slice minus the held-out city), rounded down to a multiple of 1,000. The
ALL arm draws N proportionally from all six cities. Every arm: 60,000 steps, batch
128, AdamW lr 1e-3, cosine schedule, 1,000 warmup steps, bf16 autocast. Arms
differ only in `--exclude-city` (and `--no-map` for PC1).

Seeds vary both initialisation and the training-sample draw.

**A7 (dose).** All three rejection arms reject exactly 20% of scenarios, computed
within each (fold, city) cell, so rejection dose is identical by construction. The
realised coverage per arm and cell is printed by the analysis script.

**A5 (positive controls).**
- PC1: NOMAP must have MR at least 10% (relative) higher than ALL on the full val
  split. If it does not, the model is not using the map, and a null H1 cannot be
  read as "city geometry does not matter".
- PC2: on the map-swap val set, ALL-ensemble MR must rise by at least 10%
  relative, and U1 must separate swapped from original scenarios with AUROC > 0.6.
  If it does not, U1 is blind even to a large, known shift, and a null H2 is
  uninterpretable.

## Step 3. Confounds

- **A1 onset.** Not applicable: no online detector firing over an episode. Each
  scenario is scored once at t = 5 s.
- **A2 exposure.** Every scenario has 50 observed and 60 future steps, one focal
  agent, one score. Equal exposure by construction. Focal-agent type mix differs
  by city (for example Miami has 8.4% pedestrian focal agents, Dearborn 1.9%); this
  is a composition difference, handled under A4.
- **A3 seeds.** 3 seeds per arm; inference unit for the pooled tests is the
  city fold (n = 6). Exact sign-flip permutation over 6 folds has a two-sided floor
  of p = 2/64 = 0.031, below alpha = 0.05. Seeds enter as an ensemble; per-seed
  results are reported as a stability check.
- **A4 bundling.** "City" bundles map geometry, driver behaviour, and agent-type
  and speed mix. H1 compares both arms on the same city-c scenarios, so the scenario
  mix is held fixed; what differs is only whether training included city c.
  Training size and steps are held fixed across arms. Secondary: H1 restricted to
  vehicle focal agents.
- **A6 outcome.** MR is the forecaster's task metric, not a proxy for it. There is
  no closed-loop planner, so the result says nothing about driving outcomes. Stated
  as a limitation, not claimed.
- **A8 ceiling / floor.** Expected MR roughly 0.2 to 0.4 (Argoverse 2 paper,
  Table 5: WIMP K=6 MR 0.42; leaderboard 0.17 to 0.22). Neither near 0 nor 1. If ALL
  MR on a city is below 0.05, H1 for that city is reported but flagged.
- **A9 selection.** Metric, coverage (80%), primary signal (U1), N rule, and steps
  are fixed here. One pilot run on the ALL recipe may be used to fix bugs and adjust
  only lr/steps, judged on the dev slice of the train split, never on val. Any
  change is logged under Deviations before the confirmatory runs start.
- **A10 leakage.** Val is used only for final scoring. The dev slice (2% of train,
  fixed seed) never trains any arm. The Mahalanobis Gaussian (U4) is fitted on each
  model's own training scenarios. No threshold is fitted anywhere; 80% coverage is
  a fixed rank cut.

## Step 4. Analysis plan

- **H1 primary.** Per fold c: `rel_c = (MR_LOCO-c(c) - MR_ALL(c)) / MR_ALL(c)` on
  city-c val scenarios. Pooled effect = mean of rel_c over the 6 folds. 95% CI:
  bootstrap resampling scenarios within each city (10,000 draws), pairing
  preserved. p-value: exact sign-flip test over the 6 folds.
- **H2 primary.** Per fold: CF of U1 on held-out-city scenarios. Pooled = mean over
  folds; same bootstrap and sign-flip test. Sham: 1,000 random 20% rejections per
  fold, reported as mean and 95% interval.
- **H3.** Per fold: `CF_heldout - CF_indist` (in-dist = the other 5 cities' val
  scenarios, same LOCO-c ensemble). Same bootstrap and sign-flip test, two-sided.
- **Multiplicity.** Holm-Bonferroni across H1, H2, H3 at family-wise alpha 0.05.
- **Exploratory, labelled as such permanently:** U2 entropy, U3 spread, U4
  Mahalanobis; minADE, minFDE, brier-minFDE; per-type strata; AURC curves; city
  detection AUROC of each signal.

## Stopping rule and kill criteria

All 22 training runs are completed; there is no early stopping on val. A run that
diverges (NaN loss) is rerun once with the same seed and logged.

- **H1 dead** if the pooled relative change has a 95% CI that includes 0, or a
  point estimate below +5%. Published as "no measurable city effect at this model
  scale".
- **H2 dead** if the pooled CF 95% CI includes 0. Published as "the ensemble does
  not know when it is wrong in a new city". If the CI excludes 0 but the point
  estimate is below 0.25, reported as "detectable but below the pre-set useful
  magnitude".
- **H3** is reported whichever way it lands.
- If PC1 or PC2 fails, the corresponding null is labelled uninterpretable.

Negative results will be published in the README with the same prominence as
positive ones.

## Deployment gate (engineering, not a claim)

The ALL seed-0 model is exported to ONNX and TensorRT. Pass if: TensorRT FP32 vs
PyTorch FP32 max absolute trajectory difference < 0.01 m on 2,000 val scenarios,
and TensorRT FP16 changes val minFDE and MR by < 1% relative. Latency reported at
batch 1 and 32 (p50/p99), with GPU class only described generically.

## Deviations

1. 2026-09-28, disclosed rather than a change: before registration, throwaway smoke
   runs used the val split as *training* data (the train split was still
   downloading) to debug the pipeline and the TensorRT export. Their val numbers
   were used only to find bugs (an FP16 NaN, TF32 in the FP32 engine). No
   hyperparameter, threshold, metric, or analysis choice was made from them. All
   later pipeline checks use the train split.
