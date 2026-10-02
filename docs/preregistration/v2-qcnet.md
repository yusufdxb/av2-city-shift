# Pre-registration, v2 study Q: does a stronger, published forecaster show the stopped-agent failure?

Registered 2026-10-02, before any QCNet forecast was scored on validation. Stages 1 to 4, the follow-ups and their
results are frozen. Everything below is fixed; changes are logged under Deviations.

## Why

Every result so far uses one 1.44M-parameter model trained only on focal agents. QCNet (Zhou et al., CVPR 2023) is a
7.66M-parameter query-centric forecaster whose released Argoverse 2 checkpoint is near the top of the public benchmark,
and it was trained with its loss on focal **and** scored agents. Study Q asks whether the stopped-agent failure and the
PATCH effect are properties of a weak focal-only model or persist in a strong published one.

## Model and adapter

- Checkpoint `QCNet_AV2.ckpt` from the official repository (SHA-256
  `b9f852ec888d6fb966a38e3e307af231145b8e1e69cf7b0b9dddd1ae33120f21`), code from the official repository, unmodified.
  It was trained on the Argoverse 2 training split; the validation split is therefore held out for it. (The Stage 4
  pool and the v2 reserve are training-split scenes and are not used.)
- `src/cityshift/qcnet_adapter.py` builds inputs with QCNet's own preprocessing on a time-shifted copy of the raw
  scenario (the 50 steps ending at the replan; in the closed loop the AV track is replaced by the simulated ego up to the
  replan), and converts outputs to world coordinates as QCNet's `test_step` does. QCNet's three compiled-extension
  calls (`radius`, `radius_graph`, `segment_csr`) are replaced by pure-PyTorch functions that match a brute-force
  definition of torch_cluster's semantics, including the neighbour cap (`tests/test_qcnet_shim.py`).
- Selected agents that QCNet does not forecast (not valid at the replan and the step before) get constant velocity;
  their count is recorded per arm.

## Positive control Q0: the adapter reproduces QCNet

On all 24,988 validation scenarios at t=49 (QCNet's native setting), focal-agent minFDE (K=6) must be within 5% of the
published validation value (1.25 m, i.e. at most 1.31 m) and miss rate within 0.015 of 0.16 (at most 0.175). A
200-scenario development check before registration gave 1.33 m and 0.195 (sampling noise at that size); the full set
decides. If Q0 fails, every Q verdict is **uninterpretable** (the adapter is not faithful), and the result is
published as such.

## Q-open: stopped and moving non-focal planner agents at the handoff (all validation scenarios)

Same agents, memberships and scoring as Stage 3a (`multiagent_eval.memberships` and `score`: planner-relevant agents
selected at t=49 with the logged ego, best-of-six endpoint miss at 2 m, partial futures scored at their last observed
point). QCNet rows join the released Stage 3a CV-6 and ALL rows agent for agent.

**Q1 (claim).** On stopped (< 0.5 m/s) non-focal planner agents, QCNet's miss rate exceeds CV-6's by at least +0.15
absolute (the H6a bar). Estimator: per-city agent-pooled miss rates, city-equal difference, 10,000 scenario-cluster
bootstraps within city (seed 20261002). Supported if point >= +0.15 and lower bound > 0; killed if point < +0.15 or
lower bound <= 0. If the upper bound is below 0, QCNet is reported as **better than CV-6 on stopped agents**.

**Descriptive:** the same for moving (>= 2 m/s) agents (H6b's comparison), for the complete-future subset, QCNet's
probability on modes ending more than 2 m away for stopped agents (the moving-mode mass behind the planner's braking),
and QCNet versus ALL on every group.

## Q-closed: the stop-line harness on a fixed validation subset

A fixed-seed uniform draw of 6,000 validation scenarios (seed 20261002, `study_q.validation_subset`), sized so the run
fits the compute budget (about 18 QCNet forward passes per scenario). A 24-scenario development-slice smoke run took
111 s on a GPU shared with three training jobs; only its runtime and integrity were examined (19 of 1,362 stopped-agent
triggers had no QCNet forecast and used constant velocity; the sham never exceeded QPATCH's per-replan dose).

| Arm | Role |
|---|---|
| QBASE | QCNet forecasts for every selected agent (no-treatment control) |
| QPATCH | QBASE, with CV and probability one for selected agents moving < 0.5 m/s (treatment) |
| QSHAM2 | QBASE, with CV for a random subset of selected agents of QPATCH's realised per-replan size (dose-matched sham) |
| ALL seeds 0 to 2 | The study's own focal-only model on the same scenes (paired comparison) |
| ORACLE | Ceiling and brake positive control |
| CV, STATIC, LOG | Descriptive comparators and checker calibration |

**Q2 (claim).** QPATCH reduces unnecessary hard braking by at least 30% relative to QBASE. **Q3 (claim).** QPATCH's
reduction exceeds QSHAM2's by at least 20 percentage points.

**Controls.** Brake positive control: QBASE's unnecessary-brake rate is at least twice ORACLE's; if it is not, Q2 and Q3
are uninterpretable and the finding is that QCNet does not produce excess braking in this harness. Dose: QSHAM2's
realised total at least 99% of QPATCH's, never above it per replan. Power: at least 100 QBASE brake events, no city with
zero, at most 1% undefined bootstrap draws; otherwise inconclusive.

**Estimator and decisions.** Per scenario (the QCNet arms are single runs; ALL averages seeds 0 to 2), within-city
relative reductions, equal city weights, 10,000 within-city scenario bootstraps (seed 20261002). The family Q1 to Q3 is
decided at the Bonferroni 98.33% level: supported if the point meets the bar (30%, 20 pp) and the lower bound is above
0; killed otherwise (when powered and the controls pass).

**Secondary (95%, labelled secondary):** QBASE versus ALL unnecessary-brake rates on the same scenes (relative
difference): does the stronger forecaster brake less? Collisions are descriptive (no working positive control in the
stop-line harness, study E).

## Confounds

- **A1, A2:** identical scenarios, harness, replans and agent selection for every arm; ego histories diverge by arm.
- **A3:** one public checkpoint, so inference is conditional on it; QCNet's own seed variability is not estimable here.
- **A4:** QCNet differs from ALL in architecture, size and training targets at once; Q asks whether the failure
  persists in such a model, not which difference matters.
- **A5:** Q0 (adapter fidelity), the brake positive control, the shim unit tests and the dose gate.
- **A6:** Q1 is open-loop accuracy; Q2 and Q3 are the registered braking proxy.
- **A7:** QSHAM2 matched per replan.
- **A8:** power gates above.
- **A9:** bars copied from H6a, H10 and H11; the subset size was fixed from runtime on the development slice before
  any validation forecast; no validation outcome was seen.
- **A10:** QCNet never trained on validation; the planner has no fitted parameters.

## Stopping

One open-loop pass over all validation scenarios and one closed-loop pass over the subset; one fixed analysis. Smoke
runs use at most 100 development-slice scenarios, and only their runtime and integrity are examined. Negative and
inconclusive results are published with the same prominence.

## Deviations

None at registration.
