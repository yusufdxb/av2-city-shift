# Pre-registration, v2 studies H and F: shared-fold covariance and fixed-horizon coverage

Registered 2026-10-02 for v2, before either full analysis below produced output. Study H is a re-analysis amendment of
already published Stage 1 rows. Study F is exploratory and pre-registered here before its new scoring pass. Stages 1
to 4 and follow-ups A to E are frozen. Neither study changes a registered verdict; H reports a sensitivity verdict
beside H3's original verdict. Changes after registration are dated under Deviations.

What is already known (A9): H3 was reported as "no detectable difference"; Stage 3a H6 showed the stopped non-focal
ALL versus CV reversal at t=49, using six-second forecasts and last-observed-endpoint scoring for partial futures.
These motivate the studies and cannot be treated as blind. No fixed-horizon effect is known at drafting. Small CPU
checks may verify implementation and counts; they cannot select thresholds or alter claims.

## Study H: H3 covariance correction (CPU re-analysis amendment)

**Question.** Is H3's verdict "no detectable difference" unchanged when the bootstrap accounts for the shared
validation scenarios across LOCO folds?

**Estimand.** Exactly as implemented in the registered Stage 1 analysis: per LOCO-c ensemble, capture fraction (CF)
on held-out city c minus CF on the pooled validation scenarios from the other five cities. CF uses seed-averaged
misses, ensemble disagreement U1, an 80% retained set and an oracle ranked by the true seed-averaged minFDE. Retention
uses rounded 80% counts and stable index tie breaking. The pooled effect is the mean over the six folds. The five-city
in-distribution CF is computed on the pooled rows, not averaged from five separate city CFs.

**Correction.** The registered interval independently bootstrapped each fold's held-out and in-distribution sets.
Instead draw scenarios with replacement within each of the six cities once per draw and reuse those same scenario
indices in every fold, including both its held-out and in-distribution sets. Place sampled city rows into their
original city slots; all folds use the same sampled table order for stable ties. Recompute retention, oracle ranks
and each CF inside every draw. Average defined folds as the registered code does; report undefined fold and pooled
draw shares. Use 10,000 draws, generator seed 20261002. Report two-sided percentile 98.75% and 95% intervals, with the
original registered intervals alongside. Recompute H2 (mean held-out CF) with the same draws and report both levels;
its point is unchanged, while finite-draw interval endpoints need not be numerically identical to the old bootstrap.

**Positive control.** H3's point must equal `reports/confirmatory/stage1_results.json` to 1e-12; H2's point must also
match to 1e-12. A mismatch aborts before reporting amended inference. No new forecasts or fitting.

**Decision and kill.** Use H3's registered two-sided 98.75% rule: a strictly positive lower bound or strictly negative
upper bound means "differs"; otherwise "no detectable difference". Publish whether the original verdict survives.
The unchanged-verdict claim is killed if the sensitivity verdict differs. Undefined intervals are inconclusive.
The original registered decision remains frozen even if the sensitivity decision changes. H2 is reported for context,
with no additional hypothesis or verdict change.

**Confounds.** A1: fixed t=49 observations. A2: same scenario pool and CF coverage; shared draws repair dependence,
not unequal city composition. A3: three fixed seeds averaged before CF; inference is conditional on checkpoints.
A4: city bundles agent mix and geometry; the estimator is unchanged. A5: exact registered-point parity and a synthetic
shared-draw check. A6: CF is selective prediction performance, not driving performance. A7: 20% rejection, rounding
and tie handling fixed. A8: undefined denominators and draws are reported; no replacement of missing folds beyond the
registered nanmean convention. A9: this is explicitly a post-result amendment. A10: existing evaluations only, no
fitting or validation tuning.

## Study F: fixed-horizon coverage with the logged ego (exploratory)

**Claim.** The stopped-agent reversal persists on fully observed agents at fixed horizons: at both t=49 and t=89,
the two-second stopped non-focal miss difference ALL minus CV-6 is at least +0.15 absolute, with its two-sided 95%
interval lower bound strictly above zero. Both cells must pass. This is an exploratory conjunction, with no additional
multiplicity adjustment. Every cell below is reported, including negative and inconclusive cells.

| Predictor | Role |
|---|---|
| ALL seeds 0 to 2 | Existing focal-only `runs/ALL/seed{0,1,2}/model.pt`; mean of three per-agent metrics |
| CV-6 | `baselines.constant_velocity`, the six fixed speed/yaw modes registered for Stage 3a |

**Selection and scoring.** Use every AV2 validation scenario, with the ego following its logged states and all
histories logged, without a closed-loop rollout. At t in {49, 59, 69, 79, 89}, call `closedloop.select_agents` with the
logged ego position and remaining logged route: up to 16 dynamic agents within 60 m, ranked by nearest edge to the
next 80 m of route. The remaining route is anchored at the exact logged position at each replan, with the planner's
0.05 m vertex filter and straight extension; t=49 reproduces the planner's handoff path. Use `scene.build_input`,
`closedloop.load_models` and `closedloop.predict` for identical agent-centric inputs and prediction conventions.

Score two seconds (20 steps) at every replan and four seconds (40 steps) only at 49, 59 and 69. Require the current
observation and every logged future step t+1 through t+20 or t+40, including the endpoint. Interior gaps exclude the
agent even if its endpoint is observed. The last scene step is 109; no endpoint fill or shortened horizon is allowed.
Miss = best-of-six endpoint error at exactly that horizon >2 m, and minFDE is that minimum error. **The threshold is
2 m at both horizons**, a deliberate horizon-specific choice that retains the Stage 3a distance threshold without
scaling it by time. Score each ALL seed separately and average metrics, never merge modes or average trajectories.

**Groups and rows.** Speed is the norm of logged velocity at the replan. Stopped = <0.5 m/s; moving = >=2 m/s;
intermediate speeds are excluded from grouped effects. Both groups exclude the focal agent; the AV is excluded by
selection. Write one row per (scenario, replan, agent, predictor), with separate completeness flags and miss/minFDE
columns for two and four seconds. Keep selected agents with incomplete futures as unscored rows (NaN metrics) so
the exclusion denominator is auditable. Four-second cells after t=69 are unscheduled, not exclusions. Report counts
and excluded shares per cell and city, plus completeness coverage over all selected agents.

**Estimator.** Within each city and (replan, horizon, group), calculate the agent-weighted mean miss rate for each
predictor; equal-weight the six city rates and differences. For each cell, bootstrap 10,000 times, resampling whole
scenarios with replacement within each city from scenarios containing fully observed group agents. All eligible
agents in a sampled scenario move together, preserving predictor pairing. This keeps the Stage 3a agent-weighted
estimand instead of weighting sparse and crowded scenes equally. Generator seed 20261002; two-sided 95% percentile
intervals on ALL minus CV-6. Report per-city miss and minFDE means, selected and scored agents, eligible scenarios,
exclusions, city-equal miss rates and paired differences. A missing group in any city makes the pooled cell
inconclusive; never renormalize to the cities present.

**Positive controls.** Synthetic tests check the exact endpoint, strict >2 m threshold, interior-gap and endpoint
exclusions, handoff selection parity and logged-ego anchoring, and scenario-cluster rather than agent resampling.
Implementation failures stop scoring until corrected. An oracle target would score zero trivially and is not a
substantive positive control for this comparison.

**Decision and kill.** Supported only if both primary two-second stopped cells have point difference >=+0.15 and
95% lower bound >0. A fully defined primary failing either condition kills the persistence claim, published as
"stopped-agent reversal not established at both fixed-horizon replans". A missing city/group or undefined primary
interval is inconclusive unless the other primary already kills the conjunction. Other cells are descriptive.

**Confounds.** A1: five replans fixed in advance; no adaptive onset. A2: identical logged ego, selection and horizon
within predictor pairs; report complete-future exclusions, which may select more persistent tracks. A3: average three
seed metrics per agent; no independent-seed inference. A4: paired inputs hold composition fixed within a cell;
composition can change across replans, so this is not a within-agent causal trend. A5: synthetic scoring, selection
and clustering controls above. A6: endpoint coverage measures open-loop error, not collision or brake outcomes;
mode probabilities are irrelevant to best-of-six miss. A7: every eligible agent receives both forecasts; no trigger
or sham. A8: report counts and rates; zero misses are valid for an absolute difference, and missing cities are
inconclusive. A9: prior Stage 3a results motivate the claim; horizons, thresholds and decision fixed before this
scoring pass, with no optional stopping. A10: existing checkpoints and unfitted CV-6 only; logged futures determine
eligibility and scoring after selection, never inputs or model choice. The logged future route remains privileged
navigation information inherited from the planner.

## Stopping and deviations

One full CPU analysis for H and one full scoring and analysis pass for F. No training, tuning, threshold search or
registered-verdict replacement. A 20-scenario CPU smoke may check runtime, row integrity and counts only and is
labelled smoke, never inferential. Publish negative and inconclusive results with the same prominence.

None at drafting.
