# Pre-registration, Stage 3a: same-data baselines and planner-agent accuracy

Registered 2026-09-29, before any Stage 3a validation scoring. Development checks use only the fixed 2% dev slice of the training split. This registration does not alter the Stage 1 or Stage 2 results.

## Claim and scope

**H5.** On AV2 validation scenarios, the three-seed ALL predictor has a miss rate at least **10% lower relative** than fixed lane following for the agents selected at t=49 by the Stage 2 planner, averaging the six city-specific relative reductions equally. A 10% improvement is a practical threshold for a learned model over a training-free geometric baseline. This threshold and the baseline geometry are set before validation scoring.

The observation point is t=49 only, with the logged ego. Stage 2 also replans at later times with a simulated ego, so this measures a defined subset of the forecasts used by that planner, not all of them. The AV itself is excluded. Focal, fully observed SCORED (object_category 2), and planner-relevant sets may overlap. For partly observed planner agents, minADE uses observed future steps and minFDE, miss, and brier-minFDE use the last observed endpoint. Full-horizon agents use the AV2 endpoint convention: select the mode by endpoint error, then report that mode's ADE. A miss is endpoint error greater than 2 m.

**H6 (added 2026-09-29 before any Stage 3a validation scoring; motivated by the train-dev smoke test only).**
Stage 1 trained on focal agents. In the training split 13.1% of focal agents are stopped (< 0.5 m/s) at t=49 but only
3.5% move less than 2 m over the next 6 s: a stopped focal agent is usually about to move. On the dev slice the learned
model therefore beat CV on stopped *focal* agents but lost badly on stopped *non-focal* planner agents (parked cars). H6
tests this selection-bias mechanism on validation, with speed measured at t=49 from the agent's logged velocity:
- **H6a:** among planner-relevant non-focal agents stopped at t=49 (< 0.5 m/s), the three-seed ALL miss rate exceeds the
  CV miss rate by at least **0.15 absolute** (pooled difference over the six cities, city-equal weights). The ratio
  ALL/CV is reported per city as descriptive only. (Amended before any validation scoring, same day: the first draft used
  a 2x ratio, which the dev smoke test showed is undefined under resampling when CV almost never misses, as in the
  50 stopped Palo Alto dev agents with one CV miss.)
- **H6b:** among planner-relevant non-focal agents moving at t=49 (>= 2 m/s), the ALL miss rate is at least **30% lower**
  (relative) than CV's.
Both use the same scenario-cluster bootstrap. Decision for each: supported if the 98.75% CI excludes the no-effect value
(difference 0, reduction 0) and the point estimate meets its magnitude bar; killed otherwise. The Stage 3a confirmatory family is
H5, H6a, H6b (three tests); the 98.75% level is kept, which is conservative for three.

## Arms

| Arm | Role | Forecasts |
|---|---|---|
| ALL seeds 0, 1, 2 | Treatment | Existing all-city checkpoints, per-agent metrics averaged across seeds |
| LANE | No-treatment control and H5 comparator | Up to six heading-compatible nearby lane centerlines, constant speed, fixed offset/heading softmax; CV fallback |
| CV | Same-data non-map control | Six fixed speed/yaw modes, no training |
| LOCO-c seeds 0, 1, 2 | Secondary replication comparator | Existing held-out-city checkpoints on city-c agents only |
| STATIC | Positive control | Six stationary paths, probability one on the first; its miss rate must exceed CV by at least 10% relative for planner agents moving faster than 2 m/s at t=49 |
| Dose-matched sham | Not applicable | No trigger decides whether to intervene and every arm forecasts every eligible agent |
| Oracle/ceiling | Not applicable | Ground-truth futures define the scoring target; an oracle would trivially score zero error and does not bound the learned-vs-geometric comparison |

CV grid in (speed scale, yaw rate rad/s): (1, 0), (0.75, 0), (1.25, 0), (1, -0.15), (1, 0.15), (0, 0), with probabilities (0.4, 0.1, 0.1, 0.15, 0.15, 0.1). Lane following requires lateral projection at most 3 m and heading error at most 60 degrees. Its candidate logit is `-0.5*(offset/2 m)^2 - 0.5*(heading_error/30 degrees)^2`; probabilities are the softmax over up to six best candidates. These are fixed a priori, with no validation tuning.

## Confounds and controls

- **A1 onset and A7 dose:** no trigger, treatment timing, or intervention dose; all arms forecast at t=49.
- **A2 exposure:** every predictor scores the same eligible (scenario, agent) pairs and the same valid future steps. The number of agents per scenario varies; scenario-cluster resampling preserves that variation.
- **A3 seeds:** ALL and LOCO each average three per-agent metric values, as in Stage 1. CV, LANE, and STATIC are deterministic.
- **A4 bundling:** paired forecasts on identical agent inputs hold the agent and city mix fixed within each fold. Agent type and city strata are reported, without additional confirmatory tests.
- **A6 outcome:** open-loop trajectory error, not a driving outcome. The Stage 2 planner's reaction to modes and probabilities can differ even at similar minFDE.
- **A8 floor/ceiling:** if LANE miss rate in a city is zero, its relative reduction is undefined and H5 is inconclusive. If fewer than 100 planner-relevant agents have a LANE miss across the six cities, H5 is underpowered.
- **A9 selection:** fixed geometry and statistics above. Any correction after registration is dated below. No validation outcome is used to change design.
- **A10 leakage:** no baseline is fitted. Only the train dev slice can inform implementation checks. Validation results are read once for final scoring and analysis.

## Analysis and decision

One row per (scenario, agent, predictor) records city, type, set flags, minADE, minFDE, miss, and brier-minFDE. ALL and LOCO values are means of their three seed metrics per agent, not merged trajectories. For each city, `reduction_c = (MR_LANE - MR_ALL) / MR_LANE` among planner-relevant agents; H5 is the unweighted mean of the six reductions. Bootstrap 10,000 times by resampling scenarios with replacement within each city, carrying all agents in a sampled scenario together and keeping predictor pairing. The pre-set decision uses a **98.75% scenario-cluster bootstrap CI**, conservatively matching the original four-hypothesis Bonferroni level; the Stage 3a family is H5, H6a and H6b (see H6). H5 is supported only if the CI lower bound exceeds zero, the point reduction is at least 10%, and the positive control passes. If the control passes but those effect criteria fail, H5 is **killed** at this model scale. A failed positive control makes a null uninterpretable. Undefined city ratios or fewer than 100 LANE misses make it inconclusive, not supported. No optional stopping: score all 24,988 validation scenarios once, then run the fixed analysis.

Report descriptive means of all four metrics by predictor, city, agent type, and set (focal, SCORED, planner-relevant). Report the focal and SCORED comparisons without inferential claims.

**Secondary replication of H1, descriptive only.** On planner-relevant agents in each held-out city, calculate `(MR_LOCO-c - MR_ALL) / MR_ALL` from the seed-averaged per-agent miss values. Average the six city ratios and use the same paired, within-city scenario-cluster bootstrap with a 98.75% CI. This is a descriptive replication of Stage 1 H1 on agents the planner selects, with no pass/fail threshold or additional inferential claim. Report city ratios and the pooled interval even if it is null.

## Deviations

None at registration.
