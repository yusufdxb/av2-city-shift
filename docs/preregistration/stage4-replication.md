# Pre-registration, Stage 4: independent replication and mechanism tests

Registered 2026-09-29 before scoring the replication pool. The pool is 8,140 TRAIN scenarios excluded from ALL training and the fixed TRAIN development slice; its sorted, newline-joined ID SHA-256 is `a7489d7bd7b95986807ee360e62a01c45a15f4c3297258ba4751f8520caee0c8`. MULTI trained on this pool and is ineligible. Only TRAIN-dev informs implementation checks. Stage 1, 2 and 3 results and code are frozen.

## Directional claims

- **H10 replication:** PATCH reduces unnecessary hard braking by at least 30% relative to ALL on the fresh pool, with at-fault collision-rate increase below +0.3 percentage points.
- **H11 matched sham:** PATCH's relative brake reduction exceeds SHAM2's by at least 20 percentage points, with the paired confidence interval excluding zero.
- **H12 probability mechanism:** TRIM reduces unnecessary hard braking by at least 30% relative to ALL, with at-fault collision-rate increase below +0.3 percentage points.
- **H13 learned fix:** MIX reduces unnecessary hard braking by at least 30% relative to ALL, with at-fault collision-rate increase below +0.3 percentage points, and MIX's focal open-loop miss rate exceeds ALL's by less than +0.02 absolute.

The 30% reduction and +0.3-point collision margin repeat Stage 3's registered H8 bars. Stage 3 PATCH reduced brakes 38.4% (99.17% CI 34.1% to 42.4%) and increased collisions 0.09 points (CI -0.03 to +0.20); this motivates a 30% replication bar. A 20-point matched-sham gap is about half that observed PATCH reduction and resolves Stage 3's 43%-dose sham limitation. The +0.02 focal margin repeats Stage 3's H9 protection criterion. These are design bars, not estimates from this pool.

## Arms and dose

| Arm | Role | Forecast |
|---|---|---|
| ALL seeds 0,1,2 | No-treatment control | Existing focal predictor |
| PATCH seeds 0,1,2 | Replication treatment | CV with probability one for selected agents moving <0.5 m/s at that replan |
| TRIM seeds 0,1,2 | Probability mechanism | For selected stopped agents retain own modes ending <=2 m from current position, renormalise; CV fallback if none |
| SHAM2 seeds 0,1,2 | Dose-matched sham | At every scenario, replan and seed, CV with probability one for a uniform random subset of all selected agents (stopped or moving) whose size equals PATCH's realised substitution count at that same replan |
| MIX seeds 0,1,2 | Learned treatment | Stage 1 model and schedule, 75% focal and 25% stopped non-focal samples |
| CV | Training-free comparator | Stage 2 constant velocity |
| ORACLE | Ceiling | Logged true agent futures |
| STATIC | Positive control | Stage 2 stationary forecast |
| LOG | Checker control | Logged ego trajectory |

SHAM2's random seed is fixed by scenario ID, replan and checkpoint seed. PATCH is scored first for each seed and its per-replan dose is SHAM2's reference. Because the arms' ego histories diverge, SHAM2 can select fewer agents than PATCH's dose at a replan; the dose is then capped at the agents available, the shortfall is recorded, and H11 is interpretable only if SHAM2's realised total dose is at least 99% of PATCH's. Record trigger and substitution counts. MIX draws exactly 96,000 focal samples and 32,000 stopped (<0.5 m/s at t=49) non-focal samples without replacement. It excludes every replication-pool and fixed dev scenario ID from both sources. Three seeds vary initialisation and sample draw. The architecture, 60,000 steps, batch 128, AdamW learning rate 1e-3, warmup 1,000, cosine schedule and bf16 match Stage 1. Evaluate the focal and multi-agent dev splits during training. The pool's focal agents are evaluated open loop once for ALL and MIX.

## Confounds and controls

- **A1 onset:** fixed handoff t=49 and six fixed replans; the stopped trigger uses observed speed only.
- **A2 exposure:** identical scenarios, route, planner constants, agent-selection rule, six-second horizon, speed cap and contact-based score from closedloop_v2. Arm-dependent ego histories can select different agents; SHAM2's per-replan dose copies PATCH's realised dose (capped where fewer agents are selected).
- **A3 seeds:** average three seed outcomes per scenario before city aggregation; seeds are never independent observations.
- **A4 bundling:** TRIM isolates removal of moving-mode mass while retaining native stationary trajectories when available; its CV fallback frequency is reported. PATCH and SHAM2 share substitution count, though chosen agents differ. MIX changes training mixture, so any benefit need not have the same mechanism.
- **A5 positive controls:** STATIC at-fault collisions must be at least 2x ALL's; LOG at-fault collision rate must be <1%. Failure makes closed-loop claims uninterpretable.
- **A6 outcomes:** unnecessary hard braking and at-fault collision are the Stage 3 v2 metrics. Risk calibration is descriptive only and cannot change confirmatory decisions.
- **A7 dose:** SHAM2 substitutes exactly PATCH's realised count at each scenario, seed and replan, capped only where its own selected set is smaller; a SHAM2 dose above PATCH's aborts the analysis, and a realised total below 99% of PATCH's makes H11 uninterpretable. The realised ratio and the share of capped replans are reported.
- **A8 floor:** if the sum of seed-averaged ALL unnecessary-brake events is <100, closed-loop claims are underpowered. A zero ALL brake rate in any city or >1% undefined bootstrap ratios makes affected reductions inconclusive.
- **A9 selection:** all arms, thresholds, 10,000 draws and one-pass decisions are fixed here. TRAIN-dev smoke checks may correct implementation bugs before full scoring, with dated deviations below.
- **A10 leakage:** the pool is disjoint by scenario ID from ALL and MIX training and dev. No validation scenario or outcome is read. Only ORACLE sees future agent motion during planning; true futures for calibration are read after candidate choice.

## Analysis and stopping

For each scenario, average seeds 0-2 for ALL, PATCH, TRIM, SHAM2 and MIX. Within each of six cities compute each treatment's relative brake reduction `(ALL - arm) / ALL` and collision difference `arm - ALL`. Equal-weight the six city effects. H11 uses the paired within-city difference between PATCH and SHAM2 relative reductions, equal-weighted. For focal miss, average three seed misses per focal agent, then calculate MIX minus ALL per city and equal-weight cities. Bootstrap **10,000 times**, resampling scenarios with replacement **within city**, preserving arm and agent pairing, then calculate two-sided **98.75% percentile confidence intervals**. The family has eight directional components: H10 brake and collision; H11 sham gap; H12 brake and collision; H13 brake, collision and focal miss. The 98.75% level is the same conservative level used in earlier registrations (0.05/4 two-sided); for this eight-component family use **99.375% CIs** (0.05/8 two-sided) for decisions. Report 98.75% CIs descriptively alongside decision CIs.

A brake component passes if its point reduction is >=0.30 and CI lower bound >0. Collision non-inferiority passes if CI upper bound <+0.003. H11 passes if its point gap is >=0.20 and CI lower bound >0. Focal non-inferiority passes if CI upper bound <+0.02. Each H10-H13 requires all its components and positive controls. A claim is killed if a magnitude bar or CI rule fails when powered and controls pass. The numeric kill criteria are reduction <0.30, sham gap <0.20, collision upper bound >=0.003, or focal miss upper bound >=0.02. Controls, dose mismatch, missing city, insufficient events or undefined ratios stop interpretation as specified above.

Per (scenario, replan, arm, selected agent), report the chosen acceleration candidate's predicted hit probability and observed contact with its four-second candidate plan under the logged future, using contact-based fault geometry. Describe Brier score, reliability bins (fixed deciles [0,0.1), ..., [0.9,1]) and AUROC overall and separately for stopped and moving selected agents; exclude LOG, which has no predicted risk. The raw risk log includes all six replans and records the observed future length. Since the scene ends at t=109, only replans 49, 59 and 69 have a complete four-second observed future; calibration summaries exclude the three censored later replans. No inference or model selection uses calibration results.

Complete one scoring pass over the 8,140-scenario pool, one fixed analysis, and no pool-driven retraining, threshold adjustment or optional stopping. Smoke tests use at most 100 fixed TRAIN-dev scenarios and at most 200 CPU MIX optimizer steps. Publish negative and inconclusive results.

## Deviations

None at registration. Review note: the dose rule above was finalised during code review, before registration and before any pool scoring; the train-dev smoke test (50 scenarios) realised equal totals (2,463 each).
