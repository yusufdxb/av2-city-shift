# Pre-registration, Stage 3: repair stopped-agent phantom braking

Registered 2026-09-29 before any Stage 3 validation scoring. Stage 3a H6 and diagnostics on the fixed 2% development slice of the TRAIN split motivated this design. No Stage 3 validation outcome has been read. The Stage 1/2 code and published results remain frozen.

## Claims and design

**H7 (root-cause intervention).** Under closed-loop v2 on AV2 validation, the MULTI predictor trained on fully observed focal and SCORED agents reduces the unnecessary-hard-brake rate at least **30% relative** to the focal-only ALL predictor, while the MULTI minus ALL at-fault collision-rate difference has a **99.1667% CI upper bound below +0.3 percentage points**.

**H8 (stopped-agent mechanism).** In the same harness PATCH, which uses the ALL model but substitutes constant-velocity forecasts for selected agents whose observed speed at a replan is below 0.5 m/s, reduces unnecessary hard brakes at least **30% relative** to ALL with the same +0.3 percentage-point collision non-inferiority bound. A H8 success supports the stopped-agent mechanism; H7 success without H8 success suggests broader distribution correction. PATCH is an intervention on predictions, not a learned model.

**H9 (open-loop sanity).** At t=49 among stopped, non-focal planner agents, MULTI's miss rate is at least **0.15 absolute lower** than ALL's; on focal agents MULTI's miss rate is no worse than ALL's by more than **+0.02 absolute**, with the latter's 99.1667% CI upper bound below +0.02. H9 requires both components. The stopped group includes agents with partial futures, using Stage 3a's last-observed-endpoint miss rule; sample sizes and valid-step distributions are reported. Focal agents have complete futures.

Thirty percent is half of the observed Stage 2 brake excess of ALL over CV (6.0% versus 2.4%), a material reduction without demanding CV's collision tradeoff. The +0.3-point collision margin is less than half of ALL's 0.8-point advantage over CV (1.3% versus 2.1%). The 0.15 stopped miss bar is conservative relative to the roughly 0.4 versus 0.1 train-dev gap, and +0.02 focal tolerance is less than one tenth of Stage 1's roughly 0.23 focal miss rate. These are design bars based only on existing published Stage 2 or TRAIN-dev diagnostics, never Stage 3 validation scoring.

## Arms

| Arm | Role | Forecast |
|---|---|---|
| ALL seeds 0,1,2 | No-treatment control | Existing focal-only checkpoints |
| MULTI seeds 0,1,2 | Root-cause treatment | Stage 1 architecture and recipe, 128,000 samples drawn without replacement from fully observed category-3 focal and category-2 SCORED TRAIN agents, 60,000 steps, batch 128, learning rate 1e-3 |
| PATCH seeds 0,1,2 | Mechanism treatment | Corresponding ALL checkpoint, CV for selected stopped agents at each replan |
| SHAM seeds 0,1,2 | Dose-matched sham | Corresponding ALL checkpoint; at each replan, the same NUMBER of selected agents as PATCH would substitute get constant-velocity forecasts, drawn at random (fixed seed per scenario and replan) from the selected agents that are moving |
| CV | Training-free comparator | Stage 2 constant velocity |
| ORACLE | Ceiling | Logged true agent futures, as in Stage 2 |
| STATIC | Positive control | Stage 2 stationary forecast |
| LOG | Checker calibration | Logged ego trajectory |

The fixed dev scenario IDs in `runs/dev_scenarios.npy`, equivalently `cityshift.data.dev_indices(focal_train_meta, 0.02)`, are excluded **by scenario ID** from MULTI training. The multi-agent preprocessing output contains separate `train` and `dev` splits and records scenario, focal, and non-focal counts. Three seeds independently vary model initialisation and sample draw. The only intended training contrast with ALL is the centre-agent distribution. Sampling is uniform over eligible samples, so city and agent-type proportions may differ; these are reported, not corrected using validation.

## Closed-loop v2, fixed before scoring

Stage 2's 6-second window, route, agent selection, six replans, 11 acceleration choices, weights, forecast probabilities, hard-brake rule, non-reactive agents, and scoring window remain exactly as implemented. The logged future **path** is retained and explicitly treated as a navigation route. Two corrections apply to every arm, including ORACLE, CV, STATIC, and LOG where relevant:

1. Ego speed cap is `max(v0 + 3 m/s, 15 m/s)`, where `v0` is observed ego speed at t=49. This cap never uses future speed. The 15 m/s floor keeps typical urban progress possible while avoiding the Stage 2 cap's future-speed privilege. It is frozen for validation.
2. At-fault collision attribution uses the centroid of the oriented-box overlap polygon at the **first overlapping step**. The collision is at fault only when that contact centroid lies in the front half of the ego box. A rear contact is excluded. The first step and all simultaneously overlapping agents are checked. The same exact box-overlap detection, dimensions, and logged-agent scoring remain.

PATCH's trigger is observed speed <0.5 m/s at each replan, before future information is available. The sham applies the same dose of constant-velocity substitution to moving agents instead, so it removes the stopped-agent information while keeping the amount of intervention. H8 is read as stopped-specific only if SHAM's pooled brake reduction versus ALL is less than half of PATCH's; otherwise H8 is reported as a real reduction that is not stopped-specific.

## Confounds and controls

- **A1 onset:** fixed handoff t=49, no online onset detector.
- **A2 exposure:** identical scenarios, 6 s windows, replan times, route, and agent-selection algorithm for every arm; closed-loop ego histories may diverge by arm.
- **A3 seeds:** average three seed outcomes per scenario for ALL, MULTI, PATCH, and SHAM before city aggregation. Never treat seeds as independent scenarios.
- **A4 bundling:** MULTI changes training-centre distribution only; all planner settings and scoring rules are common. Report city, type, speed, and focal proportions of training samples to expose residual composition differences.
- **A5 positive controls:** STATIC must cause at least 2x ALL collisions; LOG must stay below 1% at-fault collisions.
- **A6 outcomes:** H7/H8 are driving outcomes in this longitudinal, non-reactive-agent harness; H9 is open-loop miss rate and cannot alone establish driving benefit. The logged future path remains privileged navigation information.
- **A7 dose:** PATCH and SHAM substitute exactly the same number of agents at every replan of every scenario (fewer only when a replan has fewer moving than stopped agents, which is reported). Report realised substitution counts per arm.
- **A8 floor:** if ALL has fewer than 100 seed-averaged unnecessary-brake events on validation, H7/H8 are underpowered regardless of effect estimate. Undefined city ratios make the corresponding claim inconclusive.
- **A9 selection:** no validation tuning. Any bug correction after registration is logged below before scoring; planner constants are otherwise frozen. TRAIN-dev alone may be used for implementation checks.
- **A10 leakage:** only ORACLE sees future agent motion. The common logged future route is a navigation route, while speed cap and PATCH trigger use current observations only. Never train on a dev scenario or make decisions from validation outcomes.

STATIC at-fault collision rate must be at least **2x** ALL's, or H7/H8 are uninterpretable. LOG at-fault collision rate must be **<1%** with the new attribution rule, or H7/H8 stop for checker investigation. SHAM gates only the stopped-specific reading of H8, as above. ORACLE, CV, and progress are reported as context, without additional confirmatory claims.

## Analysis and stopping

For each closed-loop arm, average the three seed outcomes per scenario. Within each city, compute the unnecessary-brake relative reduction `(ALL - treatment) / ALL` and collision difference `treatment - ALL`. H7/H8 pooled effects are **equal-weight means of six city effects**. For H9, average three seed miss values per agent, calculate each city's MULTI minus ALL miss difference within the specified agent group, then average six cities equally. Use **10,000 paired scenario-cluster bootstrap draws within each city**, keeping all agents of a sampled scenario together for H9. All six directional inequalities (H7 brake and collision, H8 brake and collision, H9 stopped and focal) use two-sided **99.1667% percentile CIs** (six-comparison Bonferroni family at 5%). H7/H8 brake success requires CI lower bound >0 and point reduction >=30%. Collision non-inferiority requires CI upper bound <+0.003. H9 stopped success requires CI upper bound <0 and point difference <=-0.15; focal non-inferiority requires CI upper bound <+0.02. These component decisions are joined within each claim. Report component CIs and per-city estimates even on failure.

**Kill criteria:** H7 or H8 is killed if its point brake reduction is <30%, its brake CI includes zero, or its collision CI upper bound is >=+0.003, provided controls pass and it is powered. H9 is killed if its stopped-agent point improvement is <0.15, its stopped CI includes zero, or its focal CI upper bound is >=+0.02. An absent group in any city, failed control, undefined ratio, or insufficient event count yields an inconclusive or uninterpretable result as specified above, never support.

Stop after exactly one scoring pass over all 24,988 validation scenarios and one fixed analysis. No validation-driven retraining, threshold adjustment, added hypothesis, or optional stopping. Exploratory breakdowns are labelled permanently. Smoke runs are limited to at most 200 fixed TRAIN-dev scenarios and are never inferential.

## Deviations

1. 2026-09-29, before any Stage 3 validation scoring or MULTI training: the first draft's SHAM permuted mode order and
   probabilities. The planner sums risk over all modes, so that sham was a no-op by construction (an identity check,
   not a sham). Replaced by a dose-matched sham: constant-velocity substitution on an equal number of randomly chosen
   moving agents per replan. The analysis now uses SHAM only to decide whether H8 is stopped-specific, and no longer
   gates H7 on it.
