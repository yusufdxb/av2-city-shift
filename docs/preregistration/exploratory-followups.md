# Pre-registration, exploratory follow-ups: departing stopped agents, route-end censoring, TRIM without fallback

Registered 2026-09-30, after Stage 4 and before any of the analyses below produced output. All three are
**exploratory**: they reuse the Stage 4 replication pool (8,140 TRAIN scenarios, ID SHA-256
`a7489d7bd7b95986807ee360e62a01c45a15f4c3297258ba4751f8520caee0c8`), whose registered H10 to H13 results are already
public, so none of them can change a registered verdict and every result stays labelled exploratory. Stages 1 to 4,
their code and their decisions are frozen.

What is already known and therefore cannot be treated as blind (A9): the Stage 4 per-arm aggregate rates and effects,
and from the exploratory sweep that 82% of ALL and 95% of PATCH drives at the registered settings end past the end of
the logged route. Not yet computed by anyone at registration: the identity of any collided agent, any outcome
restricted to a censoring window, and any TRIM-NF or PATCH-SUB rollout.

## Shared machinery: exact replay of the Stage 4 drives (CPU)

The closed loop is deterministic given the scenario and the six executed accelerations: the ego follows the logged
route (extended straight past its end), starts at the logged handoff speed, is capped by `closedloop_v2.speed_cap`,
and executes each chosen constant acceleration for 1 s with speed clipped to [0, cap]. The Stage 4 table stores the
six accelerations per scenario, arm and seed (`<arm>_accels`). Studies B and C rebuild every ego trajectory from
these and rescore it; no forecasts or planning are rerun.

**Replay parity gate (positive control for B and C).** Before any B or C outcome is computed, the replay must
reproduce, for every scenario and every stored arm and seed (ALL, PATCH, TRIM, SHAM2, MIX for seeds 0 to 2; CV,
ORACLE, STATIC), the stored `collision`, `first_collision_step`, `planner_hard_brake`, `logged_hard_brake`,
`unnecessary_hard_brake` exactly and `progress` to 1e-9 (NaN where stored NaN). Any mismatch stops B and C until the
cause is found and recorded as a deviation. The censored scorer of study C, run with the cutoff at the last step
(t=109), must also reproduce the stored outcomes exactly, and the Stage 4 estimator applied to those outputs must
reproduce the registered H10 point estimates to 1e-12.

## Shared estimator

Unless stated otherwise, as registered for Stage 4: per scenario, average seeds 0 to 2 for model arms; within each of
the six cities compute the effect; equal-weight the six city effects; bootstrap 10,000 times, resampling scenarios with
replacement within city, preserving arm and seed pairing. Intervals are two-sided 95% percentile intervals, with the
Bonferroni 98.33% interval (three primaries) reported alongside; no other multiplicity correction. Relative reductions
`(ALL - arm) / ALL` are undefined in draws where ALL's rate is zero; the undefined share is reported, and a primary
whose undefined share exceeds 1% is inconclusive. One analysis pass, no optional stopping; negative and inconclusive
results are published.

## Study B: collisions with stopped agents that depart (CPU)

**Claim.** On the Stage 4 pool, PATCH (constant velocity for selected agents moving <0.5 m/s) raises the rate of
at-fault collisions with *departing stopped agents* by less than +0.3 percentage points relative to ALL.

**Definitions.** An agent is a *departing stopped agent* for a drive if it is a dynamic agent, observed and moving
<0.5 m/s at the handoff (t=49), and its logged position later moves more than 2 m from its t=49 position at any
observed step up to t=109. This uses logged agent motion only, so the set is identical for every arm. A drive's
*collision partner(s)* are the agents that overlap the ego at the first overlap step and pass the contact-point
at-fault test, exactly as `closedloop_v2.score` decides `collision`; the replay records their identity. The outcome is
1 if the drive has an at-fault collision and any partner is a departing stopped agent.

| Arm | Role |
|---|---|
| ALL seeds 0 to 2 | No-treatment control (Stage 4 rows, replayed) |
| PATCH seeds 0 to 2 | Treatment (Stage 4 rows, replayed) |
| CV | Comparator: forecasts every agent, stopped ones included, as continuing at its current speed |
| ORACLE | Ceiling: true futures |
| STATIC | Positive control: forecasts every agent as stationary |

**Primary.** PATCH minus ALL departing-stopped collision rate, shared estimator. Descriptive: the same for CV, ORACLE
and STATIC against ALL; all at-fault collisions split into departing-stopped, other-stopped (stopped at t=49, never
moves 2 m) and moving partners; and a secondary definition that counts an agent stopped at *any* replan (49, 59, ...,
99) and moving more than 2 m from that replan's position later.

**Positive control.** STATIC minus ALL departing-stopped collision rate must have a 95% lower bound >0. STATIC plans as
if every agent stays put, so if the attribution works it must show extra collisions with agents that then move. If it
fails, B is uninterpretable.

**Decision and kill.** Upper 95% bound < +0.003: PATCH's collision non-inferiority holds for departing stopped
agents. Upper bound >= +0.003: the claim is dead and published as "not established". A lower bound >0 is reported
as a detected increase whatever its size.

**Confounds.** A1: fixed handoff and replans; the departing set is log-defined. A2: PATCH drives further than ALL
(Stage 4 progress 2.14x vs 1.83x), so it is exposed to more agents; the primary is the per-drive rate (what matters
for safety), and per-metre-driven rates are reported descriptively. A3: three seeds averaged within scene. A4: PATCH
changes only stopped agents' forecasts. A5: STATIC control above plus replay parity. A6: at-fault collision is the
registered outcome. A7: no trigger comparison is claimed, so no sham is needed; PATCH's substitution counts are
reported. A8: if ALL, PATCH and STATIC together have fewer than 20 seed-averaged departing-stopped collisions, B is
underpowered and reported as such. A9: definitions fixed here before any partner identity was computed. A10: no
fitting.

## Study C: route-end censoring (CPU)

**Claim.** When each ALL and PATCH pair (same scenario and seed) is scored only up to the first moment either drive
passes the end of the logged route, PATCH still reduces unnecessary hard braking by at least 30% relative to ALL,
with the at-fault collision increase below +0.3 percentage points (the registered H10 bars).

**Censoring window.** For each scenario and seed, `cross(arm)` is the first step at which that arm's distance along
the route exceeds the logged route length (t=110 if never). The cutoff is `c = min(cross(ALL), cross(PATCH)) - 1`,
capped at t=109. Within the window: a replan's executed 1 s deceleration counts if its segment ends at or before `c`
(`t + 10 <= c`); the logged human hard-brake test uses only the 1 s windows ending at or before `c`; a collision
counts if the first at-fault overlap is at or before `c`. Unnecessary hard brake = a counted replan at or below -4
m/s^2 while no counted human window is. Drives with `c < 59` have no scored segment; their share is reported and they
contribute zero events to both arms. Routes of 1 m or less (Stage 4 `progress` NaN) get whatever window this rule
gives them; their count and window lengths are reported separately.

| Arm | Role |
|---|---|
| ALL seeds 0 to 2 | No-treatment control |
| PATCH seeds 0 to 2 | Treatment |
| Uncensored (c = 109) | Reference: must equal the registered H10 point estimates |

**Primary.** PATCH relative brake reduction and collision difference over the common window, shared estimator.
Descriptive: window length per scene; each arm censored at its own crossing (unequal exposure, not comparable); the
share of brakes and collisions that the registered analysis counted after the cutoff.

**Positive control.** The degenerate-cutoff parity above.

**Decision and kill.** Supported if the reduction point is >=0.30 with 95% lower bound >0 and the collision upper
bound is <+0.003. Killed, and published as "PATCH's braking reduction is not established within the logged route",
if the reduction point is <0.30 or its lower bound is <=0, or the collision upper bound is >=+0.003, provided A8
holds.

**Confounds.** A2 is the reason for this study: a common window per pair gives both arms identical exposure time on
the logged route. A8: fewer than 100 seed-averaged ALL brake events in the window, a city with zero, or >1% undefined
draws makes C inconclusive. A9: the censoring rule is fixed here before any censored outcome was computed. Other
items as in B.

## Study A: TRIM without the constant-velocity fallback (GPU, not yet run)

**Claim.** On stopped agents whose model forecast has a native stationary mode, removing the moving modes' probability
(TRIM-NF) gives at least half the brake reduction of replacing the same agents' forecasts with constant velocity
(PATCH-SUB), both relative to ALL.

| Arm | Role | Forecast |
|---|---|---|
| ALL seeds 0 to 2 | No-treatment control | Stage 4 rows; a 100-scenario rerun must match them exactly |
| TRIM-NF seeds 0 to 2 | Treatment | For selected agents moving <0.5 m/s whose own modes include one ending <=2 m from the current position (with positive probability): keep those modes, renormalise. Other agents untouched (no fallback) |
| PATCH-SUB seeds 0 to 2 | Dose-matched comparator | The same eligibility test, decided on the same model forecast; eligible agents get constant velocity with probability one |
| ORACLE | Ceiling | Stage 4 rows |

**Primary.** Ratio `R = reduction(TRIM-NF) / reduction(PATCH-SUB)`, shared estimator with R computed per bootstrap
draw from the equal-weighted city reductions.

**Positive control.** PATCH-SUB's reduction must have a 95% lower bound >0; otherwise the eligible subset is too small
to test anything and A is uninterpretable.

**Decision and kill.** Supported if R's 95% lower bound is >=0.5. Killed if its upper bound is <0.5 (probability
removal alone gives less than half), published. Otherwise inconclusive. Secondary: TRIM-NF collision difference
against the +0.003 margin.

**Confounds.** A4: both arms act on the same eligibility rule and the same model forecast and differ only in how the
forecast is changed. A7: ego histories diverge after the first replan, so the eligible sets can drift; realised
eligible counts per arm are recorded per replan and the total ratio must be within 0.95 to 1.05, else A is
uninterpretable. A8 as in C. Estimated compute: about 49,000 drives (two arms, three seeds, 8,140 scenarios), roughly
45 minutes on one GPU at the exploratory sweep's measured rate.

## Study D: route-end censored scoring of every registered closed-loop component (CPU), added 2026-09-30

Registered as an amendment before any D outcome was computed, after studies B and C had reported. Study C showed that
77 to 84% of Stage 4 at-fault collisions first make contact past the end of the logged route. D adds a versioned
scoring mode, **route-end censoring**, and applies it to every registered closed-loop component of Stage 3 (H7, H8)
and Stage 4 (H10, H11, H12, H13), reported beside the registered numbers. It replaces nothing: registered verdicts stay
as registered.

**Claim.** Under route-end censoring, every registered Stage 3 and Stage 4 closed-loop component keeps its registered
verdict when the registered estimator, CI level, decision rule, controls and power floor are applied to the censored
outcomes.

**Scoring mode.** For a drive, `cross` is the first step whose distance along the route exceeds the logged route
length. For a comparison between a set of arms, the cutoff for a scenario and model seed is `min(cross) - 1` over those
arms, capped at t=109, and every arm in that comparison is scored with the rules of study C inside it. Comparison sets:
H7 {ALL, MULTI}, H8 {ALL, PATCH}, the Stage 3 sham {ALL, SHAM} (descriptive, as registered), H10 {ALL, PATCH}, H11 {ALL,
PATCH, SHAM2}, H12 {ALL, TRIM}, H13 {ALL, MIX}. Controls: STATIC and ALL use one window per scenario, the minimum over
STATIC and ALL seeds 0 to 2; LOG is the human's own drive, which never passes its route, so it is unchanged. Open-loop
components (H9, H13 focal miss) and the SHAM2 dose are unaffected and carried over.

**Estimator.** The registered analysis code of each stage (`analysis_stage3.closedloop_analysis` with generator seed
3407, `analysis_stage4.analyze` with its fixed seed), run once per comparison set on a table whose outcome columns are
the censored ones, keeping only that comparison's components.

**Positive controls.** (1) With every cutoff at t=109, the same pipeline must reproduce the registered Stage 3 and
Stage 4 closed-loop points and intervals exactly. (2) D's H10 point must equal study C's primary point (0.676265).
Either failing stops D.

**Decision and kill.** Per component, the registered rule decides. A component whose censored verdict differs from its
registered verdict is reported as a flip, prominently, and the claim above is dead for it. Underpowered (fewer than 100
seed-averaged ALL brake events in the window), inconclusive and uninterpretable statuses are reported as such.

**Confounds.** A2: common windows per comparison equalise exposure time within each comparison; windows differ across
comparisons, so censored effects are not comparable across components. A9: the mode, windows and estimator are fixed
here before any D outcome. Other items as in B and C.

## Deviations

None at registration.

1. 2026-09-30, before any B or C outcome was computed: the bootstrap generator seed, not fixed above, is set to
   20260930 for both studies (the uncensored reference check uses its own draws and only its point estimate). The
   replay also records each arm censored at its own crossing, which the registration lists as a descriptive output.
   Replay parity passed on all 8,140 scenarios and 18 drives each (reports/followups/replay_parity.json).
2. 2026-09-30, after the first analysis output: the descriptive "counted only after the cutoff" shares were wrong
   because the common-window flags loaded as object dtype and `~` on a Python bool is -2 (truthy). The flags are now
   cast to booleans; the primaries, which average rather than negate the flags, were unchanged (reduction 0.6763,
   collision difference -0.00022, B difference -0.00019) and match an independent plain-pandas recomputation.
