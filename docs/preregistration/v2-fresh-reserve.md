# Pre-registration, v2 studies R and G: fresh-reserve replication with new checkpoints, and a geometry/probability factorial

Registered 2026-10-02, before the v2 checkpoints were trained and before any reserve scenario was scored. Stages 1 to 4,
the follow-ups and their results are frozen. Everything below is fixed; changes are logged under Deviations.

## Why

Stage 4 replicated PATCH on fresh *scenes* but with the *same* focal-only checkpoints that Stage 3 had used, and every
follow-up (A to E) reused the Stage 4 pool. Study R repeats the braking test with **newly trained checkpoints on a
reserve that no study has scored and those checkpoints never trained on**. PATCH changes a stopped agent's trajectory
and its probabilities together, and TRIM mostly fell back to constant velocity, so the mechanism was not isolated;
study G splits PATCH into its geometry and probability parts on the same drives.

## Reserve and checkpoints

- **Reserve:** 10,000 TRAIN scenarios drawn uniformly (seed 20261002) from the 187,770 TRAIN scenarios outside the
  fixed 2% development slice and outside the Stage 4 replication pool, which are the only TRAIN scenarios any study has
  scored. Script `scripts/make_reserve.py`. Sorted, newline-joined ID SHA-256
  `286a9423ffd907eebf7d67a6c7f66b2edfc6d1c4309e2a7f03e2633f8aac52ed`. Cities: Miami 2,644, Austin 2,218, Pittsburgh
  2,103, Washington DC 1,220, Dearborn 1,203, Palo Alto 612.
- **ALLR seeds 0, 1, 2:** the Stage 1 ALL recipe unchanged (128,000 examples drawn proportionally from all six cities,
  60,000 updates, batch 128, AdamW 1e-3, cosine, 1,000 warm-up, bf16), except that the reserve is removed from the
  draw (`cityshift.train --exclude-ids`; without the flag the draw is byte-identical to the registered one, checked for
  seed 0). Zero reserve scenarios can be drawn (asserted at training time). Every checkpoint passes
  `cityshift.validate_runs` (finite weights, losses and development metrics) before scoring; a diverged run is rerun
  once with the same seed, as registered in Stage 1.

## Harnesses

- **Primary: the stop-line harness** (`closedloop_v4`, study E): the ego may only choose accelerations after which it
  can still stop at the end of the logged route at 3 m/s^2 or less. Braking is the outcome; in this harness collisions
  have no working positive control (study E pilot), so they are descriptive.
- **Secondary: the Stage 4 full-drive harness** (`closedloop_v2`), for direct comparison with H10 to H12. Its collision
  components are reported, but most of its collisions occur past the logged route and the on-route collision control
  fails (study D), so they cannot support a collision-safety reading.

## Study R: replication with new checkpoints

**Claim.** On the fresh reserve, with newly trained focal-only checkpoints, PATCH reduces unnecessary hard braking by at
least 30% relative to ALLR, its reduction exceeds a dose-matched sham's by at least 20 percentage points, and TRIM
reduces it by at least 30% (the registered H10, H11 and H12 bars).

| Arm | Role |
|---|---|
| ALLR seeds 0 to 2 | No-treatment control |
| PATCH seeds 0 to 2 | Treatment: CV with probability one for selected agents moving < 0.5 m/s at the replan |
| SHAM2 seeds 0 to 2 | Dose-matched sham: PATCH's realised per-replan count, random selected agents |
| TRIM seeds 0 to 2 | Stage 4 TRIM, with its CV fallback |
| ORACLE | Ceiling and brake positive control |
| CV, STATIC, LOG | Descriptive comparators and checker calibration |

## Study G: geometry versus probability

For every selected agent moving < 0.5 m/s at a replan (PATCH's trigger), let k* be the agent's own mode with the
smallest maximum displacement from its current position over the planner's 4 s look-ahead.

| Arm | Change to the stopped agent's forecast |
|---|---|
| GEO seeds 0 to 2 | Replace mode k*'s trajectory with constant velocity; keep every probability |
| PROB seeds 0 to 2 | Keep every trajectory; put probability one on mode k* |
| PATCH | Both (one CV mode with probability one; zero-probability modes carry no planner risk) |

**Claim.** The probability change accounts for at least half of PATCH's brake reduction: G = reduction(PROB) /
reduction(PATCH) >= 0.5.

**Positive controls (unit tests, `tests/test_study_g.py`, run before scoring):** permuting mode order never changes the
planner's choice, and GEO followed by PROB gives the same plan as PATCH. Dose: GEO, PROB and PATCH act on the same
trigger (every stopped selected agent); their realised totals are reported.

## Controls

- **Brake positive control:** ALLR's pooled unnecessary-brake rate is at least twice ORACLE's, else every brake verdict in
  that harness is uninterpretable.
- **Dose:** SHAM2's realised total dose is at least 99% of PATCH's and never above it at any replan, else R's sham
  component is uninterpretable.
- **Power (A8):** at least 100 seed-averaged ALLR brake events, no city with zero ALLR brakes, at most 1% undefined
  bootstrap draws per component; otherwise the component is inconclusive.
- **Checker calibration:** replayed human drive at-fault collisions below 1% (reported).
- **Secondary harness only:** STATIC collides at least twice as often as ALLR, else its collision components are
  uninterpretable.

## Estimator and decisions

As registered for Stage 4 and study E: per scenario, average seeds 0 to 2; within each city, relative reduction
`(ALLR - arm) / ALLR` on the unnecessary-hard-brake rate; equal city weights; 10,000 bootstrap draws resampling scenarios
with replacement within city, arms and seeds paired, generator seed 20261002. The ratio G is computed per draw from the
same city-equal reductions.

**Primary family (stop-line harness), four components, decisions at the Bonferroni 98.75% level:**

| Component | Supported if | Killed if (powered, controls pass) |
|---|---|---|
| R1 PATCH reduction | point >= 0.30 and lower bound > 0 | point < 0.30 or lower bound <= 0 |
| R2 PATCH minus SHAM2 reduction | point >= 0.20 and lower bound > 0 | point < 0.20 or lower bound <= 0 |
| R3 TRIM reduction | point >= 0.30 and lower bound > 0 | point < 0.30 or lower bound <= 0 |
| G1 ratio PROB / PATCH | lower bound >= 0.5: probability sufficient | upper bound < 0.5: geometry needed; otherwise inconclusive |

**Secondary (95% intervals, labelled secondary):** R1 to R3 and G1 in the full-drive harness; collision changes against
the +0.3 percentage-point margin there; GEO's reduction and the interaction reduction(PATCH) - reduction(PROB) -
reduction(GEO); per-seed reductions (training-seed variability, descriptive).

**Secondary outcome, defined now:** an *excess hard brake* is a replan at which the chosen acceleration is -4 m/s^2 or
harder, the logged human never decelerated that hard, every selected agent's logged future covers the 4 s look-ahead,
and some candidate acceleration of -2 m/s^2 or gentler has zero planner hit probability when the selected agents are
given their logged futures (the ORACLE forecast) and, in the stop-line harness, satisfies the stop-line rule. A drive
counts if any replan qualifies. Computed after scoring by replaying stored accelerations on the CPU; PATCH's reduction is
reported at 95%.

## Confounds

- **A1 onset, A2 exposure:** fixed handoff, six replans, identical scenarios and route for every arm; ego histories
  diverge by arm, which is the closed loop.
- **A3 seeds:** three new checkpoints, averaged per scenario; intervals are conditional on these checkpoints; per-seed
  effects reported.
- **A4 bundling:** GEO and PROB each change one axis of PATCH; TRIM's fallback rate is reported.
- **A5:** brake positive control (ORACLE), mode-permutation and PATCH-identity unit tests, dose gate.
- **A6:** the braking outcome is a registered proxy; the excess-hard-brake outcome tightens it against logged futures.
- **A7 dose:** SHAM2 matched per replan; GEO and PROB share PATCH's trigger.
- **A8:** power gates above.
- **A9:** arms, thresholds, harnesses and estimator are copied from Stage 4 and study E; GEO, PROB, k* and the excess
  brake were defined here before any training or scoring.
- **A10:** the reserve is disjoint from ALLR training (asserted), from the development slice and from the Stage 4 pool.

## Stopping

Train once; score the full reserve once per harness; one fixed analysis. Smoke runs are limited to at most 100
development-slice scenarios and only their runtime and memory are examined. Negative and inconclusive results are
published with the same prominence.

## Deviations

None at registration.
