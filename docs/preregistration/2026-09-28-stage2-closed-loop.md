# Pre-registration, Stage 2: does the city shift reach driving outcomes?

Registered 2026-09-28 (late evening), before any Stage 2 run on the validation split. Stage 1
(`2026-09-28-city-shift.md`) asks whether prediction error rises in an unseen
city. Stage 2 asks whether a planner that consumes those predictions drives
worse there. Harness development and debugging use only scenarios from the
**train** split's dev slice; the validation split is touched once, for the
confirmatory run below.

## The harness (ego-replay closed loop)

Every Argoverse 2 scenario contains the self-driving car's own track (`track_id
== "AV"`, all 110 steps; checked on 200/200 sampled val scenarios). Per scenario:

1. **Handoff.** The ego takes its logged state at t = 49 (5.0 s). From then on its
   motion is simulated; every other agent replays its logged trajectory
   (non-reactive).
2. **Route.** The ego follows its own logged future path (arc-length
   parametrised, extended straight past its end). The planner chooses only the
   speed profile along it (longitudinal planning).
3. **Predict.** At each replan time t in {49, 59, 69, 79, 89, 99} the predictor
   forecasts the M = 12 nearest dynamic agents (vehicle, bus, motorcyclist,
   cyclist, pedestrian) that are within 60 m of the ego and observed at t. Inputs
   are built exactly as in training (agent-centric, 50-step history ending at t),
   with the ego's history replaced by its simulated states after t = 49.
4. **Plan.** 11 candidate profiles: constant acceleration a in {-8, -6, -4, -3,
   -2, -1, -0.5, 0, 0.5, 1, 2} m/s^2, speed clamped to [0, max(v0 + 5, 1.2 v_max_logged)].
   Cost = 100 * sum over agents of P_hit + 1.0 * (1 - progress / progress_max) +
   0.05 * |a|, where P_hit = sum of mode probabilities whose trajectory brings the
   agent's box within 0.5 m of the ego's box in the ego's front half-plane within
   the next 4 s. The cheapest candidate is executed for 1 s (10 steps), then the
   planner replans.
5. **Score** against the logged futures of *all* dynamic agents, steps 50..109.

Boxes (length x width, m): ego 4.9 x 2.0; vehicle 4.5 x 2.0; bus 12.0 x 2.6;
motorcyclist and cyclist 2.0 x 0.8; pedestrian 0.6 x 0.6. Overlap is an exact
oriented-box (separating-axis) test.

## Outcomes (per scenario)

- **At-fault collision:** ego box overlaps a logged agent box at some step, with
  the agent's centre in the ego's front half-plane. Rear-end hits by a
  non-reactive follower are excluded, as in nuPlan's at-fault rule.
- **Unnecessary hard brake:** the executed profile uses a <= -4 m/s^2 at some
  replan while the logged ego never decelerates below -4 m/s^2 in steps 50..109
  (finite difference of logged speed, 0.5 s window).
- **Planning failure (primary):** at-fault collision OR unnecessary hard brake.
  Two failure directions, under-reaction and phantom braking, in one binary.
- Secondary: each component alone, progress ratio (distance travelled / logged
  distance), minimum front clearance.

## Step 1. Claim

**H4.** On city-c validation scenarios, the planner using the LOCO-c predictor
has a higher planning-failure rate than the planner using the ALL predictor, by
at least **+10% relative**, pooled over the six folds.

## Step 2. Arms

| Arm | Purpose |
|---|---|
| **P-LOCO** (treatment) | planner + LOCO-c predictor, per seed 0..2, city-c scenarios |
| **P-ALL** (no-treatment control) | planner + ALL predictor, per seed 0..2, same scenarios |
| **P-ORACLE** (ceiling) | planner given each agent's logged future as one mode with p = 1 (constant-velocity fill where the log ends) |
| **P-CV** (floor) | planner + constant-velocity forecasts, one mode |
| **P-LOG** (checker calibration) | the logged ego speed profile, no planner |
| **P-STATIC** (positive control PC3) | planner told every agent stands still |

Everything except the predictor is identical across P-LOCO, P-ALL, P-ORACLE,
P-CV and P-STATIC: same scenarios, same agent set, same planner, same scoring.
Seeds enter by averaging the per-scenario failure over the three seeds of an
arm (values in {0, 1/3, 2/3, 1}), as in Stage 1.

**Positive controls, must pass or the corresponding null is uninterpretable:**
- PC3: P-STATIC at-fault collision rate is at least 2x P-ALL's. If not, the
  planner is not using the predictions.
- Checker calibration: P-LOG at-fault collision rate < 1%. If higher, the
  collision checker or the box sizes are wrong, and Stage 2 stops for a fix
  (logged as a deviation, re-registered).

## Step 3. Confounds

- **A1 onset.** Not an online detector. Scored over a fixed 6 s window.
- **A2 exposure.** Every scenario scores the same window, same replan times, same
  agent-selection rule. Agent count differs by scenario but is identical across
  arms within a scenario (pairing).
- **A3 seeds.** 3 per predictor arm; inference unit is the fold (n = 6), exact
  sign-flip floor two-sided 0.031.
- **A4 bundling.** The arms differ only in the predictor. The agent set is chosen
  at each replan from logged positions (independent of the predictor); the
  ego's own simulated history does depend on the arm, which is the closed loop.
- **A6 outcome.** Collision and hard brake are driving outcomes, the point of
  Stage 2. Limits, stated rather than claimed away: other agents do not react;
  the planner is longitudinal only; there is no perception noise; the
  simulation window is 6 s.
- **A7 dose.** No trigger or rejection arm; not applicable.
- **A8 floor.** Failure events may be rare. Pre-set rule: if P-ALL has fewer than
  100 failure events summed over all folds (seed-averaged), H4 is reported as
  underpowered, whatever its point estimate.
- **A9 selection.** Planner constants above are fixed now. During harness
  development on the dev slice, only bugs may be fixed; if a constant must
  change for the P-LOG or PC3 check to be meaningful, it is logged below before
  the val run.
- **A10 leakage.** The planner has no fitted parameters. The oracle arm uses the
  future by design and is labelled as such.

## Step 4. Analysis plan

- **H4 primary:** per fold c, `rel_c = (F_LOCO(c) - F_ALL(c)) / F_ALL(c)`,
  F = mean seed-averaged failure on city-c val scenarios. Pooled = mean over
  folds; 95% CI by scenario bootstrap within city (10,000 draws, pairing kept);
  exact sign-flip p over folds, two-sided.
- Multiplicity: superseded by Stage 1 deviation 3 (Bonferroni-level bootstrap CIs across H1 to H4; the fold sign-flip p is fold consistency only).
- Reported for context: P-ORACLE, P-CV, P-LOG, P-STATIC rates per city.
- Exploratory, labelled permanently: collision and hard-brake components alone,
  progress, the per-scenario association between Stage 1 prediction miss on the
  ego-relevant agents and Stage 2 failure.

## Kill criterion

H4 is dead if the pooled 95% CI includes 0, or the point estimate is below +10%.
Published as "the accuracy loss in an unseen city did not propagate to driving
outcomes in this harness", with the harness limits listed above. The negative is
reported with the same prominence as a positive.

## Deviations

All made during harness development on the train split's dev slice, before any
Stage 2 run on val, under the A9 clause.

1. **Agent selection.** Registered: the 12 nearest agents to the ego. Changed to:
   up to 16 agents within 60 m, ranked by how close their nearest edge (centre
   distance minus half length) comes to the ego's route over the next 80 m.
   Reason: with oracle forecasts the planner still hit 3/200 dev scenarios; in the
   traced case a stopped 12 m bus was outside the 12-nearest set (centre-distance
   ranking, crowded scene) until it was 12 m away at 13 m/s.
2. **Speed cap.** Registered: max(v0 + 5, 1.2 * v_max_logged). Changed to
   max(v0, v_max_logged) + 1 m/s, to keep the ego near the human's pace, a partial
   proxy for traffic controls the planner cannot see.
3. **Logged hard brake.** Registered: logged speed finite difference over 0.5 s.
   The raw velocity channel flagged 41% of logged drives (200 dev scenarios);
   changed to speed from positions, 0.5 s moving average, decel over 1 s windows
   (0.2% of 1,000 dev scenarios). The planner's hard brake is the realised speed
   change over each executed 1 s segment, <= -4 m/s^2.
4. Front clearance (exploratory) is computed as centre distance minus the two
   half lengths, over agents in the front half-plane.

Dev-slice check after these changes (1,000 scenarios, analytic arms only):
at-fault collision LOG 0.5%, ORACLE 0.2%, CV 2.5%, STATIC 3.1%; the ego still
drives 1.55 to 1.73x the logged distance, recorded as a harness limitation.

5. 2026-09-28, analysis plan (before any val run): decision by the 98.75%
   (Bonferroni, four hypotheses) scenario-bootstrap CI, per Stage 1 deviation 3.
   A fold whose ALL failure rate is exactly zero has no defined relative change;
   it is excluded from the pooled effect and the fold-consistency test and listed
   in the output as `folds_undefined`. A fold is also excluded if its ratio is
   undefined in more than 1% of its bootstrap resamples, so the pooled interval is
   not conditioned on non-zero-event draws (share reported per city) (previously a NaN could reach the sign-flip
   test and yield a spurious p = 0). The file was also renamed from a 09-29 date
   to 09-28, the date it was actually written and committed.
