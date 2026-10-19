# Planner-objective transfer and departure-preserving correction

Frozen before outcome collection for this extension, 19 October 2026. The released V2 experiments remain unchanged. This protocol allows a negative feasibility result: a large campaign is conditional on development controls, not on obtaining an attractive effect.

## Question and scope

Does the stopped-agent predictor-planner mismatch persist under a second planning objective, and can a held-out correction reduce excessive braking while retaining useful predictions of actual departures? This is an offline, non-reactive longitudinal replay experiment. Other actors replay their logs, the ego follows its privileged logged route, and the planner controls speed only. It is not a reactive benchmark, a nuPlan implementation, or a demonstration of driving safety.

nuPlan separates reactive and non-reactive settings and evaluates progress, collisions and comfort. That motivates broader measurement here, without equating these harnesses. See the [official competition description](https://github.com/motional/nuplan-devkit/blob/master/docs/competition.md) and [metric aggregation documentation](https://github.com/motional/nuplan-devkit/blob/master/docs/nuplan_metrics_aggeregation.md).

## Cohorts and frozen predictors

Use the three released ALLR predictors. No predictor retraining. Reproduce and validate every seed's training draw before selecting scenes. The existing 3,998-scene development set was excluded from gradient training but has been used for monitoring and earlier investigations. Subsets of it are development evidence, not an untouched test.

Deterministically allocate development fit (1,500), tune (500), and feasibility pilot (500) scenes, stratified by city. The remaining 1,498 development scenes are unused. For a possible final test, select 2,500 scenes from the 5,389 TRAIN scenes excluded from the union of all three ALLR training draws, the development set, the original replication pool, the V2 reserve and the additional previously scored development-smoke scene. Leave 2,889 eligible fresh scenes unused.

Membership digests use SHA-256 of sorted IDs joined by newline without a trailing newline:

| Set | Count | SHA-256 |
|---|---:|---|
| Fit | 1,500 | `89a71e4c3f3e2ab71a9a0f6f440c01046cc1f85aaaff95cdf9a56988f3ad3c1d` |
| Tune | 500 | `33a013da09f783ab666be90725be9e85f9425e6a1a8713e94c25d0d8c10a9d9a` |
| Pilot | 500 | `3a0587dfb34c5cc67c51e86abd6ccf3b9eb1b5dada9527703eaed7cc661f528f` |
| Conditional test | 2,500 | `f742f57401ddb520a9a0f7278c002c9d03a5c8b85cfb32460f5e3827c3f88840` |
| All fresh eligible | 5,389 | `d049998d71766823e670f1c7e3f817303c221e2888820ad0894fa850690e7cf8` |

`scripts/make_transfer_cohort.py` records portable checkpoint, metadata and training-membership fingerprints. Selection ranks IDs by SHA-256 with a role-specific prefix and uses city-proportional largest-remainder allocation. No outcome-dependent cohort selection.

## Second objective

Keep the existing stop-line dynamics, candidate accelerations, route, forecast inputs, four-second risk kernel, six-second progress horizon and one-second replanning. Compare:

1. **Weighted:** the released stop-line objective, risk weight 100, progress weight 1, absolute-acceleration weight 0.05.
2. **Risk budget:** among stop-line-feasible candidates with forecast risk at most 0.05, maximize route-capped progress, then minimize absolute acceleration, then candidate index. If none meets the budget, minimize forecast risk before those tie breakers. If the stop-line constraint is infeasible, execute the hardest brake, as in the original harness.

The risk is a sum over agents of probability-weighted mode overlap indicators. It is a union-bound surrogate, not a calibrated collision probability. Its inherited front-only gate may miss side or rear contacts. Independent all-direction contact scoring exposes this limitation; changing that kernel would confound the objective comparison.

## Correction and held-out departure evaluation

Compare NATIVE, PATCH, CALIB and SELECTIVE for every predictor seed and objective. PATCH replaces stopped-agent forecasts with constant velocity. CALIB preserves all native trajectories and reweights their departing versus stationary probability groups. Append a constant-velocity stationary alternative when native stationary support is absent. Retain at least 2% of original departing probability mass and preserve conditional weights within native groups. Do not invent moving geometry when native moving support is absent.

Fit a deterministic ridge-1 logistic departure calibrator on fit scenes only, pooling the three seeds with each scenario weighted equally. Inputs are the native departure logit and ten history-only features: current speed, one-second mean speed, two-second maximum speed, one- and two-second displacement, stopped and observed fractions over two seconds, consecutive stopped duration, recent acceleration and observed history span. No city feature or future input.

The departure label is maximum displacement greater than 2 m during the next four seconds, conditional on complete observations from now through that horizon. Evaluate currently stopped vehicles and buses (speed below 0.5 m/s) selected with the fixed logged ego at steps 49 and 69. Incomplete futures are unknown, never extrapolated or labeled stationary. Forecast corrections operate on selected stopped dynamic actors during planning; the primary preservation analysis concerns vehicles and buses.

SELECTIVE applies only downward corrections with calibrated departure probability below a threshold. Choose from 0.05, 0.10 and 0.20 on tune scenes only: minimize scenario-equal departure Brier score subject to departure recall at probability at least 0.10 falling by no more than 0.02 relative to NATIVE. Require positive Brier improvement. Resolve ties toward the smaller threshold. If none qualifies, use threshold zero (identity) and report that no admissible selective correction was found. Freeze the fitted model and threshold before pilot collection.

Report Brier score, clipped binary log loss, departure recall at probability at least 0.10, departing probability mass, and probability-top-1 four-second endpoint error separately for true departures and true stationary cases. Report independent scene counts, not just repeated actor/seed rows. Retaining trajectory geometry alone is not evidence of retained useful predictions.

## Controls and independent planning measures

Run oracle logged-future forecasts, constant velocity, and an optimistic blind planner that ignores forecasts. The oracle extrapolates only for planning when logged futures end; scoring labels never do. Synthetic fixtures must demonstrate both a harmful phantom forecast and a stopped vehicle that actually crosses: informative forecasts prevent contact while static or blind forecasts fail. Also test rear/side contacts, geometric clearance, stop-line feasibility, incomplete futures, and equality of the explicit weighted objective to the released implementation.

For every executed drive report realized and commanded hard braking, the original braking proxy, route-capped progress in metres and fraction, mean absolute jerk, all-direction uninflated observed box contacts, clearance below 1 m, and minimum Euclidean box-edge clearance. Report contacts with actors that were stopped and then departed. Do not censor contacts after the first event or by a front-only fault rule.

At steps 49, 59 and 69, form a common actor union across compared policies and all seeds within each scene and objective. Score a decision only if that union has complete four-second logged futures. Reconstruct each policy's own state from its executed commands. A genuine candidate conflict is uninflated all-direction overlap along its chosen four-second candidate. An excessive brake is a realized hard-brake decision for which a stop-line-feasible non-hard-braking candidate avoids all such contacts. Report common-support coverage and keep these counterfactual candidate measures separate from contacts on the executed six-second drive.

## Feasibility, inference and stopping

Run the 500-scene development pilot first. A final 2,500-scene campaign is allowed only if both objectives have at least 20 NATIVE excessive-braking scenes, at least 20 independent stopped-then-departing vehicle scenes overall, at least a twofold NATIVE-to-oracle excessive-braking rate, and at least five scenes where blind replay makes observed contact and oracle does not. These controls must distinguish useful from poor forecasts empirically. Failure stops the large planning campaign and is itself a reported outcome. Synthetic controls prove implementation behavior, not empirical feasibility.

Average predictor seeds within each scene before inference, then weight cities equally. Use paired scene bootstrap within cities, 10,000 replicates and a fixed random seed. SELECTIVE is the primary correction; CALIB is exploratory. A strong joint claim requires four 98.75% confidence bounds: less excessive braking than NATIVE, progress loss below 0.02, observed-contact harm below 0.003, and held-out departure-recall loss below 0.02. This Bonferroni allocation controls the four primary components for SELECTIVE. Report every component and PATCH comparison even when the joint claim fails. Do not interpret superiority on braking to PATCH as a requirement, since suppressing every departure is precisely its limitation.

For rare contact harm, supplement bootstrap intervals with conservative city-specific one-sided exact harmful-discordance bounds with simultaneous coverage, averaged across cities. A zero-event bootstrap interval cannot establish noninferiority. Require at least 100 baseline excessive-braking scenes and 100 independent departure scenes for a strong final claim. If these floors or precision requirements fail, label the claim underpowered. No threshold, risk budget or cohort changes after inspecting pilot or test outcomes; any follow-up is separately exploratory.

Publish executable procedures, hashes, control results and limitations. Keep the released V2 artifacts frozen. The application narrative leads with the investigation: predictor-planner mismatch, mechanism isolation, independent intervention replication, and the conditions where it stopped working.
