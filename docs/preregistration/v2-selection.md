# Pre-registration, v2 study S: is focal-agent selection the cause of the stopped-agent failure?

Registered 2026-10-02, before any study S model was trained or scored. Everything below is fixed; changes are logged
under Deviations.

## Why

The report's explanation for the stopped-agent failure is focal selection: Argoverse 2 focal agents are chosen for
interesting motion, so in the training split 13.1% of focal agents are stopped at the prediction time and 84% of
those move more than 2 m within 6 s (reports/census/focal_stopped.json). A model trained only on focal agents can learn
that "stopped" means "about to leave". So far this fits the evidence but was never tested. Study S intervenes on the
training data directly.

## Arms

All arms use the Stage 1 ALL recipe (128,000 examples drawn proportionally from all six cities, 60,000 updates, batch
128, AdamW 1e-3, cosine, 1,000 warm-up, bf16), seeds 0 to 2, and exclude the v2 reserve (so all are scored on it).

| Arm | Training exclusion | Role |
|---|---|---|
| ALLR | reserve only (study R checkpoints) | No-treatment control |
| SEL | reserve + the 20,515 TRAIN scenarios whose focal agent is stopped (< 0.5 m/s) at t=49 and ends the next 6 s more than 2 m away | Treatment: removes the stopped-then-departing association |
| SELSHAM | reserve + 20,515 TRAIN scenarios drawn uniformly (seed 20261002) from those whose focal agent is moving (>= 0.5 m/s) | Dose-matched sham: the same number of examples removed, without the association |

Exclusion lists: `scripts/make_selection_sets.py` (removed-set SHA-256: SEL
`2c471f3b64923663fad5160c3f40c423a8798d10f489e0c82e08c2bbeab1925e`, SELSHAM
`19a6893a20dce669ec426604afd360b0af773f840fdc411f6e8d7ab8cfd57017`). After SEL's removal 3,824 stopped focal examples
remain in its pool, all of them staying within 2 m. Every checkpoint passes `cityshift.validate_runs` before scoring.

## Claims

- **S1 (mechanism).** On stopped (< 0.5 m/s) non-focal planner agents of the reserve at the handoff (Stage 3a scoring,
  `multiagent_eval`), SEL's miss rate is lower than ALLR's by at least 0.15 absolute.
- **S2 (specificity).** SEL's miss rate on those agents is lower than SELSHAM's by at least 0.15 absolute.
- **S3 (braking).** In the stop-line harness on the reserve, SEL (focal-only, no forecast policy) reduces unnecessary
  hard braking by at least 30% relative to ALLR.

**Estimator.** S1, S2: per-city agent-pooled miss rates (each family's per-agent miss averaged over its three seeds),
city-equal difference, 10,000 scenario-cluster bootstraps within city (seed 20261002). S3: the study R estimator (seeds
averaged per scenario, city-equal relative reduction, within-city scenario bootstrap). The family S1 to S3 is decided at
the Bonferroni 98.33% level. S1, S2 supported if point <= -0.15 and upper bound < 0, killed otherwise. S3 supported if
point >= 0.30 and lower bound > 0, killed otherwise.

**Controls.** S3 uses study R's ALLR stop-line rows and its brake positive control (ALLR at least twice ORACLE) and
power gates (at least 100 seed-averaged ALLR brake events, no city with zero). A failed control makes S3
uninterpretable. S1 and S2 are open-loop and need no planner control; the moving non-focal group is reported as a
check that SEL did not simply degrade the model.

**Descriptive:** SELSHAM's braking change; focal-agent miss rates (overall and stopped focal agents) for the cost of
removing the examples; moving non-focal miss rates.

## Confounds

- **A3:** three seeds per arm; intervals condition on the checkpoints.
- **A4:** SEL changes both the stopped-focal share (13.1% to about 2%) and the departure association among stopped
  focal agents; it does not separate the two. A positive S1 with a null S2 would mean that removing any 20,515 examples
  matters, i.e. no selection effect.
- **A5:** ALLR is expected to reproduce the stopped-agent failure on the reserve (as ALL did on validation); this is
  reported.
- **A6:** S1 and S2 are open-loop; S3 is the registered braking proxy.
- **A7:** SELSHAM removes the same number of examples.
- **A9:** the 2 m and 0.5 m/s thresholds and the bars are the study's existing ones (H6a, H10).
- **A10:** all arms exclude the reserve; the exclusion uses training-split futures only, never reserve outcomes.

## Stopping

Train once per seed; score once; one fixed analysis. If the compute budget runs out before all six models finish,
the completed seeds are reported as such and the claims are decided only with all three seeds per arm (otherwise
inconclusive). Negative and inconclusive results are published with the same prominence.

## Deviations

None at registration.
