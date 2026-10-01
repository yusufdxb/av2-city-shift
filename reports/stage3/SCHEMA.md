# Stage 3a and Stage 3 per-row tables

Attached to GitHub release v1.1 as `stage3-per-row-results.tar.gz`; unpack in the repository root. Checksums are in
[`SHA256SUMS`](SHA256SUMS); `python scripts/audit_stage3.py` verifies them and recomputes the H5 to H9 point estimates.
The tables are derived from Argoverse 2 (© 2021 Argo AI, LLC, CC BY-NC-SA 4.0, non-commercial); the MIT license of
the code does not apply to them.

## `evals/stage3a/per_agent.parquet` (Stage 3a, H5, H6a, H6b)

2,867,859 rows: one per validation agent (318,651 agents in 24,988 scenarios) and predictor, forecast at t=49
(`python -m cityshift.multiagent_eval ... --out evals/stage3a/per_agent.parquet`, analysed by
`python -m cityshift.analysis_stage3a --parquet evals/stage3a/per_agent.parquet --out evals/stage3a/results.json`).

| Column | Meaning |
|---|---|
| `scenario_id`, `agent_id` | Argoverse 2 scenario and track ID (the pair is unique per predictor) |
| `city` | one of the six cities |
| `agent_type` | Argoverse 2 object type |
| `focal`, `scored`, `planner_relevant` | membership: the focal agent; a SCORED track observed at every future step; selected by the closed-loop planner at t=49 |
| `n_valid` | observed future steps (of 60) used for scoring |
| `speed` | agent speed at t=49, m/s |
| `predictor` | `CV`, `LANE`, `STATIC` baselines; `ALL_s{0,1,2}`; `LOCO-<city>_s{0,1,2}` (only on that held-out city's agents) |
| `min_ade`, `min_fde` | K=6 minimum displacement errors (m) of the mode with the best final point, over observed steps |
| `miss` | 1.0 if that mode's final error exceeds 2 m, else 0.0 |
| `brier_min_fde` | `min_fde + (1 - p)^2` for that mode |

## `runs/stage3/openloop_val.parquet` (Stage 3, H9)

1,911,906 rows: the same 318,651 agents and columns as above, with `predictor` in `ALL_s{0,1,2}` and `MULTI_s{0,1,2}`.

## `runs/stage3/closedloop_val.parquet` (Stage 3, H7, H8, controls)

24,988 rows, one per validation scenario (`scenario_id`, `city`). For each arm `log`, `oracle`, `cv`, `static`,
`ALL_s{0,1,2}`, `MULTI_s{0,1,2}`, `PATCH_s{0,1,2}`, `SHAM_s{0,1,2}` there are columns `<arm>_<field>`:

| Field | Meaning |
|---|---|
| `collision` | contact with the ego vehicle's front (at-fault) |
| `first_collision_step` | first overlapping step, -1 if none |
| `front_gap_m` | smallest gap to an agent in front (m) |
| `planner_hard_brake`, `logged_hard_brake` | planner / logged human decelerated by 4 m/s or more within 1 s |
| `unnecessary_hard_brake` | planner hard brake where the human did not brake hard |
| `failure` | `collision` or `unnecessary_hard_brake` |
| `progress` | ego distance travelled over the logged distance (NaN when the logged path is under 1 m) |
| `accels` | chosen acceleration at each of the six replans (absent for `log`) |
| `trigger_count`, `substituted_count` | PATCH and SHAM only: stopped planner agents seen, and forecasts replaced by CV |
