"""EXPLORATORY per-replan mechanism audit (not registered; on already-scored replication-pool scenes).

For a focal-only model (ALL, one seed) in closed loop v2, at every replan that chooses a hard brake (<= -4 m/s^2)
instead of the choice it would make with no risk term, find the agents whose predicted hit probability ruled that choice
out ("blockers"), and record for each: stopped or moving, its predicted risk mass on the rejected plan, the part of that
mass from modes whose endpoint moves more than 2 m, whether its TRUE logged future actually conflicts with the rejected
plan (same box test and front rule), and whether its future is fully observed. Also records whether PATCH (CV for stopped
agents) would have removed that brake at that replan, and the REALISED speed change over the executed 1 s segment, so the
subset that counts toward the registered unnecessary-hard-brake outcome can be separated from commanded brakes. Every
recomputed choice is asserted equal to closedloop.plan, and the scene-level outcome is asserted to match the replans.

Per-decision, per-blocker and per-scene rows go to CSV next to --out; the summary (with scenario-clustered bootstrap
95% intervals) is recomputed from those rows by summarize(), which scripts/summarize_mechanism.py also uses.

    PYTHONPATH=src python -m cityshift.mechanism --raw <train dir> --seed 0 --out reports/mechanism/mechanism_audit.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from . import closedloop as base
from . import closedloop_v2 as v2
from .closedloop import (ACCELS, CHECK_STEPS, DIMS, DT, EGO_DIMS, END, EXEC_STEPS, HARD_BRAKE, MARGIN_M, PLAN_STEPS,
                         REPLANS, W_ACCEL, W_PROGRESS, boxes_overlap, in_front, profiles)
from .closedloop_v3 import POOL_SHA256, checked_ids
from .scene import build_input

NOTE = ("EXPLORATORY, not registered. Focal-only model (ALL), closed loop v2 (registered settings), {n} replication-pool "
        "scenes (fixed random subset, seed 7), {replans} replans per model seed. Association at the decision, not "
        "causation: an agent can carry risk on the rejected plan without being decisive.")
LIMITATIONS = {
    "true_conflict": "Planner's own look-ahead test on the LOGGED future: oriented boxes at the agent's logged centre and "
                     "heading (type-default dimensions) against the ego box inflated by a 0.5 m margin on every side, "
                     "with the in-front rule on agent centres, over 4 s. The margin counts near misses (loosens it); "
                     "unobserved future steps count as NO conflict (tightens it). Shares are approximate, not bounds.",
    "full_future": "True when all 40 look-ahead steps are observed and inside the log. "
                   "*_full_future stats restrict to those agents.",
    "cohorts": "commanded: chosen acceleration <= -4 m/s^2 and different from the no-risk choice. realised: the commanded "
               "cohort restricted to replans whose executed 1 s speed change is <= -4 m/s^2 in a scene where the logged "
               "human never brakes that hard (the registered unnecessary_hard_brake definition, closedloop_v2.score).",
    "risk_weighting": "unweighted_mean = mean over stopped blockers of each blocker's moving-mode share of its own risk; "
                      "risk_weighted = sum of moving-mode risk / sum of risk over stopped blockers.",
    "rerun": "The closed loop is re-run here scene by scene, so model inference batches differ from the registered "
             "Stage 4 run; a few borderline replans can flip (agreement_with_registered_stage4 counts them).",
    "bootstrap": "95% percentile intervals, 2000 resamples of scenarios (with replacement) among scenarios that have at "
                 "least one decision in the cohort; rng seed 0.",
}
N_BOOT = 2000


def hit_tensor(sc, ego, t, agents, traj):
    """[C, M, K] mode hits over the 4 s look-ahead, exactly as closedloop.plan computes them."""
    v, dist = profiles(ego.v[t], ego.vmax, PLAN_STEPS)
    ep, eh = ego.path.at(ego.s[t] + dist[:, :CHECK_STEPS])
    ap = traj[:, :, :CHECK_STEPS]
    prev = np.concatenate([np.repeat(sc.pos[agents, t][:, None, None], ap.shape[1], 1), ap[:, :, :-1]], 2)
    dxy = ap - prev
    ah = np.where(np.linalg.norm(dxy, axis=-1) < 0.05, sc.head[agents, t][:, None, None], np.arctan2(dxy[..., 1], dxy[..., 0]))
    dims = DIMS[sc.types[agents]]
    c1, h1 = ep[:, None, None], eh[:, None, None]
    hit = boxes_overlap(c1, h1, EGO_DIMS[0] + 2 * MARGIN_M, EGO_DIMS[1] + 2 * MARGIN_M, ap[None], ah[None],
                        dims[:, 0][:, None, None][None], dims[:, 1][:, None, None][None])
    return (hit & in_front(c1, h1, ap[None])).any(-1), dist, ep, eh


def choose(p_hit, dist):
    progress = dist[:, -1] / max(dist[:, -1].max(), 1e-6)
    return int(np.argmin(base.W_RISK * p_hit.sum(-1) + W_PROGRESS * (1.0 - progress) + W_ACCEL * np.abs(ACCELS)))


def true_conflict(sc, t, i, ep, eh, c) -> tuple[bool, bool, int]:
    """Does agent i's logged future overlap the ego's candidate-c plan in the look-ahead?

    Returns (conflict, full, observed_steps). Unobserved steps count as no conflict."""
    steps = np.arange(t + 1, min(t + 1 + CHECK_STEPS, END + 1))
    valid = sc.valid[i, steps]
    n = len(steps)
    if not valid.any():
        return False, False, 0
    dims = DIMS[sc.types[i]]
    ov = boxes_overlap(ep[c, :n], eh[c, :n], EGO_DIMS[0] + 2 * MARGIN_M, EGO_DIMS[1] + 2 * MARGIN_M,
                       sc.pos[i, steps], sc.head[i, steps], dims[0], dims[1])
    ov &= in_front(ep[c, :n], eh[c, :n], sc.pos[i, steps]) & valid
    return bool(ov.any()), bool(valid.all() and n == CHECK_STEPS), int(valid.sum())


def audit(raw: str, ids: list[str], seed: int, device) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the closed loop for one model seed; return (decisions, blockers, scenes) rows."""
    key = f"ALL_s{seed}"
    models = base.load_models({key: f"runs/ALL/seed{seed}/model.pt"}, device)
    base.init_ego = v2.init_ego
    base.score = v2.score
    decisions, blockers, scenes = [], [], []
    for sid in ids:
        sc = base.load_scene(os.path.join(raw, sid))
        ego = v2.init_ego(sc)
        scene_dec, realised_dv = [], []
        for t in REPLANS:
            sce = base.scene_with_ego(sc, ego, t)
            agents = base.select_agents(sce, ego, t)
            if agents:
                preds = base.predict(models, [(key, build_input(sce, i, t)) for i in agents], device)
                traj, prob = np.stack([p[0] for p in preds]), np.stack([p[1] for p in preds])
            else:
                traj, prob = np.zeros((0, 6, 60, 2)), np.zeros((0, 6))
            planned = base.plan(sc, ego, t, agents, traj, prob)
            dec = None
            if agents:
                hits, dist, ep, eh = hit_tensor(sc, ego, t, agents, traj)
                p_hit = (hits * prob[None]).sum(-1)  # [C, M]
                c = choose(p_hit, dist)
                assert c == planned, "mechanism recomputation diverged from closedloop.plan"
                c0 = choose(np.zeros_like(p_hit), dist)
                if ACCELS[c] <= -4 and c != c0:
                    stopped = np.array([np.linalg.norm(sc.vel[i, t]) < 0.5 for i in agents])
                    # PATCH at this replan: CV (p=1) for stopped agents
                    ptraj, pprob = traj.copy(), prob.copy()
                    for j in np.flatnonzero(stopped):
                        ptraj[j] = base.cv_forecast(sc, agents[j], t)[None]
                        pprob[j] = 0.0
                        pprob[j, 0] = 1.0
                    phits, _, _, _ = hit_tensor(sc, ego, t, agents, ptraj)
                    c_patch = choose((phits * pprob[None]).sum(-1), dist)
                    moving_mode = np.linalg.norm(traj[:, :, -1] - sc.pos[agents, t][:, None], axis=-1) > 2.0  # [M, K]
                    decision_id = f"{sid}_{t}"
                    rows = []
                    for j in np.flatnonzero(p_hit[c0] > 0):
                        conflict, full, observed = true_conflict(sc, t, agents[j], ep, eh, c0)
                        mass = float(p_hit[c0, j])
                        mv = float((hits[c0, j] * prob[j] * moving_mode[j]).sum())
                        rows.append({"seed": seed, "decision_id": decision_id, "scenario_id": sid, "replan_t": t,
                                     "agent_index": int(agents[j]), "agent_id": str(sc.track_ids[agents[j]]),
                                     "agent_type": int(sc.types[agents[j]]),
                                     "speed_mps": float(np.linalg.norm(sc.vel[agents[j], t])), "stopped": bool(stopped[j]),
                                     "risk_mass": mass, "moving_mode_risk": mv,
                                     "moving_mode_share": mv / mass if mass else 0.0,
                                     "true_conflict": conflict, "full_future": full, "observed_future_steps": observed})
                    blockers.extend(rows)
                    dec = {"seed": seed, "decision_id": decision_id, "scenario_id": sid, "replan_t": t,
                           "chosen_accel": float(ACCELS[c]), "c0_accel": float(ACCELS[c0]),
                           "patch_accel": float(ACCELS[c_patch]), "n_selected_agents": len(agents),
                           "n_blockers": len(rows), "n_stopped_blockers": int(sum(r["stopped"] for r in rows)),
                           "blocking_risk_mass": float(sum(r["risk_mass"] for r in rows)),
                           "any_stopped_blocker": any(r["stopped"] for r in rows),
                           "all_blockers_stopped": bool(rows) and all(r["stopped"] for r in rows),
                           "any_true_conflict": any(r["true_conflict"] for r in rows),
                           "patch_removes_brake": bool(ACCELS[c_patch] > -4)}
            base.execute(ego, t, planned)
            dv = float((ego.v[t + EXEC_STEPS] - ego.v[t]) / (EXEC_STEPS * DT))
            realised_dv.append(dv)
            if dec is not None:
                dec["realised_dv_mps2"] = dv
                dec["realised_hard_brake"] = dv <= HARD_BRAKE + 1e-9
                scene_dec.append(dec)
        r = base.score_sim(sc, ego)
        assert r["planner_hard_brake"] == any(dv <= HARD_BRAKE + 1e-9 for dv in realised_dv)
        for dec in scene_dec:
            dec["scene_logged_hard_brake"] = bool(r["logged_hard_brake"])
            dec["realised_unnecessary"] = bool(dec["realised_hard_brake"] and not r["logged_hard_brake"])
        decisions.extend(scene_dec)
        n_realised_unnec = sum(dv <= HARD_BRAKE + 1e-9 for dv in realised_dv) if not r["logged_hard_brake"] else 0
        scenes.append({"seed": seed, "scenario_id": sid, "unnecessary_hard_brake": bool(r["unnecessary_hard_brake"]),
                       "planner_hard_brake": bool(r["planner_hard_brake"]),
                       "logged_hard_brake": bool(r["logged_hard_brake"]), "collision": bool(r["collision"]),
                       "n_commanded_decisions": len(scene_dec),
                       "n_realised_decisions": int(sum(d["realised_unnecessary"] for d in scene_dec)),
                       "n_realised_unnecessary_replans": int(n_realised_unnec)})
    dcols = ["seed", "decision_id", "scenario_id", "replan_t", "chosen_accel", "c0_accel", "patch_accel",
             "realised_dv_mps2", "realised_hard_brake", "scene_logged_hard_brake", "realised_unnecessary",
             "n_selected_agents", "n_blockers", "n_stopped_blockers", "blocking_risk_mass", "any_stopped_blocker",
             "all_blockers_stopped", "any_true_conflict", "patch_removes_brake"]
    bcols = ["seed", "decision_id", "scenario_id", "replan_t", "agent_index", "agent_id", "agent_type", "speed_mps",
             "stopped", "risk_mass", "moving_mode_risk", "moving_mode_share", "true_conflict", "full_future",
             "observed_future_steps"]
    return pd.DataFrame(decisions, columns=dcols), pd.DataFrame(blockers, columns=bcols), pd.DataFrame(scenes)


# ------------------------------------------------------------------ summary (from rows only)
def _ratio(num: pd.Series, den: pd.Series, sid: pd.Series, clusters: np.ndarray, w: np.ndarray) -> dict:
    """Point ratio sum(num)/sum(den) and its scenario-clustered bootstrap interval."""
    g = pd.DataFrame({"num": num.astype(float).values, "den": den.astype(float).values, "sid": sid.values})
    g = g.groupby("sid").sum().reindex(clusters, fill_value=0.0)
    den_tot = float(g["den"].sum())
    if den_tot == 0:
        return {"share": None, "ci95": None}
    bn, bd = w @ g["num"].values, w @ g["den"].values
    boot = np.divide(bn, bd, out=np.full_like(bn, np.nan), where=bd > 0)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {"share": float(g["num"].sum()) / den_tot, "ci95": [float(lo), float(hi)]}


def summarize_cohort(dec: pd.DataFrame, blk: pd.DataFrame) -> dict:
    blk = blk[blk["decision_id"].isin(dec["decision_id"])]
    clusters = np.array(sorted(dec["scenario_id"].unique()))
    w = np.random.default_rng(0).multinomial(len(clusters), np.full(len(clusters), 1.0 / max(len(clusters), 1)),
                                             size=N_BOOT).astype(float) if len(clusters) else np.zeros((N_BOOT, 0))
    one_d, one_b = pd.Series(1.0, index=dec.index), pd.Series(1.0, index=blk.index)
    st, tc, full = blk["stopped"].astype(bool), blk["true_conflict"].astype(bool), blk["full_future"].astype(bool)

    def d(num, den):
        return {"n": int(den.sum()), **_ratio(num, den, dec["scenario_id"], clusters, w)}

    def b(num, den):
        return {"n": int((den > 0).sum()), **_ratio(num, den, blk["scenario_id"], clusters, w)}

    return {
        "hard_brake_decisions": int(len(dec)),
        "scenarios": int(len(clusters)),
        "decisions": {k: d(dec[k].astype(float), one_d) for k in ("any_stopped_blocker", "all_blockers_stopped",
                                                                  "any_true_conflict", "patch_removes_brake")},
        "patch_removes_brake_when_all_blockers_stopped": d((dec["patch_removes_brake"] & dec["all_blockers_stopped"]).astype(float),
                                                           dec["all_blockers_stopped"].astype(float)),
        "blockers": {
            "n": int(len(blk)),
            "stopped_share": b(st.astype(float), one_b),
            "true_conflict_stopped": b((tc & st).astype(float), st.astype(float)),
            "true_conflict_moving": b((tc & ~st).astype(float), (~st).astype(float)),
            "moving_mode_share_of_risk_stopped_unweighted_mean": b(blk["moving_mode_share"] * st, st.astype(float)),
            "moving_mode_share_of_risk_stopped_risk_weighted": b(blk["moving_mode_risk"] * st, blk["risk_mass"] * st),
            "stopped_share_of_blocking_risk_mass": b(blk["risk_mass"] * st, blk["risk_mass"]),
            "true_conflict_stopped_risk_weighted": b(blk["risk_mass"] * (tc & st), blk["risk_mass"] * st),
            "true_conflict_stopped_full_future": b((tc & st & full).astype(float), (st & full).astype(float)),
            "true_conflict_stopped_partial_future": b((tc & st & ~full).astype(float), (st & ~full).astype(float)),
            "risk_mass_total": float(blk["risk_mass"].sum()),
            "risk_mass_stopped": float(blk.loc[st, "risk_mass"].sum()),
            "moving_mode_risk_stopped": float(blk.loc[st, "moving_mode_risk"].sum()),
        },
    }


def summarize(dec: pd.DataFrame, blk: pd.DataFrame, scn: pd.DataFrame) -> dict:
    real = dec[dec["realised_unnecessary"].astype(bool)]
    unnec = scn["unnecessary_hard_brake"].astype(bool)
    return {
        "scenes": int(len(scn)),
        "replans": int(len(scn) * len(REPLANS)),
        "commanded": summarize_cohort(dec, blk),
        "realised": summarize_cohort(real, blk),
        "registered_outcome_context": {
            "scenes_with_unnecessary_hard_brake": int(unnec.sum()),
            "realised_unnecessary_replans_total": int(scn["n_realised_unnecessary_replans"].sum()),
            "realised_unnecessary_replans_in_cohort": int(len(real)),
            "scenes_with_unnecessary_hard_brake_and_a_cohort_decision": int((unnec & (scn["n_realised_decisions"] > 0)).sum()),
            "commanded_decisions_not_realised_hard": int((~dec["realised_hard_brake"].astype(bool)).sum()),
            "commanded_decisions_realised_hard_but_human_braked_hard": int(
                (dec["realised_hard_brake"].astype(bool) & dec["scene_logged_hard_brake"].astype(bool)).sum()),
        },
    }


def rows_paths(out_dir: str, seed: int) -> dict[str, str]:
    return {k: os.path.join(out_dir, f"{k}_seed{seed}.csv") for k in ("decisions", "blockers", "scenes")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--pool", default="runs/replication_pool.npy")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0, help="model seed (runs/ALL/seed<seed>/model.pt)")
    ap.add_argument("--out", required=True, help="summary JSON for this seed; rows CSVs go next to it")
    args = ap.parse_args()
    ids = sorted(checked_ids(args.pool, POOL_SHA256))
    chosen_ids = [str(x) for x in np.random.default_rng(7).choice(ids, size=args.n, replace=False)]
    start = time.time()
    dec, blk, scn = audit(args.raw, chosen_ids, args.seed, torch.device("cuda"))
    runtime = time.time() - start
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    paths = rows_paths(out_dir, args.seed)
    dec.to_csv(paths["decisions"], index=False)
    blk.to_csv(paths["blockers"], index=False)
    scn.to_csv(paths["scenes"], index=False)
    summary = {"_note": NOTE.format(n=args.n, replans=len(scn) * len(REPLANS)), "_limitations": LIMITATIONS,
               "per_seed": {f"seed{args.seed}": summarize(dec, blk, scn) | {"runtime_sec": round(runtime)}}}
    json.dump(summary, open(args.out, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
