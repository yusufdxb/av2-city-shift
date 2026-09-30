"""Independent re-derivation of every registered decision (H1 to H13) from the per-row tables.

For each claim this recomputes the point estimate and a fresh scenario-cluster bootstrap interval (own code, own random
seed, city-stratified, paired, seeds averaged first) at the registered level, then re-applies the registered bar.
Pass means: every verdict equals the committed one, every point estimate matches within 1e-9, and every interval
endpoint agrees with the committed one within Monte Carlo tolerance (15% of the committed interval width).
Imports nothing from cityshift. Needs releases v1.0, v1.1 and v1.2 unpacked in the repository root.

    python scripts/audit_decisions.py [--draws 4000]
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

CITIES = ["austin", "dearborn", "miami", "palo-alto", "pittsburgh", "washington-dc"]
rng = np.random.default_rng(20260930)


def pct(level):
    a = (1 - level) / 2 * 100
    return a, 100 - a


def city_boot(frames, stat, draws):
    """frames: {city: DataFrame of scenario rows}; stat(dict of resampled frames) -> float. City-stratified bootstrap."""
    out = np.empty(draws)
    idx = {c: rng.integers(0, len(f), size=(draws, len(f))) for c, f in frames.items()}
    for b in range(draws):
        out[b] = stat({c: f.iloc[idx[c][b]] for c, f in frames.items()})
    return out


def seed_mean(df, prefix, metric, seeds=3):
    return df[[f"{prefix}_s{s}_{metric}" for s in range(seeds)]].astype(float).mean(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=4000)
    B = ap.parse_args().draws
    rows = []

    def record(name, point, reported_point, boots, level, reported_ci, verdict, reported_verdict):
        lo, hi = np.nanpercentile(boots, pct(level))
        width = abs(reported_ci[1] - reported_ci[0])
        ok_ci = abs(lo - reported_ci[0]) <= 0.15 * width and abs(hi - reported_ci[1]) <= 0.15 * width
        rows.append({"claim": name, "point": point, "point_ok": abs(point - reported_point) <= 1e-9,
                     "ci": [round(float(lo), 5), round(float(hi), 5)], "reported_ci": [round(v, 5) for v in reported_ci],
                     "ci_ok": ok_ci, "verdict": verdict(lo, hi), "reported_verdict": reported_verdict})

    # ---------------- Stage 1: H1 (miss rate, LOCO vs ALL on held-out city), 98.75% ----------------
    s1 = json.load(open("reports/confirmatory/stage1_results.json"))
    A = pd.read_parquet("evals/ALL.parquet")
    L = {c: pd.read_parquet(f"evals/LOCO-{c}.parquet") for c in CITIES}
    # Stage 1 tables store float32; average in float64 as the committed analysis does
    frames = {c: pd.DataFrame({"a": A.loc[A.city == c, [f"s{i}_miss" for i in range(3)]].astype(float).mean(axis=1).to_numpy(),
                               "l": L[c].loc[L[c].city == c, [f"s{i}_miss" for i in range(3)]].astype(float).mean(axis=1).to_numpy()})
              for c in CITIES}
    h1 = lambda fr: np.mean([f.l.mean() / f.a.mean() - 1 for f in fr.values()])  # noqa: E731
    boots = city_boot(frames, h1, B)
    bar = lambda lo, hi, p=h1(frames): "supported" if lo > 0 and p >= 0.05 else "killed"  # noqa: E731
    record("H1", h1(frames), s1["H1"]["pooled_rel_change"], boots, 0.9875, s1["H1"]["ci_bonferroni"],
           bar, "supported" if s1["decisions"]["H1"] == "supported" else s1["decisions"]["H1"])

    # ---------------- Stage 1: H2 / H3 (capture fraction of ensemble disagreement), 98.75% ----------------
    def cf(miss, score, err):
        n = int(round(0.8 * len(miss)))
        base = miss.mean()
        denom = base - miss[np.argsort(err, kind="stable")[:n]].mean()
        return (base - miss[np.argsort(score, kind="stable")[:n]].mean()) / denom if denom > 0 else np.nan

    held, indist = {}, {}
    for c in CITIES:
        d = L[c]
        f = pd.DataFrame({"m": d[[f"s{i}_miss" for i in range(3)]].astype(float).mean(axis=1),
                          "e": d[[f"s{i}_min_fde" for i in range(3)]].astype(float).mean(axis=1),
                          "u": d.disagree.astype(float), "held": d.city == c})
        held[c], indist[c] = f[f.held].reset_index(drop=True), f[~f.held].reset_index(drop=True)
    h2 = lambda fr: np.nanmean([cf(f.m.to_numpy(), f.u.to_numpy(), f.e.to_numpy()) for f in fr.values()])  # noqa: E731
    boots2 = city_boot(held, h2, max(B // 4, 500))
    p2 = h2(held)
    record("H2", p2, s1["H2"]["pooled_CF_U1_heldout"], boots2, 0.9875, s1["H2"]["ci_bonferroni"],
           lambda lo, hi: "supported" if lo > 0 else "killed", s1["decisions"]["H2"])
    boots_in = city_boot(indist, h2, max(B // 4, 500))
    boots3 = boots2 - boots_in
    p3 = p2 - h2(indist)
    record("H3", p3, s1["H3"]["pooled_CF_diff_heldout_minus_indist"], boots3, 0.9875, s1["H3"]["ci_bonferroni"],
           lambda lo, hi: "differs" if lo > 0 or hi < 0 else "no detectable difference", s1["decisions"]["H3"])

    # ---------------- Stage 2: H4 (planning failures, LOCO vs ALL), 98.75% ----------------
    s2 = json.load(open("reports/confirmatory/stage2_results.json"))
    C2 = pd.read_parquet("evals/stage2_closedloop.parquet")
    fr4 = {c: pd.DataFrame({"a": seed_mean(C2[C2.city == c], "ALL", "failure").to_numpy(),
                            "l": seed_mean(C2[C2.city == c], "LOCO", "failure").to_numpy()}) for c in CITIES}
    h4 = lambda fr: np.mean([f.l.mean() / f.a.mean() - 1 for f in fr.values()])  # noqa: E731
    record("H4", h4(fr4), s2["H4"]["pooled_rel_change"], city_boot(fr4, h4, B), 0.9875, s2["H4"]["ci_bonferroni"],
           lambda lo, hi, p=h4(fr4): "supported" if lo > 0 and p >= 0.10 else "dead", s2["H4"]["decision"])

    # ---------------- Stage 3a: H5, H6a, H6b (agent level, scenario clusters), 98.75% ----------------
    s3a = json.load(open("reports/stage3a/results.json"))
    P = pd.read_parquet("evals/stage3a/per_agent.parquet", columns=["scenario_id", "agent_id", "city", "focal",
                                                                     "planner_relevant", "speed", "predictor", "miss"])
    P = P[P.predictor.isin(["CV", "LANE", "ALL_s0", "ALL_s1", "ALL_s2"])]
    P["arm"] = np.where(P.predictor.str.startswith("ALL"), "ALL", P.predictor)
    W = P.groupby(["scenario_id", "agent_id", "arm"]).miss.mean().unstack()
    meta = P[P.predictor == "CV"].set_index(["scenario_id", "agent_id"])[["city", "focal", "planner_relevant", "speed"]]
    W = W.join(meta).reset_index()

    def cluster_frames(mask, cols):
        out = {}
        for c in CITIES:
            sub = W[mask & (W.city == c)]
            out[c] = sub.groupby("scenario_id")[cols].agg(["sum", "count"]).reset_index(drop=True)
        return out

    def ratio(fr, t, ctrl):
        return np.mean([f[(t, "sum")].sum() / f[(ctrl, "sum")].sum() for f in fr.values()])

    fr5 = cluster_frames(W.planner_relevant, ["ALL", "LANE"])
    h5 = lambda fr: 1 - ratio(fr, "ALL", "LANE")  # noqa: E731
    record("H5", h5(fr5), s3a["H5"]["pooled_reduction"], city_boot(fr5, h5, B), 0.9875, s3a["H5"]["ci_98_75"],
           lambda lo, hi, p=h5(fr5): "supported" if lo > 0 and p >= 0.10 else "killed", s3a["H5"]["decision"])
    base6 = W.planner_relevant & ~W.focal
    fr6a = cluster_frames(base6 & (W.speed < 0.5), ["ALL", "CV"])
    h6a = lambda fr: np.mean([(f[("ALL", "sum")].sum() - f[("CV", "sum")].sum()) / f[("CV", "count")].sum()  # noqa: E731
                              for f in fr.values()])
    record("H6a", h6a(fr6a), s3a["H6a"]["pooled_difference_ALL_minus_CV"], city_boot(fr6a, h6a, B), 0.9875,
           s3a["H6a"]["ci_98_75"], lambda lo, hi, p=h6a(fr6a): "supported" if lo > 0 and p >= 0.15 else "killed",
           s3a["H6a"]["decision"])
    fr6b = cluster_frames(base6 & (W.speed >= 2.0), ["ALL", "CV"])
    h6b = lambda fr: 1 - ratio(fr, "ALL", "CV")  # noqa: E731
    record("H6b", h6b(fr6b), s3a["H6b"]["pooled_reduction_ALL_vs_CV"], city_boot(fr6b, h6b, B), 0.9875,
           s3a["H6b"]["ci_98_75"], lambda lo, hi, p=h6b(fr6b): "supported" if lo > 0 and p >= 0.30 else "killed",
           s3a["H6b"]["decision"])

    # ---------------- Stage 3: H7, H8 (closed loop v2), 99.1667% ----------------
    s3 = json.load(open("reports/stage3/results.json"))
    C3 = pd.read_parquet("runs/stage3/closedloop_val.parquet")
    lvl3 = 1 - 0.05 / 6

    def closed_frames(C, arms):
        return {c: pd.DataFrame({f"{a}_{m}": seed_mean(C[C.city == c], a, m).to_numpy()
                                 for a in arms for m in ("unnecessary_hard_brake", "collision")}) for c in CITIES}

    fr3 = closed_frames(C3, ["ALL", "MULTI", "PATCH"])
    red = lambda arm: (lambda fr: np.mean([1 - f[f"{arm}_unnecessary_hard_brake"].mean() / f["ALL_unnecessary_hard_brake"].mean()  # noqa: E731
                                           for f in fr.values()]))
    dif = lambda arm: (lambda fr: np.mean([f[f"{arm}_collision"].mean() - f["ALL_collision"].mean() for f in fr.values()]))  # noqa: E731
    for name, arm in (("H7", "MULTI"), ("H8", "PATCH")):
        rep = s3["closedloop"][name]
        rb, rc = red(arm), dif(arm)
        pb, pc = rb(fr3), rc(fr3)
        bb, bc = city_boot(fr3, rb, B), city_boot(fr3, rc, B)
        lob, _ = np.nanpercentile(bb, pct(lvl3))
        _, hic = np.nanpercentile(bc, pct(lvl3))
        joint = "supported" if lob > 0 and pb >= 0.30 and hic < 0.003 else "killed"
        record(f"{name} brake", pb, rep["brake_reduction"], bb, lvl3, rep["brake_ci"], lambda lo, hi: joint, rep["status"])
        record(f"{name} collision", pc, rep["collision_difference"], bc, lvl3, rep["collision_ci"], lambda lo, hi: joint, rep["status"])

    # ---------------- Stage 3: H9 (open loop, MULTI vs ALL), 99.1667% ----------------
    O3 = pd.read_parquet("runs/stage3/openloop_val.parquet", columns=["scenario_id", "agent_id", "city", "focal",
                                                                      "planner_relevant", "speed", "predictor", "miss"])
    O3["arm"] = O3.predictor.str.split("_").str[0]
    G = O3.groupby(["scenario_id", "agent_id", "arm"]).miss.mean().unstack()
    G = G.join(O3[O3.predictor == "ALL_s0"].set_index(["scenario_id", "agent_id"])[["city", "focal", "planner_relevant", "speed"]]).reset_index()
    G["d"] = G.MULTI - G.ALL

    def diff_frames(mask):
        return {c: G[mask & (G.city == c)].groupby("scenario_id").d.agg(["sum", "count"]).reset_index(drop=True) for c in CITIES}

    mean_diff = lambda fr: np.mean([f["sum"].sum() / f["count"].sum() for f in fr.values()])  # noqa: E731
    fr9s = diff_frames(G.planner_relevant & ~G.focal & (G.speed < 0.5))
    fr9f = diff_frames(G.focal)
    b9s, b9f = city_boot(fr9s, mean_diff, B), city_boot(fr9f, mean_diff, B)
    _, his = np.nanpercentile(b9s, pct(lvl3))
    _, hif = np.nanpercentile(b9f, pct(lvl3))
    joint9 = "supported" if his < 0 and mean_diff(fr9s) <= -0.15 and hif < 0.02 else "killed"
    h9 = s3["H9"]
    record("H9 stopped", mean_diff(fr9s), h9["stopped_nonfocal_difference"], b9s, lvl3, h9["stopped_nonfocal_ci"], lambda lo, hi: joint9, h9["status"])
    record("H9 focal", mean_diff(fr9f), h9["focal_difference"], b9f, lvl3, h9["focal_ci"], lambda lo, hi: joint9, h9["status"])

    # ---------------- Stage 4: H10 to H13, 99.375% ----------------
    s4 = json.load(open("reports/stage4/results.json"))
    C4 = pd.read_parquet("runs/stage4/closedloop_pool.parquet")
    lvl4 = s4["ci_level"]
    fr4c = closed_frames(C4, ["ALL", "PATCH", "TRIM", "SHAM2", "MIX"])
    eff = s4["effects"]
    claims = s4["claims"]
    boot = {}
    for arm in ("PATCH", "TRIM", "SHAM2", "MIX"):
        boot[arm] = city_boot(fr4c, red(arm), B)
    gap = lambda fr: red("PATCH")(fr) - red("SHAM2")(fr)  # noqa: E731
    bgap = city_boot(fr4c, gap, B)
    O4 = pd.read_parquet("runs/stage4/openloop_pool.parquet")
    O4 = O4[O4.focal].assign(arm=lambda f: f.predictor.str.split("_").str[0])
    F4 = O4.groupby(["city", "scenario_id", "agent_id", "arm"]).miss.mean().unstack().reset_index()
    F4["d"] = F4.MIX - F4.ALL
    fr13 = {c: F4[F4.city == c].groupby("scenario_id").d.agg(["sum", "count"]).reset_index(drop=True) for c in CITIES}
    b13f = city_boot(fr13, mean_diff, B)
    for name, arm in (("H10", "PATCH"), ("H12", "TRIM"), ("H13", "MIX")):
        pb, pc = red(arm)(fr4c), dif(arm)(fr4c)
        bc = city_boot(fr4c, dif(arm), B)
        lob, _ = np.nanpercentile(boot[arm], pct(lvl4))
        _, hic = np.nanpercentile(bc, pct(lvl4))
        ok = lob > 0 and pb >= 0.30 and hic < 0.003
        if name == "H13":
            _, hif = np.nanpercentile(b13f, pct(lvl4))
            ok = ok and hif < 0.02
        joint = "supported" if ok else "killed"
        record(f"{name} brake", pb, eff[f"{arm}_reduction"]["point"], boot[arm], lvl4, eff[f"{arm}_reduction"]["ci"],
               lambda lo, hi: joint, claims[name]["status"])
        record(f"{name} collision", pc, eff[f"{arm}_collision_diff"]["point"], bc, lvl4, eff[f"{arm}_collision_diff"]["ci"],
               lambda lo, hi: joint, claims[name]["status"])
        if name == "H13":
            record("H13 focal", mean_diff(fr13), eff["focal_miss_diff"]["point"], b13f, lvl4, eff["focal_miss_diff"]["ci"],
                   lambda lo, hi: joint, claims[name]["status"])
    record("H11 gap", gap(fr4c), eff["sham_gap"]["point"], bgap, lvl4, eff["sham_gap"]["ci"],
           lambda lo, hi, p=gap(fr4c): "supported" if lo > 0 and p >= 0.20 else "killed", claims["H11"]["status"])

    ok = True
    for r in rows:
        good = r["point_ok"] and r["ci_ok"] and r["verdict"] == r["reported_verdict"]
        ok &= good
        print(f"{'OK ' if good else 'BAD'} {r['claim']:14s} point {r['point']: .5f} ci {r['ci']} reported {r['reported_ci']} "
              f"verdict {r['verdict']} (reported {r['reported_verdict']})")
    print("ALL DECISIONS REPRODUCED" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
