"""Render the README figures from the evaluation outputs and two checkpoints.

Usage: PYTHONPATH=src python scripts/plot_results.py --evals evals --runs runs --root <preprocessed data> --out docs/figures
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from cityshift.data import CITIES, Split  # noqa: E402
from cityshift.evaluate import load_model, run_model  # noqa: E402

# Validated categorical slots (light surface), plus neutrals.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#a3a29b", "#e6e5e0"
CITY_NAMES = {"austin": "Austin", "dearborn": "Dearborn", "miami": "Miami", "palo-alto": "Palo Alto",
              "pittsburgh": "Pittsburgh", "washington-dc": "Washington DC"}  # fmt: skip

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
})  # fmt: skip


def seed_mean(df: pd.DataFrame, m: str) -> np.ndarray:
    return df[[f"s{i}_{m}" for i in range(3)]].to_numpy(float).mean(1)


def fig_h1(evals: str, out: str) -> None:
    A = pd.read_parquet(f"{evals}/ALL.parquet")
    rng = np.random.default_rng(0)
    rows = []
    for c in CITIES:
        L = pd.read_parquet(f"{evals}/LOCO-{c}.parquet")
        m = (A.city == c).to_numpy()
        a, lo = seed_mean(A, "miss")[m], seed_mean(L, "miss")[m]
        ii = rng.integers(0, m.sum(), size=(4000, m.sum()))
        b = lo[ii].mean(1) / a[ii].mean(1) - 1
        rows.append((CITY_NAMES[c], lo.mean() / a.mean() - 1, *np.percentile(b, [2.5, 97.5])))
    import json

    h1 = json.load(open(f"{evals}/results.json"))["H1"]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(rows))
    ax.axvline(0, color=MUTED, lw=1)
    ax.axvline(5, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(5.15, len(rows) + 0.35, "registered bar +5%", color=INK2, fontsize=8.5, va="center")
    for yi, (name, v, lo, hi) in zip(y, rows):
        ax.plot([lo * 100, hi * 100], [yi, yi], color=BLUE, lw=2, solid_capstyle="round")
        ax.plot(v * 100, yi, "o", color=BLUE, ms=8, mec=SURFACE, mew=2)
        ax.text(hi * 100 + 0.4, yi, f"{v * 100:+.1f}%", va="center", fontsize=9, color=INK)
    yp = len(rows) + 0.35 - 1.2 + 0.25
    yp = -1.2
    lo, hi = h1["ci_bonferroni"]
    ax.plot([lo * 100, hi * 100], [yp, yp], color=INK, lw=2, solid_capstyle="round")
    ax.plot(h1["pooled_rel_change"] * 100, yp, "D", color=INK, ms=8, mec=SURFACE, mew=2)
    ax.text(hi * 100 + 0.4, yp, f"{h1['pooled_rel_change'] * 100:+.1f}%", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(list(y) + [yp], [r[0] for r in rows] + ["Pooled (98.75% CI)"])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_ylim(-1.8, len(rows) + 0.7)
    ax.set_xlabel("Change in miss rate when the city was never seen in training (relative, %)")
    ax.set_title("H1: miss rate rises in all six cities when the city is unseen", loc="left")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    fig.text(0.01, 0.01, "Point estimates are positive in every city; Austin's 95% interval crosses zero. Per-city bars: 95% scenario bootstrap; seeds averaged.", color=INK2, fontsize=7.6)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(f"{out}/h1_per_city.png", dpi=200)
    plt.close(fig)


def fig_h2(evals: str, out: str) -> None:
    cov = np.linspace(0.5, 1.0, 51)
    curves = {"disagree": [], "random": [], "oracle": []}
    for c in CITIES:
        L = pd.read_parquet(f"{evals}/LOCO-{c}.parquet")
        m = (L.city == c).to_numpy()
        miss, err, dis = seed_mean(L, "miss")[m], seed_mean(L, "min_fde")[m], L.disagree.to_numpy()[m]
        for key, score in (("disagree", dis), ("oracle", err)):
            order = np.argsort(score, kind="stable")
            curves[key].append([miss[order[: int(round(k * len(miss)))]].mean() for k in cov])
        curves["random"].append([miss.mean()] * len(cov))
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    style = {"random": (MUTED, "Random rejection (sham)", (0, (4, 3))), "disagree": (BLUE, "Reject where 3 seeds disagree", "-"),
             "oracle": (AQUA, "Error-ranked oracle (true minFDE)", "-")}  # fmt: skip
    for key in ("random", "disagree", "oracle"):
        col, lab, ls = style[key]
        yv = np.mean(curves[key], 0) * 100
        ax.plot(cov * 100, yv, color=col, lw=2, ls=ls)
        ax.text(cov[0] * 100 - 1.2, yv[0], lab, color=INK, fontsize=8.5, ha="left", va="center")
        ax.plot(cov[0] * 100, yv[0], "o", color=col, ms=6, mec=SURFACE, mew=1.5)
    ax.axvline(80, color=MUTED, lw=1, ls=(0, (2, 3)))
    ax.text(79.4, 1.0, "registered\n80% coverage", fontsize=8, color=INK2, va="bottom", ha="left")
    ax.set_xlim(22, 101)
    ax.set_xticks([50, 60, 70, 80, 90, 100])
    ax.invert_xaxis()
    ax.set_xlabel("Share of scenarios the predictor keeps (%)")
    ax.set_ylabel("Miss rate of kept scenarios (%)")
    ax.set_title("H2: disagreement catches a third of what an error-ranked oracle removes", loc="left")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    fig.text(0.01, 0.01, "Held-out-city validation scenarios, mean of the six leave-one-city-out folds.", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(f"{out}/h2_risk_coverage.png", dpi=200)
    plt.close(fig)


def fig_closed_loop(evals: str, out: str) -> None:
    D = pd.read_parquet(f"{evals}/stage2_closedloop.parquet")

    def rate(arm, m):
        if arm in ("ALL", "LOCO"):
            return D[[f"{arm}_s{i}_{m}" for i in range(3)]].astype(float).mean(1).mean() * 100
        return D[f"{arm}_{m}"].astype(float).mean() * 100

    arms = [("oracle", "Oracle (true futures)"), ("ALL", "All-city model"), ("LOCO", "Unseen-city model"),
            ("cv", "Constant velocity"), ("static", "Everyone stands still")]  # fmt: skip
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for yi, (arm, name) in enumerate(arms[::-1]):
        # Segments are disjoint so the bar length equals the failure rate (their union):
        # any at-fault collision, then hard brakes in drives without a collision.
        c = rate(arm, "collision")
        h = rate(arm, "failure") - c
        ax.barh(yi, c, height=0.56, color=ORANGE, edgecolor=SURFACE, linewidth=2)
        ax.barh(yi, h, left=c, height=0.56, color=BLUE, edgecolor=SURFACE, linewidth=2)
        ax.text(c + h + 0.12, yi, f"{rate(arm, 'failure'):.1f}%", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(arms)), [a[1] for a in arms[::-1]])
    ax.set_xlabel("Share of drives with a planning failure (%)")
    ax.set_title("Closed loop: learned forecasts crash less, phantom-brake more", loc="left")
    ax.bar(0, 0, color=ORANGE, label="At-fault collision")
    ax.bar(0, 0, color=BLUE, label="Unnecessary hard brake, no collision")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlim(0, 8.2)
    fig.text(0.01, 0.01, "24,988 validation drives; human drive replayed: 0.4% (collision-checker calibration).", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(f"{out}/closed_loop_failures.png", dpi=200)
    plt.close(fig)


def fig_examples(evals: str, runs: str, root: str, out: str, city: str = "palo-alto") -> None:
    """Two held-out-city scenarios, seen vs unseen model (seed 0). Selection rule is in the caption."""
    A, L = pd.read_parquet(f"{evals}/ALL.parquet"), pd.read_parquet(f"{evals}/LOCO-{city}.parquet")
    m = ((A.city == city) & (A.focal_type == "vehicle")).to_numpy()
    diff = L.s0_min_fde.to_numpy() - A.s0_min_fde.to_numpy()
    cand = np.where(m)[0]
    worst = cand[np.argmax(diff[cand])]
    typical = cand[np.argsort(A.s0_min_fde.to_numpy()[cand])[len(cand) // 2]]
    idx = np.array(sorted([typical, worst]))
    val = Split(root, "val")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds = {}
    for key, path in (("seen", f"{runs}/ALL/seed0/model.pt"), ("unseen", f"{runs}/LOCO-{city}/seed0/model.pt")):
        model, _ = load_model(path, device)
        preds[key] = run_model(model, val, idx, device)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 9.2))
    for r, vi in enumerate([typical, worst]):
        j = int(np.where(idx == vi)[0][0])
        hist = np.asarray(val.arrays["agent_hist"][vi])
        valid = np.asarray(val.arrays["agent_valid"][vi])
        lanes = np.asarray(val.arrays["lane_pts"][vi])
        lattr = np.asarray(val.arrays["lane_attr"][vi])
        tgt = np.asarray(val.arrays["target"][vi])
        for cidx, (key, col, title) in enumerate((("seen", BLUE, "Trained with Palo Alto"), ("unseen", ORANGE, "Never saw Palo Alto"))):
            ax = axes[r, cidx]
            for ln, la in zip(lanes, lattr):
                if la[0] >= 0:
                    ax.plot(ln[:, 0], ln[:, 1], color=GRID, lw=1.2, zorder=1)
            for a in range(1, hist.shape[0]):
                v = valid[a]
                if v.any():
                    ax.plot(hist[a, v, 0], hist[a, v, 1], color=MUTED, lw=1, zorder=2)
                    last = np.where(v)[0][-1]
                    ax.plot(hist[a, last, 0], hist[a, last, 1], "s", color=MUTED, ms=3, zorder=2)
            traj = preds[key]["traj"][j].numpy()
            prob = preds[key]["prob"][j].numpy()
            for k in np.argsort(prob):
                ax.plot(traj[k, :, 0], traj[k, :, 1], color=col, lw=1.2 + 2.5 * prob[k], alpha=0.35 + 0.65 * prob[k] / prob.max(), zorder=3)
                ax.plot(traj[k, -1, 0], traj[k, -1, 1], "o", color=col, ms=4, zorder=3)
            ax.plot(hist[0, :, 0], hist[0, :, 1], color=INK, lw=2, zorder=4)
            ax.plot(tgt[:, 0], tgt[:, 1], color=INK, lw=2, ls=(0, (2, 2)), zorder=4)
            ax.plot(tgt[-1, 0], tgt[-1, 1], "*", color=INK, ms=11, mec=SURFACE, zorder=5)
            fde = np.linalg.norm(traj[:, -1] - tgt[-1], axis=1).min()
            pts = np.vstack([tgt, hist[0][valid[0]][:, :2], traj.reshape(-1, 2)])
            cx, cy = (pts.min(0) + pts.max(0)) / 2
            half = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) / 2 + 12
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(f"{title}: best-of-6 endpoint error {fde:.1f} m", loc="left", fontsize=9.5, color=col)
            if cidx == 0:
                ax.text(-0.02, 0.5, "Typical scenario" if r == 0 else "Largest gap", transform=ax.transAxes, rotation=90,
                        ha="right", va="center", fontsize=10, fontweight="bold", color=INK)
    fig.text(0.02, 0.012, "Black: focal vehicle history (solid) and true future (dashed, star). Colour: the 6 predicted futures, thicker = more\n"
             "probable. Grey: other agents and lanes. Seed-0 models, Palo Alto validation vehicles. Top: the median-error scenario for\n"
             "the trained-with model. Bottom: selected as the scenario where the unseen-city model does worst relative to the other.",
             color=INK2, fontsize=7.8)  # fmt: skip
    fig.tight_layout(rect=(0.02, 0.07, 1, 1))
    fig.savefig(f"{out}/example_scenarios.png", dpi=170)
    plt.close(fig)


def fig_stage3(stage3: str, out: str) -> None:
    """Closed loop v2: unnecessary hard brakes vs at-fault collisions per forecast source (two panels, one axis each)."""
    import json

    r = json.load(open(f"{stage3}/results.json"))["closedloop"]["rates"]
    arms = [("oracle", "Oracle (true futures)"), ("ALL", "Focal-only model"), ("MULTI", "Focal + scored-agent model (MULTI)"),
            ("PATCH", "Focal-only + CV for stopped agents"), ("SHAM", "Sham: CV for moving agents"),
            ("cv", "Constant velocity"), ("static", "Everyone stands still")]  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    for ax, key, title in ((axes[0], "brake", "Unnecessary hard brakes"), (axes[1], "collision", "At-fault collisions")):
        for yi, (arm, _) in enumerate(arms[::-1]):
            v = r[key][arm] * 100
            col = ORANGE if arm == "PATCH" else (BLUE if arm in ("ALL", "MULTI") else MUTED)
            ax.barh(yi, v, height=0.56, color=col, edgecolor=SURFACE, linewidth=2)
            ax.text(v + 0.1, yi, f"{v:.1f}%", va="center", fontsize=8.5, fontweight="bold" if arm == "PATCH" else "normal")
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.set_xlim(0, max(r[key].values()) * 100 * 1.25)
        ax.set_xlabel("Share of validation drives (%)")
    axes[0].set_yticks(range(len(arms)), [a[1] for a in arms[::-1]])
    fig.suptitle("Stage 3: CV for stopped agents cuts phantom brakes 38%, collisions within the registered margin",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.01, "Closed loop v2 (no future speed cap, contact-based fault), 24,988 validation drives, model arms averaged "
             "over 3 seeds. Sham substituted 43% of PATCH's dose (see audit).", color=INK2, fontsize=7.6)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(f"{out}/stage3_closed_loop.png", dpi=200)
    plt.close(fig)


def fig_stage4(closedloop: str, out: str) -> None:
    """Stage 4 replication on fresh scenes: unnecessary hard brakes and at-fault collisions per forecast source."""
    D = pd.read_parquet(closedloop)

    def rate(arm: str, metric: str) -> float:
        cols = [f"{arm}_s{i}_{metric}" for i in range(3)]
        if cols[0] in D:
            return float(D[cols].astype(float).mean(axis=1).mean() * 100)
        return float(D[f"{arm}_{metric}"].astype(float).mean() * 100)

    arms = [("oracle", "Oracle (true futures)"), ("ALL", "Focal-only model"),
            ("PATCH", "PATCH: CV for stopped agents"), ("TRIM", "TRIM: drop moving modes of stopped agents"),
            ("SHAM2", "Dose-matched sham (random agents)"), ("MIX", "MIX: retrained, 25% stopped agents"),
            ("cv", "Constant velocity"), ("static", "Everyone stands still")]  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), sharey=True)
    for ax, key, title in ((axes[0], "unnecessary_hard_brake", "Unnecessary hard brakes"),
                           (axes[1], "collision", "At-fault collisions")):  # fmt: skip
        values = {a: rate(a, key) for a, _ in arms}
        for yi, (arm, _) in enumerate(arms[::-1]):
            v = values[arm]
            col = ORANGE if arm in ("PATCH", "TRIM") else (BLUE if arm in ("ALL", "MIX") else MUTED)
            ax.barh(yi, v, height=0.56, color=col, edgecolor=SURFACE, linewidth=2)
            ax.text(v + 0.1, yi, f"{v:.1f}%", va="center", fontsize=8.5, fontweight="bold" if arm in ("PATCH", "TRIM") else "normal")
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.set_xlim(0, max(values.values()) * 1.25)
        ax.set_xlabel("Share of drives (%)")
    axes[0].set_yticks(range(len(arms)), [a[1] for a in arms[::-1]])
    fig.suptitle("Stage 4 (8,140 fresh scenes): the stopped-agent fix replicates; a dose-matched sham shows no detectable effect",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.01, "Scenes no compared model trained on; model arms averaged over 3 seeds. TRIM fell back to CV for about 67% "
             "of stopped agents (no stationary mode).", color=INK2, fontsize=7.6)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(f"{out}/stage4_replication.png", dpi=200)
    plt.close(fig)


def fig_sensitivity(sweep: str, out: str) -> None:
    """EXPLORATORY planner sweep: phantom braking and collisions vs risk weight, per speed cap (small multiples)."""
    d = pd.read_parquet(sweep)
    d["brake"] = ((d.min_exec_decel <= -4 + 1e-9) & ~(d.min_log_decel <= -4 + 1e-9)).astype(float)
    d["coll"] = d.collision.astype(float)
    g = d.groupby(["cap", "risk_weight", "arm"])[["brake", "coll"]].mean().mul(100).reset_index()
    style = {"ALL": (BLUE, "Focal-only model", "-"), "PATCH": (ORANGE, "PATCH", "-"),
             "cv": (MUTED, "Constant velocity", (0, (4, 3))), "oracle": (AQUA, "Oracle", (0, (1, 2)))}  # fmt: skip
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), sharex=True)
    for col, (cap, cap_title) in enumerate((("v2", "Stage 3/4 speed cap"), ("tight", "Tighter speed cap"))):
        for row, (metric, ylabel) in enumerate((("brake", "Unnecessary hard brakes (%)"), ("coll", "At-fault collisions (%)"))):
            ax = axes[row, col]
            for arm, (color, label, ls) in style.items():
                x = g[(g.cap == cap) & (g.arm == arm)].sort_values("risk_weight")
                ax.plot(x.risk_weight, x[metric], color=color, lw=2, ls=ls, marker="o", ms=4.5, mec=SURFACE, mew=1.2)
                if row == 0 and col == 1:
                    nudge = {"PATCH": 0.25, "cv": -0.25}.get(arm, 0.0)  # keep adjacent labels apart
                    ax.text(x.risk_weight.iloc[-1] * 1.25, x[metric].iloc[-1] + nudge, label, color=INK, fontsize=8.5, va="center")
            ax.axvline(100, color=MUTED, lw=1, ls=(0, (2, 3)))
            ax.set_xscale("log")
            if metric == "coll":
                ax.set_yscale("log")
            ax.grid(axis="y", which="major")
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(cap_title, loc="left", fontsize=10)
            if row == 1:
                ax.set_xlabel("Planner risk weight (log; 100 = registered)")
    axes[0, 1].set_xlim(0.2, 600)
    fig.suptitle("Exploratory: PATCH cuts phantom braking by about 30 to 50% wherever the planner weighs risk (weight >= 3)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.01, "3,000 replication-pool scenes, model seed 0, rates pooled over scenes, no intervals; brake threshold -4 m/s^2. Below weight 3 the "
             "planner barely weighs risk and collides in 3 to 30% of drives.", color=INK2, fontsize=7.6)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(f"{out}/sensitivity_sweep.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default="evals")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--stage3", default="reports/stage3", help="directory with the Stage 3 results.json")
    ap.add_argument("--stage4", default="runs/stage4/closedloop_pool.parquet", help="Stage 4 closed-loop table (release v1.2)")
    ap.add_argument("--sweep", default="runs/sensitivity/sweep.parquet", help="exploratory planner sweep table")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_h1(args.evals, args.out)
    fig_h2(args.evals, args.out)
    fig_closed_loop(args.evals, args.out)
    fig_examples(args.evals, args.runs, args.root, args.out)
    fig_stage3(args.stage3, args.out)
    fig_stage4(args.stage4, args.out)
    fig_sensitivity(args.sweep, args.out)
    print("wrote", sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()
