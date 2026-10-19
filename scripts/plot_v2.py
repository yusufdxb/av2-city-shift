"""Render the v2 evidence overview from completed result summaries, without raw data.

    python scripts/plot_v2.py --out docs/figures/v2_evidence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, GREEN, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29b"
INK, GRID = "#30302e", "#e6e5e0"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports", type=Path, default=Path("reports/v2"))
    ap.add_argument("--out", type=Path, default=Path("docs/figures/v2_evidence"), help="output stem, PNG and vector PDF")
    args = ap.parse_args()

    def load(name):
        return json.loads((args.reports / f"{name}.json").read_text())

    r, q, s, f = (load(n) for n in ("study_r_v4", "study_q_open_corrected", "study_s_open", "study_f_coverage"))
    if r["scenarios"] != 10000 or q["cohort"]["omitted_agents"] != 0:
        raise ValueError("complete registered cohorts required")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
                         "font.size": 10, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "grid.color": GRID, "grid.linewidth": .6, "pdf.fonttype": 42, "legend.frameon": False})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), layout="constrained")
    a, b, c, d = axes.flat

    labels = ["PATCH", "Random sham", "TRIM", "Geometry only", "Probability only"]
    effects = [r["effects"][name+"_reduction"] for name in ("PATCH", "SHAM2", "TRIM", "GEO", "PROB")]
    y = np.arange(len(labels))
    for i, (effect, color) in enumerate(zip(effects, (BLUE, GRAY, GREEN, GRAY, ORANGE))):
        point, (lo, hi) = 100*effect["point"], np.array(effect["ci9875"])*100
        a.plot([lo, hi], [i, i], color=color, lw=2)
        a.plot(point, i, "o", color=color, ms=5)
    a.axvline(0, color=INK, lw=.6)
    a.axvline(30, color=GRAY, lw=.7, ls=":")
    a.set(yticks=y, yticklabels=labels, ylim=(4.6, -.6), xlim=(-4, 75),
          xlabel="Braking-proxy reduction (%)", title="(a) Fresh reserve: braking")
    a.text(.99, .03, "98.75% intervals", transform=a.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK)
    a.grid(axis="x")

    rates = q["stopped_all"]["miss_rates_city_equal"]
    bars = b.barh(["Small model", "CV-6", "QCNet"], [100*rates[n] for n in ("ALL", "CV", "QCNET")],
                  color=[BLUE, GRAY, ORANGE], height=.55)
    b.bar_label(bars, fmt="%.2f", padding=4, fontsize=10)
    b.invert_yaxis()
    b.set(xlim=(0, 63), ylim=(2.9, -.5), xlabel="Stopped-agent miss (%)",
          title="(b) Stronger forecaster: QCNet")
    b.text(.99, .03, "Validation; last-observed endpoint", transform=b.transAxes,
           ha="right", va="bottom", fontsize=8.5, color=INK)
    b.grid(axis="x")
    b.set_axisbelow(True)

    rates = s["stopped_nonfocal"]["miss_city_equal"]
    bars = c.barh(["ALLR", "Selection", "Selection sham"], [100*rates[n] for n in ("ALLR", "SEL", "SELSHAM")],
                  color=[BLUE, GREEN, GRAY], height=.55)
    c.bar_label(bars, fmt="%.2f", padding=4, fontsize=10)
    c.invert_yaxis()
    c.set(xlim=(0, max(100*rates[n] for n in ("ALLR", "SEL", "SELSHAM"))*1.22), ylim=(2.9, -.5),
          xlabel="Stopped-agent miss (%)", title="(c) Training selection")
    c.text(.99, .03, "Reserve; last-observed endpoint", transform=c.transAxes,
           ha="right", va="bottom", fontsize=8.5, color=INK)
    c.grid(axis="x")
    c.set_axisbelow(True)

    for i, t in enumerate((49, 89)):
        cell = next(x for x in f["cells"] if x["replan"] == t and x["horizon_s"] == 2 and x["group"] == "stopped")
        point, (lo, hi) = 100*cell["difference_ALL_minus_CV6"], 100*np.array(cell["ci95"])
        d.plot([lo, hi], [i, i], color=BLUE, lw=2)
        d.plot(point, i, "o", color=BLUE, ms=5)
    d.axvline(0, color=INK, lw=.6)
    d.set(yticks=[0, 1], yticklabels=["Handoff", "Late replan"], ylim=(1.6, -.6), xlim=(-3, .4),
          xlabel="Miss-rate difference vs CV-6 (pp)", title="(d) Two-second endpoints")
    d.text(.99, .03, "Complete futures; 95% intervals", transform=d.transAxes,
           ha="right", va="bottom", fontsize=8.5, color=INK)
    d.grid(axis="x")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(args.out.with_suffix("."+extension), dpi=220, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


if __name__ == "__main__":
    main()
