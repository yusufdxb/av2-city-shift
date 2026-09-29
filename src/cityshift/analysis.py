"""Pre-registered analysis (docs/preregistration/stage1-city-shift.md).

Inputs: evaluation parquets from ``evaluate.py``:
    <evals>/ALL.parquet, <evals>/LOCO-<city>.parquet, <evals>/NOMAP.parquet,
    <evals>/ALL_mapswap.parquet
Output: <evals>/results.json and a printed summary.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

from .data import CITIES

COVERAGE = 0.8
N_BOOT = 10000
# Decision rule (prereg deviation 3): a hypothesis is decided by its scenario-bootstrap CI at
# the Bonferroni level for the four-hypothesis family (H1-H4). The fold sign-flip p-value is
# reported only as fold consistency: with 6 folds its floor (2/64 two-sided) cannot pass any
# multiplicity correction over 4 tests.
N_TESTS = 4
CI_BONF = (100 * 0.05 / N_TESTS / 2, 100 * (1 - 0.05 / N_TESTS / 2))  # 98.75% interval


def seed_cols(df: pd.DataFrame, name: str) -> list[str]:
    return sorted(c for c in df.columns if c.startswith("s") and c.endswith(f"_{name}") and c[1:].split("_")[0].isdigit())


def seed_mean(df: pd.DataFrame, name: str) -> np.ndarray:
    return df[seed_cols(df, name)].to_numpy(float).mean(1)


def keep_mask(score: np.ndarray, coverage: float = COVERAGE) -> np.ndarray:
    """Keep the ``coverage`` fraction with the lowest score (ties broken by index)."""
    n_keep = int(round(coverage * len(score)))
    order = np.argsort(score, kind="stable")
    keep = np.zeros(len(score), bool)
    keep[order[:n_keep]] = True
    return keep


def capture_fraction(miss: np.ndarray, score: np.ndarray, oracle: np.ndarray) -> float:
    base = miss.mean()
    denom = base - miss[keep_mask(oracle)].mean()
    if denom <= 0:
        return float("nan")
    return float((base - miss[keep_mask(score)].mean()) / denom)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUROC, ties counted half."""
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank(method="average").to_numpy()
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def sign_flip_p(x: np.ndarray, two_sided: bool = True) -> float:
    """Exact sign-flip permutation p-value for mean(x) != 0 (or > 0)."""
    x = np.asarray(x, float)
    obs = x.mean()
    stats = np.array([(x * np.array(s)).mean() for s in itertools.product([1, -1], repeat=len(x))])
    if two_sided:
        return float(np.mean(np.abs(stats) >= abs(obs) - 1e-12))
    return float(np.mean(stats >= obs - 1e-12))


def decide(res: dict) -> dict[str, str]:
    """Apply the registered decision rules, including the positive-control gates."""
    out = {}
    h1_lo, _ = res["H1"]["ci_bonferroni"]
    if h1_lo > 0 and res["H1"]["pooled_rel_change"] >= 0.05:
        out["H1"] = "supported"
    else:
        out["H1"] = "dead" if res.get("PC1", {}).get("pass", False) else "null, uninterpretable (PC1 failed or missing)"
    h2_lo, _ = res["H2"]["ci_bonferroni"]
    if h2_lo > 0:
        out["H2"] = "supported" if res["H2"]["pooled_CF_U1_heldout"] >= 0.25 else "detectable, below the 0.25 useful magnitude"
    else:
        out["H2"] = "dead" if res.get("PC2", {}).get("pass", False) else "null, uninterpretable (PC2 failed or missing)"
    lo3, hi3 = res["H3"]["ci_bonferroni"]
    out["H3"] = "differs" if (lo3 > 0 or hi3 < 0) else "no detectable difference"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    all_df = pd.read_parquet(os.path.join(args.evals, "ALL.parquet"))
    loco = {c: pd.read_parquet(os.path.join(args.evals, f"LOCO-{c}.parquet")) for c in CITIES}
    for c, d in loco.items():
        assert (d.val_index.to_numpy() == all_df.val_index.to_numpy()).all(), f"row order mismatch for {c}"
    city = all_df.city.to_numpy()
    miss_all = seed_mean(all_df, "miss")
    res: dict = {"per_city": {}}

    # ---------- H1 ----------
    rel, boot_rel = [], []
    for c in CITIES:
        m = city == c
        a, lo = miss_all[m], seed_mean(loco[c], "miss")[m]
        rel.append((lo.mean() - a.mean()) / a.mean())
        # paired bootstrap over this city's scenarios
        ii = rng.integers(0, m.sum(), size=(N_BOOT, m.sum()))
        boot_rel.append((lo[ii].mean(1) - a[ii].mean(1)) / a[ii].mean(1))
        res["per_city"][c] = {
            "n": int(m.sum()),
            "MR_ALL": float(a.mean()),
            "MR_LOCO": float(lo.mean()),
            "rel_change": float(rel[-1]),
            "minFDE_ALL": float(seed_mean(all_df, "min_fde")[m].mean()),
            "minFDE_LOCO": float(seed_mean(loco[c], "min_fde")[m].mean()),
            "per_seed_MR_LOCO": [float(loco[c][col][m].mean()) for col in seed_cols(loco[c], "miss")],
            "per_seed_MR_ALL": [float(all_df[col][m].mean()) for col in seed_cols(all_df, "miss")],
        }
        veh = m & (all_df.focal_type.to_numpy() == "vehicle")
        res["per_city"][c]["rel_change_vehicle_only"] = float(
            (seed_mean(loco[c], "miss")[veh].mean() - miss_all[veh].mean()) / miss_all[veh].mean()
        )
    pooled = np.mean(boot_rel, 0)
    res["H1"] = {
        "pooled_rel_change": float(np.mean(rel)),
        "ci95": [float(np.percentile(pooled, 2.5)), float(np.percentile(pooled, 97.5))],
        "ci_bonferroni": [float(np.percentile(pooled, CI_BONF[0])), float(np.percentile(pooled, CI_BONF[1]))],
        "fold_consistency_p_two_sided": sign_flip_p(np.array(rel)),
        "per_fold": dict(zip(CITIES, map(float, rel))),
    }

    # ---------- H2 / H3 ----------
    def cf_block(d: pd.DataFrame, m: np.ndarray, signal: str) -> float:
        miss = seed_mean(d, "miss")[m]
        oracle = seed_mean(d, "min_fde")[m]
        score = d["disagree"].to_numpy()[m] if signal == "disagree" else seed_mean(d, signal)[m]
        return capture_fraction(miss, score, oracle)

    def boot_cf(d: pd.DataFrame, m: np.ndarray, signal: str, n: int = N_BOOT) -> np.ndarray:
        miss = seed_mean(d, "miss")[m]
        oracle = seed_mean(d, "min_fde")[m]
        score = d["disagree"].to_numpy()[m] if signal == "disagree" else seed_mean(d, signal)[m]
        out = np.empty(n)
        for b in range(n):
            ii = rng.integers(0, len(miss), len(miss))
            out[b] = capture_fraction(miss[ii], score[ii], oracle[ii])
        return out

    signals = ["disagree", "entropy", "spread", "maha"]
    cf = {s: {"heldout": [], "indist": []} for s in signals}
    boots_h2, boots_h3, sham = [], [], {}
    for c in CITIES:
        d = loco[c]
        mh, mi = city == c, city != c
        for s in signals:
            cf[s]["heldout"].append(cf_block(d, mh, s))
            cf[s]["indist"].append(cf_block(d, mi, s))
        bh, bi = boot_cf(d, mh, "disagree"), boot_cf(d, mi, "disagree")
        boots_h2.append(bh)
        boots_h3.append(bh - bi)
        miss_h = seed_mean(d, "miss")[mh]
        oracle_h = seed_mean(d, "min_fde")[mh]
        sham[c] = [capture_fraction(miss_h, rng.random(mh.sum()), oracle_h) for _ in range(1000)]
        res["per_city"][c]["coverage_check"] = {
            "U1": float(keep_mask(d["disagree"].to_numpy()[mh]).mean()),
            "oracle": float(keep_mask(oracle_h).mean()),
            "random": float(keep_mask(rng.random(mh.sum())).mean()),
        }
        res["per_city"][c]["CF_U1_heldout"] = cf["disagree"]["heldout"][-1]
        res["per_city"][c]["CF_U1_indist"] = cf["disagree"]["indist"][-1]
        res["per_city"][c]["sham_CF_mean_ci95"] = [
            float(np.mean(sham[c])), float(np.percentile(sham[c], 2.5)), float(np.percentile(sham[c], 97.5))
        ]
    h2 = np.array(cf["disagree"]["heldout"])
    h3 = h2 - np.array(cf["disagree"]["indist"])
    pb2, pb3 = np.nanmean(boots_h2, 0), np.nanmean(boots_h3, 0)
    res["H2"] = {
        "pooled_CF_U1_heldout": float(np.nanmean(h2)),
        "ci95": [float(np.nanpercentile(pb2, 2.5)), float(np.nanpercentile(pb2, 97.5))],
        "ci_bonferroni": [float(np.nanpercentile(pb2, CI_BONF[0])), float(np.nanpercentile(pb2, CI_BONF[1]))],
        "fold_consistency_p_one_sided": sign_flip_p(h2[np.isfinite(h2)], two_sided=False),
        "sham_pooled_mean": float(np.mean([np.nanmean(v) for v in sham.values()])),
        "folds_undefined": [c for c, x in zip(CITIES, h2) if not np.isfinite(x)],
    }
    res["H3"] = {
        "pooled_CF_diff_heldout_minus_indist": float(np.nanmean(h3)),
        "ci95": [float(np.nanpercentile(pb3, 2.5)), float(np.nanpercentile(pb3, 97.5))],
        "ci_bonferroni": [float(np.nanpercentile(pb3, CI_BONF[0])), float(np.nanpercentile(pb3, CI_BONF[1]))],
        "fold_consistency_p_two_sided": sign_flip_p(h3[np.isfinite(h3)]),
    }
    res["exploratory_CF"] = {s: {k: [float(x) for x in v] for k, v in cf[s].items()} for s in signals}
    # city-detection AUROC of each signal (proxy, exploratory): held-out vs in-dist, per fold
    res["exploratory_city_auroc"] = {
        s: {
            c: auroc(
                (loco[c]["disagree"] if s == "disagree" else pd.Series(seed_mean(loco[c], s))).to_numpy()[city == c],
                (loco[c]["disagree"] if s == "disagree" else pd.Series(seed_mean(loco[c], s))).to_numpy()[city != c],
            )
            for c in CITIES
        }
        for s in signals
    }

    # ---------- positive controls ----------
    nomap_path = os.path.join(args.evals, "NOMAP.parquet")
    if os.path.exists(nomap_path):
        nm = pd.read_parquet(nomap_path)
        a = all_df[seed_cols(all_df, "miss")[0]].to_numpy().mean()  # seed-0 vs seed-0, like for like
        b = nm[seed_cols(nm, "miss")[0]].to_numpy().mean()
        res["PC1"] = {"MR_ALL_seed0": float(a), "MR_NOMAP": float(b), "rel": float((b - a) / a), "pass": bool((b - a) / a >= 0.10)}
    swap_path = os.path.join(args.evals, "ALL_mapswap.parquet")
    if os.path.exists(swap_path):
        sw = pd.read_parquet(swap_path)
        mr_o, mr_s = miss_all.mean(), seed_mean(sw, "miss").mean()
        au = auroc(sw["disagree"].to_numpy(), all_df["disagree"].to_numpy())
        res["PC2"] = {
            "MR_orig": float(mr_o),
            "MR_swapped": float(mr_s),
            "rel": float((mr_s - mr_o) / mr_o),
            "U1_auroc_swapped_vs_orig": au,
            "pass": bool((mr_s - mr_o) / mr_o >= 0.10 and au > 0.6),
        }

    res["decisions"] = decide(res)
    with open(os.path.join(args.evals, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_city", "exploratory_CF", "exploratory_city_auroc")}, indent=2))


if __name__ == "__main__":
    main()
