"""Study H: shared within-city scenario bootstrap of the registered H2/H3 CFs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import CI_BONF, N_BOOT, capture_fraction, seed_mean
from .data import CITIES

SEED = 20261002
H2_POINT = "pooled_CF_U1_heldout"
H3_POINT = "pooled_CF_diff_heldout_minus_indist"


def shared_indices(city: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Draw each city's scenarios once, for reuse by every LOCO fold."""
    out = {}
    for c in CITIES:
        ii = np.flatnonzero(city == c)
        if not len(ii):
            raise ValueError(f"no scenarios in {c}")
        out[c] = ii[rng.integers(0, len(ii), len(ii))]
    return out


def fold_cfs(
    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], indices: dict[str, np.ndarray], city: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute retention, oracle ranking and CF on each fold's sampled rows."""
    heldout, differences = [], []
    sampled = np.empty(len(city), np.int64)
    for c in CITIES:
        sampled[city == c] = indices[c]
    for c in CITIES:
        miss, score, oracle = arrays[c]
        h, other = sampled[city == c], sampled[city != c]
        ch = capture_fraction(miss[h], score[h], oracle[h])
        ci = capture_fraction(miss[other], score[other], oracle[other])
        heldout.append(ch)
        differences.append(ch - ci)
    return np.asarray(heldout), np.asarray(differences)


def _summary(point: float, draws: np.ndarray, registered: dict, name: str) -> dict:
    finite = np.isfinite(draws)
    return {
        name: point,
        "ci95": np.nanpercentile(draws, [2.5, 97.5]).tolist(),
        "ci_bonferroni": np.nanpercentile(draws, CI_BONF).tolist(),
        "bootstrap_undefined_share": float(1 - finite.mean()),
        "original_registered": registered,
    }


def analyze(
    all_df: pd.DataFrame, loco: dict[str, pd.DataFrame], registered: dict, n_boot: int = N_BOOT, seed: int = SEED
) -> dict:
    """Preserve the registered point estimands and replace their covariance model."""
    if n_boot < 1 or all_df.val_index.duplicated().any():
        raise ValueError("positive draw count and unique scenario indices required")
    city = all_df.city.to_numpy()
    if set(city) != set(CITIES):
        raise ValueError("expected all six registered cities")
    arrays = {}
    for c in CITIES:
        d = loco[c]
        if not np.array_equal(d.val_index.to_numpy(), all_df.val_index.to_numpy()):
            raise ValueError(f"row order mismatch for {c}")
        if not np.array_equal(d.city.to_numpy(), city):
            raise ValueError(f"city mismatch for {c}")
        arrays[c] = seed_mean(d, "miss"), d.disagree.to_numpy(float), seed_mean(d, "min_fde")
        if not all(np.isfinite(a).all() for a in arrays[c]):
            raise ValueError(f"nonfinite metrics in {c}")
    # The point must use the registered global row order (including oracle ties).
    h2, h3 = [], []
    for c in CITIES:
        miss, score, oracle = arrays[c]
        mh, mi = city == c, city != c
        ch = capture_fraction(miss[mh], score[mh], oracle[mh])
        ci = capture_fraction(miss[mi], score[mi], oracle[mi])
        h2.append(ch)
        h3.append(ch - ci)
    p2, p3 = float(np.nanmean(h2)), float(np.nanmean(h3))
    assert abs(p3 - registered["H3"][H3_POINT]) <= 1e-12, "H3 registered point parity failed"
    assert abs(p2 - registered["H2"][H2_POINT]) <= 1e-12, "H2 registered point parity failed"
    rng = np.random.default_rng(seed)
    b2, b3 = np.empty(n_boot), np.empty(n_boot)
    undefined = np.zeros((2, len(CITIES)), int)
    for b in range(n_boot):
        ch, diff = fold_cfs(arrays, shared_indices(city, rng), city)
        undefined += np.stack([~np.isfinite(ch), ~np.isfinite(diff)])
        b2[b] = np.nanmean(ch) if np.isfinite(ch).any() else np.nan
        b3[b] = np.nanmean(diff) if np.isfinite(diff).any() else np.nan
    result = {
        "study": "H",
        "n_boot": n_boot,
        "seed": seed,
        "H2": _summary(p2, b2, registered["H2"], H2_POINT),
        "H3": _summary(p3, b3, registered["H3"], H3_POINT),
    }
    lo, hi = result["H3"]["ci_bonferroni"]
    verdict = "differs" if lo > 0 or hi < 0 else "no detectable difference"
    if not np.isfinite([lo, hi]).all():
        verdict = "inconclusive"
    result["H3"].update(
        {
            "shared_draw_verdict": verdict,
            "registered_verdict": registered["decisions"]["H3"],
            "verdict_unchanged": verdict == registered["decisions"]["H3"],
        }
    )
    result["per_fold"] = {
        c: {
            "CF_heldout": float(h2[i]),
            "CF_difference": float(h3[i]),
            "H2_undefined_draw_share": float(undefined[0, i] / n_boot),
            "H3_undefined_draw_share": float(undefined[1, i] / n_boot),
        }
        for i, c in enumerate(CITIES)
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--registered", default="reports/confirmatory/stage1_results.json")
    args = ap.parse_args()
    root = Path(args.evals)
    result = analyze(
        pd.read_parquet(root / "ALL.parquet"),
        {c: pd.read_parquet(root / f"LOCO-{c}.parquet") for c in CITIES},
        json.loads(Path(args.registered).read_text()),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("H2", "H3")}, indent=2))


if __name__ == "__main__":
    main()
