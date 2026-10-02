"""Synthetic parity and shared-scenario covariance checks for study H."""

import numpy as np
import pandas as pd
import pytest

from cityshift import h3_shared
from cityshift.analysis import capture_fraction, seed_mean


def test_shared_bootstrap_matches_one_city_stratified_table_per_draw():
    city = np.tile(h3_shared.CITIES, 30)
    all_df = pd.DataFrame({"val_index": np.arange(len(city)), "city": city})
    rng = np.random.default_rng(123)
    loco, arrays = {}, {}
    points_h, points_d = [], []
    for c in h3_shared.CITIES:
        d = all_df.copy()
        oracle = rng.permutation(len(city)).astype(float)
        # Seed-averaged misses with a strictly useful oracle on every resample.
        miss = oracle / len(city)
        d["disagree"] = oracle + rng.normal(0, 45, len(city))
        for s in range(3):
            d[f"s{s}_miss"] = miss
            d[f"s{s}_min_fde"] = oracle
            d[f"s{s}_brier_min_fde"] = -oracle  # never part of oracle ranking
        loco[c] = d
        arrays[c] = seed_mean(d, "miss"), d.disagree.to_numpy(), seed_mean(d, "min_fde")
        m, u, o = arrays[c]
        mask = city == c
        heldout = capture_fraction(m[mask], u[mask], o[mask])
        points_h.append(heldout)
        points_d.append(heldout - capture_fraction(m[~mask], u[~mask], o[~mask]))
    registered = {
        "H2": {h3_shared.H2_POINT: float(np.mean(points_h))},
        "H3": {h3_shared.H3_POINT: float(np.mean(points_d))},
        "decisions": {"H3": "no detectable difference"},
    }
    expected_h, expected_d = [], []
    rng = np.random.default_rng(h3_shared.SEED)
    for _ in range(40):
        sample = np.empty(len(city), int)
        for c in h3_shared.CITIES:
            mask = city == c
            original = np.flatnonzero(mask)
            sample[mask] = original[rng.integers(0, len(original), len(original))]
        heldouts, diffs = [], []
        for c in h3_shared.CITIES:
            miss, score, oracle = (x[sample] for x in arrays[c])
            mask = city == c
            h = capture_fraction(miss[mask], score[mask], oracle[mask])
            heldouts.append(h)
            diffs.append(h - capture_fraction(miss[~mask], score[~mask], oracle[~mask]))
        expected_h.append(np.mean(heldouts))
        expected_d.append(np.mean(diffs))
    result = h3_shared.analyze(all_df, loco, registered, n_boot=40)
    assert result["H3"][h3_shared.H3_POINT] == pytest.approx(registered["H3"][h3_shared.H3_POINT], abs=1e-12)
    np.testing.assert_allclose(result["H2"]["ci95"], np.percentile(expected_h, [2.5, 97.5]))
    np.testing.assert_allclose(result["H3"]["ci95"], np.percentile(expected_d, [2.5, 97.5]))
    np.testing.assert_allclose(result["H3"]["ci_bonferroni"], np.percentile(expected_d, h3_shared.CI_BONF))
    registered["H3"][h3_shared.H3_POINT] += 1e-5
    with pytest.raises(AssertionError, match="H3 registered point parity"):
        h3_shared.analyze(all_df, loco, registered, n_boot=1)
