"""End-to-end run of analysis.py on synthetic eval tables with a planted effect."""

import json
import sys

import numpy as np
import pandas as pd

from cityshift import analysis
from cityshift.data import CITIES


def _table(rng, city, err_scale, disagree_signal):
    n = len(city)
    df = pd.DataFrame({"val_index": np.arange(n), "city": city, "focal_type": "vehicle", "scenario_id": "x", "speed": 1.0})
    base = rng.exponential(1.0, n)
    for s in range(3):
        fde = base * err_scale + rng.normal(0, 0.1, n).clip(0)
        df[f"s{s}_min_fde"] = fde
        df[f"s{s}_min_ade"] = fde / 2
        df[f"s{s}_miss"] = (fde > 2.0).astype(float)
        df[f"s{s}_brier_min_fde"] = fde + 0.1
        for u in ("entropy", "spread", "maha"):
            df[f"s{s}_{u}"] = rng.random(n)
    df["disagree"] = disagree_signal * base * err_scale + rng.random(n)
    return df


def test_analysis_detects_planted_effects(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    city = np.repeat(CITIES, 2000)
    _table(rng, city, np.ones(len(city)), 1.0).to_parquet(tmp_path / "ALL.parquet")
    for c in CITIES:
        scale = np.where(city == c, 1.4, 1.0)  # held-out city is harder
        _table(rng, city, scale, 1.0).to_parquet(tmp_path / f"LOCO-{c}.parquet")
    monkeypatch.setattr(sys, "argv", ["analysis", "--evals", str(tmp_path)])
    monkeypatch.setattr(analysis, "N_BOOT", 500)
    analysis.main()
    res = json.load(open(tmp_path / "results.json"))
    assert res["H1"]["pooled_rel_change"] > 0.2 and res["H1"]["ci95"][0] > 0
    assert res["H1"]["fold_consistency_p_two_sided"] == 2 / 64
    assert res["H1"]["ci_bonferroni"][0] > 0
    # no positive-control files were written, so a supported H1 stands but nulls could not be read
    assert res["decisions"]["H1"] == "supported" and res["decisions"]["H2"] == "supported"
    assert res["H2"]["pooled_CF_U1_heldout"] > 0.2 and res["H2"]["ci95"][0] > 0
    assert abs(res["H2"]["sham_pooled_mean"]) < 0.05
    for c in CITIES:
        cov = res["per_city"][c]["coverage_check"]
        assert abs(cov["U1"] - 0.8) < 1e-3 and abs(cov["oracle"] - 0.8) < 1e-3 and abs(cov["random"] - 0.8) < 1e-3
