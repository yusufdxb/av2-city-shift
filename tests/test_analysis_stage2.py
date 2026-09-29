import json
import sys

import numpy as np
import pandas as pd

from cityshift import analysis_stage2
from cityshift.data import CITIES


def _df(rng, city, p_all, p_loco):
    n = len(city)
    df = pd.DataFrame({"city": city, "scenario_id": "x"})
    for arm, p in (("ALL", p_all), ("LOCO", p_loco)):
        for s in range(3):
            f = rng.random(n) < p
            df[f"{arm}_s{s}_failure"] = f
            df[f"{arm}_s{s}_collision"] = f
            df[f"{arm}_s{s}_unnecessary_hard_brake"] = False
            df[f"{arm}_s{s}_progress"] = 1.0
    for arm in ("oracle", "cv", "static", "log"):
        rate = {"oracle": 0.0, "cv": 0.05, "static": 0.2, "log": 0.0}[arm]
        df[f"{arm}_failure"] = rng.random(n) < rate
        df[f"{arm}_collision"] = df[f"{arm}_failure"]
    return df


def test_stage2_handles_zero_event_fold(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    city = np.repeat(CITIES, 3000)
    p_all = np.where(city == "palo-alto", 0.0, 0.05)  # one fold with no ALL failures at all
    df = _df(rng, city, p_all, np.where(city == "palo-alto", 0.0, 0.08))
    df.to_parquet(tmp_path / "cl.parquet")
    monkeypatch.setattr(sys, "argv", ["a", "--parquet", str(tmp_path / "cl.parquet"), "--out", str(tmp_path / "r.json")])
    monkeypatch.setattr(analysis_stage2, "N_BOOT", 300)
    analysis_stage2.main()
    r = json.load(open(tmp_path / "r.json"))["H4"]
    assert r["folds_undefined"] == ["palo-alto"]
    assert np.isfinite(r["pooled_rel_change"]) and r["pooled_rel_change"] > 0.3
    assert r["fold_consistency_p_two_sided"] == 2 / 32  # 5 defined folds, all positive
    assert r["decision"] == "supported"
