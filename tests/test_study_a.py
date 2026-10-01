import numpy as np
import pandas as pd

from cityshift.analysis_stage3 import CITIES
from cityshift.study_a import analyze, decide


def _data(trim_share):
    """Per city 400 scenes: ALL brakes in 100, PATCHSUB in 50, TRIMNF in 100 - trim_share * 50."""
    rows = []
    for c in CITIES:
        for i in range(400):
            rows.append({"city": c, "ALL_b": float(i < 100), "PATCHSUB_b": float(i < 50),
                         "TRIMNF_b": float(i < 100 - int(trim_share * 50)), "ALL_c": 0.0, "TRIMNF_c": 0.0,
                         "PATCHSUB_c": 0.0})
    return pd.DataFrame(rows)


def test_ratio_point_and_decisions():
    eff = analyze(_data(1.0), n_boot=300)
    assert eff["PATCHSUB_reduction"]["point"] == 0.5 and eff["R"]["point"] == 1.0
    assert decide(eff, 600, [], 1.0) == "supported"
    eff = analyze(_data(0.1), n_boot=300)
    assert np.isclose(eff["R"]["point"], 0.1) and decide(eff, 600, [], 1.0) == "killed"
    assert decide(eff, 600, [], 0.9).startswith("uninterpretable")
    assert decide(eff, 50, [], 1.0).startswith("inconclusive")
