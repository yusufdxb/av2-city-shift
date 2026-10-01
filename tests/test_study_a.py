import numpy as np
import pandas as pd
import pytest

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


def test_study_e_effects_and_decide():
    from cityshift.study_e import analyze as analyze_e, decide as decide_e
    rows = []
    for c in CITIES:
        for i in range(400):
            rows.append({"city": c, "ALL": float(i < 100), "PATCH": float(i < 50), "SHAM2": float(i < 95),
                         "TRIM": float(i < 55)})
    e = analyze_e(pd.DataFrame(rows), n_boot=300)
    assert np.isclose(e["PATCH_reduction"]["point"], 0.5) and np.isclose(e["sham_gap"]["point"], 0.45)
    assert decide_e(e["PATCH_reduction"], 0.30, True, True) == "supported"
    assert decide_e(e["PATCH_reduction"], 0.30, True, False) == "uninterpretable"
    assert decide_e(e["PATCH_reduction"], 0.60, True, True) == "killed"


def test_failed_control_is_reported_not_crashed():
    rows = [{"city": c, "ALL_b": float(i < 100), "PATCHSUB_b": float(i < 100), "TRIMNF_b": float(i < 100),
             "ALL_c": 0.0, "TRIMNF_c": 0.0, "PATCHSUB_c": 0.0} for c in CITIES for i in range(400)]
    eff = analyze(pd.DataFrame(rows), n_boot=50)
    assert np.isnan(eff["R"]["point"])
    assert decide(eff, 600, [], 1.0) == "uninterpretable (PATCHSUB positive control failed)"
    assert decide(eff, 600, [], float("nan")).startswith("uninterpretable")


def test_no_brakes_or_eligibility():
    data = _data(1.0)
    data[["ALL_b", "PATCHSUB_b", "TRIMNF_b"]] = 0.0
    eff = analyze(data, n_boot=50)
    assert decide(eff, 0, list(CITIES), 1.0).startswith("uninterpretable")
    assert decide(eff, 0, list(CITIES), float("nan")).startswith("uninterpretable")


@pytest.mark.parametrize("bad", [-1, 3, float("nan"), float("inf"), 0.5])
def test_study_e_rejects_invalid_per_replan_dose(bad):
    from cityshift.study_e import validate_dose
    patch, sham = np.full((2, 6), 2.0), np.ones((2, 6))
    sham[0, 1] = bad
    with pytest.raises(ValueError, match="invalid SHAM2 dose"):
        validate_dose(patch, sham)


def test_study_e_dose_shape_and_valid_counts():
    from cityshift.study_e import validate_dose
    validate_dose(np.full((2, 6), 2), np.ones((2, 6)))
    validate_dose(np.zeros((2, 6)), np.zeros((2, 6)))
    with pytest.raises(ValueError):
        validate_dose(np.ones((2, 5)), np.ones((2, 5)))
    with pytest.raises(ValueError):
        validate_dose(np.ones((2, 6)), np.ones((1, 6)))
    with pytest.raises(ValueError):
        validate_dose(np.full((2, 6), -1), np.zeros((2, 6)))
