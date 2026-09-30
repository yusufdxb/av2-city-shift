import pandas as pd
import pytest

from cityshift.sensitivity import summarize, with_brake


def _rows():
    """4 scenes; ALL/PATCH for model seeds 0 and 1, cv/oracle for seed 0 only; the human never brakes."""
    rows = []
    brakes = {("ALL", 0): {"a", "b"}, ("ALL", 1): {"a", "b"}, ("PATCH", 0): {"a"}, ("PATCH", 1): set()}
    for scene in "abcd":
        for arm, seeds in (("ALL", (0, 1)), ("PATCH", (0, 1)), ("cv", (0,)), ("oracle", (0,))):
            for seed in seeds:
                rows.append({"scenario_id": scene, "city": "x", "model_seed": seed, "cap": "v2", "risk_weight": 100.0,
                             "arm": arm, "collision": arm == "PATCH" and seed == 1 and scene == "d", "progress": 1.0,
                             "unnecessary_hard_brake": False, "beyond_logged_route": False,
                             "min_exec_decel": -4.5 if scene in brakes.get((arm, seed), set()) else -1.0,
                             "min_log_decel": -0.5})
    return pd.DataFrame(rows)


def test_with_brake_excludes_drives_where_the_human_braked_as_hard():
    d = pd.DataFrame({"min_exec_decel": [-4.5, -4.5, -3.9], "min_log_decel": [-0.5, -4.2, -0.5]})
    assert with_brake(d, -4.0).unnecessary_hard_brake.tolist() == [True, False, False]
    assert with_brake(d, -3.0).unnecessary_hard_brake.tolist() == [True, False, True]


def test_summarize_seed_averaged_effects_and_intervals():
    r = summarize(_rows(), n_boot=500)
    assert r["model_seeds"] == [0, 1] and r["n_scenes"] == 4 and len(r["configs"]) == 3
    by_t = {c["brake_threshold"]: c for c in r["configs"]}
    e = by_t[-4.0]["PATCH_vs_ALL"]
    # ALL brakes in 2 of 4 scenes; PATCH in 1 scene for 1 of 2 seeds -> 0.125 vs 0.5
    assert e["brake_reduction"]["point"] == pytest.approx(0.75)
    assert e["brake_reduction"]["per_seed"] == {"0": pytest.approx(0.5), "1": pytest.approx(1.0)}
    lo, hi = e["brake_reduction"]["ci95"]
    assert lo <= 0.75 <= hi
    # PATCH collides in scene d for seed 1 only: seed-averaged 0.5 in 1 of 4 scenes
    assert e["collision_diff_pp"]["point"] == pytest.approx(12.5)
    assert by_t[-4.0]["rates"]["cv"]["model_seeds"] == [0]
    # nobody decelerates to -5, so the reduction is undefined rather than 0
    assert by_t[-5.0]["PATCH_vs_ALL"]["brake_reduction"]["point"] is None
    assert r["seed0_legacy"]["configs"][1]["PATCH_pooled_brake_reduction"] == pytest.approx(0.5)
    assert r["seed0_legacy"]["configs"][2]["PATCH_pooled_brake_reduction"] is None


def test_summarize_rejects_missing_model_arm_rows():
    d = _rows()
    with pytest.raises(AssertionError):
        summarize(d[~((d.arm == "PATCH") & (d.model_seed == 1) & (d.scenario_id == "c"))], n_boot=10)
