"""Fixed endpoints, complete-future masks and scenario-cluster study F inference."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cityshift import coverage
from cityshift.closedloop import init_ego, select_agents
from cityshift.preprocess import OBJECT_TYPES


def test_horizons_and_complete_future_mask():
    valid = np.ones(110, bool)
    assert [coverage.horizons(t) for t in coverage.REPLANS] == [(2, 4), (2, 4), (2, 4), (2,), (2,)]
    assert coverage.complete_future(valid, 89, 2)  # endpoint is exactly t=109
    assert coverage.complete_future(valid, 69, 4)
    assert not coverage.complete_future(valid, 79, 4)
    valid[109] = False
    assert not coverage.complete_future(valid, 89, 2)
    assert not coverage.complete_future(valid, 69, 4)
    assert coverage.complete_future(valid, 49, 2)
    valid[60] = False  # an interior gap also excludes an otherwise observed endpoint
    assert not coverage.complete_future(valid, 49, 2)
    assert not coverage.complete_future(np.ones(109, bool), 89, 2)
    with pytest.raises(ValueError, match="unregistered replan"):
        coverage.horizons(99)


def test_endpoint_at_exact_horizon_and_strict_two_metre_threshold():
    traj = np.zeros((6, 60, 2))
    target = np.zeros((60, 2))
    target[19, 0] = 2
    target[39, 0] = 3
    target[59, 0] = 100
    assert coverage.score_endpoint(traj, target, 2) == {"min_fde_2s": 2, "miss_2s": 0}
    assert coverage.score_endpoint(traj, target, 4) == {"min_fde_4s": 3, "miss_4s": 1}
    traj[5, 39, 0] = 3
    assert coverage.score_endpoint(traj, target, 4)["miss_4s"] == 0


def test_score_scene_averages_seed_metrics_and_retains_excluded_rows(monkeypatch):
    sc = SimpleNamespace(
        scenario_id="synthetic",
        city="austin",
        track_ids=["AV", "full", "gap"],
        av=0,
        focal=0,
        valid=np.ones((3, 110), bool),
        pos=np.zeros((3, 110, 2)),
        vel=np.zeros((3, 110, 2)),
        types=np.full(3, OBJECT_TYPES.index("vehicle")),
    )
    sc.valid[2, 50:] = False
    monkeypatch.setattr(coverage, "logged_ego", lambda scene, t: object())
    monkeypatch.setattr(coverage, "select_agents", lambda scene, ego, t: [1, 2] if t == 49 else [])
    monkeypatch.setattr(
        coverage,
        "build_input",
        lambda scene, i, t: {
            "theta": np.pi / 2,
            "origin": np.zeros(2),
            "agent_hist": np.zeros((1, 50, 6)),
        },
    )

    # The helper returns world trajectories; score the three seeds separately.
    def fake_predict(models, batch, device):
        return [(np.full((6, 60, 2), [3 * int(key[-1]), 0.0]), np.ones(6) / 6) for key, _ in batch]

    monkeypatch.setattr(coverage, "predict", fake_predict)
    rows = coverage.score_scene(sc, {f"ALL_s{s}": object() for s in range(3)}, "cpu")
    all_row = next(r for r in rows if r["agent_id"] == "full" and r["predictor"] == "ALL")
    assert all_row["min_fde_2s"] == pytest.approx(3)
    assert all_row["miss_2s"] == pytest.approx(2 / 3)
    assert all_row["miss_4s"] == pytest.approx(2 / 3)
    excluded = [r for r in rows if r["agent_id"] == "gap"]
    assert len(excluded) == 2
    assert all(not r["complete_2s"] and np.isnan(r["miss_2s"]) for r in excluded)


def test_logged_ego_selection_matches_handoff_and_tracks_later_position():
    pos = np.zeros((3, 110, 2))
    pos[0, :, 0] = np.arange(110)
    pos[1, :, 0] = 50  # in radius at 49, outside at 89
    pos[2, :, 0] = 145  # outside at 49, in radius at 89
    sc = SimpleNamespace(
        pos=pos,
        vel=np.ones_like(pos),
        head=np.zeros((3, 110)),
        valid=np.ones((3, 110), bool),
        av=0,
        types=np.full(3, OBJECT_TYPES.index("vehicle")),
        track_ids=["AV", "a", "b"],
    )
    assert select_agents(sc, coverage.logged_ego(sc, 49), 49) == select_agents(sc, init_ego(sc), 49)
    np.testing.assert_allclose(coverage.logged_ego(sc, 89).path.at(np.array([0.0]))[0][0], sc.pos[0, 89])
    assert 2 in select_agents(sc, coverage.logged_ego(sc, 89), 89)


def test_city_equal_cluster_bootstrap_and_exclusion_counts():
    rows = []
    for k, city in enumerate(coverage.CITIES):
        for scenario, n_agents in (("large", 10), ("small", 1)):
            for agent in range(n_agents + 1):
                complete = agent < n_agents
                common = {
                    "scenario_id": f"{city}-{scenario}",
                    "replan": 49,
                    "agent_id": str(agent),
                    "city": city,
                    "agent_type": 0,
                    "focal": False,
                    "speed": 0.0,
                    "complete_2s": complete,
                    "complete_4s": complete,
                }
                for predictor in ("ALL", "CV-6"):
                    miss = float(scenario == "small") if predictor == "ALL" else 0.0
                    rows.append(
                        common
                        | {
                            "predictor": predictor,
                            **{f"{m}_{h}s": miss if complete else np.nan for h in (2, 4) for m in ("miss", "min_fde")},
                        }
                    )
    df = pd.DataFrame(rows)
    result = coverage.analyze(df, n_boot=200)
    cell = result["cells"][0]
    assert cell["difference_ALL_minus_CV6"] == pytest.approx(1 / 11)
    assert cell["n_agents"] == 66
    assert cell["n_excluded"] == 12
    assert cell["excluded_share"] == pytest.approx(2 / 13)
    assert cell["status"] == "killed"  # magnitude bar fails
    assert result["decision"] == "killed"
    # Independent draws must retain all ten agents in the large scenario together.
    rng = np.random.default_rng(coverage.SEED)
    boots = []
    for _ in coverage.CITIES:
        ii = rng.integers(0, 2, size=(200, 2))
        boots.append((ii == 1).sum(1) / np.where(ii == 0, 10, 1).sum(1))
    np.testing.assert_allclose(cell["ci95"], np.percentile(np.mean(boots, axis=0), [2.5, 97.5]))
    missing = coverage.analyze(df.loc[df.city != coverage.CITIES[0]], n_boot=10)
    assert missing["cells"][0]["status"] == "inconclusive"
    with pytest.raises(ValueError, match="share agents and metadata"):
        coverage.analyze(df.iloc[:-1], n_boot=10)
