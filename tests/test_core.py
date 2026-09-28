import numpy as np
import pytest
import torch

from cityshift.analysis import auroc, capture_fraction, holm, keep_mask, sign_flip_p
from cityshift.metrics import per_sample_metrics, wta_loss
from cityshift.model import Predictor
from cityshift.preprocess import FUT, HIST, LANE_PTS, MAX_AGENTS, MAX_LANES, resample_polyline


def test_resample_polyline_even_spacing():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    r = resample_polyline(xy, 5)
    assert r.shape == (5, 2)
    np.testing.assert_allclose(np.linalg.norm(np.diff(r, axis=0), axis=1), 5.0, atol=1e-9)
    np.testing.assert_allclose(r[0], xy[0])
    np.testing.assert_allclose(r[-1], xy[-1])


def test_resample_degenerate():
    assert (resample_polyline(np.array([[1.0, 2.0]]), 4) == [1.0, 2.0]).all()
    assert (resample_polyline(np.array([[1.0, 2.0], [1.0, 2.0]]), 4) == [1.0, 2.0]).all()


def test_metrics_pick_best_endpoint_mode():
    target = torch.zeros(1, FUT, 2)
    target[0, :, 0] = torch.linspace(0.1, 6.0, FUT)
    traj = torch.stack([target[0] + torch.tensor([0.0, 3.0]), target[0] + torch.tensor([0.0, 1.0])])[None]
    logits = torch.tensor([[0.0, 0.0]])
    m = per_sample_metrics(traj, logits, target)
    assert m["best_mode"].item() == 1
    assert m["min_fde"].item() == pytest.approx(1.0)
    assert m["min_ade"].item() == pytest.approx(1.0)
    assert m["miss"].item() == 0.0
    assert m["brier_min_fde"].item() == pytest.approx(1.0 + 0.25)


def test_wta_loss_only_regresses_winner():
    target = torch.zeros(1, FUT, 2)
    traj = torch.zeros(1, 2, FUT, 2, requires_grad=True)
    with torch.no_grad():
        traj[0, 1] += 5.0
    loss, _ = wta_loss(traj, torch.zeros(1, 2), target)
    loss.backward()
    assert traj.grad[0, 1].abs().sum() == 0  # loser gets no regression gradient


def _batch(b=2, n_agents=3, n_lanes=5):
    g = torch.Generator().manual_seed(0)
    hist = torch.randn(b, MAX_AGENTS, HIST, 6, generator=g)
    valid = torch.zeros(b, MAX_AGENTS, HIST, dtype=torch.bool)
    valid[:, :n_agents] = True
    atype = torch.full((b, MAX_AGENTS), -1, dtype=torch.int8)
    atype[:, :n_agents] = 0
    lanes = torch.randn(b, MAX_LANES, LANE_PTS, 2, generator=g) * 10
    lattr = torch.full((b, MAX_LANES, 2), -1, dtype=torch.int8)
    lattr[:, :n_lanes] = 0
    return hist, valid, atype, lanes, lattr


def test_model_shapes_and_padding_invariance():
    torch.manual_seed(0)
    m = Predictor().eval()
    hist, valid, atype, lanes, lattr = _batch()
    traj, logits, emb = m(hist, valid, atype, lanes, lattr)
    assert traj.shape == (2, 6, FUT, 2) and logits.shape == (2, 6) and emb.shape == (2, 128)
    # changing padded (masked) content must not change the output
    hist2, lanes2 = hist.clone(), lanes.clone()
    hist2[:, 10:] += 100.0
    lanes2[:, 20:] += 100.0
    traj2, logits2, _ = m(hist2, valid, atype, lanes2, lattr)
    torch.testing.assert_close(traj, traj2, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(logits, logits2, atol=1e-4, rtol=1e-4)


def test_no_map_model_ignores_map():
    torch.manual_seed(0)
    m = Predictor(use_map=False).eval()
    hist, valid, atype, lanes, lattr = _batch()
    a = m(hist, valid, atype, lanes, lattr)[0]
    b = m(hist, valid, atype, lanes * 3 + 7, lattr)[0]
    torch.testing.assert_close(a, b)


def test_keep_mask_exact_coverage():
    s = np.random.default_rng(0).random(1001)
    k = keep_mask(s, 0.8)
    assert k.sum() == round(0.8 * 1001)
    assert s[k].max() <= s[~k].min()


def test_capture_fraction_bounds():
    rng = np.random.default_rng(0)
    err = rng.exponential(2.0, 5000)
    miss = (err > 2).astype(float)
    assert capture_fraction(miss, err, err) == pytest.approx(1.0)
    assert capture_fraction(miss, -err, err) < 0
    rand = np.mean([capture_fraction(miss, rng.random(5000), err) for _ in range(200)])
    assert abs(rand) < 0.05


def test_sign_flip_floor_and_holm():
    assert sign_flip_p(np.ones(6)) == pytest.approx(2 / 64)
    assert sign_flip_p(np.ones(6), two_sided=False) == pytest.approx(1 / 64)
    h = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert h == {"a": 0.03, "c": 0.06, "b": 0.06}


def test_auroc():
    assert auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert auroc(np.array([1.0]), np.array([1.0])) == 0.5
