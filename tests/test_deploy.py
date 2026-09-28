import torch

from cityshift.export_trt import Deployable
from cityshift.model import Predictor
from cityshift.preprocess import HIST, LANE_PTS, MAX_AGENTS, MAX_LANES


def test_dummy_lanes_do_not_change_outputs_and_remove_zero_length_lanes():
    torch.manual_seed(0)
    m = Predictor().eval()
    b = 3
    hist = torch.randn(b, MAX_AGENTS, HIST, 6)
    valid = torch.zeros(b, MAX_AGENTS, HIST)
    valid[:, :5] = 1
    atype = torch.full((b, MAX_AGENTS), -1, dtype=torch.int32)
    atype[:, :5] = 0
    lanes = torch.randn(b, MAX_LANES, LANE_PTS, 2) * 10
    lattr = torch.full((b, MAX_LANES, 2), -1, dtype=torch.int32)
    lattr[:, :40] = 0
    lanes[:, 40:] = 0.0  # padded lanes are all-zero, as produced by preprocessing
    with torch.no_grad():
        raw = m(hist, valid > 0.5, atype, lanes, lattr)[0]
        dep = Deployable(m).eval()(hist, valid, atype, lanes, lattr)[0]
    torch.testing.assert_close(raw, dep, atol=1e-5, rtol=0)
