"""build_input at (focal, t=49) must reproduce the training preprocessor exactly."""

import glob
import os

import numpy as np
import pytest

from cityshift.preprocess import process_scenario
from cityshift.scene import build_input, load_scene

RAW = os.path.expanduser("~/datasets/av2/motion-forecasting/train")
DIRS = sorted(glob.glob(os.path.join(RAW, "*")))[:40]


@pytest.mark.skipif(not DIRS, reason="raw Argoverse 2 data not present")
@pytest.mark.parametrize("d", DIRS)
def test_build_input_matches_preprocess(d):
    ref = process_scenario(d)
    sc = load_scene(d)
    got = build_input(sc, sc.focal, 49)
    np.testing.assert_allclose(got["agent_hist"], ref["agent_hist"], atol=1e-4)
    np.testing.assert_array_equal(got["agent_valid"], ref["agent_valid"])
    np.testing.assert_array_equal(got["agent_type"], ref["agent_type"])
    # lane order may differ only among exact distance ties; compare as sets of rows
    a = np.round(got["lane_pts"].reshape(len(got["lane_pts"]), -1), 3)
    b = np.round(ref["lane_pts"].reshape(len(ref["lane_pts"]), -1), 3)
    assert sorted(map(tuple, a)) == sorted(map(tuple, b))
    fut = (sc.pos[sc.focal, 50:110] - got["origin"]) @ np.array(
        [[np.cos(got["theta"]), -np.sin(got["theta"])], [np.sin(got["theta"]), np.cos(got["theta"])]]
    )
    np.testing.assert_allclose(fut, ref["target"], atol=1e-4)
