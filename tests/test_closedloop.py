import numpy as np

from cityshift.closedloop import ACCELS, Path, boxes_overlap, in_front, profiles


def test_boxes_overlap_basic():
    z = np.zeros(2)
    assert boxes_overlap(z, 0.0, 4.0, 2.0, np.array([3.9, 0.0]), 0.0, 4.0, 2.0)
    assert not boxes_overlap(z, 0.0, 4.0, 2.0, np.array([4.1, 0.0]), 0.0, 4.0, 2.0)
    assert not boxes_overlap(z, 0.0, 4.0, 2.0, np.array([0.0, 2.1]), 0.0, 4.0, 2.0)
    # rotated 90 degrees: the other box's long side now spans y
    assert boxes_overlap(z, 0.0, 4.0, 2.0, np.array([0.0, 2.9]), np.pi / 2, 4.0, 2.0)
    assert not boxes_overlap(z, 0.0, 4.0, 2.0, np.array([0.0, 3.1]), np.pi / 2, 4.0, 2.0)
    # diagonal gap that axis-aligned bounds would call a hit
    assert not boxes_overlap(z, np.pi / 4, 4.0, 0.5, np.array([1.5, -1.5]), np.pi / 4, 4.0, 0.5)


def test_boxes_overlap_broadcasts():
    c2 = np.stack([np.linspace(0.5, 10.5, 11), np.zeros(11)], -1)  # avoid the touching case at exactly 4 m
    hit = boxes_overlap(np.zeros(2), 0.0, 4.0, 2.0, c2, np.zeros(11), 4.0, 2.0)
    assert hit.shape == (11,) and hit[:4].all() and not hit[4:].any()


def test_in_front():
    assert in_front(np.zeros(2), 0.0, np.array([1.0, 5.0]))
    assert not in_front(np.zeros(2), 0.0, np.array([-1.0, 5.0]))


def test_path_follows_polyline_and_extends():
    class Sc:
        pass

    p = Path(s=np.array([0.0, 10.0, 20.0]), xy=np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]), length_logged=20.0)
    xy, h = p.at(np.array([5.0, 15.0]))
    np.testing.assert_allclose(xy, [[5.0, 0.0], [10.0, 5.0]])
    np.testing.assert_allclose(h, [0.0, np.pi / 2])


def test_profiles_clamp_speed():
    v, d = profiles(5.0, 6.0, 30)
    assert v.shape == (len(ACCELS), 30)
    assert (v >= 0).all() and (v <= 6.0).all()
    i8 = list(ACCELS).index(-8)
    assert v[i8, -1] == 0.0 and d[i8, -1] < 5.0 * 3.0
