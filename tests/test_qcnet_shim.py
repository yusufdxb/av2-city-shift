"""The pure-PyTorch radius/radius_graph stand-ins used for QCNet match a brute-force torch_cluster definition."""

from __future__ import annotations

import torch

from cityshift.qcnet_adapter import _gather_csr, _radius, _radius_graph, _segment_csr


def brute_radius(x, y, r, bx, by, k):
    """For each y in order, the first k x points (index order) in the same batch with squared distance < r^2."""
    pairs = []
    for i in range(len(y)):
        found = 0
        for j in range(len(x)):
            if bx[j] == by[i] and float(((x[j] - y[i]) ** 2).sum()) < r * r and found < k:
                pairs.append((i, j))
                found += 1
    return sorted(pairs)


def test_radius_matches_brute_force_including_the_neighbour_cap() -> None:
    g = torch.Generator().manual_seed(0)
    x, y = torch.rand(60, 2, generator=g) * 10, torch.rand(40, 2, generator=g) * 10
    bx, by = torch.sort(torch.randint(0, 3, (60,), generator=g)).values, torch.sort(torch.randint(0, 3, (40,), generator=g)).values
    for k in (3, 300):
        got = _radius(x, y, 4.0, bx, by, max_num_neighbors=k)
        assert sorted(map(tuple, got.t().tolist())) == brute_radius(x, y, 4.0, bx, by, k)


def test_radius_graph_drops_self_loops_and_uses_source_to_target() -> None:
    g = torch.Generator().manual_seed(1)
    x = torch.rand(30, 2, generator=g) * 5
    b = torch.zeros(30, dtype=torch.long)
    edge = _radius_graph(x, 2.0, b, loop=False, max_num_neighbors=300)
    expected = sorted((j, i) for i, j in brute_radius(x, x, 2.0, b, b, 301) if i != j)
    assert sorted(map(tuple, edge.t().tolist())) == expected


def test_segment_and_gather_csr() -> None:
    src = torch.arange(6.0)
    ptr = torch.tensor([0, 2, 2, 6])
    assert _segment_csr(src, ptr, reduce="sum").tolist() == [1.0, 0.0, 14.0]
    assert _gather_csr(torch.tensor([1.0, 2.0, 3.0]), ptr).tolist() == [1.0, 1.0, 3.0, 3.0, 3.0, 3.0]
