"""Adapter for the released QCNet Argoverse 2 checkpoint (v2 study Q).

QCNet (Zhou et al., CVPR 2023, https://github.com/ZikangZhou/QCNet, Apache-2.0) is not vendored: point ``QCNET_ROOT``
at a checkout and pass the released ``QCNet_AV2.ckpt``. Inputs are built with QCNet's own preprocessing
(``ArgoverseV2Dataset.get_agent_features`` and ``get_map_features``) on a time-shifted copy of the raw scenario, so a
forecast at replan ``t`` sees exactly the 50 steps ending at ``t``, with the self-driving car's track replaced by the
simulated ego when one is given. Outputs are rotated back to world coordinates as QCNet's ``test_step`` does.

QCNet calls three functions from compiled PyG extensions; when ``torch_cluster`` or ``torch_scatter`` is not
installed, pure-PyTorch versions with the same semantics are registered (``radius`` keeps, for each query point, the
first ``max_num_neighbors`` points in index order with squared distance strictly below r^2, as torch_cluster does).
"""

from __future__ import annotations

import inspect
import os
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HIST, FUT = 50, 60


def _radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32, num_workers=1, batch_size=None):
    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y
    bx = batch_x if batch_x is not None else x.new_zeros(x.size(0), dtype=torch.long)
    by = batch_y if batch_y is not None else y.new_zeros(y.size(0), dtype=torch.long)
    if x.size(0) == 0 or y.size(0) == 0:
        return torch.zeros(2, 0, dtype=torch.long, device=x.device)
    # Both batch vectors are sorted (PyG batching); process runs of whole batch ids so each block is one dense
    # distance matrix, with cross-batch pairs masked out before the per-row neighbour count.
    nb = int(max(bx.max().item(), by.max().item())) + 1
    px = [0] + torch.cumsum(torch.bincount(bx, minlength=nb), 0).tolist()
    py = [0] + torch.cumsum(torch.bincount(by, minlength=nb), 0).tolist()
    rows, cols, b0 = [], [], 0
    while b0 < nb:
        b1 = b0 + 1
        while b1 < nb and (py[b1 + 1] - py[b0]) * (px[b1 + 1] - px[b0]) <= 40_000_000:
            b1 += 1
        y0, y1, x0, x1 = py[b0], py[b1], px[b0], px[b1]
        if y1 > y0 and x1 > x0:
            d2 = torch.cdist(y[y0:y1].float(), x[x0:x1].float()).pow(2)
            mask = (d2 < r * r) & (by[y0:y1, None] == bx[None, x0:x1])
            mask &= torch.cumsum(mask.to(torch.int32), dim=1) <= max_num_neighbors
            ry, cx = mask.nonzero(as_tuple=True)
            rows.append(ry + y0)
            cols.append(cx + x0)
        b0 = b1
    if not rows:
        return torch.zeros(2, 0, dtype=torch.long, device=x.device)
    return torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)


def _radius_graph(x, r, batch=None, loop=False, max_num_neighbors=32, flow="source_to_target", num_workers=1,
                  batch_size=None):
    edge = _radius(x, x, r, batch, batch, max_num_neighbors if loop else max_num_neighbors + 1)
    row, col = edge[0], edge[1]  # row: query (centre), col: neighbour
    if not loop:
        keep = row != col
        row, col = row[keep], col[keep]
    return torch.stack([col, row], 0) if flow == "source_to_target" else torch.stack([row, col], 0)


def _segment_csr(src, indptr, out=None, reduce="sum"):
    lengths = (indptr[1:] - indptr[:-1]).to(torch.long)
    return torch.segment_reduce(src, "sum" if reduce == "add" else reduce, lengths=lengths, axis=0, unsafe=True)


def _gather_csr(src, indptr, out=None):
    return src.repeat_interleave((indptr[1:] - indptr[:-1]).to(torch.long), dim=0)


def _install_shims() -> dict[str, bool]:
    """Register the stand-ins only for QCNet. PyG is imported first so it never routes its own internals to them."""
    import importlib.machinery

    import torch_geometric  # noqa: F401

    used = {}
    for name, funcs in (("torch_cluster", {"radius": _radius, "radius_graph": _radius_graph}),
                        ("torch_scatter", {"segment_csr": _segment_csr, "gather_csr": _gather_csr})):
        try:
            __import__(name)
            used[name] = False
        except ImportError:
            module = types.ModuleType(name)
            module.__dict__.update(funcs)
            module.__spec__ = importlib.machinery.ModuleSpec(name, None)
            sys.modules[name] = module
            used[name] = True
    return used


def _qcnet_path(root: str | None) -> str:
    root = os.path.expanduser(root or os.environ.get("QCNET_ROOT", "~/third_party/QCNet"))
    if not os.path.isdir(os.path.join(root, "predictors")):
        raise FileNotFoundError(f"QCNet checkout not found at {root}; set QCNET_ROOT")
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def load_qcnet(checkpoint: str, device: torch.device, root: str | None = None):
    """The released checkpoint in eval mode, plus which shims were installed (recorded in run manifests)."""
    shims = _install_shims()
    _qcnet_path(root)
    from predictors import QCNet  # noqa: E402  (QCNet's own package layout)

    model = QCNet.load_from_checkpoint(checkpoint, map_location=device)
    model.eval().to(device)
    return model, shims


def _dataset_stub():
    """An ArgoverseV2Dataset whose feature builders can be called without a dataset directory."""
    from datasets import ArgoverseV2Dataset  # noqa: E402  (QCNet's package)

    ds = ArgoverseV2Dataset.__new__(ArgoverseV2Dataset)
    ds.dim, ds.num_historical_steps, ds.num_future_steps = 3, HIST, FUT
    ds.num_steps, ds.predict_unseen_agents, ds.vector_repr = HIST + FUT, False, True
    ds.split = "val"  # only switches test-split category handling, which needs no labels here
    # The type tables (agent types, categories, polygon/point types) are executed from QCNet's own __init__ source,
    # so they cannot drift from the checkpoint's training code; the base-class init (which would download) is skipped.
    source = inspect.getsource(ArgoverseV2Dataset.__init__)
    start = source.rindex("\n", 0, source.index("self._agent_types")) + 1
    tables = source[start:source.rindex("\n", 0, source.index("super(ArgoverseV2Dataset")) + 1]
    exec(textwrap.dedent(tables), {}, {"self": ds})
    return ds


class QCNetScenario:
    """One raw Argoverse 2 scenario, with map features computed once and agent features rebuilt per replan."""

    def __init__(self, scenario_dir: str, root: str | None = None):
        _qcnet_path(root)
        from av2.map.map_api import ArgoverseStaticMap
        from av2.map.map_primitives import Polyline
        from av2.utils.io import read_json_file

        sid = os.path.basename(os.path.normpath(scenario_dir))
        self.df = pd.read_parquet(os.path.join(scenario_dir, f"scenario_{sid}.parquet"))
        map_path = sorted(Path(scenario_dir).glob("log_map_archive_*.json"))[0]
        map_data = read_json_file(map_path)
        centerlines = {ls["id"]: Polyline.from_json_data(ls["centerline"]) for ls in map_data["lane_segments"].values()}
        self.ds = _dataset_stub()
        self.map_features = self.ds.get_map_features(ArgoverseStaticMap.from_json(map_path), centerlines)
        self.scenario_id = str(self.df["scenario_id"].values[0])
        self.city = str(self.df["city"].values[0])

    def data_at(self, t: int, ego: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None):
        """HeteroData for the 50 steps ending at original step ``t`` (49 is the handoff).

        ``ego`` = (positions [110, 2], headings [110], speeds [110]) of the simulated ego; steps up to ``t`` replace
        the logged AV track (z is kept from the log), later steps of the AV are dropped.
        """
        from torch_geometric.data import HeteroData

        from transforms import TargetBuilder  # noqa: E402  (QCNet's package)

        shift = t - (HIST - 1)
        df = self.df[(self.df.timestep >= shift) & (self.df.timestep <= shift + HIST + FUT - 1)].copy()
        if ego is not None:
            pos, head, speed = ego
            av = df.track_id == "AV"
            df = df[~(av & (df.timestep > t))]
            av = df.track_id == "AV"
            steps = df.loc[av, "timestep"].to_numpy()
            df.loc[av, "position_x"] = pos[steps, 0]
            df.loc[av, "position_y"] = pos[steps, 1]
            df.loc[av, "heading"] = head[steps]
            df.loc[av, "velocity_x"] = speed[steps] * np.cos(head[steps])
            df.loc[av, "velocity_y"] = speed[steps] * np.sin(head[steps])
        df["timestep"] = df["timestep"] - shift
        data = {"scenario_id": self.scenario_id, "city": self.city, "agent": self.ds.get_agent_features(df)}
        data.update(self.map_features)
        # TargetBuilder's own __call__, unbound: newer PyG makes BaseTransform abstract (it requires forward()).
        return TargetBuilder.__call__(types.SimpleNamespace(num_historical_steps=HIST, num_future_steps=FUT),
                                      HeteroData(data))


@torch.no_grad()
def predict(model, data_list: list, device: torch.device) -> list[dict[str, tuple[np.ndarray, np.ndarray]]]:
    """World-frame forecasts per scenario: {track_id: (trajectories [6, 60, 2], probabilities [6])}.

    Only agents valid at the current step are returned (the planner only forecasts those).
    """
    from torch_geometric.data import Batch

    batch = Batch.from_data_list(data_list).to(device)
    batch["agent"]["av_index"] += batch["agent"]["ptr"][:-1]
    pred = model(batch)
    traj = pred["loc_refine_pos"][..., :2]
    pi = torch.softmax(pred["pi"], dim=-1)
    origin = batch["agent"]["position"][:, HIST - 1, :2]
    theta = batch["agent"]["heading"][:, HIST - 1]
    cos, sin = theta.cos(), theta.sin()
    rot = torch.stack([torch.stack([cos, sin], -1), torch.stack([-sin, cos], -1)], -2)  # as QCNet's test_step
    world = torch.matmul(traj, rot.unsqueeze(1)) + origin.reshape(-1, 1, 1, 2)
    world, pi = world.cpu().numpy().astype(np.float64), pi.cpu().numpy().astype(np.float64)
    valid = batch["agent"]["valid_mask"][:, HIST - 1].cpu().numpy()
    out = []
    ptr = batch["agent"]["ptr"].cpu().numpy()
    for g, ids in enumerate(batch["agent"]["id"]):
        lo = ptr[g]
        out.append({str(tid): (world[lo + a], pi[lo + a]) for a, tid in enumerate(ids) if valid[lo + a]})
    return out
