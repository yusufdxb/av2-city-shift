"""Score a group of seed checkpoints (one arm, one fold) on validation scenarios.

Writes one parquet row per scenario with per-seed errors, per-seed single-model
uncertainty scores, and the cross-seed ensemble disagreement.

Uncertainty scores (all fixed in the pre-registration, none fitted on val):
    disagree  ensemble: mean pairwise distance between the seeds' top-probability
              endpoints (primary signal, U1)
    entropy   single model: entropy of the mode probabilities (U2)
    spread    single model: probability-weighted std of mode endpoints (U3)
    maha      single model: Mahalanobis distance of the focal scene embedding to
              the model's own training embeddings (U4, exploratory)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np
import torch

from .data import Split, to_device
from .metrics import per_sample_metrics
from .model import Predictor


def load_model(path: str, device) -> tuple[Predictor, dict]:
    # checkpoints hold only tensors and a plain config dict, so refuse arbitrary pickled objects
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    m = Predictor(use_map=not cfg.get("no_map", False))
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), cfg


@torch.no_grad()
def run_model(model: Predictor, split: Split, idx: np.ndarray, device, lane_src: np.ndarray | None = None, bs: int = 512):
    """Returns traj [N,K,T,2], probs [N,K], focal embedding [N,d], target [N,T,2] in ``idx`` order.

    ``lane_src`` optionally replaces each scenario's map with another scenario's map
    (the map-swap positive control).
    """
    outs = {"traj": [], "prob": [], "emb": [], "target": []}
    for s in range(0, len(idx), bs):
        ii = idx[s : s + bs]
        b = {k: torch.from_numpy(np.ascontiguousarray(v[ii])) for k, v in split.arrays.items()}
        if lane_src is not None:
            jj = lane_src[s : s + bs]
            b["lane_pts"] = torch.from_numpy(np.ascontiguousarray(split.arrays["lane_pts"][jj]))
            b["lane_attr"] = torch.from_numpy(np.ascontiguousarray(split.arrays["lane_attr"][jj]))
        b = to_device(b, device)
        traj, logits, emb = model(b["agent_hist"], b["agent_valid"], b["agent_type"], b["lane_pts"], b["lane_attr"])
        outs["traj"].append(traj.float().cpu())
        outs["prob"].append(logits.float().softmax(-1).cpu())
        outs["emb"].append(emb.float().cpu())
        outs["target"].append(b["target"].cpu())
    return {k: torch.cat(v) for k, v in outs.items()}


def fit_gaussian(emb: np.ndarray, shrink: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    mu = emb.mean(0)
    cov = np.cov(emb - mu, rowvar=False)
    cov += shrink * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    return mu, np.linalg.inv(cov)


def mahalanobis(emb: np.ndarray, mu: np.ndarray, prec: np.ndarray) -> np.ndarray:
    d = emb - mu
    return np.sqrt(np.einsum("nd,de,ne->n", d, prec, d))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True, help="one checkpoint per seed")
    ap.add_argument("--out", required=True, help="output parquet")
    ap.add_argument("--map-swap", action="store_true", help="positive control: give every scenario another scenario's map")
    ap.add_argument("--fit-n", type=int, default=20000, help="training scenarios used to fit the Mahalanobis Gaussian")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    val = Split(args.root, "val")
    train = Split(args.root, "train")
    idx = np.arange(len(val))
    if args.limit:
        idx = idx[: args.limit]
    lane_src = None
    if args.map_swap:
        # Derangement within the same city, so the swap changes geometry but not city.
        rng = np.random.default_rng(7)
        lane_src = idx.copy()
        city = val.meta.city.to_numpy()[idx]
        for c in np.unique(city):
            perm = rng.permutation(np.where(city == c)[0])
            lane_src[perm] = idx[np.roll(perm, 1)]  # a cycle: nobody keeps their own map

    df = val.meta.iloc[idx][["scenario_id", "city", "focal_type", "speed"]].reset_index(drop=True)
    df["val_index"] = idx
    top_endpoints = []
    for s, path in enumerate(args.ckpts):
        model, cfg = load_model(path, device)
        r = run_model(model, val, idx, device, lane_src)
        m = per_sample_metrics(r["traj"], r["prob"].log(), r["target"])
        for k in ("min_ade", "min_fde", "miss", "brier_min_fde"):
            df[f"s{s}_{k}"] = m[k].numpy()
        p = r["prob"]
        df[f"s{s}_entropy"] = (-(p * (p + 1e-12).log()).sum(-1)).numpy()
        ep = r["traj"][:, :, -1]  # [N,K,2]
        mean_ep = (p.unsqueeze(-1) * ep).sum(1, keepdim=True)
        df[f"s{s}_spread"] = ((p * ((ep - mean_ep) ** 2).sum(-1)).sum(-1)).sqrt().numpy()
        top_endpoints.append(ep[torch.arange(len(ep)), p.argmax(-1)].numpy())
        # Mahalanobis fitted on this model's own training scenarios (never on val).
        tr_idx = np.load(os.path.join(os.path.dirname(path), "train_indices.npy"))
        fit_idx = np.sort(np.random.default_rng(s).choice(tr_idx, size=min(args.fit_n, len(tr_idx)), replace=False))
        mu, prec = fit_gaussian(run_model(model, train, fit_idx, device)["emb"].numpy().astype(np.float64))
        df[f"s{s}_maha"] = mahalanobis(r["emb"].numpy().astype(np.float64), mu, prec)
        df[f"s{s}_exclude_city"] = cfg.get("exclude_city")
    te = np.stack(top_endpoints)  # [S,N,2]
    pairs = list(itertools.combinations(range(len(te)), 2))
    df["disagree"] = np.mean([np.linalg.norm(te[i] - te[j], axis=-1) for i, j in pairs], 0) if pairs else np.nan
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_parquet(args.out)
    json.dump({"ckpts": args.ckpts, "map_swap": args.map_swap, "n": len(df)}, open(args.out + ".json", "w"), indent=2)
    print(f"wrote {args.out}: {len(df)} rows")


if __name__ == "__main__":
    main()
