"""Stage 1 training recipe on 128,000 multi-agent samples with scenario-ID dev exclusion."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

import numpy as np
import pandas as pd

from . import train
from .preprocess_multi import dev_scenario_ids
from .metrics import wta_loss
from .model import Predictor


def sample_indices(meta: pd.DataFrame, dev_ids: set[str], n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Select samples by scenario ID; no dev scenario may enter the training pool."""
    is_dev = meta.scenario_id.astype(str).isin(dev_ids).to_numpy()
    pool = np.flatnonzero(~is_dev)
    dev = np.flatnonzero(is_dev)
    if n > len(pool):
        raise ValueError(f"requested {n} samples, only {len(pool)} eligible")
    return np.sort(np.random.default_rng(seed).choice(pool, size=n, replace=False)), dev


def cpu_smoke(argv: list[str], dev_ids: set[str]) -> None:
    """Exercise 200 optimizer steps on a tiny CPU batch without using the GPU."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args(argv)
    if args.steps > 200 or args.batch > 2:
        ap.error("CPU smoke is limited to 200 steps and batch 2")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    split = train.Split(args.root, "train")
    idx, _ = sample_indices(split.meta, dev_ids, args.n_train, 1000 + args.seed)
    model = Predictor()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    os.makedirs(args.out, exist_ok=True)
    step, epoch, first_loss, last_loss = 0, 0, None, None
    while step < args.steps:
        for batch in train.loader(split, idx, args.batch, True, args.seed * 100003 + epoch, workers=0):
            if step >= args.steps:
                break
            for group in opt.param_groups:
                group["lr"] = 0.001 * (step + 1) / 1000
            traj, logits, _ = model(batch["agent_hist"], batch["agent_valid"], batch["agent_type"],
                                    batch["lane_pts"], batch["lane_attr"])
            loss, _ = wta_loss(traj.float(), logits.float(), batch["target"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if first_loss is None:
                first_loss = loss.detach().item()
            last_loss = loss.detach().item()
            step += 1
        epoch += 1
    torch.save({"model": model.state_dict(), "config": vars(args) | {"cpu_smoke": True}},
               os.path.join(args.out, "model.pt"))
    print(json.dumps({"steps": step, "train_samples": len(idx), "first_loss": first_loss, "last_loss": last_loss}))


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--focal-root", required=True)
    ap.add_argument("--dev-scenarios", default="runs/dev_scenarios.npy")
    ap.add_argument("--cpu-smoke", action="store_true")
    known, rest = ap.parse_known_args()
    dev_ids = dev_scenario_ids(known.focal_root, known.dev_scenarios)
    if known.cpu_smoke:
        cpu_smoke(rest, dev_ids)
        return

    def training_indices(meta, exclude_city, n, dev_frac, seed):
        if exclude_city is not None or dev_frac != 0.02:
            raise ValueError("MULTI uses all cities and the fixed 2% dev scenarios")
        return sample_indices(meta, dev_ids, n, seed)[0]

    def dev_indices(meta, dev_frac):
        if dev_frac != 0.02:
            raise ValueError("MULTI uses the fixed 2% dev scenarios")
        dev = sample_indices(meta, dev_ids, 0, 0)[1]
        if not len(dev):
            raise ValueError("multi-agent data has no dev samples")
        return dev

    train.training_indices = training_indices
    train.dev_indices = dev_indices
    sys.argv = [sys.argv[0], *rest]
    train.main()


if __name__ == "__main__":
    main()
