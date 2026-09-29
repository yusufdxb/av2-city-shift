"""Train a fixed 75% focal, 25% stopped non-focal predictor without scenario leakage."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

from . import train
from .data import Split, dev_indices
from .metrics import per_sample_metrics, wta_loss
from .model import Predictor
from .preprocess_multi import dev_scenario_ids

POOL_SHA256 = "a7489d7bd7b95986807ee360e62a01c45a15f4c3297258ba4751f8520caee0c8"


def pool_ids(path: str) -> set[str]:
    from .closedloop_v3 import checked_ids

    values = checked_ids(path, POOL_SHA256)
    if len(values) != 8140:
        raise ValueError("replication pool must have 8,140 scenario IDs")
    return set(values)


def sample_indices(focal: pd.DataFrame, multi: pd.DataFrame, excluded: set[str], n: int,
                   seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw exactly three focal samples per stopped non-focal sample."""
    if n % 4:
        raise ValueError("mixture size must be divisible by four")
    focal_ok = ~focal.scenario_id.astype(str).isin(excluded).to_numpy()
    multi_ok = (~multi.scenario_id.astype(str).isin(excluded).to_numpy()
                & ~multi.focal.to_numpy(bool) & (multi.speed.to_numpy(float) < 0.5))
    f_pool, m_pool = np.flatnonzero(focal_ok), np.flatnonzero(multi_ok)
    n_f, n_m = n * 3 // 4, n // 4
    if len(f_pool) < n_f or len(m_pool) < n_m:
        raise ValueError(f"insufficient eligible samples: focal {len(f_pool)}/{n_f}, multi {len(m_pool)}/{n_m}")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(f_pool, n_f, replace=False)), np.sort(rng.choice(m_pool, n_m, replace=False))


class MixSplit:
    """Read-only concatenation of focal and multi preprocessed splits."""

    def __init__(self, focal: Split, multi: Split):
        self.focal, self.multi = focal, multi
        self.meta = pd.concat([focal.meta.assign(source="focal"), multi.meta.assign(source="multi")],
                              ignore_index=True)

    def __len__(self) -> int:
        return len(self.meta)

    def batch(self, idx: np.ndarray) -> dict[str, torch.Tensor]:
        idx = np.asarray(idx)
        f_mask = idx < len(self.focal)
        left = self.focal.batch(idx[f_mask]) if f_mask.any() else None
        right = self.multi.batch(idx[~f_mask] - len(self.focal)) if (~f_mask).any() else None
        if left is None:
            result = right
        elif right is None:
            result = left
        else:
            result = {k: torch.cat([left[k], right[k]], 0) for k in left}
        assert result is not None
        result["index"] = torch.from_numpy(np.sort(idx))
        return result


def evaluate_dev(model: Predictor, split: Split, idx: np.ndarray, device: torch.device) -> dict[str, float]:
    """Score both fixed TRAIN-dev sources and retain source-specific metrics."""
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in train.loader(split, idx, 256, False, workers=4):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                traj, logits, _ = model(batch["agent_hist"], batch["agent_valid"], batch["agent_type"],
                                        batch["lane_pts"], batch["lane_attr"])
            metrics = per_sample_metrics(traj.float(), logits.float(), batch["target"])
            for key in ("min_ade", "min_fde", "miss", "brier_min_fde"):
                sums[key] = sums.get(key, 0) + float(metrics[key].sum())
            count += len(batch["target"])
    model.train()
    return {key: value / count for key, value in sums.items()}


def cpu_smoke(focal: Split, multi: Split, excluded: set[str], out: str, steps: int,
              n_train: int, seed: int) -> None:
    """Exercise a small, genuine mixture on CPU for up to 200 steps."""
    if not 1 <= steps <= 200 or n_train % 4:
        raise ValueError("CPU smoke requires 1..200 steps and mixture size divisible by four")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    fi, mi = sample_indices(focal.meta, multi.meta, excluded, n_train, 1000 + seed)
    split = MixSplit(focal, multi)
    idx = np.r_[fi, mi + len(focal)]
    model = Predictor()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    step, epoch, first, last = 0, 0, None, None
    while step < steps:
        for batch in train.loader(split, idx, 2, True, seed * 100003 + epoch, workers=0):
            if step >= steps:
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
            first = float(loss) if first is None else first
            last = float(loss)
            step += 1
        epoch += 1
    os.makedirs(out, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": {"cpu_smoke": True, "seed": seed}},
               os.path.join(out, "model.pt"))
    print(json.dumps({"steps": step, "focal_samples": len(fi), "stopped_nonfocal_samples": len(mi),
                      "first_loss": first, "last_loss": last}))


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--focal-root", required=True)
    ap.add_argument("--multi-root", required=True)
    ap.add_argument("--pool", default="runs/replication_pool.npy")
    ap.add_argument("--dev-scenarios", default="runs/dev_scenarios.npy")
    ap.add_argument("--cpu-smoke", action="store_true")
    known, rest = ap.parse_known_args()
    excluded = pool_ids(known.pool) | dev_scenario_ids(known.focal_root, known.dev_scenarios)
    focal = Split(known.focal_root, "train")
    multi = Split(known.multi_root, "train")
    multi_dev = Split(known.multi_root, "dev")
    if set(multi_dev.meta.scenario_id.astype(str)) - excluded:
        raise ValueError("multi dev contains non-dev scenarios")
    if known.cpu_smoke:
        smoke = argparse.ArgumentParser()
        smoke.add_argument("--out", required=True)
        smoke.add_argument("--steps", type=int, default=200)
        smoke.add_argument("--n-train", type=int, default=128)
        smoke.add_argument("--seed", type=int, default=99)
        args = smoke.parse_args(rest)
        cpu_smoke(focal, multi, excluded, args.out, args.steps, args.n_train, args.seed)
        return

    def patched_split(root: str, part: str) -> MixSplit:
        if part != "train":
            raise ValueError("MIX only uses TRAIN preprocessing")
        return MixSplit(focal, multi)

    def patched_training(meta, exclude_city, n, dev_frac, seed):
        if exclude_city is not None or dev_frac != 0.02 or n != 128000:
            raise ValueError("MIX requires all cities, 128,000 samples and the fixed dev slice")
        fi, mi = sample_indices(focal.meta, multi.meta, excluded, n, seed)
        return np.sort(np.r_[fi, mi + len(focal)])

    def patched_dev(meta, dev_frac):
        return np.arange(len(focal.meta))[focal.meta.scenario_id.astype(str).isin(excluded).to_numpy()]

    def patched_evaluate(model, split, idx, device, batch_size=256):
        focal_metrics = evaluate_dev(model, focal, dev_indices(focal.meta, 0.02), device)
        multi_metrics = evaluate_dev(model, multi_dev, np.arange(len(multi_dev)), device)
        return {f"focal_{k}": v for k, v in focal_metrics.items()} | \
            {f"multi_{k}": v for k, v in multi_metrics.items()}

    train.Split = patched_split
    train.training_indices = patched_training
    train.dev_indices = patched_dev
    train.evaluate = patched_evaluate
    sys.argv = [sys.argv[0], *rest]
    train.main()


if __name__ == "__main__":
    main()
