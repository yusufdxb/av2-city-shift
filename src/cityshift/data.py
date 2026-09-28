"""Memory-mapped dataset over preprocessed splits, plus the city-split logic."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .preprocess import ARRAY_KEYS

CITIES = ("austin", "dearborn", "miami", "palo-alto", "pittsburgh", "washington-dc")


class Split:
    """One preprocessed split (``train`` or ``val``) opened read-only."""

    def __init__(self, root: str, split: str):
        d = os.path.join(root, split)
        self.arrays = {k: np.load(os.path.join(d, f"{k}.npy"), mmap_mode="r") for k in ARRAY_KEYS}
        self.meta = pd.read_parquet(os.path.join(d, "meta.parquet"))
        assert len(self.meta) == len(self.arrays["target"])

    def __len__(self) -> int:
        return len(self.meta)

    def batch(self, idx: np.ndarray) -> dict[str, torch.Tensor]:
        idx = np.sort(idx)  # sorted fancy-indexing is much faster on memmaps
        return {k: torch.from_numpy(np.ascontiguousarray(v[idx])) for k, v in self.arrays.items()} | {
            "index": torch.from_numpy(idx)
        }


class _BatchDataset(Dataset):
    def __init__(self, split: Split, batches: list[np.ndarray]):
        self.split, self.batches = split, batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, i: int):
        return self.split.batch(self.batches[i])


class _Identity(Sampler):
    def __init__(self, n: int):
        self.n = n

    def __iter__(self):
        return iter(range(self.n))

    def __len__(self) -> int:
        return self.n


def loader(split: Split, indices: np.ndarray, batch_size: int, shuffle: bool, seed: int = 0, workers: int = 8) -> DataLoader:
    idx = np.asarray(indices)
    if shuffle:
        idx = np.random.default_rng(seed).permutation(idx)
    batches = [idx[i : i + batch_size] for i in range(0, len(idx), batch_size)]
    return DataLoader(
        _BatchDataset(split, batches),
        batch_size=None,
        sampler=_Identity(len(batches)),
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
    )


def training_indices(meta: pd.DataFrame, exclude_city: str | None, n: int, dev_frac: float, seed: int) -> np.ndarray:
    """Draw ``n`` training scenarios from the eligible cities, proportional to city size.

    The dev slice (``dev_frac`` of every city, fixed by a seed independent of the run
    seed) is removed first so it never trains any arm.
    """
    rng = np.random.default_rng(seed)
    dev = dev_indices(meta, dev_frac)
    pool = np.setdiff1d(np.arange(len(meta)), dev)
    if exclude_city is not None:
        pool = pool[meta.city.to_numpy()[pool] != exclude_city]
    if n > len(pool):
        raise ValueError(f"requested {n} training scenarios, only {len(pool)} eligible")
    return np.sort(rng.choice(pool, size=n, replace=False))


def dev_indices(meta: pd.DataFrame, dev_frac: float) -> np.ndarray:
    rng = np.random.default_rng(12345)
    return np.sort(rng.choice(len(meta), size=int(dev_frac * len(meta)), replace=False))


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
