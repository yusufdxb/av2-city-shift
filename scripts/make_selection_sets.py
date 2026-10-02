"""v2 study S training exclusions: remove the focal examples that teach "stopped means about to leave", or a sham.

SEL excludes every TRAIN scenario whose focal agent is stopped (< 0.5 m/s) at t=49 and ends the next 6 s more than 2 m
from where it stopped (the departing stopped focal agents counted in reports/census/focal_stopped.json). SELSHAM
excludes the same number of TRAIN scenarios drawn uniformly (seed 20261002) from those whose focal agent is moving
(>= 0.5 m/s) at t=49. Both also exclude the v2 reserve, so both can be scored on it. Sorted-ID SHA-256 digests are
printed for the registration.

    PYTHONPATH=src python scripts/make_selection_sets.py --root <preprocessed data> --out-dir runs/v2/selection
"""

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from cityshift.data import dev_indices

SEED = 20261002


def digest(ids: np.ndarray) -> str:
    return hashlib.sha256("\n".join(sorted(ids.tolist())).encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--reserve", default="runs/v2/reserve_ids.npy")
    ap.add_argument("--out-dir", default="runs/v2/selection")
    args = ap.parse_args()
    meta = pd.read_parquet(os.path.join(args.root, "train", "meta.parquet"))
    target = np.load(os.path.join(args.root, "train", "target.npy"), mmap_mode="r")
    ids = meta.scenario_id.astype(str).to_numpy()
    reserve = np.load(args.reserve, allow_pickle=True).astype(str)
    eligible = ~np.isin(ids, reserve)
    eligible[dev_indices(meta, 0.02)] = False  # never drawn for training anyway; kept out of both lists
    stopped = (meta.speed < 0.5).to_numpy()
    end_disp = np.linalg.norm(np.asarray(target[:, -1]), axis=-1)  # focal frame: origin at the t=49 position
    departing = stopped & (end_disp > 2.0) & eligible
    moving = ~stopped & eligible
    sel = ids[departing]
    sham = np.sort(np.random.default_rng(SEED).choice(ids[moving], size=len(sel), replace=False))
    os.makedirs(args.out_dir, exist_ok=True)
    out = {}
    for name, x in (("SEL", sel), ("SELSHAM", sham)):
        full = np.sort(np.concatenate([x, reserve]))  # passed to cityshift.train --exclude-ids
        np.save(os.path.join(args.out_dir, f"{name}_exclude.npy"), full.astype(object), allow_pickle=True)
        out[name] = {"removed": int(len(x)), "removed_sha256": digest(x), "exclude_file_sha256": digest(full)}
    out["eligible_training_scenarios"] = int(eligible.sum())
    out["stopped_staying_left_in_pool"] = int((stopped & (end_disp <= 2.0) & eligible).sum())
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
