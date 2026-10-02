"""Select the v2 fresh reserve: TRAIN scenarios that no study has scored and that the v2 checkpoints never train on.

Eligible: every TRAIN scenario outside the fixed 2% development slice and outside the Stage 4 replication pool (the
only TRAIN scenarios any study has scored). A fixed-seed uniform draw of 10,000 from the eligible set; the sorted,
newline-joined ID SHA-256 is printed and written into the v2 registration.

    PYTHONPATH=src python scripts/make_reserve.py --root <preprocessed data> --out runs/v2/reserve_ids.npy
"""

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from cityshift.closedloop_v3 import POOL_SHA256, checked_ids
from cityshift.data import dev_indices

N_RESERVE = 10_000
SEED = 20261002


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--pool", default="runs/replication_pool.npy")
    ap.add_argument("--out", default="runs/v2/reserve_ids.npy")
    args = ap.parse_args()
    meta = pd.read_parquet(os.path.join(args.root, "train", "meta.parquet"))
    ids = meta.scenario_id.astype(str).to_numpy()
    dev = set(ids[dev_indices(meta, 0.02)])
    pool = set(checked_ids(args.pool, POOL_SHA256))
    eligible = np.array(sorted(set(ids) - dev - pool))
    reserve = np.sort(np.random.default_rng(SEED).choice(eligible, size=N_RESERVE, replace=False))
    digest = hashlib.sha256("\n".join(reserve.tolist()).encode()).hexdigest()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, reserve.astype(object), allow_pickle=True)
    cities = pd.Series(meta.set_index("scenario_id").loc[reserve, "city"]).value_counts().to_dict()
    print(json.dumps({"train": len(ids), "dev": len(dev), "stage4_pool": len(pool), "eligible": len(eligible),
                      "reserve": len(reserve), "sha256": digest, "cities": cities}, indent=2))


if __name__ == "__main__":
    main()
