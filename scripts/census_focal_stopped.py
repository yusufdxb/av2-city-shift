"""Census of stopped focal agents in the training split (descriptive, CPU only).

How often is a focal agent stopped at the prediction time, and how often does a stopped focal agent stay put over
the next 6 s? This is the training-data fact behind the focal-selection explanation of H6. Reads only the
preprocessed training arrays (meta.parquet speed, focal future positions in the focal frame, whose origin is the
focal position at t=49).

    python scripts/census_focal_stopped.py --root <preprocessed data> --out reports/census/focal_stopped.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

STOPPED_MPS = 0.5  # same threshold as the PATCH trigger and H6a
STAY_M = 2.0  # same threshold as the miss definition and TRIM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="reports/census/focal_stopped.json")
    args = ap.parse_args()
    meta = pd.read_parquet(os.path.join(args.root, "train", "meta.parquet"))
    target = np.load(os.path.join(args.root, "train", "target.npy"), mmap_mode="r")
    stopped = (meta.speed < STOPPED_MPS).to_numpy()
    dist = np.linalg.norm(np.asarray(target[stopped]), axis=-1)  # [n_stopped, 60] metres from the t=49 position
    n, n_stop = len(meta), int(stopped.sum())
    end_within = int((dist[:, -1] <= STAY_M).sum())
    always_within = int((dist.max(axis=1) <= STAY_M).sum())
    out = {
        "_note": "Descriptive census of the full training split (all focal agents, including the 2% dev slice). "
        "Stopped: logged speed < 0.5 m/s at t=49. Positions are the focal agent's logged future, t=50..109.",
        "focal_agents": n,
        "stopped_at_handoff": n_stop,
        "stopped_share": n_stop / n,
        "stopped_ending_within_2m": end_within,
        "stopped_ending_within_2m_share": end_within / n_stop,
        "stopped_staying_within_2m_throughout": always_within,
        "stopped_staying_within_2m_throughout_share": always_within / n_stop,
        "stopped_moving_more_than_2m_by_6s_share": 1 - end_within / n_stop,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
