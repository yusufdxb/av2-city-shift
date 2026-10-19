"""Recover Study Q's omitted agents using its registered single-CV fallback, without rerunning QCNet."""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from . import closedloop as base
from .multiagent_eval import score
from .v2_validation import AGENT_KEYS

_RAW = ""


def fallback_scene(job) -> list[dict]:
    sid, frame = job
    sc = base._load(str(Path(_RAW) / sid))
    by_id = {a: i for i, a in enumerate(sc.track_ids)}
    rows = []
    for row in frame.to_dict("records"):
        i = by_id[row["agent_id"]]
        traj = np.repeat(base.cv_forecast(sc, i, base.HANDOFF)[None], 6, axis=0)
        prob = np.array([1., 0, 0, 0, 0, 0])
        valid = np.ones(60, bool) if row["focal"] else sc.valid[i, base.HANDOFF + 1:]
        metadata = {k: row[k] for k in AGENT_KEYS + ["city", "agent_type", "focal", "scored",
                                                        "planner_relevant", "n_valid", "speed"]}
        moving_mass = float(np.linalg.norm(traj[0, -1] - sc.pos[i, base.HANDOFF]) > 2.0)
        rows.append(metadata | {"predictor": "QCNET", "qcnet_missing": True, "moving_mode_mass": moving_mass}
                    | score(traj, prob, sc.pos[i, base.HANDOFF + 1:], valid))
    return rows


def main() -> None:
    global _RAW
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--native", default="runs/v2/q_open")
    ap.add_argument("--stage3a", default="evals/stage3a/per_agent.parquet")
    ap.add_argument("--out", default="runs/v2/q_open_fallback.parquet")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    _RAW = args.raw
    paths = sorted(glob.glob(str(Path(args.native) / "part_*.parquet")))
    native = pd.concat([pd.read_parquet(p, columns=AGENT_KEYS) for p in paths], ignore_index=True)
    if native.duplicated(AGENT_KEYS).any():
        raise ValueError("duplicate native QCNet rows")
    reference = pd.read_parquet(args.stage3a)
    reference = reference[reference.predictor == "CV"]
    missing = reference.merge(native, on=AGENT_KEYS, how="left", indicator=True, validate="one_to_one")
    missing = missing[missing._merge == "left_only"].drop(columns="_merge")
    with mp.get_context("fork").Pool(args.workers) as pool:
        batches = pool.map(fallback_scene, list(missing.groupby("scenario_id", sort=True)), chunksize=8)
    rows = pd.DataFrame([r for batch in batches for r in batch])
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(str(target) + ".tmp", index=False)
    Path(str(target) + ".tmp").replace(target)
    print(json.dumps({"native_agents": len(native), "reference_agents": len(reference), "fallback_agents": len(rows),
                      "stopped_nonfocal_fallbacks": int((~rows.focal & rows.planner_relevant & (rows.speed < .5)).sum())}))


if __name__ == "__main__":
    main()
