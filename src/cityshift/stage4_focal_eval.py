"""Pool-only focal open-loop evaluation using multiagent_eval scoring internals."""

from __future__ import annotations

import argparse
import os

import pandas as pd
import torch

from .closedloop_v3 import POOL_SHA256, checked_ids
from .data import Split
from .evaluate import load_model
from .multiagent_eval import _entries, _flush
from .scene import load_scene


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--mix-runs", default="runs/stage4/MIX")
    args = ap.parse_args()
    ids = checked_ids(args.scenarios, POOL_SHA256)
    if len(ids) != 8140 or os.path.basename(args.scenarios) != "replication_pool.npy":
        ap.error("expected the full registered replication pool")
    split = Split(args.root, "train")
    by_id = {str(sid): i for i, sid in enumerate(split.meta.scenario_id)}
    if not set(ids) <= by_id.keys():
        ap.error("pool scenarios missing from focal preprocessed TRAIN split")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for seed in range(3):
        models[f"ALL_s{seed}"] = load_model(f"{args.runs}/ALL/seed{seed}/model.pt", device)[0]
        models[f"MIX_s{seed}"] = load_model(f"{args.mix_runs}/seed{seed}/model.pt", device)[0]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = None
    pending: list[dict] = []
    count = 0
    try:
        for number, sid in enumerate(ids, 1):
            sc = load_scene(os.path.join(args.raw, sid))
            pending.extend(e for e in _entries(sc, {}, split, by_id[sid]) if e["focal"])
            if len(pending) >= 128:
                writer, written = _flush(pending, [], models, device, writer, args.out)
                count += written
                pending.clear()
            if number % 1000 == 0:
                print(f"{number}/{len(ids)} scenarios, {count} focal rows", flush=True)
        if pending:
            writer, written = _flush(pending, [], models, device, writer, args.out)
            count += written
    finally:
        if writer is not None:
            writer.close()
    if count != len(ids) * 6:
        raise ValueError(f"expected {len(ids) * 6} seed-level focal rows, got {count}")
    print(pd.DataFrame({"scenarios": [len(ids)], "rows": [count]}).to_json(orient="records"))


if __name__ == "__main__":
    main()
