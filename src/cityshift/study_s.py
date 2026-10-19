"""Study S: frozen Stage 3a agent scoring on the registered fresh TRAIN reserve."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import torch

from . import multiagent_eval as scoring
from .closedloop_v3 import checked_ids
from .data import Split, dev_indices
from .evaluate import load_model
from .run_manifest import file_sha256, run_settings, validate_manifest
from .study_r import RESERVE_SHA256
from .v2_validation import agent_cohort
from .validate_runs import check

_SPLIT = None
_RAW = ""
_INDEX = {}
FAMILIES = ("ALLR", "SEL", "SELSHAM")
REMOVED_DIGESTS = {"SEL": "2c471f3b64923663fad5160c3f40c423a8798d10f489e0c82e08c2bbeab1925e",
                   "SELSHAM": "19a6893a20dce669ec426604afd360b0af773f840fdc411f6e8d7ab8cfd57017"}


def entries(sid: str) -> list[dict]:
    directory = os.path.join(_RAW, sid)
    scene = scoring.load_scene(directory)
    frame = pq.read_table(os.path.join(directory, f"scenario_{sid}.parquet"),
                          columns=["track_id", "object_category"]).to_pandas().drop_duplicates("track_id")
    return scoring._entries(scene, dict(zip(frame.track_id, frame.object_category)), _SPLIT, _INDEX[sid])


def validate_training(directory: Path, reserve: set[str], excluded: set[str], meta: pd.DataFrame) -> None:
    problems = check(str(directory))
    if problems:
        raise ValueError(f"invalid checkpoint {directory.name}: {problems}")
    cfg = json.loads((directory / "config.json").read_text())
    expected = {"n_train": 128000, "steps": 60000, "batch": 128, "lr": 0.001, "wd": 0.01,
                "dev_frac": 0.02, "no_map": False, "exclude_city": None, "seed": int(directory.name[-1])}
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError("checkpoint does not follow the registered training recipe")
    import numpy as np

    idx = np.load(directory / "train_indices.npy")
    if len(idx) != 128000 or hashlib.sha1(idx.tobytes()).hexdigest() != cfg["train_indices_sha1"]:
        raise ValueError("training draw fingerprint mismatch")
    if set(idx) & set(dev_indices(meta, 0.02)):
        raise ValueError("development example found in checkpoint draw")
    if set(meta.scenario_id.iloc[idx].astype(str)) & (reserve | excluded):
        raise ValueError("reserve or excluded training example found in checkpoint draw")
    pool = np.setdiff1d(np.arange(len(meta)), dev_indices(meta, 0.02))
    pool = pool[~meta.scenario_id.iloc[pool].astype(str).isin(reserve | excluded).to_numpy()]
    expected_draw = np.sort(np.random.default_rng(1000 + cfg["seed"]).choice(pool, 128000, replace=False))
    if not np.array_equal(idx, expected_draw):
        raise ValueError("checkpoint draw does not match its registered seed and exclusions")


def main() -> None:
    global _SPLIT, _RAW, _INDEX
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--pool", default="runs/v2/reserve_ids.npy")
    ap.add_argument("--allr", default="runs/v2/ALLR")
    ap.add_argument("--runs", default="runs/v2/completion")
    ap.add_argument("--selection", default="runs/v2/selection")
    ap.add_argument("--input-fingerprints", help="content inventory recorded before this scoring pass")
    ap.add_argument("--out-dir", default="runs/v2/study_s_open")
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = ap.parse_args()
    if min(args.chunk, args.batch, args.workers) < 1:
        ap.error("chunk, batch and workers must be positive")
    import numpy as np

    ids = sorted(checked_ids(args.pool, RESERVE_SHA256))
    _SPLIT, _RAW = Split(args.root, "train"), args.raw
    _INDEX = {str(s): i for i, s in enumerate(_SPLIT.meta.scenario_id)}
    if not set(ids) <= _INDEX.keys():
        raise ValueError("registered reserve missing from preprocessed TRAIN split")
    reserve = set(ids)
    paths = {}
    for family in FAMILIES:
        excluded = set() if family == "ALLR" else set(np.load(Path(args.selection) / f"{family}_exclude.npy",
                                                             allow_pickle=True).astype(str))
        if family != "ALLR":
            removed = sorted(excluded - reserve)
            if len(removed) != 20515 or hashlib.sha256("\n".join(removed).encode()).hexdigest() != REMOVED_DIGESTS[family]:
                raise ValueError("Study S training exclusions differ from registration")
        root = Path(args.allr) if family == "ALLR" else Path(args.runs) / family
        for seed in range(3):
            directory = root / f"seed{seed}"
            validate_training(directory, reserve, excluded, _SPLIT.meta)
            paths[f"{family}_s{seed}"] = str(directory / "model.pt")
    device = torch.device(args.device)
    settings = run_settings("open_stage3a", str(device), [0, 1, 2], args.pool, ids, args.raw, paths,
                            runner="study_s", chunk=args.chunk, batch=args.batch)
    source = Path(__file__).parent
    settings["source_sha256"].update({f"{n}.py": file_sha256(source / f"{n}.py") for n in
                                       ("multiagent_eval", "metrics", "baselines", "evaluate", "validate_runs")})
    if args.input_fingerprints:
        settings["input_fingerprints_sha256"] = file_sha256(args.input_fingerprints)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    validate_manifest(output / "manifest.json", settings, list(output.glob("part_*.parquet")))
    predictors = list(paths) + ["CV"]
    expected = {f"part_{i:05d}.parquet": ids[i:i + args.chunk] for i in range(0, len(ids), args.chunk)}
    for part in output.glob("part_*.parquet"):
        if part.name not in expected:
            raise ValueError("unexpected Study S part")
        agent_cohort(pd.read_parquet(part), predictors, expected[part.name])
    models = {name: load_model(path, device)[0] for name, path in paths.items()}
    torch.set_num_threads(1)
    for name, cohort in expected.items():
        target = output / name
        if target.exists():
            continue
        with mp.get_context("fork").Pool(args.workers) as pool:
            batches = pool.map(entries, cohort, chunksize=4)
        pending, writer = [], None
        tmp = str(target) + ".tmp"
        try:
            for batch in batches:
                pending.extend(batch)
                if len(pending) >= args.batch:
                    writer, _ = scoring._flush(pending, ["CV"], models, device, writer, tmp)
                    pending.clear()
            if pending:
                writer, _ = scoring._flush(pending, ["CV"], models, device, writer, tmp)
        finally:
            if writer is not None:
                writer.close()
        agent_cohort(pd.read_parquet(tmp), predictors, cohort)
        os.replace(tmp, target)
        print(json.dumps({"part": name, "scenarios": len(cohort)}), flush=True)


if __name__ == "__main__":
    main()
