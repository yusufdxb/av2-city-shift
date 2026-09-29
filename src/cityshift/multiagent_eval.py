"""Score fixed baselines and checkpoints on focal, SCORED, and planner agents at t=49.

Examples::

    python -m cityshift.multiagent_eval --root data/pp --raw data/raw/train --split train \
        --scenarios runs/dev_scenarios.npy --limit 100 --baseline CV LANE STATIC \
        --checkpoint pilot=runs/pilot/seed99/model.pt --out runs/stage3a_dev.parquet
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .baselines import BASELINES
from .closedloop import HANDOFF, init_ego, select_agents
from .data import Split
from .evaluate import load_model
from .metrics import MISS_THRESHOLD_M
from .preprocess import FUT, OBJECT_TYPES
from .scene import Scene, build_input, load_scene

INPUTS = ("agent_hist", "agent_valid", "agent_type", "lane_pts", "lane_attr")


def memberships(sc: Scene, categories: dict[str, int]) -> dict[int, tuple[bool, bool, bool]]:
    """Return focal, fully observed SCORED, and t=49 planner membership."""
    selected = set(select_agents(sc, init_ego(sc), HANDOFF))
    out = {}
    for i, track_id in enumerate(sc.track_ids):
        if i == sc.av or not sc.valid[i, HANDOFF]:
            continue
        focal = i == sc.focal
        scored = categories.get(track_id) == 2 and bool(sc.valid[i, HANDOFF + 1 :].all())
        planner = i in selected
        if (focal or scored or planner) and sc.valid[i, HANDOFF + 1 :].any():
            out[i] = focal, scored, planner
    return out


def score(traj: np.ndarray, prob: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    """Use the final observed endpoint and average only observed future steps."""
    assert traj.shape == (6, FUT, 2) and prob.shape == (6,)
    assert target.shape == (FUT, 2) and valid.shape == (FUT,) and valid.any()
    last = int(np.flatnonzero(valid)[-1])
    err = np.linalg.norm(traj[:, valid] - target[valid][None], axis=-1)
    fde = np.linalg.norm(traj[:, last] - target[last], axis=-1)
    best = int(fde.argmin())
    return {
        "min_ade": float(err[best].mean()),
        "min_fde": float(fde[best]),
        "miss": float(fde[best] > MISS_THRESHOLD_M),
        "brier_min_fde": float(fde[best] + (1 - prob[best]) ** 2),
    }


def _focal_input(split: Split, index: int) -> dict[str, np.ndarray]:
    return {name: np.asarray(split.arrays[name][index]) for name in INPUTS}


def _entries(sc: Scene, categories: dict[str, int], split: Split, index: int) -> list[dict]:
    members = memberships(sc, categories)
    out = []
    for i, (focal, scored, planner) in members.items():
        inp = _focal_input(split, index) if focal else build_input(sc, i, HANDOFF)
        if focal:
            target = np.asarray(split.arrays["target"][index])
            valid = np.ones(FUT, bool)
        else:
            origin, theta = sc.pos[i, HANDOFF], sc.head[i, HANDOFF]
            c, s = np.cos(theta), np.sin(theta)
            rot = np.array([[c, -s], [s, c]])
            target = (sc.pos[i, HANDOFF + 1 :] - origin) @ rot
            valid = sc.valid[i, HANDOFF + 1 :]
        out.append({
            "input": inp, "target": target, "valid": valid,
            "scenario_id": sc.scenario_id, "agent_id": sc.track_ids[i], "city": sc.city,
            "agent_type": OBJECT_TYPES[sc.types[i]], "focal": focal, "scored": scored,
            "planner_relevant": planner, "n_valid": int(valid.sum()),
            "speed": float(np.linalg.norm(sc.vel[i, HANDOFF])),
        })
    return out


def _append(rows: list[dict], entries: list[dict], name: str, preds: list[tuple[np.ndarray, np.ndarray]]) -> None:
    for entry, (traj, prob) in zip(entries, preds):
        meta = {k: v for k, v in entry.items() if k not in ("input", "target", "valid")}
        rows.append(meta | {"predictor": name} | score(traj, prob, entry["target"], entry["valid"]))


@torch.no_grad()
def _flush(entries: list[dict], baselines: list[str], models: dict, device: torch.device, writer: pq.ParquetWriter | None,
           out: str) -> tuple[pq.ParquetWriter, int]:
    rows: list[dict] = []
    for name in baselines:
        _append(rows, entries, name, [BASELINES[name](e["input"]) for e in entries])
    if models:
        batch = {key: torch.from_numpy(np.stack([e["input"][key] for e in entries])).to(device) for key in INPUTS}
        for name, model in models.items():
            match = re.fullmatch(r"LOCO-(.*)_s[0-2]", name)
            selected = [i for i, e in enumerate(entries) if e["city"] == match.group(1)] if match else list(range(len(entries)))
            if not selected:
                continue
            traj, logits, _ = model(*(batch[key][selected] for key in INPUTS))
            t = traj.float().cpu().numpy()
            p = logits.float().softmax(-1).cpu().numpy()
            _append(rows, [entries[i] for i in selected], name, list(zip(t, p)))
    table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(out, table.schema, compression="zstd")
    writer.write_table(table)
    return writer, len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="preprocessed root")
    ap.add_argument("--raw", required=True, help="raw directory containing scenario IDs")
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--scenarios", help=".npy of scenario IDs; required for train")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--baseline", nargs="*", choices=list(BASELINES), default=[])
    ap.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.split == "train" and not args.scenarios:
        ap.error("--scenarios is required for train to restrict evaluation to the dev slice")
    if args.batch < 1 or args.limit < 0:
        ap.error("--batch must be positive and --limit nonnegative")
    names = list(args.baseline)
    checkpoints = {}
    for item in args.checkpoint:
        name, sep, path = item.partition("=")
        if not sep or not name or not path or name in names or name in checkpoints:
            ap.error(f"invalid or duplicate checkpoint: {item}")
        checkpoints[name] = path
    if not names and not checkpoints:
        ap.error("provide --baseline or --checkpoint")
    split = Split(args.root, args.split)
    by_id = {str(sid): i for i, sid in enumerate(split.meta.scenario_id)}
    if args.scenarios:
        ids = [str(x) for x in np.load(args.scenarios, allow_pickle=True)]
        if args.split == "train":
            from .data import dev_indices

            dev_ids = set(split.meta.scenario_id.iloc[dev_indices(split.meta, 0.02)].astype(str))
            if not set(ids) <= dev_ids:
                ap.error("train --scenarios must be a subset of cityshift.data.dev_indices(meta, 0.02)")
    else:
        ids = list(by_id)
    if len(ids) != len(set(ids)) or not set(ids) <= by_id.keys():
        ap.error("scenario IDs must be unique and present in the preprocessed split")
    if args.limit:
        ids = ids[: args.limit]
    if not ids:
        ap.error("no scenarios selected")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {name: load_model(path, device)[0] for name, path in checkpoints.items()}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = None
    pending: list[dict] = []
    n_rows = 0
    try:
        for j, sid in enumerate(ids, 1):
            path = os.path.join(args.raw, sid)
            sc = load_scene(path)
            cats = pq.read_table(os.path.join(path, f"scenario_{sid}.parquet"), columns=["track_id", "object_category"])
            frame = cats.to_pandas().drop_duplicates("track_id")
            categories = dict(zip(frame.track_id, frame.object_category))
            pending.extend(_entries(sc, categories, split, by_id[sid]))
            if len(pending) >= args.batch:
                writer, count = _flush(pending, names, models, device, writer, args.out)
                n_rows += count
                pending.clear()
            if j % 1000 == 0:
                print(f"{j}/{len(ids)} scenarios, {n_rows} rows", flush=True)
        if pending:
            writer, count = _flush(pending, names, models, device, writer, args.out)
            n_rows += count
    finally:
        if writer is not None:
            writer.close()
    print(f"wrote {args.out}: {n_rows} rows from {len(ids)} scenarios")


if __name__ == "__main__":
    main()
