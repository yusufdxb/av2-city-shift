"""Freeze development partitions and a fresh, training-disjoint planning-transfer test.

This reads memberships and training provenance only, never experiment outcomes.
The recorded preprocessing digest fixes the interpretation of saved training indices.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


META_SHA256 = "0789309c3396535d21aa0063c5c8a004cdcfe6e068d027dd62b9bdaeb85db10e"
STAGE4_SHA256 = "a7489d7bd7b95986807ee360e62a01c45a15f4c3297258ba4751f8520caee0c8"
V2_SHA256 = "286a9423ffd907eebf7d67a6c7f66b2edfc6d1c4309e2a7f03e2633f8aac52ed"
ELIGIBLE_SHA256 = "d049998d71766823e670f1c7e3f817303c221e2888820ad0894fa850690e7cf8"
PREVIOUS_SMOKE_ID = "045fdc95-4a64-4ec6-8f63-e446f9a1c313"
PARTITION_SIZES = {"fit": 1500, "tune": 500, "pilot": 500, "test": 2500}
HASH_PREFIX = "transfer-v1"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ids_sha256(ids) -> str:
    """Hash sorted unique scenario IDs, newline-joined without a trailing newline."""
    ids = list(map(str, ids))
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate scenario IDs")
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def stratified_ids(cities: dict[str, str], count: int, role: str) -> list[str]:
    """Largest-remainder city allocation, then deterministic per-role hash ranking.

    Tied fractional remainders use alphabetical city order. Membership is independent
    of input ordering. Each successive development role uses the remaining pool.
    """
    if count < 0 or count > len(cities):
        raise ValueError("partition size exceeds eligible pool or is negative")
    if count == 0:
        return []
    sizes = Counter(cities.values())
    total = len(cities)
    quotas = {city: count * size // total for city, size in sizes.items()}
    order = sorted(sizes, key=lambda city: (-(count * sizes[city] % total), city))
    for city in order[:count - sum(quotas.values())]:
        quotas[city] += 1
    selected = []
    for city in sorted(sizes):
        ranked = sorted(
            (sid for sid, value in cities.items() if value == city),
            key=lambda sid: (hashlib.sha256(f"{HASH_PREFIX}:{role}:{sid}".encode()).hexdigest(), sid),
        )
        selected.extend(ranked[:quotas[city]])
    return sorted(selected)


def require_disjoint(partitions: dict[str, list[str]]) -> None:
    seen = set()
    for role, ids in partitions.items():
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError(f"duplicate or overlapping partition: {role}")
        seen.update(ids)


def partition_record(ids: list[str], cities: dict[str, str]) -> dict:
    return {"ids": sorted(ids), "sha256": ids_sha256(ids),
            "cities": dict(sorted(Counter(cities[sid] for sid in ids).items()))}


def checked_pool(path: Path, digest: str, count: int, all_ids: set[str]) -> set[str]:
    values = np.load(path, allow_pickle=True)
    if values.ndim != 1:
        raise ValueError("scenario pool must be one-dimensional")
    ids = list(values.astype(str))
    if len(ids) != count or ids_sha256(ids) != digest or not set(ids) <= all_ids:
        raise ValueError("registered scenario pool membership mismatch")
    return set(ids)


def build_cohort(root: Path, runs: Path, stage4_pool: Path, v2_pool: Path) -> dict:
    meta_path = root / "train" / "meta.parquet"
    if file_sha256(meta_path) != META_SHA256:
        raise ValueError("TRAIN metadata content differs from recorded V2 preprocessing")
    meta = pd.read_parquet(meta_path)
    ids = meta.scenario_id.astype(str).to_numpy()
    cities = dict(zip(ids, meta.city.astype(str)))
    if len(cities) != len(ids):
        raise ValueError("TRAIN metadata contains duplicate scenario IDs")
    all_ids = set(cities)
    dev_indices = np.sort(np.random.default_rng(12345).choice(len(meta), int(.02 * len(meta)), replace=False))
    dev = set(ids[dev_indices])
    if len(dev) != 3998:
        raise ValueError("fixed development slice size mismatch")
    stage4 = checked_pool(stage4_pool, STAGE4_SHA256, 8140, all_ids)
    reserve = checked_pool(v2_pool, V2_SHA256, 10000, all_ids)
    training_pool = np.array([i for i, sid in enumerate(ids) if sid not in dev and sid not in reserve])
    training = set()
    fingerprints = []
    for seed in range(3):
        directory = runs / f"seed{seed}"
        cfg = json.loads((directory / "config.json").read_text())
        indices = np.load(directory / "train_indices.npy", allow_pickle=False)
        expected_cfg = {"seed": seed, "n_train": 128000, "steps": 60000,
                        "dev_frac": .02, "n_dev": 3998, "exclude_city": None}
        if any(cfg.get(key) != value for key, value in expected_cfg.items()):
            raise ValueError(f"ALLR seed {seed} configuration differs from registered recipe")
        if indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) != 128000:
            raise ValueError(f"ALLR seed {seed} invalid training indices")
        index_sha1 = hashlib.sha1(indices.tobytes()).hexdigest()
        if index_sha1 != cfg.get("train_indices_sha1"):
            raise ValueError(f"ALLR seed {seed} training-index fingerprint mismatch")
        expected = np.sort(np.random.default_rng(1000 + seed).choice(training_pool, 128000, replace=False))
        if not np.array_equal(indices, expected):
            raise ValueError(f"ALLR seed {seed} training draw does not reproduce")
        counts = dict(sorted(Counter(cities[sid] for sid in ids[indices]).items()))
        if counts != cfg.get("train_city_counts"):
            raise ValueError(f"ALLR seed {seed} training city counts mismatch")
        training.update(ids[indices])
        fingerprints.append({"family": f"ALLR/seed{seed}", "seed": seed, "training_draw_seed": 1000 + seed,
                             "n_train": len(indices), "train_indices_sha1": index_sha1,
                             "checkpoint_sha256": file_sha256(directory / "model.pt"), "cities": counts,
                             "registered_draw_reproduced": True})
    eligible = all_ids - training - dev - stage4 - reserve - {PREVIOUS_SMOKE_ID}
    if len(eligible) != 5389 or ids_sha256(eligible) != ELIGIBLE_SHA256:
        raise ValueError("fresh test eligibility differs from audited membership")
    remaining = {sid: cities[sid] for sid in dev}
    partitions = {}
    for role in ("fit", "tune", "pilot"):
        partitions[role] = stratified_ids(remaining, PARTITION_SIZES[role], role)
        for sid in partitions[role]:
            remaining.pop(sid)
    partitions["test"] = stratified_ids({sid: cities[sid] for sid in eligible}, PARTITION_SIZES["test"], "test")
    require_disjoint(partitions)
    if training.intersection(set().union(*(set(value) for value in partitions.values()))):
        raise ValueError("a partition overlaps evaluated-checkpoint gradient training")
    return {
        "schema": 1,
        "procedure": {
            "hash_prefix": HASH_PREFIX, "selection": "city-proportional largest remainder; alphabetical city ties",
            "ranking": "ascending SHA256('transfer-v1:'+role+':'+scenario_id), then scenario_id",
            "development_role_order": ["fit", "tune", "pilot"], "development_seed": 12345,
            "development_fraction": .02, "development_exposure": "previously evaluated development data",
            "id_digest": "SHA256(sorted IDs newline-joined, no trailing newline)",
        },
        "meta_sha256": META_SHA256, "training_fingerprints": fingerprints,
        "exclusions": {
            "gradient_training_union": len(training), "development": len(dev),
            "stage4": {"count": len(stage4), "sha256": STAGE4_SHA256},
            "v2_reserve": {"count": len(reserve), "sha256": V2_SHA256},
            "additional_scored": [{"id": PREVIOUS_SMOKE_ID,
                                   "reason": "previously scored in Stage 3 development-smoke closed-loop table"}],
        },
        "eligibility": {"count": len(eligible), "sha256": ids_sha256(eligible),
                        "cities": dict(sorted(Counter(cities[sid] for sid in eligible).items()))},
        "partitions": {role: partition_record(value, cities) for role, value in partitions.items()},
        "unused": {"development": partition_record(sorted(remaining), cities),
                   "fresh_test": partition_record(sorted(eligible - set(partitions["test"])), cities)},
    }


def write_cohort(path: Path, result: dict) -> bool:
    """Create once, allow identical reruns, and preserve any differing existing record."""
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise ValueError("refusing to replace a different existing cohort")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="preprocessed data root")
    parser.add_argument("--runs", type=Path, default=Path("runs/v2/ALLR"))
    parser.add_argument("--stage4-pool", type=Path, default=Path("runs/replication_pool.npy"))
    parser.add_argument("--v2-pool", type=Path, default=Path("runs/v2/reserve_ids.npy"))
    parser.add_argument("--out", type=Path, default=Path("runs/transfer/cohort.json"))
    args = parser.parse_args()
    try:
        result = build_cohort(args.root, args.runs, args.stage4_pool, args.v2_pool)
        created = write_cohort(args.out, result)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"created": created, "eligible": result["eligibility"]["count"],
                      "partitions": {role: {"count": len(value["ids"]), "sha256": value["sha256"],
                                            "cities": value["cities"]}
                                     for role, value in result["partitions"].items()}}, indent=2))


if __name__ == "__main__":
    main()
