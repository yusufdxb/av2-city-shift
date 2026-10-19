"""Record content fingerprints for the data used by v2, without including local filesystem paths.

python scripts/fingerprint_v2_inputs.py --raw-train data/raw/train --raw-val data/raw/val --pp-root data/pp
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-train", required=True)
    ap.add_argument("--raw-val", required=True)
    ap.add_argument("--pp-root")
    ap.add_argument("--reserve", default="runs/v2/reserve_ids.npy")
    ap.add_argument("--out", default="runs/v2/input_fingerprints.json")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    reserve = sorted(np.load(args.reserve, allow_pickle=True).astype(str))
    if len(reserve) != 10000 or hashlib.sha256("\n".join(reserve).encode()).hexdigest() != \
            "286a9423ffd907eebf7d67a6c7f66b2edfc6d1c4309e2a7f03e2633f8aac52ed":
        ap.error("registered reserve required")
    validation = sorted(p.name for p in Path(args.raw_val).iterdir() if p.is_dir())
    if len(validation) != 24988:
        ap.error("all 24,988 validation scenes required")
    files = {}
    for split, root, ids in (("train", Path(args.raw_train), reserve), ("val", Path(args.raw_val), validation)):
        for sid in ids:
            directory = root / sid
            maps = sorted(directory.glob("log_map_archive_*.json"))
            if len(maps) != 1:
                raise ValueError(f"expected one map for {split}/{sid}")
            for path in (directory / f"scenario_{sid}.parquet", maps[0]):
                files[f"raw/{split}/{sid}/{path.name}"] = path
    if args.pp_root:
        for split in ("train", "val"):
            for path in sorted((Path(args.pp_root) / split).iterdir()):
                if path.suffix in (".npy", ".parquet"):
                    files[f"preprocessed/{split}/{path.name}"] = path

    def record(item):
        name, path = item
        return name, {"bytes": path.stat().st_size, "sha256": sha(path)}

    with ThreadPoolExecutor(args.workers) as pool:
        records = dict(pool.map(record, sorted(files.items())))
    result = {"recorded_at": datetime.now(timezone.utc).isoformat(),
              "scope": "current completion inputs; this does not reconstruct historical run manifests",
              "train_reserve_scenarios": len(reserve), "validation_scenarios": len(validation), "files": records}
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"fingerprinted_files": len(records), "bytes": sum(r["bytes"] for r in records.values())}))


if __name__ == "__main__":
    main()
