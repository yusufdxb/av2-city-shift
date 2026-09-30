"""Summarize the EXPLORATORY multi-seed planner sweep: reports/sensitivity/summary.json and SHA256SUMS.

    PYTHONPATH=src python scripts/summarize_sensitivity.py
"""

from __future__ import annotations

import argparse
import hashlib
import json

import pandas as pd

from cityshift.sensitivity import summarize


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", default=[f"runs/sensitivity/sweep_seed{s}.parquet" for s in range(3)])
    ap.add_argument("--out", default="reports/sensitivity/summary.json")
    ap.add_argument("--sums", default="reports/sensitivity/SHA256SUMS")
    ap.add_argument("--bootstraps", type=int, default=10_000)
    args = ap.parse_args()
    d = pd.concat([pd.read_parquet(p) for p in args.rows], ignore_index=True)
    assert not d.duplicated(["scenario_id", "model_seed", "cap", "risk_weight", "arm"]).any()
    result = summarize(d, args.bootstraps)
    result["row_tables"] = {p: sha256(p) for p in args.rows}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    with open(args.sums, "w") as f:
        f.writelines(f"{h}  {p}\n" for p, h in result["row_tables"].items())
    print(f"wrote {args.out} and {args.sums}: {len(result['configs'])} configs, seeds {result['model_seeds']}")


if __name__ == "__main__":
    main()
