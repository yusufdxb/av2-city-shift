"""Check every confirmatory run before evaluation: checkpoint present, weights and losses finite.

Registered rule: a run that diverged (non-finite loss) is rerun once with the same seed.
This script only detects; it exits non-zero with the list of bad runs.
"""

from __future__ import annotations

import json
import math
import os
import sys

import torch

CITIES = ("austin", "dearborn", "miami", "palo-alto", "pittsburgh", "washington-dc")


def expected(runs: str) -> list[str]:
    out = [f"{runs}/ALL/seed{s}" for s in range(3)]
    out += [f"{runs}/LOCO-{c}/seed{s}" for c in CITIES for s in range(3)]
    return out + [f"{runs}/NOMAP/seed0"]


def check(run: str) -> list[str]:
    problems = []
    ck = os.path.join(run, "model.pt")
    if not os.path.exists(ck):
        return ["no model.pt"]
    state = torch.load(ck, map_location="cpu", weights_only=False)["model"]
    bad = [k for k, v in state.items() if v.is_floating_point() and not torch.isfinite(v).all()]
    if bad:
        problems.append(f"non-finite weights in {bad[:3]}")
    cfg = json.load(open(os.path.join(run, "config.json")))
    log = os.path.join(run, "train_log.jsonl")
    last = max(json.loads(line)["step"] for line in open(log))
    if last != cfg["steps"]:
        problems.append(f"log ends at step {last}, configured {cfg['steps']}")
    for line in open(log):
        r = json.loads(line)
        vals = [r.get("loss")] + list(r.get("dev", {}).values())
        if any(v is not None and not math.isfinite(v) for v in vals):
            problems.append(f"non-finite loss or dev metric at step {r['step']}")
            break
    return problems


def main() -> None:
    runs = sys.argv[1] if len(sys.argv) > 1 else "runs"
    report = {r: check(r) for r in expected(runs)}
    bad = {r: p for r, p in report.items() if p}
    print(json.dumps({"checked": len(report), "bad": bad}, indent=2))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
