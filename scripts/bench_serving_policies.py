"""Benchmark Stage 3 forecast policies on the fixed TRAIN development scenarios."""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench_serving import synthetic_scene, versions  # noqa: E402
from cityshift.preprocess_multi import dev_scenario_ids  # noqa: E402
from cityshift.scene import load_scene  # noqa: E402
from cityshift.serving import Backend, StageTimer, decision_agreement  # noqa: E402
from cityshift.serving_policies import POLICIES, STAGES, run_scenario  # noqa: E402

BACKENDS = ("pytorch-fp32", "trt-fp16")


def cpu_info() -> dict[str, str | int | None]:
    """Record the CPU model and logical core count without device identifiers."""
    model = platform.processor() or None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                model = line.partition(":")[2].strip()
                break
    return {"model": model, "logical_cores": os.cpu_count()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", help="raw TRAIN split directory")
    ap.add_argument("--focal-root", help="focal preprocessed root for fixed dev ID verification")
    ap.add_argument("--scenarios", default="runs/dev_scenarios.npy")
    ap.add_argument("--checkpoint", help="predictor checkpoint")
    ap.add_argument("--engine-dir", default="reports/serving/engines")
    ap.add_argument("--out", default="reports/serving/serving_policies_report.json")
    ap.add_argument("--limit", type=int, default=0, help="limit the fixed TRAIN dev scenarios")
    ap.add_argument("--smoke", action="store_true", help="synthetic CPU check, at most 20 scenarios")
    args = ap.parse_args()
    if args.limit < 0:
        ap.error("--limit must be nonnegative")
    if not args.smoke and not all((args.raw, args.focal_root, args.checkpoint)):
        ap.error("benchmark requires --raw, --focal-root, and --checkpoint")

    if args.smoke:
        torch.set_num_threads(1)
        count = min(args.limit or 20, 20)
        scenes = [synthetic_scene(i) for i in range(count)]
        for scene in scenes[1::2]:
            scene.pos[1, :, 0] = 35.0
            scene.vel[1] = 0.0
        names = ("pytorch-fp32",)
        checkpoint = None
    else:
        dev = dev_scenario_ids(args.focal_root, args.scenarios)
        dirs = sorted(d for d in glob.glob(os.path.join(args.raw, "*")) if os.path.basename(d) in dev)
        if len(dirs) != len(dev):
            ap.error(f"raw TRAIN directory contains {len(dirs)} of {len(dev)} fixed dev scenarios")
        scenes = (load_scene(d) for d in (dirs[:args.limit] if args.limit else dirs))
        names = BACKENDS
        checkpoint = args.checkpoint
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    models = {name: Backend(name, checkpoint, args.engine_dir, force_cpu=args.smoke) for name in names}
    timers = {(name, policy): StageTimer(model.device.type == "cuda") for name, model in models.items() for policy in POLICIES}
    counts = {(name, policy): {"selected_agents": 0, "inferred_agents": 0}
              for name in names for policy in POLICIES}
    start = time.perf_counter()
    n = 0

    def paired_scenes():
        nonlocal n
        for scene in scenes:
            results = {}
            for name, model in models.items():
                for policy in POLICIES:
                    result = run_scenario(scene, model, timers[name, policy], policy)
                    results[name, policy] = result
                    counts[name, policy]["selected_agents"] += sum(row["selected_agents"] for row in result[1])
                    counts[name, policy]["inferred_agents"] += sum(row["inferred_agents"] for row in result[1])
            n += 1
            if n % 20 == 0:
                print(f"completed {n} scenarios in {time.perf_counter() - start:.1f} s", flush=True)
            if "trt-fp16" in names:
                yield scene, results["pytorch-fp32", "PATCH"], results["trt-fp16", "PATCH"]

    agreement = decision_agreement(paired_scenes()) if "trt-fp16" in names else None
    if args.smoke:
        for _ in paired_scenes():
            pass
    report = {
        "data": "synthetic smoke" if args.smoke else "TRAIN fixed dev slice",
        "scenarios": n,
        "replans_per_scenario": 6,
        "policy_definitions": {
            "PATCH": "Stage 3 CV substitution for selected agents with observed speed < 0.5 m/s",
            "PATCH-skip-inference": "PATCH with stopped selected agents excluded from model inference",
            "TRIM": "For stopped selected agents, retain modes with endpoint displacement <= 2 m; renormalize; CV fallback",
        },
        "batch_policy": "one inference call per replan for selected agents requiring model forecasts, at most 16",
        "cpu": cpu_info(),
        "versions": versions(),
        "latency": {name: {policy: timers[name, policy].summary() for policy in POLICIES} for name in names},
        "agent_counts": {name: {policy: counts[name, policy] for policy in POLICIES} for name in names},
        "trt_fp16_vs_pytorch_fp32_patch_agreement": agreement,
        "elapsed_s": round(time.perf_counter() - start, 2),
    }
    if any(set(timers[key].summary()) != set(STAGES) | {"end_to_end"} for key in timers):
        raise RuntimeError("missing timing stage")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}: {n} scenarios")


if __name__ == "__main__":
    main()
