"""Measure online serving on the TRAIN development slice or smoke-test synthetic scenes."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cityshift.preprocess import LANE_PTS, OBJECT_TYPES  # noqa: E402
from cityshift.preprocess_multi import dev_scenario_ids  # noqa: E402
from cityshift.scene import Scene, load_scene  # noqa: E402
from cityshift.serving import Backend, StageTimer, decision_agreement, run_scenario  # noqa: E402


def synthetic_scene(number: int) -> Scene:
    """Two straight-moving tracks with no map, for a CPU serving smoke test."""
    steps = np.arange(110)
    pos = np.zeros((2, 110, 2), np.float64)
    pos[0, :, 0] = steps * 0.5
    pos[1, :, 0] = 25 + steps * 0.4 + number * 0.01
    vel = np.zeros_like(pos)
    vel[0, :, 0] = 5.0
    vel[1, :, 0] = 4.0
    return Scene(
        scenario_id=f"synthetic_{number:02d}", city="synthetic", track_ids=["AV", "agent"],
        types=np.array([OBJECT_TYPES.index("vehicle")] * 2), pos=pos, vel=vel,
        head=np.zeros((2, 110)), valid=np.ones((2, 110), bool), focal=1, av=0,
        poly_res=np.zeros((0, LANE_PTS, 2)), poly_attr=np.zeros((0, 2), np.int8),
        poly_orig=np.zeros((0, 2)), poly_start=np.zeros(0, np.int64),
    )


def versions() -> dict[str, str | None]:
    try:
        import tensorrt

        trt = tensorrt.__version__
    except ImportError:
        trt = None
    try:
        runtime = torch.cuda.cudart().cudaRuntimeGetVersion() if torch.cuda.is_available() else None
    except (RuntimeError, AttributeError):
        runtime = None
    return {
        "torch": torch.__version__, "tensorrt": trt,
        "cuda_runtime": str(runtime) if runtime else None,
        "cuda_compiled": torch.version.cuda,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", help="raw TRAIN split directory")
    ap.add_argument("--focal-root", help="preprocessed focal root used to verify dev IDs")
    ap.add_argument("--scenarios", default="runs/dev_scenarios.npy")
    ap.add_argument("--checkpoint", help="predictor checkpoint")
    ap.add_argument("--engine-dir", default="reports/serving/engines")
    ap.add_argument("--out", default="reports/serving/serving_report.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="20 synthetic scenarios, CPU-only PyTorch")
    args = ap.parse_args()
    if args.limit < 0:
        ap.error("--limit must be nonnegative")
    if not args.smoke and not all((args.raw, args.focal_root, args.checkpoint)):
        ap.error("full benchmark requires --raw, --focal-root, and --checkpoint")
    if args.smoke:
        torch.set_num_threads(1)
        scenes = [synthetic_scene(i) for i in range(20)]
        backends = ["pytorch-fp32"]
        checkpoint = None
    else:
        dev = dev_scenario_ids(args.focal_root, args.scenarios)
        dirs = sorted(d for d in glob.glob(os.path.join(args.raw, "*")) if os.path.basename(d) in dev)
        if len(dirs) != len(dev):
            ap.error(f"raw TRAIN directory contains {len(dirs)} of {len(dev)} fixed dev scenarios")
        scenes = (load_scene(d) for d in (dirs[: args.limit] if args.limit else dirs))
        backends = ["pytorch-fp32", "trt-fp32", "trt-fp16"]
        checkpoint = args.checkpoint
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    models = {name: Backend(name, checkpoint, args.engine_dir, force_cpu=args.smoke) for name in backends}
    timers = {name: StageTimer(model.device.type == "cuda") for name, model in models.items()}
    n = 0
    start = time.perf_counter()

    def paired_scenes():
        nonlocal n
        for scene in scenes:
            results = {name: run_scenario(scene, model, timers[name]) for name, model in models.items()}
            n += 1
            if n % 20 == 0:
                print(f"completed {n} scenarios in {time.perf_counter() - start:.1f} s", flush=True)
            if "trt-fp16" in results:
                yield scene, results["pytorch-fp32"], results["trt-fp16"]

    if args.smoke:
        for _ in paired_scenes():
            pass
        agreement = None
    else:
        agreement = decision_agreement(paired_scenes())
    report = {
        "data": "synthetic smoke" if args.smoke else "TRAIN fixed dev slice",
        "scenarios": n, "replans_per_scenario": 6, "batch_policy": "one inference for selected agents per replan, at most 16",
        "versions": versions(), "latency": {name: timer.summary() for name, timer in timers.items()},
        "decision_agreement": agreement,
        "elapsed_s": round(time.perf_counter() - start, 2),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}: {n} scenarios")


if __name__ == "__main__":
    main()
