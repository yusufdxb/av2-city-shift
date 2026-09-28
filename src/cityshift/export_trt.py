"""Export a checkpoint to ONNX, build TensorRT FP32/FP16 engines, check parity and latency.

Pass criteria are in the pre-registration's "Deployment gate" section.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import tensorrt as trt
import torch
from torch import nn

from .data import Split
from .evaluate import load_model
from .metrics import per_sample_metrics
from .preprocess import HIST, LANE_PTS, MAX_AGENTS, MAX_LANES

INPUTS = ("agent_hist", "agent_valid", "agent_type", "lane_pts", "lane_attr")
OUTPUTS = ("traj", "prob", "emb")


class Deployable(nn.Module):
    """Engine-friendly signature: float and int32 inputs only, softmax applied."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, agent_hist, agent_valid, agent_type, lane_pts, lane_attr):
        traj, logits, emb = self.model(agent_hist, agent_valid > 0.5, agent_type, lane_pts, lane_attr)
        return traj, logits.softmax(-1), emb


def to_engine_inputs(split: Split, idx: np.ndarray) -> dict[str, torch.Tensor]:
    a = split.arrays
    return {
        "agent_hist": torch.from_numpy(np.ascontiguousarray(a["agent_hist"][idx])),
        "agent_valid": torch.from_numpy(np.ascontiguousarray(a["agent_valid"][idx])).float(),
        "agent_type": torch.from_numpy(np.ascontiguousarray(a["agent_type"][idx])).int(),
        "lane_pts": torch.from_numpy(np.ascontiguousarray(a["lane_pts"][idx])),
        "lane_attr": torch.from_numpy(np.ascontiguousarray(a["lane_attr"][idx])).int(),
    }


def export_onnx(model: nn.Module, path: str) -> None:
    b = 2
    dummy = (
        torch.zeros(b, MAX_AGENTS, HIST, 6),
        torch.ones(b, MAX_AGENTS, HIST),
        torch.zeros(b, MAX_AGENTS, dtype=torch.int32),
        torch.zeros(b, MAX_LANES, LANE_PTS, 2),
        torch.zeros(b, MAX_LANES, 2, dtype=torch.int32),
    )
    dummy = tuple(t.cuda() for t in dummy)
    torch.onnx.export(
        Deployable(model).eval(),
        dummy,
        path,
        input_names=list(INPUTS),
        output_names=list(OUTPUTS),
        dynamic_axes={n: {0: "batch"} for n in INPUTS + OUTPUTS},
        opset_version=17,
        dynamo=False,
    )


def build_engine(onnx_path: str, fp16: bool, max_batch: int = 256) -> bytes:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    # TF32 is on by default in TensorRT; turn it off so "FP32" means IEEE FP32.
    config.clear_flag(trt.BuilderFlag.TF32)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        t = network.get_input(i)
        rest = tuple(t.shape)[1:]
        profile.set_shape(t.name, (1,) + rest, (32,) + rest, (max_batch,) + rest)
    config.add_optimization_profile(profile)
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("TensorRT build failed")
    return bytes(blob)


class Engine:
    def __init__(self, blob: bytes):
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(blob)
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

    def __call__(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        b = next(iter(inputs.values())).shape[0]
        outs = {}
        for name in INPUTS:
            self.ctx.set_input_shape(name, tuple(inputs[name].shape))
            self.ctx.set_tensor_address(name, inputs[name].data_ptr())
        for name in OUTPUTS:
            shape = tuple(self.ctx.get_tensor_shape(name))
            assert shape[0] == b
            outs[name] = torch.empty(shape, dtype=torch.float32, device="cuda")
            self.ctx.set_tensor_address(name, outs[name].data_ptr())
        with torch.cuda.stream(self.stream):
            ok = self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        if not ok:
            raise RuntimeError("TensorRT execution failed")
        return outs


def latency_ms(fn, inputs, iters: int = 300, warmup: int = 30) -> dict[str, float]:
    for _ in range(warmup):
        fn(inputs)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(inputs)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return {"p50": float(np.percentile(ts, 50)), "p99": float(np.percentile(ts, 99))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, _ = load_model(args.ckpt, torch.device("cuda"))
    onnx_path = os.path.join(args.out, "predictor.onnx")
    export_onnx(model, onnx_path)
    val = Split(args.root, "val")
    idx = np.random.default_rng(0).choice(len(val), size=args.n, replace=False)
    idx.sort()
    target = torch.from_numpy(np.ascontiguousarray(val.arrays["target"][idx])).cuda()
    ref_mod = Deployable(model).eval()
    report: dict = {"n": args.n}

    def run_batched(fn, bs=128):
        res = {k: [] for k in OUTPUTS}
        for s in range(0, len(idx), bs):
            inp = {k: v.cuda() for k, v in to_engine_inputs(val, idx[s : s + bs]).items()}
            o = fn(inp)
            for k in OUTPUTS:
                res[k].append(o[k].float())
        return {k: torch.cat(v) for k, v in res.items()}

    with torch.no_grad():
        ref = run_batched(lambda i: dict(zip(OUTPUTS, ref_mod(*[i[k] for k in INPUTS]))))
    ref_m = per_sample_metrics(ref["traj"], ref["prob"].log(), target)
    report["pytorch_fp32"] = {"minFDE": ref_m["min_fde"].mean().item(), "MR": ref_m["miss"].mean().item()}

    for prec in ("fp32", "fp16"):
        blob = build_engine(onnx_path, fp16=(prec == "fp16"))
        with open(os.path.join(args.out, f"predictor_{prec}.engine"), "wb") as f:
            f.write(blob)
        eng = Engine(blob)
        out = run_batched(eng)
        m = per_sample_metrics(out["traj"], out["prob"].log(), target)
        r = {
            "max_abs_traj_diff_m": (out["traj"] - ref["traj"]).abs().max().item(),
            "max_abs_prob_diff": (out["prob"] - ref["prob"]).abs().max().item(),
            "minFDE": m["min_fde"].mean().item(),
            "MR": m["miss"].mean().item(),
        }
        r["minFDE_rel_change"] = r["minFDE"] / report["pytorch_fp32"]["minFDE"] - 1
        r["MR_rel_change"] = r["MR"] / report["pytorch_fp32"]["MR"] - 1
        for bs in (1, 32):
            inp = {k: v.cuda() for k, v in to_engine_inputs(val, idx[:bs]).items()}
            r[f"latency_bs{bs}_ms"] = latency_ms(eng, inp)
        report[f"trt_{prec}"] = r
    for bs in (1, 32):
        inp = {k: v.cuda() for k, v in to_engine_inputs(val, idx[:bs]).items()}
        with torch.no_grad():
            report["pytorch_fp32"][f"latency_bs{bs}_ms"] = latency_ms(lambda i: ref_mod(*[i[k] for k in INPUTS]), inp)
    f32, f16 = report["trt_fp32"], report["trt_fp16"]
    report["gate"] = {
        "fp32_parity_lt_1cm": f32["max_abs_traj_diff_m"] < 0.01,
        "fp16_minFDE_lt_1pct": abs(f16["minFDE_rel_change"]) < 0.01,
        "fp16_MR_lt_1pct": abs(f16["MR_rel_change"]) < 0.01,
    }
    report["gate"]["pass"] = all(report["gate"].values())
    report["versions"] = {"tensorrt": trt.__version__, "torch": torch.__version__}
    with open(os.path.join(args.out, "deploy_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
