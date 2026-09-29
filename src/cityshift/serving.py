"""Single-process online serving and paired closed-loop deployment measurements."""

from __future__ import annotations

import hashlib
import os
import time
from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from . import closedloop as cl
from .preprocess import HIST, LANE_PTS, MAX_AGENTS, MAX_LANES, OBJECT_TYPES
from .scene import Scene

INPUTS = ("agent_hist", "agent_valid", "agent_type", "lane_pts", "lane_attr")
STAGES = ("agent_selection", "input_build_cpu", "host_to_device", "inference", "device_to_host", "world_transform", "planning")


def _rotate(values: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return np.matmul(values.reshape(len(values), -1, 2), rotation).reshape(values.shape)


def build_input_batch(scene: Scene, centers: list[int] | np.ndarray, t: int) -> dict[str, np.ndarray]:
    """Build agent-centric inputs for several centers at the same replan step."""
    centers = np.asarray(centers, dtype=np.int64)
    if centers.ndim != 1 or (not scene.valid[centers, t].all()):
        raise ValueError("centers must be observed at t")
    b = len(centers)
    if b == 0:
        return {
            "agent_hist": np.zeros((0, MAX_AGENTS, HIST, 6), np.float32),
            "agent_valid": np.zeros((0, MAX_AGENTS, HIST), bool),
            "agent_type": np.zeros((0, MAX_AGENTS), np.int8),
            "lane_pts": np.zeros((0, MAX_LANES, LANE_PTS, 2), np.float32),
            "lane_attr": np.zeros((0, MAX_LANES, 2), np.int8),
            "origin": np.zeros((0, 2), scene.pos.dtype), "theta": np.zeros(0, scene.head.dtype),
        }
    origin = scene.pos[centers, t].copy()
    theta = scene.head[centers, t].copy()
    c, s = np.cos(theta), np.sin(theta)
    rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
    lo = t - HIST + 1
    w0 = max(lo, 0)
    off = w0 - lo
    valid = scene.valid[:, w0 : t + 1]
    present = valid.any(1)
    last = w0 + valid.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1)
    d = np.linalg.norm(scene.pos[np.arange(len(last)), last][None] - origin[:, None], axis=-1)
    order = np.argsort(d, axis=1, kind="stable")
    selected = np.empty((b, min(MAX_AGENTS, int(present.sum()))), dtype=np.int64)
    for j, center in enumerate(centers):
        others = order[j][present[order[j]] & (order[j] != center)]
        selected[j] = np.r_[center, others[: selected.shape[1] - 1]]
    hist = np.zeros((b, MAX_AGENTS, HIST, 6), np.float32)
    agent_valid = np.zeros((b, MAX_AGENTS, HIST), bool)
    agent_type = np.full((b, MAX_AGENTS), -1, np.int8)
    if selected.size:
        xy = _rotate(scene.pos[selected, w0 : t + 1] - origin[:, None, None], rot)
        vv = _rotate(scene.vel[selected, w0 : t + 1], rot)
        h = scene.head[selected, w0 : t + 1] - theta[:, None, None]
        feat = np.concatenate([xy, vv, np.cos(h)[..., None], np.sin(h)[..., None]], axis=-1)
        observed = scene.valid[selected, w0 : t + 1]
        hist[:, : selected.shape[1], off : off + observed.shape[-1]] = np.where(observed[..., None], feat, 0)
        agent_valid[:, : selected.shape[1], off : off + observed.shape[-1]] = observed
        agent_type[:, : selected.shape[1]] = scene.types[selected]
    lane_pts = np.zeros((b, MAX_LANES, LANE_PTS, 2), np.float32)
    lane_attr = np.full((b, MAX_LANES, 2), -1, np.int8)
    if len(scene.poly_start):
        dist = np.linalg.norm(scene.poly_orig[None] - origin[:, None], axis=-1)
        mind = np.minimum.reduceat(dist, scene.poly_start, axis=1)
        lane_order = np.argsort(mind, axis=1, kind="stable")[:, :MAX_LANES]
        n = lane_order.shape[1]
        lane_pts[:, :n] = _rotate(scene.poly_res[lane_order] - origin[:, None, None], rot)
        lane_attr[:, :n] = scene.poly_attr[lane_order]
    return {
        "agent_hist": hist, "agent_valid": agent_valid, "agent_type": agent_type,
        "lane_pts": lane_pts, "lane_attr": lane_attr, "origin": origin, "theta": theta,
    }


class StageTimer:
    """Wall-clock stage samples with CUDA work completed at each boundary."""

    def __init__(self, cuda: bool):
        self.cuda = cuda
        self.samples: dict[str, list[float]] = defaultdict(list)

    def sync(self) -> None:
        if self.cuda:
            torch.cuda.synchronize()

    @contextmanager
    def stage(self, name: str):
        self.sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.samples[name].append((time.perf_counter() - start) * 1000)

    def summary(self) -> dict[str, dict[str, float]]:
        return {name: {"p50_ms": float(np.percentile(values, 50)), "p99_ms": float(np.percentile(values, 99))}
                for name, values in self.samples.items() if values}


class Backend:
    """One model or cached TensorRT engine with the common Deployable signature."""

    def __init__(self, name: str, checkpoint: str | None, engine_dir: str | None = None, force_cpu: bool = False):
        from .export_trt import Deployable
        from .model import Predictor

        if name not in ("pytorch-fp32", "trt-fp32", "trt-fp16"):
            raise ValueError(name)
        self.name = name
        self.device = torch.device("cpu" if name == "pytorch-fp32" and force_cpu else "cuda")
        if name == "pytorch-fp32" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        if name.startswith("trt") and not torch.cuda.is_available():
            raise RuntimeError("TensorRT serving requires CUDA")
        if checkpoint:
            from .evaluate import load_model

            model, _ = load_model(checkpoint, self.device)
        else:
            if name != "pytorch-fp32":
                raise ValueError("TensorRT requires a checkpoint")
            model = Predictor().to(self.device).eval()
        if name == "pytorch-fp32":
            self.runner = Deployable(model).to(self.device).eval()  # the dummy-lane buffer must follow the model
        else:
            from .export_trt import Engine, build_engine, export_onnx

            if not engine_dir:
                raise ValueError("engine_dir is required for TensorRT")
            directory = Path(engine_dir)
            directory.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()[:16]
            import tensorrt as trt

            base = f"{digest}_trt{trt.__version__}"
            path = directory / f"{base}_{name}.engine"
            if not path.exists():
                onnx = directory / f"{base}.onnx"
                if not onnx.exists():
                    export_onnx(model, str(onnx))
                blob = build_engine(str(onnx), fp16=name == "trt-fp16", max_batch=32)
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(blob)
                os.replace(tmp, path)
            self.runner = Engine(path.read_bytes())

    def infer(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if self.name == "pytorch-fp32":
                traj, prob, _ = self.runner(*[inputs[key] for key in INPUTS])
            else:
                out = self.runner(inputs)
                traj, prob = out["traj"], out["prob"]
        return traj, prob


def _tensor_inputs(batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        "agent_hist": torch.from_numpy(batch["agent_hist"]),
        "agent_valid": torch.from_numpy(batch["agent_valid"].astype(np.float32)),
        "agent_type": torch.from_numpy(batch["agent_type"].astype(np.int32)),
        "lane_pts": torch.from_numpy(batch["lane_pts"]),
        "lane_attr": torch.from_numpy(batch["lane_attr"].astype(np.int32)),
    }


def serve_replan(scene: Scene, ego: cl.EgoSim, t: int, backend: Backend, timer: StageTimer) -> dict:
    """Run one online replan with one predictor call for all selected agents."""
    timer.sync()
    start = time.perf_counter()
    with timer.stage("agent_selection"):
        simulated = cl.scene_with_ego(scene, ego, t)
        agents = cl.select_agents(simulated, ego, t)
    with timer.stage("input_build_cpu"):
        batch = build_input_batch(simulated, agents, t) if agents else None
        host = _tensor_inputs(batch) if batch is not None else None
    with timer.stage("host_to_device"):
        device_inputs = {k: v.to(backend.device) for k, v in host.items()} if host is not None else None
    with timer.stage("inference"):
        outputs = backend.infer(device_inputs) if device_inputs is not None else None
    with timer.stage("device_to_host"):
        if outputs is not None:
            local = outputs[0].float().cpu().numpy()
            prob = outputs[1].float().cpu().numpy()
        else:
            local = np.zeros((0, 6, 60, 2), np.float32)
            prob = np.zeros((0, 6), np.float32)
    with timer.stage("world_transform"):
        if agents:
            theta = batch["theta"]
            c, s = np.cos(theta), np.sin(theta)
            rot_t = np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], -2)
            traj = _rotate(local, rot_t) + batch["origin"][:, None, None]
        else:
            traj = local
    with timer.stage("planning"):
        choice = cl.plan(scene, ego, t, agents, traj, prob)
        cl.execute(ego, t, choice)
    timer.sync()
    timer.samples["end_to_end"].append((time.perf_counter() - start) * 1000)
    return {"t": t, "accel": float(cl.ACCELS[choice]), "agents": agents, "traj": traj}


def run_scenario(scene: Scene, backend: Backend, timer: StageTimer) -> tuple[dict, list[dict]]:
    ego = cl.init_ego(scene)
    trace = [serve_replan(scene, ego, t, backend, timer) for t in cl.REPLANS]
    return cl.score_sim(scene, ego), trace


def decision_agreement(pairs: Iterable[tuple[Scene, tuple[dict, list[dict]], tuple[dict, list[dict]]]]) -> dict:
    """Count paired control and outcome changes, and rank common-agent deviations."""
    replans = changed = collision = braking = either = n = 0
    deviations = []
    for scene, (ref_score, ref_trace), (test_score, test_trace) in pairs:  # may be a generator: count as we go
        n += 1
        collision_changed = bool(ref_score["collision"] != test_score["collision"])
        braking_changed = bool(ref_score["unnecessary_hard_brake"] != test_score["unnecessary_hard_brake"])
        collision += collision_changed
        braking += braking_changed
        either += collision_changed or braking_changed
        for ref, test in zip(ref_trace, test_trace):
            replans += 1
            changed += ref["accel"] != test["accel"]
            test_index = {agent: j for j, agent in enumerate(test["agents"])}
            for j, agent in enumerate(ref["agents"]):
                if agent in test_index:
                    delta = np.linalg.norm(ref["traj"][j] - test["traj"][test_index[agent]], axis=-1)
                    deviations.append({
                        "scenario_id": scene.scenario_id, "t": ref["t"], "agent_id": scene.track_ids[agent],
                        "agent_type": OBJECT_TYPES[scene.types[agent]],
                        "speed_mps": float(np.linalg.norm(scene.vel[agent, ref["t"]])),
                        "max_trajectory_deviation_m": float(delta.max()),
                    })
    return {
        "scenarios": n, "replans": replans, "acceleration_changed": changed,
        "acceleration_changed_share": changed / replans if replans else None,
        "collision_changed": collision, "collision_changed_share": collision / n if n else None,
        "unnecessary_hard_brake_changed": braking, "unnecessary_hard_brake_changed_share": braking / n if n else None,
        "either_outcome_changed": either, "either_outcome_changed_share": either / n if n else None,
        "largest_trajectory_deviations": sorted(deviations, key=lambda row: row["max_trajectory_deviation_m"], reverse=True)[:5],
    }
