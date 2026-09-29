"""Online serving with Stage 3 scoring and stopped-agent forecast policies."""

from __future__ import annotations

import time

import numpy as np

from . import closedloop as cl
from . import closedloop_v2 as cl_v2
from .scene import Scene
from .serving import Backend, StageTimer, _rotate, _tensor_inputs, build_input_batch

POLICIES = ("none", "PATCH", "PATCH-skip-inference", "TRIM")
STAGES = ("agent_selection", "inference_filter", "input_build_cpu", "host_to_device", "inference",
          "device_to_host", "world_transform", "forecast_policy", "planning")
STOPPED_MPS = 0.5
TRIM_ENDPOINT_M = 2.0


def inference_indices(scene: Scene, agents: list[int], t: int, policy: str) -> list[int]:
    """Return selected-agent positions sent to the model, preserving their order."""
    if policy not in POLICIES:
        raise ValueError(policy)
    if policy != "PATCH-skip-inference":
        return list(range(len(agents)))
    return [j for j, i in enumerate(agents) if np.linalg.norm(scene.vel[i, t]) >= STOPPED_MPS]


def apply_policy(scene: Scene, agents: list[int], t: int, traj: np.ndarray, prob: np.ndarray,
                 policy: str) -> tuple[np.ndarray, np.ndarray]:
    """Apply a forecast policy to world-space trajectories and mode probabilities."""
    if policy not in POLICIES:
        raise ValueError(policy)
    if policy == "none":
        return traj, prob
    if policy in ("PATCH", "PATCH-skip-inference"):
        out_traj, out_prob, _ = cl_v2.patch_predictions(scene, agents, t, traj, prob)
        return out_traj, out_prob
    for j, i in enumerate(agents):
        if np.linalg.norm(scene.vel[i, t]) >= STOPPED_MPS:
            continue
        displacement = np.linalg.norm(traj[j, :, -1] - scene.pos[i, t], axis=-1)
        keep = displacement <= TRIM_ENDPOINT_M
        weights = np.where(keep, prob[j], 0.0)
        total = float(weights.sum())
        if total > 0.0:
            prob[j] = weights / total
        else:
            traj[j] = cl.cv_forecast(scene, i, t)[None]
            prob[j] = 0.0
            prob[j, 0] = 1.0
    return traj, prob


def score_sim(scene: Scene, ego: cl.EgoSim) -> dict:
    """Use Stage 3 contact-based scoring on a completed served rollout."""
    pos, _, head = ego.states(cl.END)
    decel = [(ego.v[t + cl.EXEC_STEPS] - ego.v[t]) / (cl.EXEC_STEPS * cl.DT) for t in cl.REPLANS]
    result = cl_v2.score(scene, pos, head, ego.v, decel)
    result["progress"] = float(ego.s[cl.END] / ego.path.length_logged) if ego.path.length_logged > 1.0 else np.nan
    result["accels"] = ego.chosen
    return result


def serve_replan(scene: Scene, ego: cl.EgoSim, t: int, backend: Backend, timer: StageTimer, policy: str) -> dict:
    """Forecast selected agents, apply policy, plan, and execute one replan."""
    if policy not in POLICIES:
        raise ValueError(policy)
    timer.sync()
    start = time.perf_counter()
    with timer.stage("agent_selection"):
        simulated = cl.scene_with_ego(scene, ego, t)
        agents = cl.select_agents(simulated, ego, t)
    with timer.stage("inference_filter"):
        indices = inference_indices(scene, agents, t, policy)
        inferred_agents = [agents[j] for j in indices]
    with timer.stage("input_build_cpu"):
        batch = build_input_batch(simulated, inferred_agents, t) if inferred_agents else None
        host = _tensor_inputs(batch) if batch is not None else None
    with timer.stage("host_to_device"):
        device_inputs = {k: v.to(backend.device) for k, v in host.items()} if host is not None else None
    with timer.stage("inference"):
        outputs = backend.infer(device_inputs) if device_inputs is not None else None
    with timer.stage("device_to_host"):
        if outputs is not None:
            local = outputs[0].float().cpu().numpy()
            inferred_prob = outputs[1].float().cpu().numpy()
        else:
            local = np.zeros((0, 6, 60, 2), np.float32)
            inferred_prob = np.zeros((0, 6), np.float32)
    with timer.stage("world_transform"):
        traj = np.zeros((len(agents), 6, 60, 2), scene.pos.dtype)
        prob = np.zeros((len(agents), 6), np.float32)
        if inferred_agents:
            theta = batch["theta"]
            c, s = np.cos(theta), np.sin(theta)
            rot_t = np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], -2)
            traj[indices] = _rotate(local, rot_t) + batch["origin"][:, None, None]
            prob[indices] = inferred_prob
    with timer.stage("forecast_policy"):
        traj, prob = apply_policy(scene, agents, t, traj, prob, policy)
    with timer.stage("planning"):
        choice = cl.plan(scene, ego, t, agents, traj, prob)
        cl.execute(ego, t, choice)
    timer.sync()
    timer.samples["end_to_end"].append((time.perf_counter() - start) * 1000)
    return {"t": t, "accel": float(cl.ACCELS[choice]), "agents": agents, "traj": traj,
            "selected_agents": len(agents), "inferred_agents": len(indices)}


def run_scenario(scene: Scene, backend: Backend, timer: StageTimer, policy: str) -> tuple[dict, list[dict]]:
    """Serve all six Stage 3 replans and score the executed drive."""
    ego = cl_v2.init_ego(scene)
    trace = [serve_replan(scene, ego, t, backend, timer, policy) for t in cl.REPLANS]
    return score_sim(scene, ego), trace
