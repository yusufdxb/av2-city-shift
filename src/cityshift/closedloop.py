"""Ego-replay closed-loop harness (Stage 2 pre-registration).

The self-driving car ("AV" track) is handed to a longitudinal planner at t = 49 and
simulated along its own logged path; every other agent replays its log. At each
replan (every 1 s) the planner forecasts nearby agents with a predictor, picks one
of 11 constant-acceleration speed profiles, and executes it for 1 s. The executed
drive is scored against the logged futures of all dynamic agents.
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from .preprocess import OBJECT_TYPES
from .scene import DYNAMIC, Scene, build_input, load_scene

# ---- registered constants (docs/preregistration/2026-09-29-stage2-closed-loop.md) ----
DT = 0.1
HANDOFF = 49
REPLANS = (49, 59, 69, 79, 89, 99)
EXEC_STEPS = 10
END = 109
M_AGENTS = 16
RADIUS_M = 60.0
PATH_AHEAD_M = 80.0
ACCELS = np.array([-8, -6, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2], float)
CHECK_STEPS = 40  # 4 s collision look-ahead
PLAN_STEPS = 60  # 6 s horizon for the progress term
MARGIN_M = 0.5
W_RISK, W_PROGRESS, W_ACCEL = 100.0, 1.0, 0.05
HARD_BRAKE = -4.0
EGO_DIMS = (4.9, 2.0)
TYPE_DIMS = {"vehicle": (4.5, 2.0), "bus": (12.0, 2.6), "motorcyclist": (2.0, 0.8), "cyclist": (2.0, 0.8), "pedestrian": (0.6, 0.6)}
DYN_IDX = {OBJECT_TYPES.index(t) for t in DYNAMIC}
DIMS = np.zeros((len(OBJECT_TYPES), 2))
for _t, _d in TYPE_DIMS.items():
    DIMS[OBJECT_TYPES.index(_t)] = _d


# ------------------------------------------------------------------ geometry
def boxes_overlap(c1, h1, l1, w1, c2, h2, l2, w2) -> np.ndarray:
    """Separating-axis test for oriented boxes; all args broadcast (centres [..., 2])."""
    d = c2 - c1
    u1 = np.stack([np.cos(h1), np.sin(h1)], -1)
    v1 = np.stack([-np.sin(h1), np.cos(h1)], -1)
    u2 = np.stack([np.cos(h2), np.sin(h2)], -1)
    v2 = np.stack([-np.sin(h2), np.cos(h2)], -1)
    sep = np.zeros(np.broadcast_shapes(d.shape[:-1], np.shape(h2), np.shape(h1)), bool)
    for a in (u1, v1, u2, v2):
        dot = lambda x, y: (x * y).sum(-1)  # noqa: E731
        r1 = l1 / 2 * np.abs(dot(u1, a)) + w1 / 2 * np.abs(dot(v1, a))
        r2 = l2 / 2 * np.abs(dot(u2, a)) + w2 / 2 * np.abs(dot(v2, a))
        sep |= np.abs(dot(d, a)) > r1 + r2
    return ~sep


def in_front(c_ego, h_ego, c_agent) -> np.ndarray:
    return ((c_agent - c_ego) * np.stack([np.cos(h_ego), np.sin(h_ego)], -1)).sum(-1) > 0


# ------------------------------------------------------------------ ego path
@dataclass
class Path:
    s: np.ndarray  # cumulative arc length of vertices
    xy: np.ndarray  # vertices
    length_logged: float

    @staticmethod
    def from_scene(sc: Scene) -> "Path":
        p = sc.pos[sc.av, HANDOFF : END + 1]
        keep = [0]
        for i in range(1, len(p)):
            if np.linalg.norm(p[i] - p[keep[-1]]) > 0.05:
                keep.append(i)
        pts = p[keep]
        if len(pts) >= 2:
            direction = pts[-1] - pts[-2]
        else:
            h = sc.head[sc.av, HANDOFF]
            direction = np.array([np.cos(h), np.sin(h)])
        direction = direction / np.linalg.norm(direction)
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        length = float(s[-1])
        pts = np.vstack([pts, pts[-1] + 300.0 * direction])
        s = np.append(s, length + 300.0)
        return Path(s=s, xy=pts, length_logged=length)

    def at(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = np.clip(s, 0.0, self.s[-1])
        x = np.interp(s, self.s, self.xy[:, 0])
        y = np.interp(s, self.s, self.xy[:, 1])
        seg = np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, len(self.s) - 2)
        d = self.xy[seg + 1] - self.xy[seg]
        return np.stack([x, y], -1), np.arctan2(d[..., 1], d[..., 0])


def profiles(v0: float, vmax: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Speed [C, n] and travelled distance [C, n] for every candidate acceleration."""
    t = DT * np.arange(1, n + 1)
    v = np.clip(v0 + ACCELS[:, None] * t[None], 0.0, vmax)
    return v, np.cumsum(v, axis=1) * DT


# ------------------------------------------------------------------ simulation state
@dataclass
class EgoSim:
    path: Path
    vmax: float
    s: np.ndarray = field(default_factory=lambda: np.zeros(END + 1))
    v: np.ndarray = field(default_factory=lambda: np.zeros(END + 1))
    chosen: list = field(default_factory=list)

    def states(self, upto: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World pos, vel, heading for steps HANDOFF..upto (arrays over the full 110)."""
        pos = np.zeros((END + 1, 2))
        head = np.zeros(END + 1)
        vel = np.zeros((END + 1, 2))
        steps = np.arange(HANDOFF, upto + 1)
        p, h = self.path.at(self.s[steps])
        pos[steps], head[steps] = p, h
        vel[steps] = self.v[steps, None] * np.stack([np.cos(h), np.sin(h)], -1)
        return pos, vel, head


def init_ego(sc: Scene) -> EgoSim:
    path = Path.from_scene(sc)
    v0 = float(np.linalg.norm(sc.vel[sc.av, HANDOFF]))
    vlog = np.linalg.norm(sc.vel[sc.av, HANDOFF : END + 1], axis=1).max()
    e = EgoSim(path=path, vmax=max(v0, vlog) + 1.0)
    e.v[HANDOFF] = v0
    return e


def scene_with_ego(sc: Scene, ego: EgoSim, t: int) -> Scene:
    pos, vel, head = ego.states(t)
    p, v, h = sc.pos[sc.av].copy(), sc.vel[sc.av].copy(), sc.head[sc.av].copy()
    p[HANDOFF + 1 : t + 1], v[HANDOFF + 1 : t + 1], h[HANDOFF + 1 : t + 1] = (
        pos[HANDOFF + 1 : t + 1], vel[HANDOFF + 1 : t + 1], head[HANDOFF + 1 : t + 1],
    )  # fmt: skip
    return sc.with_track(sc.av, p, v, h, upto=t)


def select_agents(sc: Scene, ego: "EgoSim", t: int) -> list[int]:
    """Up to M_AGENTS dynamic agents observed at t and within RADIUS_M of the ego, ranked by
    how close their nearest edge comes to the ego's route over the next PATH_AHEAD_M."""
    ego_xy = ego.path.at(np.array([ego.s[t]]))[0][0]
    route = ego.path.at(ego.s[t] + np.arange(0.0, PATH_AHEAD_M + 1.0, 1.0))[0]  # [R, 2]
    cand = [
        i for i in range(len(sc.track_ids))
        if i != sc.av and sc.valid[i, t] and sc.types[i] in DYN_IDX and np.linalg.norm(sc.pos[i, t] - ego_xy) < RADIUS_M
    ]  # fmt: skip
    if not cand:
        return []
    c = np.array(cand)
    d = np.linalg.norm(sc.pos[c, t][:, None] - route[None], axis=-1).min(1) - DIMS[sc.types[c], 0] / 2
    return [cand[j] for j in np.argsort(d, kind="stable")[:M_AGENTS]]


# ------------------------------------------------------------------ analytic predictors
def cv_forecast(sc: Scene, i: int, t: int, n: int = 60) -> np.ndarray:
    return sc.pos[i, t] + sc.vel[i, t] * DT * np.arange(1, n + 1)[:, None]


def static_forecast(sc: Scene, i: int, t: int, n: int = 60) -> np.ndarray:
    return np.repeat(sc.pos[i, t][None], n, 0)


def oracle_forecast(sc: Scene, i: int, t: int, n: int = 60) -> np.ndarray:
    out = np.zeros((n, 2))
    last_p, last_v = sc.pos[i, t], sc.vel[i, t]
    for k in range(n):
        step = t + 1 + k
        if step <= END and sc.valid[i, step]:
            last_p, last_v = sc.pos[i, step], sc.vel[i, step]
            out[k] = last_p
        else:
            last_p = last_p + last_v * DT
            out[k] = last_p
    return out


# ------------------------------------------------------------------ planning
def plan(sc: Scene, ego: EgoSim, t: int, agents: list[int], traj: np.ndarray, prob: np.ndarray) -> int:
    """traj [M, K, 60, 2] world, prob [M, K]. Returns the chosen candidate index."""
    v, dist = profiles(ego.v[t], ego.vmax, PLAN_STEPS)
    progress = dist[:, -1] / max(dist[:, -1].max(), 1e-6)
    risk = np.zeros(len(ACCELS))
    if agents:
        ep, eh = ego.path.at(ego.s[t] + dist[:, :CHECK_STEPS])  # [C, T, 2], [C, T]
        ap = traj[:, :, :CHECK_STEPS]  # [M, K, T, 2]
        prev = np.concatenate([np.repeat(sc.pos[agents, t][:, None, None], ap.shape[1], 1), ap[:, :, :-1]], 2)
        dxy = ap - prev
        ah = np.arctan2(dxy[..., 1], dxy[..., 0])
        still = np.linalg.norm(dxy, axis=-1) < 0.05
        ah = np.where(still, sc.head[agents, t][:, None, None], ah)
        dims = DIMS[sc.types[agents]]  # [M, 2]
        al, aw = dims[:, 0][:, None, None], dims[:, 1][:, None, None]
        # broadcast to [C, M, K, T]
        c1, h1 = ep[:, None, None], eh[:, None, None]
        c2, h2 = ap[None], ah[None]
        hit = boxes_overlap(c1, h1, EGO_DIMS[0] + 2 * MARGIN_M, EGO_DIMS[1] + 2 * MARGIN_M, c2, h2, al[None], aw[None])
        hit &= in_front(c1, h1, c2)
        p_hit = (hit.any(-1) * prob[None]).sum(-1)  # [C, M]
        risk = p_hit.sum(-1)
    cost = W_RISK * risk + W_PROGRESS * (1.0 - progress) + W_ACCEL * np.abs(ACCELS)
    return int(np.argmin(cost))


def execute(ego: EgoSim, t: int, c: int) -> None:
    v, dist = profiles(ego.v[t], ego.vmax, EXEC_STEPS)
    ego.s[t + 1 : t + 1 + EXEC_STEPS] = ego.s[t] + dist[c]
    ego.v[t + 1 : t + 1 + EXEC_STEPS] = v[c]
    ego.chosen.append(float(ACCELS[c]))


# ------------------------------------------------------------------ scoring
def score(sc: Scene, pos: np.ndarray, head: np.ndarray, speed: np.ndarray, replan_decel: list[float]) -> dict:
    steps = np.arange(HANDOFF + 1, END + 1)
    others = [i for i in range(len(sc.track_ids)) if i != sc.av and sc.types[i] in DYN_IDX]
    collision, first, gap = False, -1, np.inf
    if others:
        o = np.array(others)
        c2 = sc.pos[o][:, steps]  # [N, T, 2]
        h2 = sc.head[o][:, steps]
        val = sc.valid[o][:, steps]
        dims = DIMS[sc.types[o]]
        c1, h1 = pos[steps][None], head[steps][None]
        ov = boxes_overlap(c1, h1, EGO_DIMS[0], EGO_DIMS[1], c2, h2, dims[:, 0][:, None], dims[:, 1][:, None])
        front = in_front(c1, h1, c2)
        hit = ov & front & val
        if hit.any():
            collision = True
            first = int(steps[np.where(hit.any(0))[0][0]])
        d = np.linalg.norm(c2 - c1, axis=-1) - (EGO_DIMS[0] / 2 + dims[:, 0][:, None] / 2)
        d = np.where(front & val, d, np.inf)
        gap = float(d.min())
    # Logged speed from positions, 0.5 s moving average, decel over 1 s windows: the raw
    # velocity channel is noisy enough to flag 41% of logged drives as hard braking.
    sp = np.linalg.norm(np.diff(sc.pos[sc.av], axis=0), axis=1) / DT
    vs = np.convolve(sp, np.ones(5) / 5, mode="same")
    log_decel = vs[HANDOFF + 10 : END - 4] - vs[HANDOFF : END - 14]
    logged_hard = bool((log_decel <= HARD_BRAKE + 1e-9).any())
    planner_hard = bool(any(dv <= HARD_BRAKE + 1e-9 for dv in replan_decel))
    unnecessary = planner_hard and not logged_hard
    return {
        "collision": collision,
        "first_collision_step": first,
        "front_gap_m": gap,
        "planner_hard_brake": planner_hard,
        "logged_hard_brake": logged_hard,
        "unnecessary_hard_brake": unnecessary,
        "failure": collision or unnecessary,
    }


def score_sim(sc: Scene, ego: EgoSim) -> dict:
    pos, _, head = ego.states(END)
    # realised speed change over each executed 1 s segment
    decel = [(ego.v[t + EXEC_STEPS] - ego.v[t]) / (EXEC_STEPS * DT) for t in REPLANS]
    r = score(sc, pos, head, ego.v, decel)
    r["progress"] = float(ego.s[END] / ego.path.length_logged) if ego.path.length_logged > 1.0 else np.nan
    r["accels"] = ego.chosen
    return r


def score_log(sc: Scene) -> dict:
    """Checker calibration: the logged ego drive itself."""
    r = score(sc, sc.pos[sc.av], sc.head[sc.av], np.linalg.norm(sc.vel[sc.av], axis=1), [])
    r["progress"] = 1.0
    return r


# ------------------------------------------------------------------ batched driver
_SCENES: list[Scene] = []


def _load(d: str) -> Scene:
    return load_scene(d)


def _inputs(job: tuple[int, EgoSim, int]):
    k, ego, t = job
    sc = scene_with_ego(_SCENES[k], ego, t)
    agents = select_agents(sc, ego, t)
    return k, agents, [build_input(sc, i, t) for i in agents]


def _analytic(job: tuple[int, EgoSim, int, str]):
    k, ego, t, kind = job
    sc = _SCENES[k]  # other agents are logged; the ego is not an input to these forecasts
    agents = select_agents(sc, ego, t)
    fn = {"cv": cv_forecast, "static": static_forecast, "oracle": oracle_forecast}[kind]
    traj = np.stack([fn(sc, i, t) for i in agents])[:, None] if agents else np.zeros((0, 1, 60, 2))
    return k, agents, traj, np.ones((len(agents), 1))


def _plan(job):
    k, ego, t, agents, traj, prob = job
    c = plan(_SCENES[k], ego, t, agents, traj, prob)
    execute(ego, t, c)
    return k, ego


@torch.no_grad()
def predict(models: dict[str, torch.nn.Module], batch: list[tuple[str, dict]], device) -> list[tuple[np.ndarray, np.ndarray]]:
    """batch: list of (model_key, input dict). Returns per item (world traj [K,60,2], prob [K])."""
    out: list = [None] * len(batch)
    keys = sorted({k for k, _ in batch})
    for key in keys:
        idx = [i for i, (k, _) in enumerate(batch) if k == key]
        for s in range(0, len(idx), 512):
            ii = idx[s : s + 512]
            x = {n: torch.from_numpy(np.stack([batch[i][1][n] for i in ii])).to(device) for n in ("agent_hist", "agent_valid", "agent_type", "lane_pts", "lane_attr")}
            traj, logits, _ = models[key](x["agent_hist"], x["agent_valid"], x["agent_type"], x["lane_pts"], x["lane_attr"])
            traj, prob = traj.float().cpu().numpy(), logits.float().softmax(-1).cpu().numpy()
            for j, i in enumerate(ii):
                th = float(batch[i][1]["theta"])
                c, sn = np.cos(th), np.sin(th)
                rot_t = np.array([[c, sn], [-sn, c]])  # local -> world is (x @ rot.T)
                out[i] = (traj[j] @ rot_t + batch[i][1]["origin"], prob[j])
    return out


def run_arm(pool, n: int, arm: dict, models, device) -> list[dict]:
    """arm: {"kind": "model"|"cv"|"static"|"oracle"|"log", "model_for": callable(k) -> key}."""
    if arm["kind"] == "log":
        return [score_log(_SCENES[k]) for k in range(n)]
    egos = [init_ego(_SCENES[k]) for k in range(n)]
    for t in REPLANS:
        if arm["kind"] == "model":
            res = pool.map(_inputs, [(k, egos[k], t) for k in range(n)], chunksize=8)
            flat = [(arm["model_for"](k), inp) for k, agents, inps in res for inp in inps]
            preds = predict(models, flat, device)
            jobs, p = [], 0
            for k, agents, inps in res:
                m = len(agents)
                traj = np.stack([x[0] for x in preds[p : p + m]]) if m else np.zeros((0, 6, 60, 2))
                prob = np.stack([x[1] for x in preds[p : p + m]]) if m else np.zeros((0, 6))
                p += m
                jobs.append((k, egos[k], t, agents, traj, prob))
        else:
            res = pool.map(_analytic, [(k, egos[k], t, arm["kind"]) for k in range(n)], chunksize=8)
            jobs = [(k, egos[k], t, agents, traj, prob) for k, agents, traj, prob in res]
        for k, ego in pool.map(_plan, jobs, chunksize=8):
            egos[k] = ego
    return [score_sim(_SCENES[k], egos[k]) for k in range(n)]


def load_models(spec: dict[str, str], device) -> dict[str, torch.nn.Module]:
    from .evaluate import load_model

    return {k: load_model(p, device)[0] for k, p in spec.items()}


def main() -> None:
    global _SCENES
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw split dir, e.g. .../motion-forecasting/val")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", required=True, help="output parquet")
    ap.add_argument("--scenarios", default=None, help="optional .npy of scenario ids to use (e.g. the dev slice)")
    ap.add_argument("--dev-model", default=None, help="development only: use this one checkpoint for every model arm")
    ap.add_argument("--arms", default="log,oracle,cv,static,ALL,LOCO")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    dirs = sorted(glob.glob(os.path.join(args.raw, "*")))
    if args.scenarios:
        keep = set(np.load(args.scenarios, allow_pickle=True).tolist())
        dirs = [d for d in dirs if os.path.basename(d) in keep]
    if args.limit:
        dirs = dirs[: args.limit]
    device = torch.device("cuda")
    seeds = [int(s) for s in args.seeds.split(",")]
    arms_req = args.arms.split(",")
    cities = ("austin", "dearborn", "miami", "palo-alto", "pittsburgh", "washington-dc")
    spec = {}
    if args.dev_model:
        spec["DEV"] = args.dev_model
    else:
        for s in seeds:
            if "ALL" in arms_req:
                spec[f"ALL/seed{s}"] = f"{args.runs}/ALL/seed{s}/model.pt"
            if "LOCO" in arms_req:
                for c in cities:
                    spec[f"LOCO-{c}/seed{s}"] = f"{args.runs}/LOCO-{c}/seed{s}/model.pt"
    models = load_models(spec, device)

    rows = []
    t0 = time.time()
    ctx = mp.get_context("fork")
    for c0 in range(0, len(dirs), args.chunk):
        chunk = dirs[c0 : c0 + args.chunk]
        with ctx.Pool(args.workers) as pool:
            _SCENES = pool.map(_load, chunk, chunksize=16)
        with ctx.Pool(args.workers) as pool:  # fork again so workers see the loaded scenes
            n = len(_SCENES)
            arms = []
            for a in arms_req:
                if a in ("log", "oracle", "cv", "static"):
                    arms.append((a, {"kind": a}))
                elif args.dev_model:
                    arms.append((f"{a}_dev", {"kind": "model", "model_for": lambda k: "DEV"}))
                else:
                    for s in seeds:
                        if a == "ALL":
                            arms.append((f"ALL_s{s}", {"kind": "model", "model_for": (lambda s: lambda k: f"ALL/seed{s}")(s)}))
                        elif a == "LOCO":
                            arms.append(
                                (f"LOCO_s{s}", {"kind": "model", "model_for": (lambda s: lambda k: f"LOCO-{_SCENES[k].city}/seed{s}")(s)})
                            )
            base = [{"scenario_id": sc.scenario_id, "city": sc.city} for sc in _SCENES]
            for name, arm in arms:
                for b, r in zip(base, run_arm(pool, n, arm, models, device)):
                    b.update({f"{name}_{k}": v for k, v in r.items()})
            rows.extend(base)
        print(json.dumps({"done": c0 + len(chunk), "of": len(dirs), "sec": round(time.time() - t0)}), flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_parquet(args.out)
    print(f"wrote {args.out}: {len(df)} scenarios")


if __name__ == "__main__":
    main()
