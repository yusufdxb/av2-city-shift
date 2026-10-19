"""Development-gated comparison of planning objectives and forecast corrections.

Other actors replay their observations. This experiment neither runs a reactive
simulator nor measures real-world driving safety. Historical harnesses are not
monkeypatched, and future observations enter independent scoring only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from . import closedloop as base
from . import closedloop_v2 as v2
from .closedloop_v4 import feasible
from .departure_calibration import correct_predictions
from .departure_evaluation import implementation_manifest, load_cohort
from .planning_objectives import OBJECTIVES, complete_future, plan, replay_metrics, true_candidate_contacts
from .scene import build_input, load_scene

POLICIES = ("NATIVE", "PATCH", "CALIB", "SELECTIVE")
CONTROLS = ("oracle", "cv", "blind")
COMPLETE_REPLANS = (49, 59, 69)
_SCENES = []


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path, data):
    """Atomic JSON output without nonstandard NaN values."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def raw_inventory(directories):
    """Portable content fingerprints, not local dataset paths."""
    rows = []
    for directory in directories:
        directory = Path(directory)
        files = [directory / f"scenario_{directory.name}.parquet", *sorted(directory.glob("log_map_archive_*.json"))]
        if len(files) != 2 or not all(p.is_file() for p in files):
            raise ValueError("expected one scenario parquet and one scenario map")
        rows.append({"scenario_id": directory.name, "files": {p.name: sha256_file(p) for p in files}})
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"scenes": len(rows), "sha256": digest, "records": rows}


def replay_commands(sc, commands):
    if len(commands) != len(base.REPLANS):
        raise ValueError("six executed commands required")
    ego = v2.init_ego(sc)
    for t, acceleration in zip(base.REPLANS, commands):
        matches = np.flatnonzero(np.isclose(base.ACCELS, acceleration, atol=1e-12, rtol=0))
        if len(matches) != 1:
            raise ValueError("command outside registered candidate set")
        base.execute(ego, t, int(matches[0]))
    return ego


def common_support_metrics(sc, records):
    """Score all arms on identical complete actor support within an objective.

    Selection and planning already finished before this function is called.
    Complete support is the union across policies, controls and predictor seeds.
    Later replans have incomplete four-second horizons and are not labeled safe.
    """
    results = []
    for objective in OBJECTIVES:
        group = [r for r in records if r["objective"] == objective]
        common = {}
        for j, t in enumerate(COMPLETE_REPLANS):
            actors = sorted({a for r in group for a in r["agents"][j]})
            common[t] = (actors, complete_future(sc, actors, t))
        for record in group:
            ego = replay_commands(sc, record["accels"])
            metrics = replay_metrics(sc, ego)
            decisions = []
            for j, t in enumerate(COMPLETE_REPLANS):
                actors, complete = common[t]
                item = {"t": t, "complete": complete, "actors": len(actors)}
                if complete:
                    contacts = true_candidate_contacts(sc, ego, t, actors)
                    candidate = int(np.flatnonzero(np.isclose(base.ACCELS, record["accels"][j]))[0])
                    nonhard_safe = feasible(ego, t) & (base.ACCELS > base.HARD_BRAKE) & ~contacts
                    hard = bool(ego.v[t + base.EXEC_STEPS] - ego.v[t] <= base.HARD_BRAKE + 1e-9)
                    item.update(
                        chosen_contact=bool(contacts[candidate]),
                        nonhard_safe_available=bool(nonhard_safe.any()),
                        realized_hard_brake=hard,
                        excessive_brake=hard and bool(nonhard_safe.any()),
                    )
                decisions.append(item)
            known = [d for d in decisions if d["complete"]]
            metrics.update(
                hard_brake=bool(metrics["planner_hard_brake"]),
                commanded_hard_brake=bool(any(a <= base.HARD_BRAKE for a in record["accels"])),
                genuine_conflict=bool(any(d["chosen_contact"] for d in known)) if known else None,
                excessive_brake=bool(any(d["excessive_brake"] for d in known)) if known else None,
                complete_decisions=len(known),
                eligible_decisions=len(COMPLETE_REPLANS),
                common_actor_counts=[d["actors"] for d in decisions],
                complete_support=[d["complete"] for d in decisions],
                decision_contacts=[d.get("chosen_contact") for d in decisions],
                decision_excessive_brakes=[d.get("excessive_brake") for d in decisions],
                decision_nonhard_safe=[d.get("nonhard_safe_available") for d in decisions],
                stopline_infeasible=sum(d["stopline_infeasible"] for d in record["diagnostics"]),
                risk_budget_fallback=sum(d["risk_budget_fallback"] for d in record["diagnostics"]),
                chosen_forecast_risk=[d["chosen_forecast_risk"] for d in record["diagnostics"]],
                corrected_agents=sum(d.get("n_corrected", 0) for d in record["diagnostics"]),
            )
            results.append(
                {
                    "scenario_id": sc.scenario_id,
                    "city": sc.city,
                    "objective": objective,
                    "policy": record["policy"],
                    "seed": record["seed"],
                }
                | metrics
            )
    return results


def _score(job):
    k, records = job
    return common_support_metrics(_SCENES[k], records)


def _inputs(job):
    k, ego, t = job
    scene = base.scene_with_ego(_SCENES[k], ego, t)
    agents = base.select_agents(scene, ego, t)
    return k, agents, [build_input(scene, i, t) for i in agents]


def _analytic(job):
    k, ego, t, kind = job
    scene = _SCENES[k]
    agents = base.select_agents(scene, ego, t)
    fn = base.oracle_forecast if kind == "oracle" else base.cv_forecast
    traj = np.stack([fn(scene, i, t) for i in agents])[:, None] if agents else np.zeros((0, 1, 60, 2))
    return k, agents, traj, np.ones((len(agents), 1))


def _plan(job):
    k, ego, t, agents, traj, prob, policy, objective, artifact = job
    scene = _SCENES[k]
    metadata = {}
    if policy == "PATCH":
        traj, prob, count = v2.patch_predictions(scene, agents, t, traj.copy(), prob.copy())
        metadata["n_corrected"] = count
    elif policy in ("CALIB", "SELECTIVE") and agents:
        traj, prob, metadata = correct_predictions(
            scene,
            agents,
            t,
            traj,
            prob,
            artifact["model"],
            selective=policy == "SELECTIVE",
            threshold=artifact["tuning"]["threshold"],
        )
    if policy == "blind":
        c, diagnostic = plan(scene, ego, t, [], np.zeros((0, 1, 60, 2)), np.zeros((0, 1)), objective)
    else:
        c, diagnostic = plan(scene, ego, t, agents, traj, prob, objective)
    base.execute(ego, t, c)
    return k, ego, diagnostic | {"n_corrected": metadata.get("n_corrected", 0)}


def run_arm(pool, count, policy, objective, seed, models, device, artifact):
    egos = [v2.init_ego(scene) for scene in _SCENES]
    selected = [[] for _ in egos]
    diagnostics = [[] for _ in egos]
    for t in base.REPLANS:
        if policy in POLICIES:
            inputs = pool.map(_inputs, [(k, egos[k], t) for k in range(count)], chunksize=4)
            batch = [(str(seed), inp) for _, _, inps in inputs for inp in inps]
            predicted = base.predict(models, batch, device)
            forecasts, offset = [], 0
            for k, agents, inps in inputs:
                n = len(inps)
                forecast = predicted[offset : offset + n]
                traj = np.stack([r[0] for r in forecast]) if n else np.zeros((0, 6, 60, 2))
                prob = np.stack([r[1] for r in forecast]) if n else np.zeros((0, 6))
                forecasts.append((k, agents, traj, prob))
                offset += n
        else:
            forecasts = pool.map(_analytic, [(k, egos[k], t, policy) for k in range(count)], chunksize=4)
        jobs = []
        for k, agents, traj, prob in forecasts:
            selected[k].append(agents)
            jobs.append((k, egos[k], t, agents, traj, prob, policy, objective, artifact))
        for k, ego, diagnostic in pool.map(_plan, jobs, chunksize=4):
            egos[k] = ego
            diagnostics[k].append(diagnostic)
    return [
        {
            "policy": policy,
            "objective": objective,
            "seed": seed,
            "accels": ego.chosen,
            "agents": selected[k],
            "diagnostics": diagnostics[k],
        }
        for k, ego in enumerate(egos)
    ]


def run_partition(args):
    global _SCENES
    cohort = load_cohort(Path(args.cohort))
    artifact = json.loads(Path(args.calibrator).read_text())
    ids = cohort["partitions"][args.partition]["ids"]
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    spec = {str(seed): str(Path(args.runs) / f"seed{seed}" / "model.pt") for seed in range(3)}
    checkpoint_hashes = {k: sha256_file(path) for k, path in spec.items()}
    expected_hashes = {str(r["seed"]): r["checkpoint_sha256"] for r in cohort["training_fingerprints"]}
    if checkpoint_hashes != expected_hashes:
        raise ValueError("predictor checkpoints differ from cohort provenance")
    context = artifact.get("context", {})
    expected_context = {
        "cohort_sha256": sha256_file(args.cohort),
        "protocol_sha256": sha256_file(args.protocol),
        "checkpoint_sha256": {f"seed{k}/model.pt": v for k, v in checkpoint_hashes.items()},
        "implementation_sha256": implementation_manifest(),
    }
    if any(context.get(k) != value for k, value in expected_context.items()):
        raise ValueError("frozen calibrator provenance differs from planning inputs")
    if any(artifact.get(f"{role}_cohort_sha256") != cohort["partitions"][role]["sha256"] for role in ("fit", "tune")):
        raise ValueError("calibrator fit/tune membership mismatch")
    if artifact["tuning"]["threshold"] not in (0.0, 0.05, 0.1, 0.2):
        raise ValueError("selective threshold outside frozen grid")
    if args.partition == "test":
        pilot = json.loads((output / "pilot_analysis.json").read_text())
        if pilot.get("full_test_gate") != "proceed" or not pilot.get("feasibility_passed_both_objectives"):
            raise ValueError("development feasibility failed; fresh planning test remains unscored")
        provenance = pilot.get("provenance", {})
        checks = {
            "calibrator_sha256": sha256_file(args.calibrator),
            "cohort_sha256": sha256_file(args.cohort),
            "pilot_manifest_sha256": sha256_file(output / "pilot_manifest.json"),
        }
        if any(provenance.get(k) != value for k, value in checks.items()):
            raise ValueError("pilot feasibility provenance mismatch")
    partition_ids = cohort["partitions"]
    all_ids = [sid for record in partition_ids.values() for sid in record["ids"]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("cohort partitions overlap")
    sources = [
        Path(__file__),
        Path(__file__).with_name("planning_objectives.py"),
        Path(__file__).with_name("departure_calibration.py"),
        Path(args.protocol),
    ]
    manifest = {
        "schema": 1,
        "partition": args.partition,
        "cohort_sha256": sha256_file(args.cohort),
        "membership_sha256": cohort["partitions"][args.partition]["sha256"],
        "calibrator_sha256": sha256_file(args.calibrator),
        "checkpoints": checkpoint_hashes,
        "sources": implementation_manifest() | {p.name: sha256_file(p) for p in sources},
        "objectives": list(OBJECTIVES),
        "policies": list(POLICIES),
        "controls": list(CONTROLS),
        "seeds": [0, 1, 2],
        "scenes": len(ids),
        "chunk": args.chunk,
        "scope": "non-reactive longitudinal replay with privileged logged ego route",
    }
    manifest_path = output / f"{args.partition}_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("existing partition manifest differs; refusing mixed results")
    json_write(manifest_path, manifest)
    chunks_dir = output / f"{args.partition}_chunks"
    chunks_dir.mkdir(exist_ok=True)
    # Limit native CPU thread pools before fork; CUDA inference stays in parent.
    torch.set_num_threads(1)
    device = torch.device(args.device)
    models = base.load_models(spec, device)
    started = time.time()
    context = mp.get_context("fork")
    for start in range(0, len(ids), args.chunk):
        chunk_ids = ids[start : start + args.chunk]
        path = chunks_dir / f"{start:06d}.parquet"
        if path.exists():
            old = pd.read_parquet(path)
            if set(old.scenario_id) != set(chunk_ids) or len(old) != len(chunk_ids) * 30:
                raise ValueError("incomplete saved chunk")
            inventory_path = path.with_suffix(".json")
            if not inventory_path.exists():
                raise ValueError("saved chunk lacks input inventory")
            continue
        dirs = [str(Path(args.raw) / sid) for sid in chunk_ids]
        inventory = raw_inventory(dirs)
        json_write(path.with_suffix(".json"), inventory)
        with context.Pool(args.workers) as pool:
            _SCENES = pool.map(load_scene, dirs, chunksize=4)
        with context.Pool(args.workers) as pool:
            records = [[] for _ in _SCENES]
            for objective in OBJECTIVES:
                for policy in (*CONTROLS, *POLICIES):
                    seeds = range(3) if policy in POLICIES else (-1,)
                    for seed in seeds:
                        result = run_arm(pool, len(_SCENES), policy, objective, seed, models, device, artifact)
                        for saved, record in zip(records, result):
                            saved.append(record)
            scored = pool.map(_score, list(enumerate(records)), chunksize=1)
        frame = pd.DataFrame([row for rows in scored for row in rows])
        temporary = path.with_suffix(".tmp.parquet")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
        print(
            json.dumps(
                {
                    "partition": args.partition,
                    "done": start + len(chunk_ids),
                    "of": len(ids),
                    "seconds": round(time.time() - started),
                }
            ),
            flush=True,
        )
    tables = [pd.read_parquet(path) for path in sorted(chunks_dir.glob("*.parquet"))]
    frame = pd.concat(tables, ignore_index=True)
    if len(frame) != len(ids) * 30 or set(frame.scenario_id) != set(ids):
        raise ValueError("partition inventory mismatch")
    frame.to_parquet(output / f"closedloop_{args.partition}.parquet", index=False)
    analyze_partition(output, args.partition, args.calibrator, args.cohort)
    print(json.dumps({"partition": args.partition, "rows": len(frame), "completed": True}), flush=True)


def analyze_partition(output, partition, calibrator, cohort):
    from .transfer_analysis import analyze_rollouts

    output = Path(output)
    departure_path = output / f"openloop_{partition}_summary.json"
    departure = json.loads(departure_path.read_text())
    intervals = {}
    for policy in ("CALIB", "SELECTIVE"):
        recall = departure["arms"][policy]["departure_recall_loss_vs_native"]
        if recall["coverage_complete"] and recall["ci"] is not None:
            intervals[policy] = {"recall_loss_ci98_75": recall["ci"]}
    rollout_path = output / f"closedloop_{partition}.parquet"
    result = analyze_rollouts(
        pd.read_parquet(rollout_path),
        departure_scenes=departure["n_independent_departure_scenes"],
        departure_summary=intervals,
        phase=partition,
    )
    result["provenance"] = {
        "calibrator_sha256": sha256_file(calibrator),
        "cohort_sha256": sha256_file(cohort),
        f"{partition}_manifest_sha256": sha256_file(output / f"{partition}_manifest.json"),
        "rollouts_sha256": sha256_file(rollout_path),
        "departure_summary_sha256": sha256_file(departure_path),
        "analysis_source_sha256": sha256_file(Path(__file__).with_name("transfer_analysis.py")),
    }
    json_write(output / f"{partition}_analysis.json", result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--cohort", default="runs/transfer/cohort.json")
    ap.add_argument("--runs", default="runs/v2/ALLR")
    ap.add_argument("--calibrator", default="runs/transfer/calibrator.json")
    ap.add_argument("--protocol", default="docs/preregistration/objective-transfer.md")
    ap.add_argument("--out", default="runs/transfer")
    ap.add_argument("--partition", choices=("pilot", "test"), required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.workers < 1 or args.chunk < 1:
        ap.error("positive worker and chunk counts required")
    run_partition(args)


if __name__ == "__main__":
    main()
