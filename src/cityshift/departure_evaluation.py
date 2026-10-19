"""Frozen-fit, held-out open-loop departure evaluation for planning transfer.

Actor selection uses the logged ego and present observations. Complete future
observations are consulted only to include and score events. The predictor and
correction receive history only. No raw paths are written into result metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from . import closedloop as base
from . import closedloop_v2 as v2
from .departure_calibration import (
    DISTANCE_M,
    DT,
    HISTORY_FEATURE_NAMES,
    HORIZON,
    STOPPED_MPS,
    correct_predictions,
    fit_calibrator,
    history_features,
    native_departure_mass,
    predict_departure,
)
from .planning_objectives import departure_label
from .preprocess import OBJECT_TYPES
from .scene import build_input, load_scene

SEEDS = (0, 1, 2)
TIMES = (49, 69)
THRESHOLDS = (0.05, 0.10, 0.20)
RECALL_THRESHOLD = 0.10
RECALL_LOSS_LIMIT = 0.02
MIN_RETAINED_FRACTION = 0.02
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261019
CITIES = ("austin", "dearborn", "miami", "palo-alto", "pittsburgh", "washington-dc")
ARMS = ("NATIVE", "PATCH", "CALIB", "SELECTIVE")
PROTOCOL = {
    "schema": 1,
    "seeds": list(SEEDS),
    "times": list(TIMES),
    "objects": ["vehicle", "bus"],
    "stopped_speed_lt_mps": STOPPED_MPS,
    "horizon_steps": HORIZON,
    "dt_s": DT,
    "max_displacement_gt_m": DISTANCE_M,
    "ridge": 1.0,
    "threshold_grid": list(THRESHOLDS),
    "recall_probability_ge": RECALL_THRESHOLD,
    "tune_recall_loss_le": RECALL_LOSS_LIMIT,
    "min_retained_fraction": MIN_RETAINED_FRACTION,
    "log_loss_clip": 1e-9,
    "selection": "base.select_agents with logged ego arc distance and speed at t",
    "inclusion": "complete observations t through t+40; unknown futures excluded only from scoring",
    "correction_scope": "planning corrections affect stopped dynamic actors; this evaluation concerns vehicles and buses",
    "features": list(HISTORY_FEATURE_NAMES),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def ids_digest(ids) -> str:
    ids = list(ids)
    if len(ids) != len(set(ids)) or any(not isinstance(s, str) or not s for s in ids):
        raise ValueError("scenario IDs must be unique nonempty strings")
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def load_cohort(path: Path) -> dict:
    cohort = json.loads(Path(path).read_text())
    if cohort.get("schema") != 1 or set(cohort.get("partitions", {})) != {"fit", "tune", "pilot", "test"}:
        raise ValueError("unsupported cohort schema or missing partition")
    seen = set()
    for name, part in cohort["partitions"].items():
        ids = part["ids"]
        if not ids or ids_digest(ids) != part.get("sha256") or seen.intersection(ids):
            raise ValueError(f"invalid or overlapping cohort partition: {name}")
        seen.update(ids)
    return cohort


def checkpoint_manifest(runs: Path, cohort: dict) -> dict:
    expected = {int(row["seed"]): row["checkpoint_sha256"] for row in cohort["training_fingerprints"]}
    result = {}
    if set(expected) != set(SEEDS):
        raise ValueError("cohort does not describe all three frozen predictor seeds")
    for seed in SEEDS:
        name = f"seed{seed}/model.pt"
        digest = file_sha256(Path(runs) / name)
        if digest != expected[seed]:
            raise ValueError(f"checkpoint differs from cohort fingerprint: seed {seed}")
        result[name] = digest
    return result


def implementation_manifest() -> dict:
    names = (
        "departure_evaluation.py",
        "departure_calibration.py",
        "planning_objectives.py",
        "scene.py",
        "closedloop.py",
        "closedloop_v2.py",
        "preprocess.py",
        "evaluate.py",
        "model.py",
    )
    return {name: file_sha256(Path(__file__).with_name(name)) for name in names}


def _write_json_once(path: Path, value) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"refusing to overwrite mismatched artifact: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(payload)


def _row_inventory(frame: pd.DataFrame) -> dict:
    if frame.empty:
        raise ValueError("no complete stopped-vehicle events were collected")
    keys = ["scenario_id", "track_id", "t", "seed"]
    if frame.duplicated(keys).any():
        raise ValueError("duplicate scenario/track/time/seed evaluation rows")
    groups = frame.groupby(keys[:-1], sort=False)
    if set(frame.seed.unique()) != set(SEEDS) or not groups.seed.nunique().eq(len(SEEDS)).all():
        raise ValueError("every scored event must contain all three predictor seeds")
    if not groups.label.nunique().eq(1).all():
        raise ValueError("predictor seeds disagree on a ground-truth event label")
    ordered = frame.sort_values(keys)[keys].astype(str).values.tolist()
    return {
        "rows": len(frame),
        "events": len(frame.drop_duplicates(keys[:-1])),
        "scenes": int(frame.scenario_id.nunique()),
        "key_sha256": _json_digest(ordered),
    }


def _read_cached(path: Path, context: dict) -> tuple[pd.DataFrame, dict] | None:
    metadata_path = path.with_suffix(".metadata.json")
    if not path.exists() and not metadata_path.exists():
        return None
    if not path.exists() or not metadata_path.exists():
        raise ValueError(f"incomplete existing artifact, refusing overwrite: {path.name}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("context") != context or file_sha256(path) != metadata.get("parquet_sha256"):
        raise ValueError(f"existing table context or checksum differs: {path.name}")
    frame = pd.read_parquet(path)
    if _row_inventory(frame) != metadata.get("row_inventory"):
        raise ValueError("cached row inventory mismatch")
    return frame, metadata


def _write_table(path: Path, frame: pd.DataFrame, context: dict, raw_inventory: list[dict], counts: dict | None = None) -> dict:
    if path.exists() or path.with_suffix(".metadata.json").exists():
        raise ValueError(f"refusing to overwrite existing table: {path.name}")
    inventory = _row_inventory(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.partial")
    if temporary.exists():
        raise ValueError(f"partial output already exists: {temporary.name}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)
    metadata = {
        "context": context,
        "row_inventory": inventory,
        "parquet_sha256": file_sha256(path),
        "raw_scenario_count": len(raw_inventory),
        "raw_inventory_sha256": _json_digest(sorted(raw_inventory, key=lambda r: r["scenario_id"])),
    }
    if counts is not None:
        metadata["complete_future_coverage"] = counts
    _write_json_once(path.with_suffix(".metadata.json"), metadata)
    return metadata


def _logged_ego(sc, t: int):
    # Keep every nonduplicate logged vertex. The historical Path builder drops
    # sub-5 cm movements, making cumulative raw distance inconsistent with it.
    logged_path = sc.pos[sc.av, base.HANDOFF : base.END + 1]
    keep = np.r_[True, np.any(np.diff(logged_path, axis=0) != 0, axis=1)]
    points = logged_path[keep]
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=-1))]
    direction = (
        points[-1] - points[-2]
        if len(points) > 1
        else np.array([np.cos(sc.head[sc.av, base.HANDOFF]), np.sin(sc.head[sc.av, base.HANDOFF])])
    )
    direction /= np.linalg.norm(direction)
    path = base.Path(
        s=np.r_[arc, arc[-1] + 300.0], xy=np.vstack([points, points[-1] + direction * 300.0]), length_logged=float(arc[-1])
    )
    ego = base.EgoSim(path=path, vmax=v2.speed_cap(float(np.linalg.norm(sc.vel[sc.av, base.HANDOFF]))))
    steps = slice(base.HANDOFF, t + 1)
    logged = sc.pos[sc.av, steps]
    ego.s[steps] = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(logged, axis=0), axis=-1))]
    ego.v[steps] = np.linalg.norm(sc.vel[sc.av, steps], axis=-1)
    return ego


def _prepare_scene(directory: str):
    """CPU worker: selection precedes offline future-completeness scoring."""
    directory = Path(directory)
    maps = sorted(directory.glob("log_map_archive_*.json"))
    if len(maps) != 1:
        raise ValueError("expected exactly one scenario map archive")
    raw_files = [directory / f"scenario_{directory.name}.parquet", maps[0]]
    raw_record = {"scenario_id": directory.name, "files": {p.name: file_sha256(p) for p in raw_files}}
    sc = load_scene(str(directory))
    bundles = []
    vehicle_types = {OBJECT_TYPES.index("vehicle"), OBJECT_TYPES.index("bus")}
    for t in TIMES:
        ego = _logged_ego(sc, t)
        selected = [
            i
            for i in base.select_agents(sc, ego, t)
            if sc.types[i] in vehicle_types and np.linalg.norm(sc.vel[i, t]) < STOPPED_MPS
        ]
        # The complete-future filter defines scored support, never predictor features.
        labels = [(i, departure_label(sc, i, t)) for i in selected]
        eligible = [(i, label) for i, label in labels if label is not None]
        agents = [i for i, _ in eligible]
        if agents:
            bundles.append(
                {
                    "t": t,
                    "agents": agents,
                    "labels": [bool(y) for _, y in eligible],
                    "features": history_features(sc, agents, t),
                    "inputs": [build_input(sc, i, t) for i in agents],
                    "n_selected": len(selected),
                    "n_incomplete": len(selected) - len(agents),
                }
            )
        elif selected:
            bundles.append(
                {
                    "t": t,
                    "agents": [],
                    "labels": [],
                    "features": np.empty((0, 10)),
                    "inputs": [],
                    "n_selected": len(selected),
                    "n_incomplete": len(selected),
                }
            )
    return sc, bundles, raw_record


def _top1_fde(traj, prob, truth) -> np.ndarray:
    indices = np.argmax(prob, axis=1)
    return np.linalg.norm(traj[np.arange(len(traj)), indices, HORIZON - 1] - truth, axis=-1)


def _threshold_key(threshold: float) -> str:
    return f"selective_{round(threshold * 100):02d}"


def _score_forecasts(sc, bundle, seed: int, traj, prob, model: dict | None, thresholds: tuple[float, ...]) -> list[dict]:
    agents, t = bundle["agents"], bundle["t"]
    if not agents:
        return []
    native = native_departure_mass(traj, prob, sc.pos[agents, t])
    truth = sc.pos[agents, t + HORIZON]
    rows = []
    native_fde = _top1_fde(traj, prob, truth)
    cv = np.stack([base.cv_forecast(sc, i, t, traj.shape[2]) for i in agents])
    patch_mass = (np.linalg.norm(cv[:, :HORIZON] - sc.pos[agents, t, None], axis=-1) > DISTANCE_M).any(-1).astype(float)
    patch_fde = np.linalg.norm(cv[:, HORIZON - 1] - truth, axis=-1)
    evaluated = {}
    if model is not None:
        ct, cp, cm = correct_predictions(sc, agents, t, traj, prob, model, min_retained_fraction=MIN_RETAINED_FRACTION)
        evaluated["calib"] = (cm, _top1_fde(ct, cp, truth))
        for threshold in thresholds:
            ct, cp, cm = correct_predictions(
                sc, agents, t, traj, prob, model, selective=True, threshold=threshold, min_retained_fraction=MIN_RETAINED_FRACTION
            )
            evaluated[_threshold_key(threshold)] = (cm, _top1_fde(ct, cp, truth))
    for j, i in enumerate(agents):
        row = {
            "scenario_id": sc.scenario_id,
            "city": sc.city,
            "track_id": sc.track_ids[i],
            "t": t,
            "seed": seed,
            "label": bool(bundle["labels"][j]),
            "native_mass": float(native[j]),
            "native_fde4s": float(native_fde[j]),
            "patch_mass": float(patch_mass[j]),
            "patch_fde4s": float(patch_fde[j]),
        }
        row.update({name: float(bundle["features"][j, k]) for k, name in enumerate(HISTORY_FEATURE_NAMES)})
        for prefix, (meta, fde) in evaluated.items():
            row[f"{prefix}_mass"] = meta["realized_departure_mass"][j]
            row[f"{prefix}_fde4s"] = float(fde[j])
            row[f"{prefix}_cv_admitted"] = meta["cv_admitted"][j]
            row[f"{prefix}_corrected"] = meta["corrected"][j]
            row[f"{prefix}_unrepresentable"] = meta["unrepresentable_departure_target"][j]
        rows.append(row)
    return rows


def _collect(
    raw: Path,
    ids: list[str],
    runs: Path,
    workers: int,
    chunk: int,
    model=None,
    thresholds: tuple[float, ...] = (),
    device: str = "cuda",
) -> tuple[pd.DataFrame, list[dict], dict]:
    if workers < 1 or chunk < 1:
        raise ValueError("workers and chunk must be positive")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; --device cpu is supported for explicit CPU runs")
    directories = [str(Path(raw) / sid) for sid in ids]
    if any(not Path(directory).is_dir() for directory in directories):
        raise ValueError("raw directory is missing a cohort scenario")
    torch.set_num_threads(1)
    models = base.load_models({f"seed{s}": str(Path(runs) / f"seed{s}" / "model.pt") for s in SEEDS}, torch.device(device))
    rows, inventory = [], []
    counts = {"selected_actor_times": 0, "incomplete_actor_times": 0, "complete_actor_times": 0}
    start = time.monotonic()
    for offset in range(0, len(directories), chunk):
        # Workers perform no CUDA work. A fresh fork avoids retaining chunk scenes.
        with mp.get_context("fork").Pool(workers) as pool:
            prepared = pool.map(_prepare_scene, directories[offset : offset + chunk], chunksize=1)
        batches = [
            (f"seed{seed}", inp) for seed in SEEDS for _, bundles, _ in prepared for bundle in bundles for inp in bundle["inputs"]
        ]
        predictions = base.predict(models, batches, torch.device(device))
        position = 0
        for seed in SEEDS:
            for sc, bundles, _ in prepared:
                for bundle in bundles:
                    n = len(bundle["agents"])
                    if n:
                        pred = predictions[position : position + n]
                        rows.extend(
                            _score_forecasts(
                                sc,
                                bundle,
                                seed,
                                np.stack([p[0] for p in pred]),
                                np.stack([p[1] for p in pred]),
                                model,
                                thresholds,
                            )
                        )
                    position += n
        if position != len(predictions):
            raise RuntimeError("batched predictor inventory mismatch")
        for _, bundles, record in prepared:
            inventory.append(record)
            for bundle in bundles:
                counts["selected_actor_times"] += bundle["n_selected"]
                counts["incomplete_actor_times"] += bundle["n_incomplete"]
                counts["complete_actor_times"] += len(bundle["agents"])
        print(
            json.dumps(
                {
                    "scenes_done": len(inventory),
                    "scenes_total": len(ids),
                    "rows": len(rows),
                    "elapsed_s": round(time.monotonic() - start, 1),
                }
            ),
            flush=True,
        )
    return pd.DataFrame(rows), inventory, counts


def _scene_mean(values, scenarios) -> float | None:
    if not len(values):
        return None
    values = np.asarray(values, dtype=float)
    _, inverse = np.unique(np.asarray(scenarios), return_inverse=True)
    return float(np.mean(np.bincount(inverse, weights=values) / np.bincount(inverse)))


def _city_mean(values, frame: pd.DataFrame) -> float | None:
    if not len(frame):
        return None
    temp = pd.DataFrame({"city": frame.city.to_numpy(), "scenario_id": frame.scenario_id.to_numpy(), "value": values})
    return float(temp.groupby(["city", "scenario_id"]).value.mean().groupby("city").mean().mean())


def tune_selective_threshold(frame: pd.DataFrame) -> dict:
    """Tune only precomputed, realizable corrections on the tune partition."""
    label = frame.label.to_numpy(dtype=float)
    native = frame.native_mass.to_numpy(dtype=float)
    positive = label == 1
    baseline_brier = _scene_mean((native - label) ** 2, frame.scenario_id)
    baseline_recall = _scene_mean((native[positive] >= RECALL_THRESHOLD).astype(float), frame.scenario_id[positive])
    choices = []
    for threshold in THRESHOLDS:
        mass = frame[f"{_threshold_key(threshold)}_mass"].to_numpy(dtype=float)
        brier = _scene_mean((mass - label) ** 2, frame.scenario_id)
        recall = _scene_mean((mass[positive] >= RECALL_THRESHOLD).astype(float), frame.scenario_id[positive])
        loss = None if recall is None else baseline_recall - recall
        improvement = baseline_brier - brier
        choices.append(
            {
                "threshold": threshold,
                "brier": brier,
                "brier_improvement": improvement,
                "departure_recall": recall,
                "departure_recall_loss": loss,
                "admissible": loss is not None and loss <= RECALL_LOSS_LIMIT + 1e-12 and improvement > 1e-12,
            }
        )
    admissible = [choice for choice in choices if choice["admissible"]]
    best = min(admissible, key=lambda item: (-item["brier_improvement"], item["threshold"])) if admissible else None
    return {
        "threshold": best["threshold"] if best else 0.0,
        "status": "admissible_correction" if best else "no_admissible_improvement_identity",
        "native_brier": baseline_brier,
        "native_departure_recall": baseline_recall,
        "n_independent_departure_scenes": int(frame.loc[positive, "scenario_id"].nunique()),
        "objective": "scenario_equal_brier_subject_to_scenario_equal_departure_recall",
        "candidates": choices,
    }


def _recall_loss_interval(frame: pd.DataFrame, prefix: str) -> dict:
    positive = frame[frame.label.astype(bool)].copy()
    cities = sorted(positive.city.unique())
    missing = sorted(set(CITIES) - set(cities))
    result = {
        "confidence": 0.9875,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "supported_cities": cities,
        "missing_cities": missing,
        "coverage_complete": not missing,
        "n_independent_departure_scenes": int(positive.scenario_id.nunique()),
    }
    if positive.empty:
        return result | {"estimate": None, "ci": None, "underpowered": True}
    positive["loss"] = (positive.native_mass >= RECALL_THRESHOLD).astype(float) - (
        positive[f"{prefix}_mass"] >= RECALL_THRESHOLD
    ).astype(float)
    scene = positive.groupby(["city", "scenario_id"]).loss.mean()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.zeros(BOOTSTRAP_REPLICATES)
    estimates = []
    for city in cities:
        values = scene.loc[city].to_numpy()
        estimates.append(values.mean())
        bootstrap += values[rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))].mean(1) / len(cities)
    lower, upper = np.quantile(bootstrap, [0.00625, 0.99375])
    return result | {
        "estimate": float(np.mean(estimates)),
        "ci": [float(lower), float(upper)],
        "underpowered": result["n_independent_departure_scenes"] < 100 or bool(missing),
    }


def summarize_rows(frame: pd.DataFrame, *, partition: str, selective_prefix: str = "selective") -> dict:
    """City-equal estimates and paired scene bootstrap, never actor rows as IID."""
    inventory = _row_inventory(frame)
    if frame.groupby("scenario_id").city.nunique().max() != 1:
        raise ValueError("one scenario has conflicting city labels")
    if not np.isin(frame.label.to_numpy(), [False, True, 0, 1]).all():
        raise ValueError("departure labels must be binary")
    labels = frame.label.to_numpy(dtype=float)
    native = frame.native_mass.to_numpy(dtype=float)
    result = {
        "partition": partition,
        "row_inventory": inventory,
        "n_independent_departure_scenes": int(frame.loc[frame.label.astype(bool), "scenario_id"].nunique()),
        "n_independent_departure_events": len(frame[frame.label.astype(bool)].drop_duplicates(["scenario_id", "track_id", "t"])),
        "cities": sorted(frame.city.unique()),
        "weighting": "seeds averaged within event, events within scene, then supported cities equally",
        "arms": {},
    }
    for arm, prefix in zip(ARMS, ("native", "patch", "calib", selective_prefix)):
        mass = frame[f"{prefix}_mass"].to_numpy(dtype=float)
        fde = frame[f"{prefix}_fde4s"].to_numpy(dtype=float)
        if not np.isfinite(mass).all() or not np.isfinite(fde).all() or (mass < 0).any() or (mass > 1).any():
            raise ValueError("nonfinite or invalid evaluation probabilities/errors")
        bounded = np.clip(mass, PROTOCOL["log_loss_clip"], 1 - PROTOCOL["log_loss_clip"])
        metric = {
            "brier": _city_mean((mass - labels) ** 2, frame),
            "binary_log_loss": _city_mean(-labels * np.log(bounded) - (1 - labels) * np.log1p(-bounded), frame),
            "departure_mass": _city_mean(mass, frame),
            "top1_fde4s_m": _city_mean(fde, frame),
            "by_label": {},
        }
        supported = native > 0
        metric["native_departure_mass_retained_fraction"] = _city_mean(
            np.minimum(mass[supported], native[supported]) / native[supported], frame[supported]
        )
        for value, name in ((True, "departed"), (False, "stationary")):
            mask = frame.label.astype(bool).to_numpy() == value
            sub = frame[mask]
            metric["by_label"][name] = {
                "rows": len(sub),
                "independent_scenes": int(sub.scenario_id.nunique()),
                "cities": sorted(sub.city.unique()),
                "departure_mass": _city_mean(mass[mask], sub),
                "top1_fde4s_m": _city_mean(fde[mask], sub),
            }
            if value:
                metric["departure_recall_at_0_1"] = _city_mean((mass[mask] >= RECALL_THRESHOLD).astype(float), sub)
        if arm != "NATIVE":
            metric["departure_recall_loss_vs_native"] = _recall_loss_interval(frame, prefix)
        result["arms"][arm] = metric
    return result


def _context(cohort_path: Path, cohort: dict, runs: Path, protocol: Path) -> dict:
    return {
        "protocol": PROTOCOL,
        "protocol_sha256": file_sha256(protocol),
        "cohort_sha256": file_sha256(cohort_path),
        "checkpoint_sha256": checkpoint_manifest(runs, cohort),
        "implementation_sha256": implementation_manifest(),
    }


def _get_table(raw, ids, runs, out, role, context, workers, chunk, model=None, thresholds=(), device="cuda"):
    path = Path(out) / f"openloop_{role}.parquet"
    cached = _read_cached(path, context)
    if cached:
        return cached
    frame, inventory, counts = _collect(raw, ids, runs, workers, chunk, model, thresholds, device)
    metadata = _write_table(path, frame, context, inventory, counts)
    return frame, metadata


def run_fit(
    raw: Path,
    cohort_path: Path,
    runs: Path,
    out: Path,
    *,
    workers: int = 12,
    chunk: int = 100,
    protocol: Path = Path("docs/preregistration/objective-transfer.md"),
    device: str = "cuda",
) -> dict:
    cohort = load_cohort(cohort_path)
    context = _context(cohort_path, cohort, runs, protocol)
    artifact_path = Path(out) / "calibrator.json"
    if artifact_path.exists():
        artifact = json.loads(artifact_path.read_text())
        if artifact.get("context") != context:
            raise ValueError("frozen calibrator context differs; refusing refit")
        predict_departure(artifact["model"], np.zeros((0, len(HISTORY_FEATURE_NAMES))), np.zeros(0))
        return artifact
    fit_context = context | {"partition": "fit", "partition_sha256": cohort["partitions"]["fit"]["sha256"]}
    fit, fit_meta = _get_table(
        raw, cohort["partitions"]["fit"]["ids"], runs, out, "fit", fit_context, workers, chunk, device=device
    )
    model = fit_calibrator(
        fit[list(HISTORY_FEATURE_NAMES)].to_numpy(),
        fit.native_mass.to_numpy(),
        fit.label.to_numpy(),
        fit.scenario_id.to_numpy(),
        ridge=1.0,
    )
    tune_context = context | {
        "partition": "tune",
        "partition_sha256": cohort["partitions"]["tune"]["sha256"],
        "model_sha256": _json_digest(model),
        "thresholds": list(THRESHOLDS),
    }
    tune, tune_meta = _get_table(
        raw, cohort["partitions"]["tune"]["ids"], runs, out, "tune", tune_context, workers, chunk, model, THRESHOLDS, device
    )
    artifact = {
        "schema": 1,
        "model": model,
        "tuning": tune_selective_threshold(tune),
        "context": context,
        "fit_cohort_sha256": fit_context["partition_sha256"],
        "tune_cohort_sha256": tune_context["partition_sha256"],
        "fit_inventory": fit_meta,
        "tune_inventory": tune_meta,
    }
    _write_json_once(artifact_path, artifact)
    return artifact


def run_evaluation(
    raw: Path,
    cohort_path: Path,
    runs: Path,
    out: Path,
    *,
    partition: str = "pilot",
    workers: int = 12,
    chunk: int = 100,
    protocol: Path = Path("docs/preregistration/objective-transfer.md"),
    device: str = "cuda",
) -> dict:
    if partition not in {"pilot", "test"}:
        raise ValueError("evaluation partition must be pilot or test")
    cohort = load_cohort(cohort_path)
    context = _context(cohort_path, cohort, runs, protocol)
    artifact_path = Path(out) / "calibrator.json"
    artifact = json.loads(artifact_path.read_text())
    if artifact.get("context") != context:
        raise ValueError("frozen calibrator context differs from evaluation implementation/protocol/checkpoints")
    if partition == "test":
        pilot = json.loads((Path(out) / "pilot_analysis.json").read_text())
        if pilot.get("full_test_gate") != "proceed" or not pilot.get("feasibility_passed_both_objectives"):
            raise ValueError("development feasibility failed; fresh test remains unscored")
        provenance = pilot.get("provenance", {})
        expected = {
            "calibrator_sha256": file_sha256(artifact_path),
            "cohort_sha256": file_sha256(cohort_path),
            "pilot_manifest_sha256": file_sha256(Path(out) / "pilot_manifest.json"),
        }
        if any(provenance.get(key) != value for key, value in expected.items()):
            raise ValueError("pilot feasibility provenance mismatch")
    threshold = float(artifact["tuning"]["threshold"])
    if threshold not in (*THRESHOLDS, 0.0):
        raise ValueError("frozen threshold is outside registered grid")
    evaluation_context = context | {
        "partition": partition,
        "partition_sha256": cohort["partitions"][partition]["sha256"],
        "calibrator_sha256": file_sha256(artifact_path),
        "threshold": threshold,
    }
    frame, metadata = _get_table(
        raw,
        cohort["partitions"][partition]["ids"],
        runs,
        out,
        partition,
        evaluation_context,
        workers,
        chunk,
        artifact["model"],
        (threshold,),
        device,
    )
    prefix = _threshold_key(threshold)
    summary = summarize_rows(frame, partition=partition, selective_prefix=prefix)
    summary |= {"metadata": metadata, "frozen_tuning": artifact["tuning"]}
    _write_json_once(Path(out) / f"openloop_{partition}_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, default=Path("runs/transfer/cohort.json"))
    parser.add_argument("--runs", type=Path, default=Path("runs/v2/ALLR"))
    parser.add_argument("--out", type=Path, default=Path("runs/transfer"))
    parser.add_argument("--phase", choices=("fit", "evaluate"), required=True)
    parser.add_argument("--partition", choices=("pilot", "test"), default="pilot")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk", type=int, default=100)
    parser.add_argument("--protocol", type=Path, default=Path("docs/preregistration/objective-transfer.md"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    common = dict(
        raw=args.raw,
        cohort_path=args.cohort,
        runs=args.runs,
        out=args.out,
        workers=args.workers,
        chunk=args.chunk,
        protocol=args.protocol,
        device=args.device,
    )
    if args.phase == "fit":
        artifact = run_fit(**common)
        print(json.dumps({"phase": "fit", "tuning": artifact["tuning"]}, sort_keys=True), flush=True)
    else:
        summary = run_evaluation(**common, partition=args.partition)
        print(
            json.dumps(
                {"phase": "evaluate", "partition": args.partition, "row_inventory": summary["row_inventory"]}, sort_keys=True
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
