"""Settings and input fingerprints for exploratory runners. Validate before loading models or writing rows."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import pandas as pd


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_settings(harness: str, device: str, seeds: list[int], pool: str, ids: list[str], raw: str,
                 checkpoints: dict[str, str], runner: str | None = None, **settings) -> dict:
    source = Path(__file__).parent
    names = ["run_manifest", "data", "model", "preprocess", "scene", "closedloop", "closedloop_v2", "closedloop_v3",
             runner or ("followup_a" if harness == "closedloop_v3" else "followup_e")]
    if harness == "closedloop_v4":
        names.append("closedloop_v4")
    return {"schema": 1, "harness": harness, "device": device, "seeds": seeds,
            "pool_sha256": file_sha256(pool), "scenario_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "scenarios": len(ids), "raw_root_sha256": hashlib.sha256(str(Path(raw).resolve()).encode()).hexdigest(),
            "checkpoints_sha256": {k: file_sha256(p) for k, p in checkpoints.items()},
            "source_sha256": {f"{n}.py": file_sha256(source / f"{n}.py") for n in names},
            "versions": {"python": platform.python_version(),
                         **{n: version(n) for n in ("torch", "numpy", "pandas", "pyarrow")}}, **settings}


def validate_manifest(path: str | Path, settings: dict, outputs: list[str | Path]) -> None:
    """Refuse changed settings or legacy rows without a manifest. Never infer historical provenance."""
    path = Path(path)
    if path.exists():
        with path.open() as f:
            old = json.load(f)
        if old.get("settings") != settings:
            raise SystemExit(f"refusing to resume: {path} was written with different settings or inputs")
    elif any(Path(p).exists() for p in outputs):
        raise SystemExit(f"refusing to reuse existing rows without {path}; use a new output location")
    else:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w") as f:
            json.dump({"settings": settings, "git": revision}, f, indent=2)
            f.write("\n")
        tmp.replace(path)


def validate_parts(directory: str | Path, ids: list[str], chunk: int, seeds: list[int],
                   arms: tuple[str, ...] = ("ALL", "PATCH", "SHAM2", "TRIM")) -> None:
    """Every existing E part must cover exactly its expected scenarios and contain all arm outcomes and doses."""
    directory = Path(directory)
    expected = {f"part_{i:05d}.parquet": ids[i:i + chunk] for i in range(0, len(ids), chunk)}
    metrics = ("collision", "first_collision_step", "planner_hard_brake", "logged_hard_brake",
               "unnecessary_hard_brake", "progress", "accels")
    prefixes = [f"{a}_s{s}" for a in arms for s in seeds] + ["cv", "oracle", "static", "log"]
    required = {"scenario_id", "city"} | {f"{a}_{m}" for a in prefixes for m in metrics if a != "log" or m != "accels"}
    required |= {f"{a}_s{s}_dose_by_replan" for a in arms for s in seeds}
    for part in sorted(directory.glob("part_*.parquet")):
        if part.name not in expected:
            raise SystemExit(f"unexpected part: {part}")
        rows = pd.read_parquet(part)
        if not required <= set(rows) or rows.scenario_id.tolist() != expected[part.name]:
            raise SystemExit(f"invalid scenarios or columns in {part}")
