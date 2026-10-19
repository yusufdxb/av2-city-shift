"""Build v2 release assets from completed, audited local outputs.

Run from a clean committed checkout after both independent audits pass::

    python scripts/package_v2.py --provenance completion-provenance.json --paper paper

The output defaults to the ignored runs/v2/release directory. No network or
publishing action is performed. Checkpoint tensor values are preserved; only
filesystem paths in their configuration are replaced with portable paths.
Original and distributed checkpoint hashes are recorded separately.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
NOTICE = """City Shift v2 evidence release

Argoverse 2 data is copyright 2021 Argo AI, LLC, provided under Creative
Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0):
https://creativecommons.org/licenses/by-nc-sa/4.0/
The derived tables, trained weights, example-scene capture and data-derived
figures in these assets carry that dataset license. The code is MIT licensed.
This project is not affiliated with or endorsed by Argo AI.

Extract the assets into the source checkout. Read docs/v2.md for reproduction
and limitations. SHA256SUMS identifies the assets. Each data/model archive
also contains an inventory with per-file SHA-256 values.
"""
TABLE_DIRS = ("coverage", "q_open", "q_closed", "study_r_v4", "study_r_v2", "study_s_open",
              "study_s_SEL_v4", "study_s_SELSHAM_v4")
TABLE_FILES = ("reserve_ids.npy", "coverage_empty.parquet", "q_open_fallback.parquet", "input_fingerprints.json",
               "excess_brake_v4.parquet", "excess_brake_v2.parquet",
               "selection/SEL_exclude.npy", "selection/SELSHAM_exclude.npy")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def inspect_text(value, label):
    """Refuse private paths and unreviewed process context in distributed text."""
    text = json.dumps(value) if not isinstance(value, str) else value
    for term in ("/home/", "/Users/", "Obsidian", "codex/", ".claude/", "AGENTS.md", "CLAUDE.md", "\u2014"):
        if term.lower() in text.lower():
            raise ValueError(f"public-content check failed for {label}: {term}")


def copy(source, stage, relative):
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"required release input missing: {source}")
    if source.suffix in (".json", ".jsonl", ".tex", ".bib", ".py", ".md", ".txt"):
        inspect_text(source.read_text(), relative)
    target = stage / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def archive(stage, output, label, metadata):
    inventory = {str(p.relative_to(stage)): {"bytes": p.stat().st_size, "sha256": sha(p)}
                 for p in sorted(stage.rglob("*")) if p.is_file()}
    write_json(stage / "provenance" / f"{label}-inventory.json", metadata | {"files": inventory})
    (stage / "V2-LICENSE-NOTICE.txt").write_text(NOTICE)
    with output.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as tar:
            for path in sorted(stage.rglob("*")):
                if not path.is_file():
                    continue
                info = tarfile.TarInfo(str(path.relative_to(stage)))
                info.size, info.mode, info.mtime = path.stat().st_size, 0o644, 0
                with path.open("rb") as handle:
                    tar.addfile(info, handle)


def portable_config(config, directory):
    config = dict(config)
    config["root"], config["out"] = "data/pp", str(directory)
    if config.get("exclude_ids"):
        family = directory.parent.name
        config["exclude_ids"] = ("runs/v2/reserve_ids.npy" if family == "ALLR" else
                                 f"runs/v2/selection/{family}_exclude.npy")
    inspect_text(config, str(directory / "config.json"))
    return config


def tensor_digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        h.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode())
        h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def model_asset(stage):
    inventory = {}
    roots = [Path("runs/ALL"), Path("runs/v2/ALLR"), Path("runs/v2/completion/SEL"),
             Path("runs/v2/completion/SELSHAM")]
    for root in roots:
        for seed in range(3):
            relative = root / f"seed{seed}"
            source = ROOT / relative
            checkpoint = torch.load(source / "model.pt", map_location="cpu", weights_only=False)
            config = portable_config(checkpoint["config"], relative)
            original_tensors = tensor_digest(checkpoint["model"])
            target = stage / relative
            target.mkdir(parents=True)
            torch.save({"model": checkpoint["model"], "config": config}, target / "model.pt")
            distributed = torch.load(target / "model.pt", map_location="cpu", weights_only=True)
            if tensor_digest(distributed["model"]) != original_tensors:
                raise ValueError("tensor identity changed while preparing portable checkpoint")
            external_config = portable_config(json.loads((source / "config.json").read_text()), relative)
            if external_config != config:
                raise ValueError("checkpoint and external training configuration differ")
            write_json(target / "config.json", config)
            for name in ("train_indices.npy", "train_log.jsonl"):
                copy(source / name, stage, relative / name)
            inventory[str(relative / "model.pt")] = {"original_sha256": sha(source / "model.pt"),
                                                     "distributed_sha256": sha(target / "model.pt"),
                                                     "tensor_sha256": original_tensors,
                                                     "config_change": "portable root/out/exclude_ids paths only"}
    return inventory


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/v2/release")
    ap.add_argument("--provenance", type=Path, required=True, help="reviewed completion provenance JSON")
    ap.add_argument("--paper", type=Path, required=True, help="finished report source and rendered main.pdf")
    args = ap.parse_args()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status.strip():
        ap.error("commit the reviewed source and reports before packaging")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for name in ("audit_decisions", "audit_amendments"):
        audit = json.loads((ROOT / "reports/v2" / f"{name}.json").read_text())
        if not audit.get("pass") or audit.get("draws", 0) < 10000:
            ap.error(f"{name} must pass with at least 10,000 fresh draws")
        if name == "audit_decisions" and not audit.get("all_supported_studies_requested"):
            ap.error("the R/G/Q/S audit must include every study")
        if name == "audit_amendments" and set(audit.get("studies", [])) != {"H", "F"}:
            ap.error("the amendments audit must include H and F")
    provenance = json.loads(args.provenance.read_text())
    inspect_text(provenance, "completion provenance")
    if provenance.get("complete") is not True:
        ap.error("completion provenance must explicitly record complete=true")
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {"release": "v2", "source_revision": revision, "license_notice": "V2-LICENSE-NOTICE.txt"}
    with tempfile.TemporaryDirectory(prefix="v2-assets-", dir=args.out) as temporary:
        temporary = Path(temporary)
        stage = temporary / "tables"
        stage.mkdir()
        for stem in TABLE_DIRS:
            directory = ROOT / "runs/v2" / stem
            files = sorted(p for p in directory.iterdir() if p.suffix in (".parquet", ".json"))
            if not any(p.suffix == ".parquet" for p in files):
                raise ValueError(f"no rows for {stem}")
            for path in files:
                copy(path, stage, path.relative_to(ROOT))
        for name in TABLE_FILES:
            relative = Path("runs/v2") / name
            copy(ROOT / relative, stage, relative)
        for name in ("ALL", "LOCO-austin", "LOCO-dearborn", "LOCO-miami", "LOCO-palo-alto",
                     "LOCO-pittsburgh", "LOCO-washington-dc", "stage3a/per_agent"):
            relative = Path("evals") / f"{name}.parquet"
            copy(ROOT / relative, stage, relative)
        for path in sorted((ROOT / "reports/v2").glob("*.json")):
            copy(path, stage, path.relative_to(ROOT))
        write_json(stage / "provenance/v2-completion.json", provenance)
        archive(stage, args.out / "av2-city-shift-v2-tables.tar.gz", "v2-tables", metadata)

        stage = temporary / "models"
        stage.mkdir()
        checkpoints = model_asset(stage)
        archive(stage, args.out / "av2-city-shift-v2-models.tar.gz", "v2-models", metadata | {"checkpoints": checkpoints})

        stage = temporary / "report"
        stage.mkdir()
        for pattern in ("*.tex", "*.bib", "*.py", "Makefile", "generated/*.tex", "generated/capture.pkl",
                        "figures/*.png", "figures/*.pdf"):
            for path in sorted(args.paper.glob(pattern)):
                copy(path, stage, Path("paper") / path.relative_to(args.paper))
        archive(stage, args.out / "av2-city-shift-v2-report-source.tar.gz", "v2-report", metadata)
        copy(args.paper / "main.pdf", args.out, Path("av2-city-shift-v2-report.pdf"))

    # git archive provides the exact committed public tree, with no ignored local files.
    source = subprocess.check_output(["git", "archive", "--format=tar", revision], cwd=ROOT)
    source_path = args.out / "av2-city-shift-v2-source.tar.gz"
    with source_path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped:
        shutil.copyfileobj(io.BytesIO(source), zipped)
    assets = sorted(p for p in args.out.glob("av2-city-shift-v2-*") if p.is_file())
    sums = "".join(f"{sha(path)}  {path.name}\n" for path in assets)
    (args.out / "SHA256SUMS").write_text(sums)
    write_json(args.out / "manifest.json", metadata | {"assets": {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
                                                               for p in assets}})
    print(json.dumps({"source_revision": revision, "assets": len(assets), "bytes": sum(p.stat().st_size for p in assets)}))


if __name__ == "__main__":
    main()
