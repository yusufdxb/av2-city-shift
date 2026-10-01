"""Build the local, unpublished v1.5 follow-up assets from stored rows. No inference or replay."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


NOTICE = ("These tables are derived evaluation and replay outputs computed from the Argoverse 2 Motion Forecasting "
          "dataset (https://www.argoverse.org/av2.html), (c) 2021 Argo AI, LLC, and are distributed under CC BY-NC-SA 4.0 "
          "(https://creativecommons.org/licenses/by-nc-sa/4.0/): non-commercial use only, with attribution and "
          "share-alike. The MIT license of the av2-city-shift code does not apply to them. Not affiliated with or "
          "endorsed by Argo AI.")

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def archive(path: Path, files: list[Path]) -> None:
    with path.open("wb") as out, gzip.GzipFile(filename="", fileobj=out, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for source in sorted(files):
                info = tar.gettarinfo(str(source), arcname=source.relative_to(ROOT).as_posix())
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                info.mode = 0o644
                with source.open("rb") as f:
                    tar.addfile(info, f)


def main() -> None:
    out = ROOT / "runs/release-v1.5"
    reports = ROOT / "reports/followups"
    a = [ROOT / f"runs/followups/{name}{suffix}" for name in ("study_a.parquet", "study_a_parity.parquet")
         for suffix in ("", ".report.json")]
    a += sorted((ROOT / "runs/followups").glob("study_a*.parquet.manifest.json"))
    e = sorted((ROOT / "runs/followups/study_e").glob("part_*.parquet"))
    if not e:
        raise SystemExit("no study E parts found")
    manifest = ROOT / "runs/followups/study_e/manifest.json"
    if manifest.exists():
        e.append(manifest)
    e.append(ROOT / "runs/followups/study_e_stopline_diagnostics.parquet")
    replay = [ROOT / f"runs/followups/{name}.parquet" for name in ("replay_rows", "replay_rows_stage3")]
    assets = {"study-a-row-results.tar.gz": a, "study-e-row-results.tar.gz": e, "followup-replay-rows.tar.gz": replay}
    files = sorted({p for paths in assets.values() for p in paths})
    for path in files:
        if not path.is_file():
            raise SystemExit(f"missing release input: {path.relative_to(ROOT)}")
    inventory = {"release": "v1.5", "data_license": NOTICE,
                 "historical_provenance": {
                     "study_a_manifest_present": (ROOT / "runs/followups/study_a.parquet.manifest.json").exists(),
                     "study_a_parity_manifest_present": (ROOT / "runs/followups/study_a_parity.parquet.manifest.json").exists(),
                     "study_e_manifest_present": manifest.exists(),
                     "note": "Absent historical manifests are not reconstructed. Checksums identify released bytes; "
                             "they do not establish homogeneous production settings."},
                 "assets": {name: [p.relative_to(ROOT).as_posix() for p in paths] for name, paths in assets.items()},
                 "files": {p.relative_to(ROOT).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size,
                           **({"rows": pq.read_metadata(p).num_rows} if p.suffix == ".parquet" else {})} for p in files}}
    out.mkdir(parents=True, exist_ok=True)
    checksum = reports / "SHA256SUMS"
    checksum.write_text("".join(f"{v['sha256']}  {p}\n" for p, v in inventory["files"].items()))
    inventory_path = reports / "release_v1.5.json"
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")
    shutil.copyfile(checksum, out / "SHA256SUMS")
    shutil.copyfile(inventory_path, out / "release_v1.5.json")
    notice = out / "DATA_NOTICE.txt"
    notice.write_text(NOTICE + "\n")
    # Archives carry their checksum list, inventory and notice under runs/release-v1.5/, never over the tracked copies
    # in reports/followups/, so a clone verifies the released bytes against its own committed list.
    for name, paths in assets.items():
        archive(out / name, paths + [out / "SHA256SUMS", out / "release_v1.5.json", notice])
    (out / "ASSET_SHA256SUMS").write_text("".join(f"{sha256(out / name)}  {name}\n" for name in sorted(assets)))
    print(json.dumps({"assets": list(assets), "checksummed_files": len(files)}))


if __name__ == "__main__":
    main()
