"""Portable release artifacts retain model values and refuse private configuration."""

import importlib.util
import json
from pathlib import Path
import tarfile

import pytest
import torch

spec = importlib.util.spec_from_file_location("package_v2", Path(__file__).parents[1] / "scripts/package_v2.py")
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


def test_portable_configuration_preserves_recipe_and_removes_paths():
    config = {"root": "/home/example/data", "out": "/home/example/models", "seed": 2, "batch": 128,
              "exclude_ids": "/home/example/excluded.npy", "train_indices_sha1": "abc"}
    portable = package.portable_config(config, Path("runs/v2/completion/SEL/seed2"))
    assert portable == config | {"root": "data/pp", "out": "runs/v2/completion/SEL/seed2",
                                  "exclude_ids": "runs/v2/selection/SEL_exclude.npy"}
    assert config["root"] == "/home/example/data"
    with pytest.raises(ValueError, match="public-content"):
        package.portable_config(config | {"other_path": "/home/example/private"}, Path("runs/v2/ALLR/seed2"))


def test_tensor_identity_includes_names_shapes_and_values():
    state = {"a": torch.tensor(1.), "b": torch.arange(6).reshape(2, 3), "c": torch.ones(2, dtype=torch.bfloat16)}
    original = package.tensor_digest(state)
    assert original == package.tensor_digest(dict(reversed(list(state.items()))))
    assert original != package.tensor_digest(state | {"a": torch.tensor(2.)})
    assert original != package.tensor_digest(state | {"b": state["b"].reshape(3, 2)})
    assert original != package.tensor_digest({"renamed": state["a"], "b": state["b"], "c": state["c"]})


def test_archive_inventory_and_reproducible_bytes(tmp_path):
    stage = tmp_path / "input"
    stage.mkdir()
    (stage / "rows.parquet").write_bytes(b"example row bytes")
    first, second = tmp_path / "first.gz", tmp_path / "second.gz"
    metadata = {"source_revision": "example"}
    package.archive(stage, first, "test", metadata)
    # Inventory and notice are generated outputs, not additional inputs on a repeated build.
    (stage / "provenance/test-inventory.json").unlink()
    (stage / "V2-LICENSE-NOTICE.txt").unlink()
    package.archive(stage, second, "test", metadata)
    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first) as archive:
        inventory = json.load(archive.extractfile("provenance/test-inventory.json"))
        assert inventory["files"]["rows.parquet"]["sha256"] == package.sha(stage / "rows.parquet")
        assert all(not member.name.startswith("/") and ".." not in Path(member.name).parts for member in archive)
