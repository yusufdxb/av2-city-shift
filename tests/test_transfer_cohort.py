"""Small membership tests; these do not require AV2 arrays or checkpoints."""

import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("make_transfer_cohort", Path(__file__).parents[1] / "scripts/make_transfer_cohort.py")
cohort = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cohort)


def test_stratification_has_exact_size_and_deterministic_city_tie():
    cities = {**{f"a{i}": "a" for i in range(7)}, **{f"b{i}": "b" for i in range(2)}, "c0": "c"}
    selected = cohort.stratified_ids(cities, 5, "test")
    assert len(selected) == 5
    assert sum(cities[sid] == "a" for sid in selected) == 4
    assert sum(cities[sid] == "b" for sid in selected) == 1
    assert sum(cities[sid] == "c" for sid in selected) == 0
    assert cohort.stratified_ids(dict(reversed(list(cities.items()))), 5, "test") == selected


def test_sequential_development_roles_are_disjoint_and_leave_unused_ids():
    remaining = {f"s{i}": "a" if i < 12 else "b" for i in range(20)}
    partitions = {}
    for role, size in (("fit", 7), ("tune", 3), ("pilot", 3)):
        partitions[role] = cohort.stratified_ids(remaining, size, role)
        for sid in partitions[role]:
            remaining.pop(sid)
    cohort.require_disjoint(partitions)
    assert len(remaining) == 7
    assert not set(remaining).intersection(set().union(*map(set, partitions.values())))


def test_role_hash_changes_ranking_without_using_global_random_state():
    cities = {f"s{i}": "city" for i in range(30)}
    assert cohort.stratified_ids(cities, 10, "fit") != cohort.stratified_ids(cities, 10, "test")
    assert cohort.stratified_ids(cities, 0, "test") == []
    assert cohort.stratified_ids(cities, len(cities), "test") == sorted(cities)
    for size in (-1, len(cities) + 1):
        with pytest.raises(ValueError, match="partition size"):
            cohort.stratified_ids(cities, size, "test")


def test_duplicate_and_overlapping_memberships_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        cohort.ids_sha256(["same", "same"])
    with pytest.raises(ValueError, match="overlapping"):
        cohort.require_disjoint({"fit": ["same"], "test": ["same"]})
    with pytest.raises(ValueError, match="duplicate"):
        cohort.require_disjoint({"fit": ["same", "same"]})


def test_id_digest_is_order_independent_and_has_no_trailing_newline():
    assert cohort.ids_sha256(["b", "a"]) == cohort.ids_sha256(["a", "b"])
    assert cohort.ids_sha256(["a", "b"]) == "7e18f737311b2dc3b2f269dd78396b0351f14fb66efa879f768cb23181883c78"


def test_existing_different_record_is_preserved(tmp_path):
    path = tmp_path / "nested" / "cohort.json"
    original = {"schema": 1, "partitions": {"test": ["a"]}}
    assert cohort.write_cohort(path, original)
    before = path.read_bytes()
    assert not cohort.write_cohort(path, original)
    with pytest.raises(ValueError, match="refusing to replace"):
        cohort.write_cohort(path, {"schema": 1, "partitions": {"test": ["b"]}})
    assert path.read_bytes() == before
