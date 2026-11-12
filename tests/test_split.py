from collections import Counter

import pytest

from src.data.split import (
    Entry,
    check_no_leakage,
    drop_duplicates,
    read_split_csv,
    stratified_split,
    write_split_csv,
)


def _entries() -> list[Entry]:
    out = []
    for breed in range(5):
        for i in range(40):
            out.append(Entry(f"b{breed}_{i}", breed, breed % 2, f"h{breed}_{i}"))
    return out


def test_ratios_and_stratification() -> None:
    s = stratified_split(_entries())
    assert sum(len(v) for v in s.values()) == 200
    for breed in range(5):
        counts = {k: Counter(e.breed_id for e in v)[breed] for k, v in s.items()}
        assert counts == {"train": 28, "val": 6, "test": 6}


def test_deterministic_and_seed_sensitive() -> None:
    a = stratified_split(_entries(), seed=1)
    b = stratified_split(_entries(), seed=1)
    c = stratified_split(_entries(), seed=2)
    names = lambda s: [e.name for e in s["test"]]  # noqa: E731
    assert names(a) == names(b)
    assert names(a) != names(c)


def test_duplicates_removed_and_leakage_detected() -> None:
    e = _entries() + [Entry("dup", 0, 0, "h0_0")]
    kept, removed = drop_duplicates(e)
    assert removed == 1 and len(kept) == 200
    bad = {"train": [Entry("x", 0, 0, "h")], "val": [Entry("y", 0, 0, "h")], "test": []}
    with pytest.raises(ValueError):
        check_no_leakage(bad)


def test_csv_roundtrip(tmp_path) -> None:
    s = stratified_split(_entries())
    write_split_csv(s, tmp_path / "s.csv")
    r = read_split_csv(tmp_path / "s.csv")
    assert {k: sorted(e.name for e in v) for k, v in r.items()} == {
        k: sorted(e.name for e in v) for k, v in s.items()
    }
