"""Breed-stratified Train/Val/Test split with duplicate detection.

Output is a CSV (committed to Git) with columns:
``name, breed_id, species_id, sha256, split``.
"""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Entry:
    name: str  # e.g. "Abyssinian_1"
    breed_id: int  # 0..36
    species_id: int  # 0 = cat, 1 = dog
    sha256: str = ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def drop_duplicates(entries: list[Entry]) -> tuple[list[Entry], int]:
    """Keep the first entry per content hash (entries without hash are kept)."""
    seen: set[str] = set()
    kept: list[Entry] = []
    for e in entries:
        if e.sha256 and e.sha256 in seen:
            continue
        seen.add(e.sha256)
        kept.append(e)
    return kept, len(entries) - len(kept)


def stratified_split(
    entries: list[Entry],
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[Entry]]:
    """Split per breed so every split keeps the breed distribution."""
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios must sum to 1")
    rng = np.random.default_rng(seed)
    by_breed: dict[int, list[Entry]] = defaultdict(list)
    for e in sorted(entries, key=lambda x: x.name):
        by_breed[e.breed_id].append(e)

    out: dict[str, list[Entry]] = {s: [] for s in SPLITS}
    for breed in sorted(by_breed):
        items = by_breed[breed]
        order = rng.permutation(len(items))
        n_train = round(len(items) * ratios[0])
        n_val = round(len(items) * ratios[1])
        for rank, idx in enumerate(order):
            split = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"
            out[split].append(items[idx])
    return out


def write_split_csv(splits: dict[str, list[Entry]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "breed_id", "species_id", "sha256", "split"])
        for split in SPLITS:
            for e in sorted(splits[split], key=lambda x: x.name):
                w.writerow([e.name, e.breed_id, e.species_id, e.sha256, split])


def read_split_csv(path: Path) -> dict[str, list[Entry]]:
    out: dict[str, list[Entry]] = {s: [] for s in SPLITS}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out[row["split"]].append(
                Entry(row["name"], int(row["breed_id"]), int(row["species_id"]), row["sha256"])
            )
    return out


def check_no_leakage(splits: dict[str, list[Entry]]) -> None:
    """Raise if a name or content hash appears in more than one split."""
    owner: dict[str, str] = {}
    for split, items in splits.items():
        for e in items:
            for key in (e.name, e.sha256):
                if not key:
                    continue
                if owner.setdefault(key, split) != split:
                    raise ValueError(f"leakage: {key!r} is in both {owner[key]} and {split}")
