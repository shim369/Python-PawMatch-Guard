"""Train/test split of the no-animal negative photos (see scripts/build_negatives.py).

Half goes to detector training (as images with *no* boxes), half is held out so the
false-positive rate on unseen negatives can be measured fairly. The split is saved as a CSV
and committed.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path


def split_negatives(images_dir: Path, out_csv: Path, train_n: int = 150,
                    seed: int = 42) -> dict[str, list[str]]:
    ids = sorted(p.stem for p in images_dir.glob("*.jpg"))
    random.Random(seed).shuffle(ids)
    split = {"train": sorted(ids[:train_n]), "test": sorted(ids[train_n:])}
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "split"])
        for name, items in split.items():
            w.writerows([[i, name] for i in items])
    return split


def read_negatives_split(csv_path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"train": [], "test": []}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            out[row["split"]].append(row["image_id"])
    return out
