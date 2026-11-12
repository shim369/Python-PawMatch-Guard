"""Fallback source: convert the Hugging Face mirror (timm/oxford-iiit-pet, CC BY-SA 4.0)
into the original Oxford-IIIT Pet layout (images/*.jpg + annotations/list.txt).

Use when the official server is too slow. Place ``train.parquet`` / ``test.parquet`` in
``<data_root>/hf`` (see README) and run ``pawmatch prepare-data --source hf``.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from .oxford_pet import DATASET_DIR

BASE_URL = "https://huggingface.co/datasets/timm/oxford-iiit-pet/resolve/main/data"
FILES = {
    "train.parquet": "train-00000-of-00001.parquet",
    "test.parquet": "test-00000-of-00001.parquet",
}


def download(hf_dir: Path) -> None:
    import urllib.request

    hf_dir.mkdir(parents=True, exist_ok=True)
    for local, remote in FILES.items():
        if not (hf_dir / local).exists():
            urllib.request.urlretrieve(f"{BASE_URL}/{remote}", hf_dir / local)


def convert(data_root: Path) -> Path:
    """Extract images and write list.txt (``name class_id(1-37) species(1=cat,2=dog) breed``)."""
    base = data_root / DATASET_DIR
    (base / "images").mkdir(parents=True, exist_ok=True)
    (base / "annotations").mkdir(parents=True, exist_ok=True)
    lines = ["#Image CLASS-ID SPECIES BREED ID"]
    for fname in FILES:
        pf = pq.ParquetFile(data_root / "hf" / fname)
        for batch in pf.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                name = row["image_id"]
                (base / "images" / f"{name}.jpg").write_bytes(row["image"]["bytes"])
                lines.append(f"{name} {row['label'] + 1} {row['label_cat_dog'] + 1} 0")
    (base / "annotations" / "list.txt").write_text("\n".join(lines) + "\n")
    return base
