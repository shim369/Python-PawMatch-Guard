"""Oxford-IIIT Pet: download, parse list.txt, build split CSV, torch Dataset.

License: CC BY-SA 4.0 (https://www.robots.ox.ac.uk/~vgg/data/pets/), which allows
redistribution of derived metadata such as our split CSV with attribution.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .split import Entry, sha256_file

DATASET_DIR = "oxford-iiit-pet"


def download(root: Path) -> Path:
    """Download the dataset via torchvision (no-op if already present)."""
    from torchvision.datasets import OxfordIIITPet

    OxfordIIITPet(root=str(root), split="trainval", download=True)
    OxfordIIITPet(root=str(root), split="test", download=True)
    return root / DATASET_DIR


def parse_list(list_txt: Path) -> tuple[list[Entry], list[str]]:
    """Parse ``annotations/list.txt``.

    Line format: ``<Image> <CLASS-ID 1-37> <SPECIES 1=cat 2=dog> <BREED-ID>``.
    Returns entries (0-based ids) and the 37 breed names ordered by class id.
    """
    entries: list[Entry] = []
    breeds: dict[int, str] = {}
    for line in list_txt.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, class_id, species, _ = line.split()
        cid = int(class_id) - 1
        breeds[cid] = name.rsplit("_", 1)[0].lower()
        entries.append(Entry(name=name, breed_id=cid, species_id=int(species) - 1))
    return entries, [breeds[i] for i in sorted(breeds)]


def with_hashes(entries: list[Entry], images_dir: Path) -> list[Entry]:
    """Attach a content hash; entries whose image is missing/unreadable are dropped."""
    out = []
    for e in entries:
        p = images_dir / f"{e.name}.jpg"
        if p.exists():
            out.append(Entry(e.name, e.breed_id, e.species_id, sha256_file(p)))
    return out


class PetClassificationDataset(Dataset):
    """Yields ``(image_tensor, species_id, breed_id)``."""

    def __init__(self, entries: list[Entry], images_dir: Path, transform=None) -> None:
        self.entries = entries
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, int]:
        e = self.entries[i]
        img = Image.open(self.images_dir / f"{e.name}.jpg").convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, e.species_id, e.breed_id
