"""Oxford-IIIT Pet head boxes (annotations/xmls) as a detection dataset.

Only ~3.7k of the images have an XML box. The box covers the animal's *head*, so the
detector learns a single class, ``animal_head``, used as a proxy for "an animal is present".
"""

from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .split import Entry

ANIMAL_HEAD = 1  # torchvision reserves 0 for background


def parse_xml(path: Path) -> tuple[np.ndarray, int, int]:
    """Return (boxes (N, 4) xyxy, width, height)."""
    root = ET.parse(path).getroot()
    w = int(root.findtext("size/width"))
    h = int(root.findtext("size/height"))
    boxes = [
        [float(o.findtext(f"bndbox/{k}")) for k in ("xmin", "ymin", "xmax", "ymax")]
        for o in root.iter("object")
    ]
    return np.array(boxes, dtype=np.float32).reshape(-1, 4), w, h


def with_boxes(entries: list[Entry], root: Path) -> list[Entry]:
    return [e for e in entries if (root / "annotations" / "xmls" / f"{e.name}.xml").exists()]


class PetDetectionDataset(Dataset):
    def __init__(self, entries: list[Entry], root: Path, train: bool = False) -> None:
        self.entries, self.root, self.train = entries, root, train

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        name = self.entries[i].name
        boxes, w, _ = parse_xml(self.root / "annotations" / "xmls" / f"{name}.xml")
        img = Image.open(self.root / "images" / f"{name}.jpg").convert("RGB")
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
        if self.train and random.random() < 0.5:
            x = x.flip(-1)
            boxes = np.stack([w - boxes[:, 2], boxes[:, 1], w - boxes[:, 0], boxes[:, 3]], 1)
        target = {
            "boxes": torch.from_numpy(boxes.astype(np.float32)),
            "labels": torch.full((len(boxes),), ANIMAL_HEAD, dtype=torch.int64),
        }
        return x, target


class NegativeDetectionDataset(Dataset):
    """Photos without any animal: an image with an empty target (all background)."""

    def __init__(self, paths: list[Path], train: bool = False) -> None:
        self.paths, self.train = paths, train

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
        if self.train and random.random() < 0.5:
            x = x.flip(-1)
        return x, {"boxes": torch.zeros((0, 4), dtype=torch.float32),
                   "labels": torch.zeros((0,), dtype=torch.int64)}
