"""Oxford-IIIT Pet segmentation dataset (trimap -> binary animal mask).

Trimap values: 1 = animal, 2 = background, 3 = boundary/"not classified". We treat 1 and 3 as
animal so the mask includes the fur edge.

``bug`` deliberately re-creates a preprocessing bug for docs/debugging.md:
  - ``bgr``          : image channels fed as BGR instead of RGB
  - ``linear_mask``  : trimap resized with bilinear interpolation (blends label ids)
  - ``double_norm``  : ImageNet normalization applied twice
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .split import Entry
from .transforms import IMAGENET_MEAN, IMAGENET_STD

BUGS = ("none", "bgr", "linear_mask", "double_norm")


def trimap_to_mask(trimap: np.ndarray) -> np.ndarray:
    """(H, W) trimap ids -> uint8 {0, 1} mask."""
    return np.isin(trimap, (1, 3)).astype(np.uint8)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x - mean) / std


class PetSegmentationDataset(Dataset):
    def __init__(self, entries: list[Entry], root: Path, size: int = 256, train: bool = False,
                 bug: str = "none") -> None:
        if bug not in BUGS:
            raise ValueError(f"unknown bug: {bug}")
        self.entries, self.root, self.size, self.train, self.bug = entries, root, size, train, bug

    def __len__(self) -> int:
        return len(self.entries)

    def load_mask(self, name: str) -> np.ndarray:
        tri = Image.open(self.root / "annotations" / "trimaps" / f"{name}.png")
        if self.bug == "linear_mask":
            small = np.asarray(tri.resize((self.size, self.size), Image.BILINEAR), dtype=np.float32)
            # Typical bug: blended ids (e.g. 1.5, 2.5) are truncated by an int cast.
            return np.isin(small.astype(np.int64), (1, 3)).astype(np.uint8)
        small = np.asarray(tri.resize((self.size, self.size), Image.NEAREST))
        return trimap_to_mask(small)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.entries[i].name
        img = Image.open(self.root / "images" / f"{name}.jpg").convert("RGB")
        img = img.resize((self.size, self.size), Image.BILINEAR)
        mask = self.load_mask(name)
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
        if self.train and random.random() < 0.5:
            x, mask = x.flip(-1), np.ascontiguousarray(mask[:, ::-1])
        if self.bug == "bgr":
            x = x.flip(0)
        x = _normalize(x)
        if self.bug == "double_norm":
            x = _normalize(x)
        return x, torch.from_numpy(mask.astype(np.int64))
