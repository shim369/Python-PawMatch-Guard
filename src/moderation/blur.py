"""Blur personal-information regions (faces, license plates, text)."""

from __future__ import annotations

import cv2
import numpy as np


def blur_regions(image: np.ndarray, boxes, pad: float = 0.1, strength: float = 0.35) -> np.ndarray:
    """Return a copy of ``image`` (H, W, 3 uint8) with each box Gaussian-blurred.

    ``pad`` grows each box by a fraction of its size so edges are covered; the kernel scales
    with the box so the content is unrecoverable at any size.
    """
    out = image.copy()
    h, w = image.shape[:2]
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        px, py = bw * pad, bh * pad
        a, b = max(int(x1 - px), 0), max(int(y1 - py), 0)
        c, d = min(int(x2 + px) + 1, w), min(int(y2 + py) + 1, h)
        if c <= a or d <= b:
            continue
        k = max(int(min(c - a, d - b) * strength) | 1, 3)  # odd kernel
        out[b:d, a:c] = cv2.GaussianBlur(out[b:d, a:c], (k, k), 0)
    return out
