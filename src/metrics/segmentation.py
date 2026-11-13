"""From-scratch segmentation metrics for binary masks (verified against torchmetrics)."""

from __future__ import annotations

import cv2
import numpy as np


def _b(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask).astype(bool)


def iou(pred: np.ndarray, target: np.ndarray) -> float:
    """Intersection over union. Two empty masks count as a perfect match (1.0)."""
    p, t = _b(pred), _b(target)
    union = np.logical_or(p, t).sum()
    return 1.0 if union == 0 else float(np.logical_and(p, t).sum() / union)


def dice(pred: np.ndarray, target: np.ndarray) -> float:
    p, t = _b(pred), _b(target)
    total = p.sum() + t.sum()
    return 1.0 if total == 0 else float(2 * np.logical_and(p, t).sum() / total)


def boundary_band(mask: np.ndarray, width: int = 3) -> np.ndarray:
    """Pixels of ``mask`` within ``width`` pixels of its contour."""
    m = _b(mask).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(m, kernel, iterations=width, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return (m - eroded).astype(bool)


def boundary_iou(pred: np.ndarray, target: np.ndarray, width: int = 3) -> float:
    """IoU computed only on the contour bands (sensitive to edge quality)."""
    return iou(boundary_band(pred, width), boundary_band(target, width))


def mean_over_images(fn, preds, targets) -> float:
    vals = [fn(p, t) for p, t in zip(preds, targets, strict=True)]
    return float(np.mean(vals)) if vals else 0.0
