"""Consistency checks for box and mask annotations (run before training)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.metrics.detection import box_iou


@dataclass(frozen=True)
class Issue:
    # out_of_bounds | zero_area | bad_class | mask_shape | mask_values | mask_box_mismatch
    code: str
    index: int  # box index (or -1 for image-level issues)
    detail: str = ""


def check_boxes(boxes: np.ndarray, classes: np.ndarray, width: int, height: int,
                num_classes: int) -> list[Issue]:
    issues: list[Issue] = []
    for i, (b, c) in enumerate(zip(np.asarray(boxes, float).reshape(-1, 4), classes, strict=True)):
        x1, y1, x2, y2 = b
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            issues.append(Issue("out_of_bounds", i, f"{b.tolist()} vs {width}x{height}"))
        if x2 <= x1 or y2 <= y1:
            issues.append(Issue("zero_area", i, str(b.tolist())))
        if not (0 <= int(c) < num_classes):
            issues.append(Issue("bad_class", i, f"class id {c} not in [0, {num_classes})"))
    return issues


def mask_bbox(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=float)


def check_mask_box_consistency(mask: np.ndarray, box: np.ndarray, width: int, height: int,
                               min_iou: float = 0.7, valid_values=(0, 1)) -> list[Issue]:
    """The mask must match the image size, use valid values and roughly fit its box."""
    if mask.shape != (height, width):
        return [Issue("mask_shape", -1, f"{mask.shape} vs {(height, width)}")]
    bad = np.setdiff1d(np.unique(mask), valid_values)
    if len(bad):
        return [Issue("mask_values", -1, f"unexpected values {bad.tolist()}")]
    mb = mask_bbox(mask > 0)
    if mb is None:
        return [Issue("zero_area", -1, "empty mask")]
    if box_iou(mb[None], np.asarray(box, float)[None])[0, 0] < min_iou:
        return [Issue("mask_box_mismatch", 0, f"mask bbox {mb.tolist()} vs box {list(box)}")]
    return []
