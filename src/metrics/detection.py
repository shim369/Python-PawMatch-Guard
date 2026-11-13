"""From-scratch object-detection metrics: IoU, AP@0.5, mAP@0.5:0.95, per-class recall.

Boxes are ``[x1, y1, x2, y2]``. AP uses COCO's 101-point interpolation so values can be
checked against pycocotools / torchmetrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N, 4) and (M, 4) boxes -> (N, M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = np.clip(a[:, 2:] - a[:, :2], 0, None).prod(axis=1)
    area_b = np.clip(b[:, 2:] - b[:, :2], 0, None).prod(axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    out = np.zeros_like(inter)
    np.divide(inter, union, out=out, where=union > 0)
    return out


@dataclass
class ImageDetections:
    """Predictions and ground truth for one image."""

    pred_boxes: np.ndarray  # (N, 4)
    pred_scores: np.ndarray  # (N,)
    pred_labels: np.ndarray  # (N,)
    gt_boxes: np.ndarray  # (M, 4)
    gt_labels: np.ndarray  # (M,)


def _match(images: list[ImageDetections], cls: int, thr: float):
    """Greedy matching by descending score; returns (scores, is_tp, n_gt)."""
    scores: list[float] = []
    tps: list[bool] = []
    n_gt = 0
    for im in images:
        gt = im.gt_boxes[im.gt_labels == cls]
        n_gt += len(gt)
        sel = im.pred_labels == cls
        boxes, sc = im.pred_boxes[sel], im.pred_scores[sel]
        order = np.argsort(-sc, kind="stable")
        used = np.zeros(len(gt), dtype=bool)
        ious = box_iou(boxes, gt) if len(gt) and len(boxes) else np.zeros((len(boxes), len(gt)))
        for i in order:
            hit = False
            if len(gt):
                cand = np.where(~used, ious[i], -1.0)
                j = int(cand.argmax())
                if cand[j] >= thr:
                    used[j] = True
                    hit = True
            scores.append(float(sc[i]))
            tps.append(hit)
    return np.array(scores), np.array(tps, dtype=bool), n_gt


def average_precision(images: list[ImageDetections], cls: int, thr: float = 0.5) -> float:
    """COCO-style AP (101-point interpolation) for one class at one IoU threshold.

    Returns -1.0 when the class has no ground truth (COCO convention; excluded from means).
    """
    scores, tp, n_gt = _match(images, cls, thr)
    if n_gt == 0:
        return -1.0
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = tp[order]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(~tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    for i in range(len(precision) - 2, -1, -1):  # monotone envelope
        precision[i] = max(precision[i], precision[i + 1])
    pts = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, pts, side="left")
    q = np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0.0)
    return float(q.mean())


def mean_ap(images: list[ImageDetections], classes: list[int], thresholds=IOU_THRESHOLDS) -> float:
    vals = [average_precision(images, c, float(t)) for c in classes for t in thresholds]
    vals = [v for v in vals if v >= 0]
    return float(np.mean(vals)) if vals else 0.0


def class_recall(images: list[ImageDetections], cls: int, thr: float = 0.5) -> float:
    """Fraction of GT boxes of ``cls`` matched at ``thr`` (all predictions, no score cut)."""
    _, tp, n_gt = _match(images, cls, thr)
    return float(tp.sum() / n_gt) if n_gt else 0.0
