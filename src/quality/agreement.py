"""Intra-annotator agreement: annotate the same images twice and compare with IoU."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.metrics.detection import box_iou


@dataclass(frozen=True)
class Agreement:
    mean_iou: float  # mean IoU over matched pairs
    matched: int
    unmatched_a: int
    unmatched_b: int

    @property
    def match_rate(self) -> float:
        total = self.matched * 2 + self.unmatched_a + self.unmatched_b
        return 2 * self.matched / total if total else 1.0


def box_agreement(round1: list[np.ndarray], round2: list[np.ndarray],
                  min_iou: float = 0.1) -> Agreement:
    """Greedy one-to-one matching of boxes between two rounds, per image (same-class boxes
    should be passed together; call once per class)."""
    ious, matched, ua, ub = [], 0, 0, 0
    for a, b in zip(round1, round2, strict=True):
        a = np.asarray(a, float).reshape(-1, 4)
        b = np.asarray(b, float).reshape(-1, 4)
        m = box_iou(a, b) if len(a) and len(b) else np.zeros((len(a), len(b)))
        used_a, used_b = set(), set()
        for flat in np.argsort(-m, axis=None):
            i, j = divmod(int(flat), m.shape[1])
            if m[i, j] < min_iou:
                break
            if i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            ious.append(m[i, j])
        matched += len(used_a)
        ua += len(a) - len(used_a)
        ub += len(b) - len(used_b)
    return Agreement(float(np.mean(ious)) if ious else 0.0, matched, ua, ub)
