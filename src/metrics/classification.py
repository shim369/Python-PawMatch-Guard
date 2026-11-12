"""From-scratch classification metrics (verified against scikit-learn in tests)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Return an (C, C) matrix where entry [i, j] counts true class i predicted as j."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.size and (
        y_true.min() < 0 or y_pred.min() < 0 or y_true.max() >= num_classes
        or y_pred.max() >= num_classes
    ):
        raise ValueError("class id out of range")
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def accuracy(cm: np.ndarray) -> float:
    total = cm.sum()
    return float(np.trace(cm) / total) if total else 0.0


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=np.float64)
    np.divide(num, den, out=out, where=den > 0)
    return out


@dataclass(frozen=True)
class PRF:
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    support: np.ndarray

    @property
    def macro_f1(self) -> float:
        return float(self.f1.mean()) if self.f1.size else 0.0


def precision_recall_f1(cm: np.ndarray) -> PRF:
    """Per-class precision / recall / F1 from a confusion matrix.

    Classes with no predictions (or no samples) get 0.0, matching
    scikit-learn's ``zero_division=0``.
    """
    tp = np.diag(cm).astype(np.float64)
    pred_total = cm.sum(axis=0).astype(np.float64)
    true_total = cm.sum(axis=1).astype(np.float64)
    precision = _safe_div(tp, pred_total)
    recall = _safe_div(tp, true_total)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return PRF(precision, recall, f1, true_total.astype(np.int64))
