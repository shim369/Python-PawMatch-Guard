"""Noisy-label injection and detection.

Detection signal: per-sample loss of the *given* label under a model that did not train on
the sample (e.g. a held-out split). High loss / disagreement with the prediction => suspect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def inject_label_noise(labels: np.ndarray, num_classes: int, rate: float = 0.05,
                       seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Swap ``rate`` of the labels to a different class. Returns (noisy_labels, flipped_idx)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).copy()
    n_flip = int(round(len(labels) * rate))
    idx = rng.choice(len(labels), size=n_flip, replace=False)
    shift = rng.integers(1, num_classes, size=n_flip)  # never 0 -> always a different class
    labels[idx] = (labels[idx] + shift) % num_classes
    return labels, np.sort(idx)


def label_loss(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Cross-entropy of the given label for each sample."""
    p = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return -np.log(p)


@dataclass(frozen=True)
class DetectionReport:
    flagged: int
    true_noisy: int
    caught: int

    @property
    def recall(self) -> float:  # share of injected noise that was flagged
        return self.caught / self.true_noisy if self.true_noisy else 0.0

    @property
    def precision(self) -> float:  # share of flagged samples that were really noisy
        return self.caught / self.flagged if self.flagged else 0.0


def detect_noisy(probs: np.ndarray, labels: np.ndarray, top_fraction: float = 0.05,
                 method: str = "loss") -> np.ndarray:
    """Indices flagged as suspicious.

    ``loss``: the top ``top_fraction`` samples by loss of the given label.
    ``disagree``: every sample whose argmax prediction differs from the label.
    """
    if method == "disagree":
        return np.nonzero(probs.argmax(1) != labels)[0]
    k = int(round(len(labels) * top_fraction))
    return np.sort(np.argsort(-label_loss(probs, labels))[:k])


def evaluate_detection(flagged: np.ndarray, true_noisy: np.ndarray) -> DetectionReport:
    caught = len(np.intersect1d(flagged, true_noisy))
    return DetectionReport(len(flagged), len(true_noisy), caught)
