"""Moderation rules: model outputs -> publish / review / reject.

Pure functions with no model or I/O dependency, so every rule is unit-testable.
Because a miss is costly, anything uncertain goes to a human (``REVIEW``).

Rules, in priority order (``reject > review > publish``):
 1. No animal detected                          -> REJECT
    (an animal detected with low score          -> REVIEW)
 2. Face / plate / text detected                -> blur them, then PUBLISH;
    if any such detection has a low score       -> REVIEW
 3. Classified species differs from the post    -> REVIEW
 4. Otherwise                                   -> PUBLISH
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import yaml


class Verdict(IntEnum):
    """Ordered by severity so ``max`` picks the worst verdict (used for video)."""

    PUBLISH = 0
    REVIEW = 1
    REJECT = 2


PII_LABELS = ("face", "plate", "text")


@dataclass(frozen=True)
class Detection:
    label: str  # "animal" | "face" | "plate" | "text"
    score: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass(frozen=True)
class Thresholds:
    animal_reject: float = 0.30  # best animal score below this -> no animal
    animal_ok: float = 0.50  # below this (but >= reject) -> uncertain -> review
    pii_blur_min: float = 0.30  # PII detections at/above this are blurred
    pii_confident: float = 0.60  # a blurred PII detection below this -> review

    @classmethod
    def from_yaml(cls, path: Path) -> Thresholds:
        return cls(**yaml.safe_load(path.read_text()))


@dataclass(frozen=True)
class Prediction:
    detections: tuple[Detection, ...]
    predicted_species: str | None = None  # "cat" | "dog" from the classifier
    species_confidence: float = 1.0


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    reasons: tuple[str, ...] = ()
    blur_boxes: tuple[tuple[float, float, float, float], ...] = ()
    risk: float = 0.0  # tie-breaker among equal verdicts (higher = riskier)
    extra: dict = field(default_factory=dict, compare=False)


def decide(pred: Prediction, declared_species: str | None,
           th: Thresholds = Thresholds()) -> Decision:
    reasons: list[str] = []
    verdict = Verdict.PUBLISH

    animals = [d for d in pred.detections if d.label == "animal"]
    best = max((d.score for d in animals), default=0.0)
    if best < th.animal_reject:
        return Decision(Verdict.REJECT, ("no_animal",), risk=1.0 - best,
                        extra={"species": pred.predicted_species})
    if best < th.animal_ok:
        verdict = Verdict.REVIEW
        reasons.append("uncertain_animal")

    pii = [d for d in pred.detections if d.label in PII_LABELS and d.score >= th.pii_blur_min]
    blur = tuple(d.box for d in pii)
    if pii:
        reasons.append("pii_blurred")
        if min(d.score for d in pii) < th.pii_confident:
            verdict = max(verdict, Verdict.REVIEW)
            reasons.append("pii_low_confidence")

    if (declared_species and pred.predicted_species
            and declared_species != pred.predicted_species):
        verdict = max(verdict, Verdict.REVIEW)
        reasons.append("species_mismatch")

    risk = (1.0 - best) + (0.5 if "species_mismatch" in reasons else 0.0) \
        + (0.25 if "pii_low_confidence" in reasons else 0.0)
    return Decision(verdict, tuple(reasons), blur, risk, {"species": pred.predicted_species})


def worst(decisions: list[Decision]) -> Decision:
    """The most dangerous decision of a set (video: worst frame wins).

    Blur boxes are not merged across frames; callers blur per frame.
    """
    if not decisions:
        raise ValueError("no decisions")
    return max(decisions, key=lambda d: (d.verdict, d.risk))
