"""Moderation pipeline: models -> rules, for images and videos.

The models are injected through two small protocols so the pipeline can be tested with fakes
and so a detector with face/plate/text classes can be dropped in later without changes here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from .blur import blur_regions
from .rules import Decision, Detection, Prediction, Thresholds, Verdict, decide, worst
from .video import extract_frames, pick_thumbnail


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]:
        """image: (H, W, 3) uint8 RGB -> detections (labels: animal/face/plate/text)."""


class SpeciesClassifier(Protocol):
    def predict(self, image: np.ndarray) -> tuple[str, float]:
        """image: RGB uint8 -> ("cat" | "dog", confidence)."""


@dataclass
class ModerationResult:
    decision: Decision
    frame_decisions: list[Decision] = field(default_factory=list)  # videos only
    thumbnail: np.ndarray | None = None  # blurred, safest frame (videos) / blurred image


class ModerationPipeline:
    def __init__(self, detector: Detector, classifier: SpeciesClassifier,
                 thresholds: Thresholds | None = None) -> None:
        self.detector, self.classifier = detector, classifier
        self.th = thresholds or Thresholds()

    def analyze(self, image: np.ndarray, declared_species: str | None) -> Decision:
        species, conf = self.classifier.predict(image)
        pred = Prediction(tuple(self.detector.detect(image)), species, conf)
        return decide(pred, declared_species, self.th)

    def moderate_image(self, image: np.ndarray, declared_species: str | None
                       ) -> tuple[ModerationResult, np.ndarray]:
        """Returns the result and the image to publish (PII blurred)."""
        decision = self.analyze(image, declared_species)
        blurred = blur_regions(image, decision.blur_boxes) if decision.blur_boxes else image
        return ModerationResult(decision, thumbnail=blurred), blurred

    def moderate_video(self, video: Path, work_dir: Path, declared_species: str | None,
                       fps: float = 1.0) -> ModerationResult:
        """One frame per second; the most dangerous frame decides the verdict."""
        frames = extract_frames(video, work_dir, fps)
        if not frames:
            raise RuntimeError("no frames could be extracted")
        decisions, images = [], []
        for f in frames:
            img = np.asarray(Image.open(f).convert("RGB"))
            d = self.analyze(img, declared_species)
            decisions.append(d)
            images.append(blur_regions(img, d.blur_boxes) if d.blur_boxes else img)
        thumb = images[frames.index(pick_thumbnail(frames, [d.risk for d in decisions]))]
        final = worst(decisions)
        # Only the thumbnail is blurred, not the video itself: a video that shows personal
        # information must not be auto-published.
        if final.verdict is Verdict.PUBLISH and any(d.blur_boxes for d in decisions):
            final = Decision(Verdict.REVIEW, (*final.reasons, "video_pii_unblurred"),
                             final.blur_boxes, final.risk)
        return ModerationResult(final, decisions, thumb)
