"""Torch implementations of the Detector / SpeciesClassifier protocols."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.transforms import eval_transform
from src.models import detection as det_model
from src.models.classifier import PetClassifier
from src.models.train_classifier import pick_device

from .pipeline import ModerationPipeline
from .rules import Detection, Thresholds

SPECIES = ("cat", "dog")


class TorchSpeciesClassifier:
    def __init__(self, run_dir: Path, device: torch.device) -> None:
        self.device = device
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        self.tf = eval_transform(cfg["image_size"])
        self.model = PetClassifier(pretrained=False).to(device)
        self.model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> tuple[str, float]:
        from PIL import Image

        x = self.tf(Image.fromarray(image)).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x)[0], 1)[0]
        i = int(probs.argmax())
        return SPECIES[i], float(probs[i])


class TorchAnimalDetector:
    """Animal-head detector (Phase 2). Face / plate / text classes are not trained yet, so
    this returns only ``animal`` detections; see docs/annotation_guidelines.md."""

    def __init__(self, run_dir: Path, device: torch.device) -> None:
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        self.device = device
        self.model = det_model.build_model(pretrained=False, min_size=cfg["min_size"]).to(device)
        self.model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
        self.model.eval()

    @torch.no_grad()
    def detect(self, image: np.ndarray) -> list[Detection]:
        x = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255).to(self.device)
        out = self.model([x])[0]
        return [Detection("animal", float(s), tuple(float(v) for v in b))
                for b, s in zip(out["boxes"].cpu(), out["scores"].cpu(), strict=True)]


def load_pipeline(config_path: Path = Path("configs/pipeline.yaml")) -> ModerationPipeline:
    cfg = yaml.safe_load(config_path.read_text())
    device = pick_device()
    th = Thresholds.from_yaml(Path(cfg["thresholds"]))
    return ModerationPipeline(
        TorchAnimalDetector(Path(cfg["detector_run"]), device),
        TorchSpeciesClassifier(Path(cfg["classifier_run"]), device),
        th,
    )
