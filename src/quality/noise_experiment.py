"""Run the 5% label-noise experiment with a trained classifier on held-out splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.oxford_pet import DATASET_DIR, PetClassificationDataset
from src.data.split import read_split_csv
from src.data.transforms import eval_transform
from src.models.classifier import PetClassifier
from src.models.train_classifier import pick_device
from src.quality.noisy_labels import detect_noisy, evaluate_detection, inject_label_noise


@torch.no_grad()
def breed_probs(run_dir: Path, splits: tuple[str, ...] = ("val", "test")):
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    device = pick_device()
    model = PetClassifier(pretrained=False).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.eval()
    entries = [e for s in splits for e in read_split_csv(Path(cfg["split_csv"]))[s]]
    ds = PetClassificationDataset(
        entries, Path(cfg["data_root"]) / DATASET_DIR / "images", eval_transform(cfg["image_size"])
    )
    probs = []
    for x, _, _ in DataLoader(ds, batch_size=64, num_workers=2):
        probs.append(torch.softmax(model(x.to(device))[1], 1).cpu().numpy())
    return np.concatenate(probs), np.array([e.breed_id for e in entries])


def run(run_dir: Path, rate: float = 0.05, seed: int = 42) -> dict:
    probs, labels = breed_probs(run_dir)
    noisy, flipped = inject_label_noise(labels, probs.shape[1], rate, seed)
    out = {"n": len(labels), "noise_rate": rate, "n_noisy": len(flipped), "seed": seed}
    for name, flagged in {
        "top_loss": detect_noisy(probs, noisy, rate, "loss"),
        "disagree": detect_noisy(probs, noisy, method="disagree"),
    }.items():
        r = evaluate_detection(flagged, flipped)
        out[name] = {"flagged": r.flagged, "caught": r.caught,
                     "recall": r.recall, "precision": r.precision}
    # Baseline: how many *clean* labels the model itself already disagrees with.
    out["model_error_rate_on_clean"] = float((probs.argmax(1) != labels).mean())
    return out


def main(run_dir: str, out_path: str) -> None:
    res = run(Path(run_dir))
    Path(out_path).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
