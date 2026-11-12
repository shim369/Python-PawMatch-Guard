"""Train / evaluate the species+breed classifier. Everything is driven by a YAML config."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.oxford_pet import DATASET_DIR, PetClassificationDataset, parse_list
from src.data.split import Entry, read_split_csv
from src.data.transforms import eval_transform, train_transform
from src.i18n import t
from src.metrics import accuracy, confusion_matrix, precision_recall_f1
from src.models.classifier import PetClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def class_weights(entries: list[Entry], num_classes: int = 37) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1."""
    counts = np.bincount([e.breed_id for e in entries], minlength=num_classes).astype(np.float64)
    w = counts.sum() / np.maximum(counts, 1)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def make_sampler(entries: list[Entry]) -> WeightedRandomSampler:
    w = class_weights(entries).numpy()
    sample_w = [float(w[e.breed_id]) for e in entries]
    return WeightedRandomSampler(sample_w, num_samples=len(entries), replacement=True)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-nn.functional.cross_entropy(logits, target, reduction="none"))
        return ((1 - pt) ** self.gamma * ce).mean()


def build_loss(kind: str, entries: list[Entry]) -> nn.Module:
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights(entries))
    if kind == "focal":
        return FocalLoss(weight=class_weights(entries))
    raise ValueError(f"unknown loss: {kind}")


@torch.no_grad()
def predict(model: PetClassifier, loader: DataLoader, device: torch.device):
    model.eval()
    ys, yb, ps, pb = [], [], [], []
    for x, species, breed in loader:
        sl, bl = model(x.to(device))
        ys.append(species.numpy())
        yb.append(breed.numpy())
        ps.append(sl.argmax(1).cpu().numpy())
        pb.append(bl.argmax(1).cpu().numpy())
    return (np.concatenate(a) for a in (ys, ps, yb, pb))


def summarize(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    prf = precision_recall_f1(cm)
    return {
        "accuracy": accuracy(cm),
        "macro_f1": prf.macro_f1,
        "per_class": {
            "precision": prf.precision.tolist(),
            "recall": prf.recall.tolist(),
            "f1": prf.f1.tolist(),
            "support": prf.support.tolist(),
        },
        "confusion_matrix": cm.tolist(),
    }


def run(config_path: Path, lang: str | None = None) -> Path:
    cfg = yaml.safe_load(config_path.read_text())
    set_seed(cfg["seed"])
    device = pick_device()
    root = Path(cfg["data_root"]) / DATASET_DIR
    images_dir = root / "images"
    splits = read_split_csv(Path(cfg["split_csv"]))
    _, breed_names = parse_list(root / "annotations" / "list.txt")

    size = cfg["image_size"]
    train_ds = PetClassificationDataset(splits["train"], images_dir, train_transform(size))
    val_ds = PetClassificationDataset(splits["val"], images_dir, eval_transform(size))
    test_ds = PetClassificationDataset(splits["test"], images_dir, eval_transform(size))
    kw = {"batch_size": cfg["batch_size"], "num_workers": cfg["num_workers"]}
    sampler = make_sampler(splits["train"]) if cfg["sampler"] == "weighted" else None
    train_loader = DataLoader(train_ds, shuffle=sampler is None, sampler=sampler, **kw)
    val_loader = DataLoader(val_ds, **kw)
    test_loader = DataLoader(test_ds, **kw)

    model = PetClassifier(num_breeds=len(breed_names)).to(device)
    breed_loss = build_loss(cfg["loss"], splits["train"])
    breed_loss.to(device)
    species_loss = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=1e-4
    )

    run_dir = Path(cfg["output_dir"]) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        model.freeze_backbone(epoch <= cfg["freeze_epochs"])
        model.train()
        total = 0.0
        for x, species, breed in train_loader:
            x, species, breed = x.to(device), species.to(device), breed.to(device)
            sl, bl = model(x)
            loss = species_loss(sl, species) + breed_loss(bl, breed)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        _, _, yb, pb = predict(model, val_loader, device)
        acc = float((yb == pb).mean())
        history.append({"epoch": epoch, "loss": total / len(train_ds), "val_breed_acc": acc})
        print(
            t("train.epoch", lang, epoch=epoch, epochs=cfg["epochs"],
              loss=history[-1]["loss"], acc=acc)
        )

    ys, ps, yb, pb = predict(model, test_loader, device)
    metrics = {
        "species": summarize(ys, ps, 2),
        "breed": summarize(yb, pb, len(breed_names)),
        "breed_names": breed_names,
        "history": history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    torch.save(model.state_dict(), run_dir / "model.pt")
    print(t("train.done", lang, path=run_dir))
    return run_dir
