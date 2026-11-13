"""DeepLabV3 (MobileNetV3-Large backbone) for animal / background segmentation."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

from src.data.oxford_pet import DATASET_DIR
from src.data.segmentation import PetSegmentationDataset
from src.data.split import read_split_csv
from src.metrics.segmentation import boundary_iou, dice, iou
from src.models.train_classifier import pick_device, set_seed


def build_model(pretrained: bool = True) -> nn.Module:
    # ImageNet-pretrained backbone; the segmentation head is trained from scratch.
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    return deeplabv3_mobilenet_v3_large(
        weights=None, weights_backbone=weights, num_classes=2, aux_loss=False
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    ious, dices, bious = [], [], []
    for x, y in loader:
        pred = model(x.to(device))["out"].argmax(1).cpu().numpy()
        for p, t in zip(pred, y.numpy(), strict=True):
            ious.append(iou(p, t))
            dices.append(dice(p, t))
            bious.append(boundary_iou(p, t))
    return {"iou": float(np.mean(ious)), "dice": float(np.mean(dices)),
            "boundary_iou": float(np.mean(bious)), "n": len(ious)}


def run(config_path: Path) -> Path:
    cfg = yaml.safe_load(config_path.read_text())
    set_seed(cfg["seed"])
    device = pick_device()
    root = Path(cfg["data_root"]) / DATASET_DIR
    splits = read_split_csv(Path(cfg["split_csv"]))
    bug, size = cfg.get("bug", "none"), cfg["image_size"]
    train_ds = PetSegmentationDataset(splits["train"], root, size, train=True, bug=bug)
    # ``linear_mask`` corrupts only the *training labels*; evaluation uses the clean masks.
    # ``bgr`` / ``double_norm`` corrupt the input pipeline, which train and eval share.
    eval_bug = "none" if bug == "linear_mask" else bug
    mk = lambda s: PetSegmentationDataset(splits[s], root, size, bug=eval_bug)  # noqa: E731
    val_ds, test_ds = mk("val"), mk("test")
    if cfg.get("max_train"):
        train_ds = Subset(train_ds, range(cfg["max_train"]))
    kw = {"batch_size": cfg["batch_size"], "num_workers": cfg["num_workers"]}
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **kw)
    val_loader, test_loader = DataLoader(val_ds, **kw), DataLoader(test_ds, **kw)

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    run_dir = Path(cfg["output_dir"]) / f"seg-{bug}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    history, t0 = [], time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x)["out"], y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        val = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "loss": total / n,
                        **{f"val_{k}": v for k, v in val.items()}})
        print(f"epoch {epoch}/{cfg['epochs']} loss={history[-1]['loss']:.4f} "
              f"val_iou={val['iou']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    test = evaluate(model, test_loader, device)
    (run_dir / "metrics.json").write_text(json.dumps({"test": test, "history": history,
                                                      "bug": bug}, indent=2))
    torch.save(model.state_dict(), run_dir / "model.pt")
    print("test:", test, flush=True)
    return run_dir
