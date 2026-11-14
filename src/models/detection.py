"""Faster R-CNN (MobileNetV3-Large FPN, COCO-pretrained) fine-tuned for animal heads."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.data.detection import ANIMAL_HEAD, PetDetectionDataset, with_boxes
from src.data.oxford_pet import DATASET_DIR
from src.data.split import read_split_csv
from src.metrics.detection import ImageDetections, average_precision, class_recall, mean_ap
from src.models.train_classifier import pick_device, set_seed


def build_model(pretrained: bool = True, min_size: int = 384, max_size: int = 640):
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1 if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights, weights_backbone=None, min_size=min_size, max_size=max_size
    )
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, 2)  # background + animal_head
    return model


def collate(batch):
    return tuple(zip(*batch, strict=True))


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    images: list[ImageDetections] = []
    for xs, targets in loader:
        outs = model([x.to(device) for x in xs])
        for o, t in zip(outs, targets, strict=True):
            images.append(ImageDetections(
                o["boxes"].cpu().numpy(), o["scores"].cpu().numpy(), o["labels"].cpu().numpy(),
                t["boxes"].numpy(), t["labels"].numpy()))
    cls = [ANIMAL_HEAD]
    hit = [
        float(any(p >= 0.5 for p in im.pred_scores[im.pred_labels == ANIMAL_HEAD]))
        for im in images
    ]
    return {
        "ap50": average_precision(images, ANIMAL_HEAD, 0.5),
        "map_50_95": mean_ap(images, cls),
        "recall_iou50": class_recall(images, ANIMAL_HEAD, 0.5),
        # share of images where at least one box has score >= 0.5 (rule 1: "no animal -> reject")
        "image_detect_rate_at_score_0.5": float(np.mean(hit)),
        "n": len(images),
    }


def run(config_path: Path) -> Path:
    cfg = yaml.safe_load(config_path.read_text())
    set_seed(cfg["seed"])
    device = pick_device()
    root = Path(cfg["data_root"]) / DATASET_DIR
    splits = {k: with_boxes(v, root) for k, v in read_split_csv(Path(cfg["split_csv"])).items()}
    sizes = {k: len(v) for k, v in splits.items()}
    print("images with boxes:", sizes, flush=True)
    kw = {"batch_size": cfg["batch_size"], "num_workers": cfg["num_workers"],
          "collate_fn": collate}
    train_loader = DataLoader(PetDetectionDataset(splits["train"], root, train=True),
                              shuffle=True, **kw)
    val_loader = DataLoader(PetDetectionDataset(splits["val"], root), **kw)
    test_loader = DataLoader(PetDetectionDataset(splits["test"], root), **kw)

    model = build_model(min_size=cfg["min_size"]).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=1e-4)
    run_dir = Path(cfg["output_dir"]) / f"det-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))

    history, t0 = [], time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total, n = 0.0, 0
        for xs, targets in train_loader:
            xs = [x.to(device) for x in xs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(xs, targets).values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xs)
            n += len(xs)
        val = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "loss": total / n,
                        **{f"val_{k}": v for k, v in val.items()}})
        print(f"epoch {epoch}/{cfg['epochs']} loss={total / n:.4f} val_ap50={val['ap50']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    test = evaluate(model, test_loader, device)
    (run_dir / "metrics.json").write_text(
        json.dumps({"test": test, "history": history, "split_sizes": sizes}, indent=2))
    torch.save(model.state_dict(), run_dir / "model.pt")
    print("test:", test, flush=True)
    return run_dir
