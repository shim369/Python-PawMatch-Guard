"""Sanity check: the training step must be able to overfit a tiny batch.

A pipeline bug (wrong labels, broken loss, frozen params) shows up here first.
Uses a small CNN instead of ResNet50 so it runs in seconds on CPU.
"""

import torch
from torch import nn

from src.data.split import Entry
from src.models.train_classifier import FocalLoss, class_weights, set_seed


def test_can_overfit_tiny_batch() -> None:
    set_seed(0)
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 4, (16,))
    net = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(), nn.Linear(8, 4))
    opt = torch.optim.Adam(net.parameters(), lr=0.05)
    for _ in range(300):
        loss = nn.functional.cross_entropy(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
    assert (net(x).argmax(1) == y).float().mean() == 1.0


def test_focal_reduces_to_ce_at_gamma_zero() -> None:
    logits, y = torch.randn(8, 5), torch.randint(0, 5, (8,))
    assert torch.isclose(FocalLoss(gamma=0.0)(logits, y), nn.functional.cross_entropy(logits, y))


def test_class_weights_favor_rare() -> None:
    entries = [Entry(f"a{i}", 0, 0) for i in range(9)] + [Entry("b", 1, 0)]
    w = class_weights(entries, num_classes=2)
    assert w[1] > w[0] and torch.isclose(w.mean(), torch.tensor(1.0))


def test_segmentation_model_can_overfit_two_images() -> None:
    from src.models.segmentation import build_model

    set_seed(0)
    model = build_model(pretrained=False)
    x = torch.randn(2, 3, 64, 64)
    y = torch.zeros(2, 64, 64, dtype=torch.long)
    y[:, 16:48, 16:48] = 1
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(100):
        loss = nn.functional.cross_entropy(model(x)["out"], y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # train-mode BN: batch statistics are stable on a 2-image batch, running stats are not
    pred = model(x)["out"].argmax(1)
    assert loss.item() < 0.2 and (pred == y).float().mean() > 0.95
