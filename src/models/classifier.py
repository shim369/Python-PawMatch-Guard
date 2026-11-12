"""ResNet50 transfer-learning classifier with two heads: species (2) and breed (37)."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class PetClassifier(nn.Module):
    def __init__(self, num_breeds: int = 37, pretrained: bool = True, backbone: str = "resnet50"):
        super().__init__()
        if backbone != "resnet50":
            raise ValueError(f"unsupported backbone: {backbone}")
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        net = models.resnet50(weights=weights)
        feat = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.species_head = nn.Linear(feat, 2)
        self.breed_head = nn.Linear(feat, num_breeds)

    def freeze_backbone(self, frozen: bool = True) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = not frozen

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.backbone(x)
        return self.species_head(f), self.breed_head(f)
