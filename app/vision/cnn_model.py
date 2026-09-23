"""Convolutional neural network for product-damage / defect classification.

A compact ResNet-style CNN (not a pretrained import) so the module has zero
external weight dependencies and can be trained from scratch on a customer's
own labeled return-photo dataset. For production accuracy, swap
`ProductDefectCNN` for a fine-tuned backbone (e.g. EfficientNet-B0 /
ResNet-18 via `torchvision.models`) behind the same `DefectClassifierService`
interface — nothing else in the pipeline needs to change.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFECT_CLASSES: list[str] = [
    "no_defect",
    "surface_scratch",
    "crack_or_fracture",
    "discoloration_or_stain",
    "missing_component",
    "packaging_damage",
]


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class ProductDefectCNN(nn.Module):
    """Input: (B, 3, 224, 224) RGB product photo. Output: class logits."""

    def __init__(self, num_classes: int = len(DEFECT_CLASSES)) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = nn.Sequential(_ResidualBlock(32), _ResidualBlock(32))
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(_ResidualBlock(64), _ResidualBlock(64))
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(_ResidualBlock(128), _ResidualBlock(128))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
