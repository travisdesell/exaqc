# src/models/ResNetImageEncoder.py

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
)


class ResNetImageEncoder(nn.Module):
    """Pretrained ResNet encoder for quantum encoding."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        model_name: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        activation: str = "sigmoid",
    ):
        super().__init__()

        if model_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
            feat_dim = backbone.fc.in_features

        elif model_name == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            backbone = resnet34(weights=weights)
            feat_dim = backbone.fc.in_features

        elif model_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            backbone = resnet50(weights=weights)
            feat_dim = backbone.fc.in_features

        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")

        backbone.fc = nn.Identity()

        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False

        self.backbone = backbone
        self.proj = nn.Linear(feat_dim, embedding_dim)

        if activation == "sigmoid":
            self.out_activation = nn.Sigmoid()
        elif activation == "tanh":
            self.out_activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation={activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return quantum-compatible embedding."""
        single = x.dim() == 3

        if single:
            x = x.unsqueeze(0)

        # MNIST/Fashion-MNIST: convert 1-channel to 3-channel for ResNet
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # ResNet expects ImageNet-style spatial size.
        x = nn.functional.interpolate(
            x,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        features = self.backbone(x)
        z = self.proj(features)
        z = self.out_activation(z)

        return z.squeeze(0) if single else z
