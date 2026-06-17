"""
MobileNet V2 — Lightweight transfer‑learning model for malaria detection.

Uses a pre‑trained MobileNetV2 backbone from *torchvision*.  The classifier
is replaced with a compact two‑layer head suited for binary output.

WARNING: This is an educational prototype only — NOT for clinical use.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


class MobileNetV2Model(nn.Module):
    """MobileNetV2 fine‑tuned for binary malaria classification.

    Parameters
    ----------
    num_classes : int
        Output units (1 for BCEWithLogitsLoss).
    pretrained : bool
        Load ImageNet‑pretrained weights.
    freeze_backbone : bool
        If *True*, freeze the feature‑extraction layers; only the custom
        classifier head will be trained.
    """

    def __init__(
        self,
        num_classes: int = 1,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # ── load backbone ─────────────────────────────────────────
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)

        # ── optionally freeze feature layers ──────────────────────
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # ── replace classifier head ───────────────────────────────
        # Original: Dropout(0.2) → Linear(1280, 1000)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        # Ensure classifier is always trainable
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    # ── forward pass ──────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape ``(B, 3, H, W)``

        Returns
        -------
        Tensor, shape ``(B, 1)`` — raw logits.
        """
        return self.backbone(x)

    # ── Grad‑CAM helpers ──────────────────────────────────────────
    def get_last_conv_layer(self) -> nn.Module:
        """Return the last convolutional block (``features[-1]``)."""
        return self.backbone.features[-1]

    @staticmethod
    def get_last_conv_layer_name() -> str:
        """Return the dotted attribute path to the Grad‑CAM target layer."""
        return "backbone.features"
