"""
ResNet‑18 — Transfer‑learning wrapper for malaria cell classification.

Uses a pre‑trained ResNet‑18 backbone from *torchvision*.  The final
fully‑connected layer is replaced with a two‑layer head for binary output.

WARNING: This is an educational prototype only — NOT for clinical use.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class ResNet18Model(nn.Module):
    """ResNet‑18 fine‑tuned for binary malaria classification.

    Parameters
    ----------
    num_classes : int
        Output units (1 for BCEWithLogitsLoss).
    pretrained : bool
        Load ImageNet‑pretrained weights.
    freeze_backbone : bool
        If *True*, freeze **all** layers except the custom classifier head.
        Useful for fast feature‑extraction training.
    """

    def __init__(
        self,
        num_classes=1,
        pretrained=True,
        freeze_backbone=False,
    ):
        super().__init__()

        # ── load backbone ─────────────────────────────────────────
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # ── optionally freeze backbone ────────────────────────────
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ── replace final FC layer ────────────────────────────────
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Linear(in_features, 256)
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        # Ensure classifier is always trainable
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    # ── forward pass ──────────────────────────────────────────────
    def forward(self, x):
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
    def get_last_conv_layer(self):
        """Return the last convolutional block (``layer4[-1]``)."""
        return self.backbone.layer4[-1]

    @staticmethod
    def get_last_conv_layer_name():
        """Return the dotted attribute path to the Grad‑CAM target layer."""
        return "backbone.layer4"











