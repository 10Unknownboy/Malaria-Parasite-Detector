"""
Simple CNN — A lightweight 4‑block convolutional network for malaria detection.

Architecture
------------
4 × (Conv2d → BatchNorm2d → ReLU → MaxPool2d)
→ Global Average Pooling
→ FC(256, 128) → ReLU → Dropout(0.5) → FC(128, 1)

Designed as the baseline model.  No pre‑trained weights — trains from scratch.

WARNING: This is an educational prototype only — NOT for clinical use.
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Four‑block CNN for binary cell‑image classification.

    Parameters
    ----------
    num_classes : int
        Number of output units.  Use **1** for binary classification with
        ``BCEWithLogitsLoss``.
    """

    def __init__(self, num_classes=1):
        super().__init__()

        # ── convolutional backbone ────────────────────────────────
        self.features = nn.Sequential(
            # Block 1: 3 → 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 128 → 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ── global pooling ────────────────────────────────────────
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ── classifier head ───────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    # ── forward pass ──────────────────────────────────────────────
    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor, shape ``(B, 3, H, W)``

        Returns
        -------
        Tensor, shape ``(B, 1)`` — raw logits (apply sigmoid externally).
        """
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # ── Grad‑CAM helpers ──────────────────────────────────────────
    def get_last_conv_layer(self):
        """Return the last convolutional layer for Grad‑CAM.

        This is the ``Conv2d`` in Block 4 (``self.features[12]``).
        """
        # Block 4 starts at index 12 (Conv), 13 (BN), 14 (ReLU), 15 (Pool)
        return self.features[12]

    @staticmethod
    def get_last_conv_layer_name():
        """Return the dotted attribute path to the target layer."""
        return "features.12"
