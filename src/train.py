"""
Train — Training loop with mixed‑precision, early stopping, and history logging.

Central function :func:`train_model` handles a full training run for any
model, including:

* ``BCEWithLogitsLoss`` binary objective
* ``Adam`` optimiser with weight decay
* ``ReduceLROnPlateau`` learning‑rate scheduler
* **Mixed‑precision training** (AMP) when CUDA is available
* Per‑epoch metric tracking (train/val loss & accuracy, LR)
* Best‑model checkpointing & CSV history export
* Training‑curve plots (loss & accuracy)

WARNING: This is an educational prototype only — NOT for clinical use.
"""

import csv
import os
import time
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non‑interactive backend — safe for scripts / Colab
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    MODELS_DIR,
    NUM_EPOCHS,
    TRAINING_CURVES_DIR,
    USE_AMP,
    WEIGHT_DECAY,
    create_dirs,
)


# ────────────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────────────
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
    device: torch.device = DEVICE,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = EARLY_STOPPING_PATIENCE,
    lr_patience: int = LR_SCHEDULER_PATIENCE,
    lr_factor: float = LR_SCHEDULER_FACTOR,
    use_amp: bool = USE_AMP,
) -> Dict[str, List[Any]]:
    """Train a binary classifier end‑to‑end.

    Parameters
    ----------
    model : nn.Module
        The model to train (moved to *device* internally).
    train_loader, val_loader : DataLoader
        Training and validation data.
    model_name : str
        Used for checkpoint filenames and plot titles.
    device : torch.device
        Target device.
    num_epochs : int
        Maximum number of epochs.
    learning_rate : float
        Initial learning rate for Adam.
    weight_decay : float
        L2 regularisation strength.
    patience : int
        Number of epochs without val‑loss improvement before early stopping.
    lr_patience : int
        Scheduler patience (epochs).
    lr_factor : float
        Factor by which the LR is reduced on plateau.
    use_amp : bool
        Enable automatic mixed‑precision (only effective on CUDA).

    Returns
    -------
    dict
        Training history with keys: ``train_loss``, ``val_loss``,
        ``train_acc``, ``val_acc``, ``lr``, ``epoch``.
    """
    create_dirs()
    model = model.to(device)

    # ── loss / optimiser / scheduler ──────────────────────────────
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=lr_patience, factor=lr_factor
    )

    # ── mixed‑precision scaler (no‑op on CPU) ─────────────────────
    amp_enabled = use_amp and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    # ── history tracking ──────────────────────────────────────────
    history: Dict[str, List[Any]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_path = os.path.join(MODELS_DIR, f"{model_name}_best.pth")

    print(f"\n{'=' * 65}")
    print(f"  🏋️  Training: {model_name}")
    print(f"  Device: {device}  |  AMP: {amp_enabled}  |  "
          f"Epochs: {num_epochs}  |  Patience: {patience}")
    print(f"{'=' * 65}")

    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        # ── training phase ────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            train_loader, desc=f"  Epoch {epoch:02d}/{num_epochs} [train]",
            leave=False,
        )
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad()

            with autocast(enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / total
        train_acc = correct / total

        # ── validation phase ──────────────────────────────────────
        val_loss, val_acc = _evaluate_epoch(
            model, val_loader, criterion, device, amp_enabled,
        )

        # ── scheduler step ────────────────────────────────────────
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        # ── bookkeeping ───────────────────────────────────────────
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"  Epoch {epoch:02d}  │  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  │  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  │  "
            f"LR: {current_lr:.2e}"
        )

        # ── checkpointing ────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "model_name": model_name,
            }, best_model_path)
        else:
            epochs_without_improvement += 1

        # ── early stopping ────────────────────────────────────────
        if epochs_without_improvement >= patience:
            print(f"\n  ⏹  Early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    elapsed = time.time() - t_start
    print(f"\n  ⏱  Training completed in {elapsed / 60:.1f} min")
    print(f"  💾  Best model saved → {best_model_path}")

    # ── save history CSV ──────────────────────────────────────────
    csv_path = os.path.join(MODELS_DIR, f"{model_name}_history.csv")
    _save_history_csv(history, csv_path)
    print(f"  📄  History CSV → {csv_path}")

    # ── plot curves ───────────────────────────────────────────────
    _plot_training_curves(history, model_name)

    return history


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> tuple:
    """Compute loss and accuracy on an entire DataLoader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        with autocast(enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def _save_history_csv(history: Dict[str, List], path: str) -> None:
    """Write training history to a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(history.keys())
        writer.writerows(zip(*history.values()))


def _plot_training_curves(history: Dict[str, List], model_name: str) -> None:
    """Save loss and accuracy curves as PNG files."""
    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── loss curve ────────────────────────────────────────────────
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "s-", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} — Loss Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── accuracy curve ────────────────────────────────────────────
    axes[1].plot(epochs, history["train_acc"], "o-", label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], "s-", label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{model_name} — Accuracy Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(TRAINING_CURVES_DIR, f"{model_name}_curves.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📈  Curves → {save_path}")
