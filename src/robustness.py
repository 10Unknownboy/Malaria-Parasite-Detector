"""
Robustness — Test model resilience to Gaussian blur, noise, and data reduction.

This module evaluates how gracefully a trained model degrades under:

* **Gaussian blur** — simulating out‑of‑focus microscopy.
* **Gaussian noise** — simulating sensor / acquisition noise.
* **Reduced training data** — measuring sample‑efficiency.

All results are saved as JSON + bar‑chart PNGs under ``results/robustness/``.

WARNING: This is an educational prototype only — NOT for clinical use.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    DEVICE,
    IMAGE_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    ROBUSTNESS_DIR,
    SEED,
    WEIGHT_DECAY,
    create_dirs,
    set_seed,
)


# ────────────────────────────────────────────────────────────────────
# Perturbation primitives
# ────────────────────────────────────────────────────────────────────
def apply_gaussian_blur(
    image_tensor: torch.Tensor,
    kernel_size: int = 5,
) -> torch.Tensor:
    """Apply Gaussian blur to an image tensor.

    Parameters
    ----------
    image_tensor : Tensor, shape ``(C, H, W)`` or ``(B, C, H, W)``
        Image(s) to blur.
    kernel_size : int
        Must be **odd**.  Larger → more blur.

    Returns
    -------
    Tensor — same shape as input.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # force odd

    was_3d = image_tensor.dim() == 3
    if was_3d:
        image_tensor = image_tensor.unsqueeze(0)  # add batch dim

    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    channels = image_tensor.shape[1]

    # 1‑D Gaussian kernel
    x = torch.arange(kernel_size, dtype=torch.float32, device=image_tensor.device)
    x = x - kernel_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()

    # separable 2‑D kernel
    kernel = gauss.outer(gauss)
    kernel = kernel.expand(channels, 1, kernel_size, kernel_size)

    pad = kernel_size // 2
    blurred = F.conv2d(image_tensor, kernel, padding=pad, groups=channels)

    return blurred.squeeze(0) if was_3d else blurred


def apply_gaussian_noise(
    image_tensor: torch.Tensor,
    sigma: float = 0.05,
) -> torch.Tensor:
    """Add zero‑mean Gaussian noise to an image tensor.

    Parameters
    ----------
    image_tensor : Tensor, shape ``(C, H, W)`` or ``(B, C, H, W)``
    sigma : float
        Standard deviation of the noise.

    Returns
    -------
    Tensor — same shape, **not** clamped (caller decides).
    """
    noise = torch.randn_like(image_tensor) * sigma
    return image_tensor + noise


# ────────────────────────────────────────────────────────────────────
# Evaluation under perturbation
# ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _evaluate_perturbed(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    perturbation_fn,
) -> Dict[str, float]:
    """Run inference with a perturbation applied to each batch."""
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = perturbation_fn(images)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images).squeeze(1)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {"accuracy": correct / total if total > 0 else 0.0}


def test_robustness_blur(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device = DEVICE,
    kernel_sizes: Optional[List[int]] = None,
) -> Dict[int, Dict[str, float]]:
    """Test model accuracy under increasing Gaussian blur.

    Returns
    -------
    dict[int, dict]
        ``{kernel_size: {"accuracy": float}}``
    """
    if kernel_sizes is None:
        kernel_sizes = [3, 5, 7, 9]

    results: Dict[int, Dict[str, float]] = {}
    for ks in kernel_sizes:
        fn = lambda imgs, _ks=ks: apply_gaussian_blur(imgs, kernel_size=_ks)
        metrics = _evaluate_perturbed(model, test_loader, device, fn)
        results[ks] = metrics
        print(f"    Blur k={ks:2d}  →  Acc: {metrics['accuracy']:.4f}")
    return results


def test_robustness_noise(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device = DEVICE,
    sigmas: Optional[List[float]] = None,
) -> Dict[float, Dict[str, float]]:
    """Test model accuracy under increasing Gaussian noise.

    Returns
    -------
    dict[float, dict]
        ``{sigma: {"accuracy": float}}``
    """
    if sigmas is None:
        sigmas = [0.01, 0.05, 0.1, 0.2]

    results: Dict[float, Dict[str, float]] = {}
    for sigma in sigmas:
        fn = lambda imgs, _s=sigma: apply_gaussian_noise(imgs, sigma=_s)
        metrics = _evaluate_perturbed(model, test_loader, device, fn)
        results[sigma] = metrics
        print(f"    Noise σ={sigma:.2f}  →  Acc: {metrics['accuracy']:.4f}")
    return results


# ────────────────────────────────────────────────────────────────────
# Reduced‑data experiment
# ────────────────────────────────────────────────────────────────────
def test_reduced_data(
    model_class: type,
    model_name: str,
    full_train_dataset,
    val_loader: DataLoader,
    test_loader: DataLoader,
    fractions: Optional[List[float]] = None,
    device: torch.device = DEVICE,
    num_epochs: int = 10,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    batch_size: int = BATCH_SIZE,
) -> Dict[float, Dict[str, float]]:
    """Retrain the model on progressively smaller subsets.

    Parameters
    ----------
    model_class : type
        Model class (not instance) — a fresh model is created for each run.
    model_name : str
        Descriptive name for logging.
    full_train_dataset : Dataset
        Complete training dataset.
    val_loader, test_loader : DataLoader
        Validation and test loaders (unchanged across runs).
    fractions : list[float]
        Fractions of the full training set to use.

    Returns
    -------
    dict[float, dict]
        ``{fraction: {"accuracy": float}}``
    """
    # Import here to avoid circular dependency
    from src.train import train_model

    if fractions is None:
        fractions = [1.0, 0.5, 0.25, 0.1]

    results: Dict[float, Dict[str, float]] = {}
    n_total = len(full_train_dataset)

    for frac in fractions:
        set_seed(SEED)
        n_subset = max(1, int(n_total * frac))
        indices = np.random.permutation(n_total)[:n_subset]
        subset = Subset(full_train_dataset, indices.tolist())
        sub_loader = DataLoader(
            subset, batch_size=batch_size, shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )

        model = model_class()
        print(f"\n    Training with {frac:.0%} data ({n_subset:,} samples)")
        history = train_model(
            model, sub_loader, val_loader,
            model_name=f"{model_name}_frac{frac}",
            device=device, num_epochs=num_epochs,
            learning_rate=learning_rate, weight_decay=weight_decay,
        )

        # Evaluate on test set
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                preds = (torch.sigmoid(model(images).squeeze(1)) >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total if total > 0 else 0.0
        results[frac] = {"accuracy": acc, "n_samples": n_subset}
        print(f"    Fraction {frac:.0%}  →  Test Acc: {acc:.4f}")

    return results


# ────────────────────────────────────────────────────────────────────
# All‑in‑one robustness suite
# ────────────────────────────────────────────────────────────────────
def run_all_robustness(
    model: nn.Module,
    model_name: str,
    test_loader: DataLoader,
    device: torch.device = DEVICE,
) -> Dict[str, Any]:
    """Run blur and noise robustness tests and save results + plots.

    Returns
    -------
    dict
        ``{"blur": {...}, "noise": {...}}``.
    """
    create_dirs()

    print(f"\n{'=' * 55}")
    print(f"    Robustness Tests — {model_name}")
    print(f"{'=' * 55}")

    print("\n  ─── Gaussian Blur ───")
    blur_results = test_robustness_blur(model, test_loader, device)

    print("\n  ─── Gaussian Noise ───")
    noise_results = test_robustness_noise(model, test_loader, device)

    combined = {"blur": blur_results, "noise": noise_results}

    # ── save JSON ─────────────────────────────────────────────────
    json_path = os.path.join(ROBUSTNESS_DIR, f"{model_name}_robustness.json")
    # Convert keys to strings for JSON serialisation
    serialisable = {
        "blur": {str(k): v for k, v in blur_results.items()},
        "noise": {str(k): v for k, v in noise_results.items()},
    }
    with open(json_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\n    Results JSON → {json_path}")

    # ── bar charts ────────────────────────────────────────────────
    _plot_robustness_bar(blur_results, "Gaussian Blur (kernel size)", model_name, "blur")
    _plot_robustness_bar(noise_results, "Gaussian Noise (σ)", model_name, "noise")

    return combined


def _plot_robustness_bar(
    results: Dict,
    xlabel: str,
    model_name: str,
    test_type: str,
) -> None:
    """Save a bar chart for a robustness experiment."""
    labels = [str(k) for k in results.keys()]
    accs = [v["accuracy"] for v in results.values()]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, accs, color="steelblue", edgecolor="black")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{model_name} — Robustness: {xlabel}")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{acc:.3f}", ha="center", va="bottom", fontsize=10,
        )

    plt.tight_layout()
    path = os.path.join(ROBUSTNESS_DIR, f"{model_name}_{test_type}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Chart → {path}")
