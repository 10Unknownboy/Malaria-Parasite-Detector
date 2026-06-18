"""
Evaluate — Test‑set metrics, visualisations, model comparison, and metadata export.

Core workflow
-------------
1. :func:`evaluate_model` — per‑model metrics + plots (confusion matrix, ROC,
   sample predictions).
2. :func:`compare_models` — cross‑model comparison table, best‑model report.
3. :func:`save_all_metadata` — write JSON / CSV artefacts consumed by the
   report generator and the Streamlit app.

WARNING: This is an educational prototype only — NOT for clinical use.
"""

import json
import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from src.config import (
    CLASS_MAPPING,
    CLASS_NAMES,
    CONFUSION_MATRIX_DIR,
    DEVICE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODELS_DIR,
    RESULTS_DIR,
    ROC_CURVES_DIR,
    SAMPLE_PREDICTIONS_DIR,
    create_dirs,
)


# ────────────────────────────────────────────────────────────────────
# Single‑model evaluation
# ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    model_name: str,
    device: torch.device = DEVICE,
) -> Dict[str, float]:
    """Compute metrics on the test set and save diagnostic plots.

    Parameters
    ----------
    model : nn.Module
        Trained model (weights already loaded).
    test_loader : DataLoader
        Test‑split DataLoader.
    model_name : str
        Used for filenames and plot titles.
    device : torch.device
        Target device.

    Returns
    -------
    dict[str, float]
        Keys: ``accuracy``, ``precision``, ``recall``, ``specificity``,
        ``f1``, ``roc_auc``.
    """
    create_dirs()
    model = model.to(device)
    model.eval()

    all_labels: List[int] = []
    all_probs: List[float] = []

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()

        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().astype(int).tolist())

    all_labels_np = np.array(all_labels)
    all_probs_np = np.array(all_probs)
    all_preds_np = (all_probs_np >= 0.5).astype(int)

    # ── compute metrics ───────────────────────────────────────────
    tn, fp, fn, tp = confusion_matrix(all_labels_np, all_preds_np).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": accuracy_score(all_labels_np, all_preds_np),
        "precision": precision_score(all_labels_np, all_preds_np, zero_division=0),
        "recall": recall_score(all_labels_np, all_preds_np, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(all_labels_np, all_preds_np, zero_division=0),
        "roc_auc": roc_auc_score(all_labels_np, all_probs_np),
    }

    # ── print summary ─────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"    Evaluation Results — {model_name}")
    print(f"{'=' * 55}")
    for k, v in metrics.items():
        print(f"  {k:>12s} : {v:.4f}")
    print(f"{'=' * 55}\n")

    # ── generate plots ────────────────────────────────────────────
    _plot_confusion_matrix(all_labels_np, all_preds_np, model_name)
    _plot_roc_curve(all_labels_np, all_probs_np, metrics["roc_auc"], model_name)
    _plot_sample_predictions(
        test_loader.dataset, all_preds_np, all_probs_np, model_name,
    )

    return metrics


# ────────────────────────────────────────────────────────────────────
# Confusion matrix
# ────────────────────────────────────────────────────────────────────
def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    path = os.path.join(CONFUSION_MATRIX_DIR, f"{model_name}_cm.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Confusion matrix → {path}")


# ────────────────────────────────────────────────────────────────────
# ROC curve
# ────────────────────────────────────────────────────────────────────
def _plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    auc_score: float,
    model_name: str,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(ROC_CURVES_DIR, f"{model_name}_roc.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    ROC curve → {path}")


# ────────────────────────────────────────────────────────────────────
# Sample predictions grid
# ────────────────────────────────────────────────────────────────────
def _plot_sample_predictions(
    dataset,
    all_preds: np.ndarray,
    all_probs: np.ndarray,
    model_name: str,
    n_correct: int = 4,
    n_incorrect: int = 4,
) -> None:
    """Save a grid showing correct and incorrect predictions."""
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)

    labels = np.array(dataset.labels)
    correct_mask = all_preds == labels
    incorrect_mask = ~correct_mask

    correct_idx = np.where(correct_mask)[0]
    incorrect_idx = np.where(incorrect_mask)[0]

    np.random.seed(42)
    sel_correct = np.random.choice(
        correct_idx, min(n_correct, len(correct_idx)), replace=False,
    )
    sel_incorrect = np.random.choice(
        incorrect_idx, min(n_incorrect, len(incorrect_idx)), replace=False,
    ) if len(incorrect_idx) > 0 else np.array([], dtype=int)

    total = len(sel_correct) + len(sel_incorrect)
    if total == 0:
        return

    cols = min(4, total)
    rows = (total + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if total == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    idx_list = list(sel_correct) + list(sel_incorrect)
    for i, idx in enumerate(idx_list):
        img_tensor, _ = dataset[idx]
        img = img_tensor.numpy().transpose(1, 2, 0)  # CHW → HWC
        img = std * img + mean  # de‑normalise
        img = np.clip(img, 0, 1)

        true_label = CLASS_NAMES[int(labels[idx])]
        pred_label = CLASS_NAMES[int(all_preds[idx])]
        prob = all_probs[idx]
        is_correct = idx in sel_correct

        axes[i].imshow(img)
        colour = "green" if is_correct else "red"
        axes[i].set_title(
            f"True: {true_label}\nPred: {pred_label} ({prob:.2f})",
            color=colour, fontsize=9,
        )
        axes[i].axis("off")

    # hide unused subplots
    for j in range(len(idx_list), len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Sample Predictions — {model_name}", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(SAMPLE_PREDICTIONS_DIR, f"{model_name}_samples.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Sample predictions → {path}")


# ────────────────────────────────────────────────────────────────────
# Multi‑model comparison
# ────────────────────────────────────────────────────────────────────
def compare_models(
    metrics_dict: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """Compare multiple models and determine the best one.

    Parameters
    ----------
    metrics_dict : dict[str, dict[str, float]]
        ``{model_name: {metric_name: value}}``.

    Returns
    -------
    dict
        Best‑model report (also saved to ``models/best_model_report.json``).
    """
    create_dirs()

    # ── build comparison DataFrame ────────────────────────────────
    df = pd.DataFrame(metrics_dict).T
    df.index.name = "model"
    df = df.sort_values("f1", ascending=False)

    csv_path = os.path.join(MODELS_DIR, "model_comparison.csv")
    df.to_csv(csv_path)

    print(f"\n{'=' * 70}")
    print("    Model Comparison")
    print(f"{'=' * 70}")
    print(df.to_string(float_format="{:.4f}".format))
    print(f"{'=' * 70}\n")
    print(f"    Comparison CSV → {csv_path}")

    # ── determine best model (F1, then recall as tiebreaker) ─────
    best_name = df.index[0]
    # If top two have same F1, use recall
    if len(df) > 1 and abs(df.iloc[0]["f1"] - df.iloc[1]["f1"]) < 1e-6:
        top_f1 = df[abs(df["f1"] - df.iloc[0]["f1"]) < 1e-6]
        best_name = top_f1.sort_values("recall", ascending=False).index[0]

    best_report = {
        "best_model": best_name,
        "metrics": metrics_dict[best_name],
        "ranking": list(df.index),
    }

    report_path = os.path.join(MODELS_DIR, "best_model_report.json")
    with open(report_path, "w") as f:
        json.dump(best_report, f, indent=2)
    print(f"    Best model: {best_name}")
    print(f"    Report → {report_path}\n")

    return best_report


# ────────────────────────────────────────────────────────────────────
# Metadata export
# ────────────────────────────────────────────────────────────────────
def save_all_metadata(
    all_metrics: Dict[str, Dict[str, float]],
    all_histories: Dict[str, Dict[str, List]],
    models_dict: Dict[str, nn.Module],
) -> None:
    """Persist training artefacts consumed downstream.

    Saves
    -----
    * ``models/model_config.json``     — architecture info & param counts
    * ``models/class_mapping.json``    — label‑to‑name mapping
    * ``models/training_metrics.json`` — per‑model per‑epoch histories
    * ``results/final_results_summary.json``
    * ``results/final_results_summary.csv``
    """
    create_dirs()

    # 1. model_config.json ─────────────────────────────────────────
    config_info: Dict[str, Any] = {}
    for name, model in models_dict.items():
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        config_info[name] = {
            "architecture": model.__class__.__name__,
            "total_parameters": total,
            "trainable_parameters": trainable,
        }
    _write_json(os.path.join(MODELS_DIR, "model_config.json"), config_info)

    # 2. class_mapping.json ────────────────────────────────────────
    _write_json(os.path.join(MODELS_DIR, "class_mapping.json"), {
        "class_mapping": CLASS_MAPPING,
        "class_names": CLASS_NAMES,
    })

    # 3. training_metrics.json ─────────────────────────────────────
    # Convert numpy / tensor values to plain Python types
    clean_histories: Dict[str, Any] = {}
    for name, hist in all_histories.items():
        clean_histories[name] = {
            k: [float(v) if isinstance(v, (float, int, np.floating)) else v
                for v in vals]
            for k, vals in hist.items()
        }
    _write_json(os.path.join(MODELS_DIR, "training_metrics.json"), clean_histories)

    # 4. final_results_summary.json ────────────────────────────────
    summary = {name: {k: float(v) for k, v in m.items()}
               for name, m in all_metrics.items()}
    _write_json(os.path.join(RESULTS_DIR, "final_results_summary.json"), summary)

    # 5. final_results_summary.csv ─────────────────────────────────
    df = pd.DataFrame(summary).T
    df.index.name = "model"
    csv_path = os.path.join(RESULTS_DIR, "final_results_summary.csv")
    df.to_csv(csv_path)

    print("    All metadata saved.")


def _write_json(path: str, data: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"     → {path}")
