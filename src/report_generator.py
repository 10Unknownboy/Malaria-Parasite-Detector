"""
Report Generator — Auto‑generate a Markdown project report from saved artefacts.

Reads model comparison CSVs, best‑model JSON, training metrics, and final
results to produce a comprehensive ``project_report.md`` under ``reports/``.

WARNING: This is an educational prototype only — NOT for clinical use.
"""

import json
import os
from datetime import datetime
from datetime import datetime

import pandas as pd

from src.config import (
    CLASS_NAMES,
    CONFUSION_MATRIX_DIR,
    DATA_DIR,
    GRADCAM_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    LR_SCHEDULER_PATIENCE,
    LR_SCHEDULER_FACTOR,
    WEIGHT_DECAY,
    SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    MODELS_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
    ROBUSTNESS_DIR,
    ROC_CURVES_DIR,
    SAMPLE_PREDICTIONS_DIR,
    TRAINING_CURVES_DIR,
    create_dirs,
)


def _safe_read_json(path):
    """Read a JSON file or return None on failure."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _safe_read_csv(path):
    """Read a CSV file or return None on failure."""
    if not os.path.isfile(path):
        return None
    try:
        return pd.read_csv(path, index_col=0)
    except Exception:
        return None


def _count_images(data_dir):
    """Count images in Parasitized / Uninfected folders."""
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    counts = {}
    for class_name in ["Parasitized", "Uninfected"]:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            n = sum(
                1 for f in os.listdir(class_dir)
                if os.path.splitext(f)[1].lower() in valid_exts
            )
        else:
            n = 0
        counts[class_name] = n
    counts["total"] = sum(counts.values())
    return counts


# ────────────────────────────────────────────────────────────────────
# Main generator
# ────────────────────────────────────────────────────────────────────
def generate_report(
    results_dir=RESULTS_DIR,
    models_dir=MODELS_DIR,
    reports_dir=REPORTS_DIR,
    data_dir=DATA_DIR,
):
    """Generate ``project_report.md`` from saved artefacts.

    Parameters
    ----------
    results_dir : str
    models_dir : str
    reports_dir : str
    data_dir : str

    Returns
    -------
    str
        Absolute path to the generated Markdown report.
    """
    create_dirs()

    # ── load artefacts ────────────────────────────────────────────
    comparison_df = _safe_read_csv(os.path.join(models_dir, "model_comparison.csv"))
    best_report = _safe_read_json(os.path.join(models_dir, "best_model_report.json"))
    model_config = _safe_read_json(os.path.join(models_dir, "model_config.json"))
    training_metrics = _safe_read_json(os.path.join(models_dir, "training_metrics.json"))
    final_summary = _safe_read_json(os.path.join(results_dir, "final_results_summary.json"))
    image_counts = _count_images(data_dir)

    # ── build Markdown ────────────────────────────────────────────
    lines = []

    # Title & Disclaimer
    lines.append("# Malaria Parasite Detector — Project Report")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")
    lines.append("> **  DISCLAIMER**: This project is an **educational prototype**.")
    lines.append("> It is **NOT** validated for clinical or diagnostic use.  Always")
    lines.append("> consult qualified medical professionals for malaria diagnosis.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. Introduction
    lines.append("## 1. Introduction & Problem Statement")
    lines.append("")
    lines.append("Malaria remains one of the most significant global health challenges,")
    lines.append("causing hundreds of thousands of deaths annually.  Microscopic examination")
    lines.append("of thin blood smears by trained technicians is the gold standard for")
    lines.append("diagnosis, but it is time‑consuming and requires expert skill.")
    lines.append("")
    lines.append("This project explores **deep‑learning‑based automated detection** of")
    lines.append("*Plasmodium*‑infected red blood cells from microscopy images.  Three")
    lines.append("convolutional neural network architectures are compared:")
    lines.append("")
    lines.append("1. **Simple CNN** — a lightweight 4‑block model trained from scratch.")
    lines.append("2. **ResNet‑18** — a pre‑trained residual network fine‑tuned for this task.")
    lines.append("3. **MobileNet V2** — a pre‑trained efficient network for mobile deployment.")
    lines.append("")

    # 2. Dataset
    lines.append("## 2. Dataset Description")
    lines.append("")
    lines.append("The dataset comes from the NIH National Library of Medicine and consists")
    lines.append("of segmented thin blood smear cell images.")
    lines.append("")
    lines.append(f"| Statistic | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| Total images | {image_counts['total']:,} |")
    lines.append(f"| Parasitized | {image_counts.get('Parasitized', 'N/A'):,} |")
    lines.append(f"| Uninfected | {image_counts.get('Uninfected', 'N/A'):,} |")
    lines.append(f"| Image size | {IMAGE_SIZE}×{IMAGE_SIZE} (resized) |")
    lines.append(f"| Train / Val / Test | {TRAIN_RATIO:.0%} / {VAL_RATIO:.0%} / {TEST_RATIO:.0%} |")
    lines.append("")

    # 3. Methodology
    lines.append("## 3. Methodology")
    lines.append("")
    lines.append("### 3.1 Preprocessing & Augmentation")
    lines.append("")
    lines.append("* Resize to `128×128` pixels")
    lines.append("* **Training augmentations**: random horizontal & vertical flip, "
                 "random rotation (±15°), colour jitter (brightness, contrast, saturation ±0.2)")
    lines.append("* **Normalisation**: ImageNet mean & std")
    lines.append("")
    lines.append("### 3.2 Architectures")
    lines.append("")

    if model_config:
        lines.append("| Model | Architecture | Trainable Parameters |")
        lines.append("|-------|-------------|---------------------|")
        for name, info in model_config.items():
            tp = info.get("trainable_parameters", "?")
            lines.append(f"| {name} | {info.get('architecture', '?')} | {tp:,} |")
        lines.append("")

    lines.append("### 3.3 Training Setup")
    lines.append("")
    lines.append(f"* **Loss**: `BCEWithLogitsLoss`")
    lines.append(f"* **Optimiser**: Adam (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
    lines.append(f"* **Scheduler**: ReduceLROnPlateau (patience={LR_SCHEDULER_PATIENCE}, factor={LR_SCHEDULER_FACTOR})")
    lines.append(f"* **Early stopping**: patience={EARLY_STOPPING_PATIENCE}")
    lines.append(f"* **Max epochs**: {NUM_EPOCHS}")
    lines.append(f"* **Batch size**: {BATCH_SIZE}")
    lines.append(f"* **Mixed precision**: enabled on CUDA")
    lines.append(f"* **Seed**: {SEED}")
    lines.append("")

    # 4. Results
    lines.append("## 4. Results")
    lines.append("")

    if comparison_df is not None:
        lines.append("### 4.1 Model Comparison")
        lines.append("")
        lines.append(comparison_df.to_markdown(floatfmt=".4f"))
        lines.append("")
    else:
        lines.append("*Model comparison data not yet available — train models first.*")
        lines.append("")

    if best_report:
        best_name = best_report.get("best_model", "N/A")
        best_metrics = best_report.get("metrics", {})
        lines.append(f"### 4.2 Best Model: **{best_name}**")
        lines.append("")
        for k, v in best_metrics.items():
            lines.append(f"* {k}: **{v:.4f}**")
        lines.append("")

    lines.append("### 4.3 Confusion Matrices")
    lines.append("")
    lines.append(f"See `{CONFUSION_MATRIX_DIR}/` for per‑model confusion matrix heatmaps.")
    lines.append("")

    lines.append("### 4.4 ROC Curves")
    lines.append("")
    lines.append(f"See `{ROC_CURVES_DIR}/` for per‑model ROC curves.")
    lines.append("")

    lines.append("### 4.5 Training Curves")
    lines.append("")
    lines.append(f"See `{TRAINING_CURVES_DIR}/` for loss and accuracy plots.")
    lines.append("")

    # 5. Robustness
    lines.append("## 5. Robustness Analysis")
    lines.append("")
    lines.append("Models were tested under Gaussian blur (kernel sizes 3–9) and")
    lines.append("additive Gaussian noise (σ = 0.01–0.2) to evaluate degradation")
    lines.append("under imperfect acquisition conditions.")
    lines.append("")
    lines.append(f"See `{ROBUSTNESS_DIR}/` for bar charts and JSON results.")
    lines.append("")

    # 6. Explainability
    lines.append("## 6. Explainability — Grad‑CAM")
    lines.append("")
    lines.append("Gradient‑weighted Class Activation Mapping (Grad‑CAM) heatmaps")
    lines.append("highlight the image regions that most influence each model's")
    lines.append("prediction.  This helps verify that the network focuses on")
    lines.append("biologically plausible features (e.g. parasites inside RBCs).")
    lines.append("")
    lines.append(f"See `{GRADCAM_DIR}/` for per‑model visualisations.")
    lines.append("")

    # 7. Limitations
    lines.append("## 7. Limitations & Future Work")
    lines.append("")
    lines.append("* **Single dataset** — results may not generalise to other staining")
    lines.append("  protocols or microscope optics.")
    lines.append("* **Binary task only** — does not identify *Plasmodium* species or")
    lines.append("  quantify parasitemia.")
    lines.append("* **No clinical validation** — performance on real clinical samples")
    lines.append("  is unknown.")
    lines.append("* **Future directions**: multi‑class species classification, attention")
    lines.append("  mechanisms, mobile deployment with ONNX / TFLite, federated learning")
    lines.append("  across hospital sites.")
    lines.append("")

    # 8. Conclusion
    lines.append("## 8. Conclusion")
    lines.append("")
    if best_report:
        bm = best_report.get("best_model", "N/A")
        bm_metrics = best_report.get("metrics", {})
        f1 = bm_metrics.get("f1", 0)
        auc = bm_metrics.get("roc_auc", 0)
        lines.append(f"The **{bm}** model achieved the best overall performance with")
        lines.append(f"an F1 score of **{f1:.4f}** and ROC AUC of **{auc:.4f}**.")
    else:
        lines.append("*Results pending — train and evaluate models to populate this section.*")
    lines.append("")
    lines.append("While promising, this remains an educational prototype.  Clinical")
    lines.append("deployment would require rigorous prospective validation, regulatory")
    lines.append("approval, and integration with existing laboratory workflows.")
    lines.append("")

    # Appendix
    lines.append("## Appendix: Hyperparameters & Reproducibility")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Seed | {SEED} |")
    lines.append(f"| Image size | {IMAGE_SIZE}×{IMAGE_SIZE} |")
    lines.append(f"| Batch size | {BATCH_SIZE} |")
    lines.append(f"| Learning rate | {LEARNING_RATE} |")
    lines.append(f"| Weight decay | {WEIGHT_DECAY} |")
    lines.append(f"| Max epochs | {NUM_EPOCHS} |")
    lines.append(f"| Early stopping patience | {EARLY_STOPPING_PATIENCE} |")
    lines.append(f"| LR scheduler patience | {LR_SCHEDULER_PATIENCE} |")
    lines.append(f"| LR scheduler factor | {LR_SCHEDULER_FACTOR} |")
    lines.append(f"| Train/Val/Test split | {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*End of report.*")

    # ── write file ────────────────────────────────────────────────
    report_md = "\n".join(lines)
    report_path = os.path.join(reports_dir, "project_report.md")
    os.makedirs(reports_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\n    Report generated → {report_path}")
    return report_path




