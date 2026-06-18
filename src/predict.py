
"""
Predict — Single‑image and batch inference with optional Grad‑CAM overlay.

Provides a clean API for loading trained models and running predictions
on new cell images, suitable for the CLI script and Streamlit app.

WARNING: This is an educational prototype only — NOT for clinical use.
"""

import os
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from src.config import (
    CLASS_MAPPING,
    CLASS_NAMES,
    DEVICE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODELS_DIR,
)
from src.dataset import get_transforms
from src.models import get_model


# ────────────────────────────────────────────────────────────────────
# Model loading
# ────────────────────────────────────────────────────────────────────
def load_model(
    model_name,
    model_path=None,
    device=DEVICE,
    **model_kwargs,
):
    """Load a trained model from a ``.pth`` checkpoint.

    Parameters
    ----------
    model_name : str
        Registry key (``"simple_cnn"``, ``"resnet18"``, ``"mobilenetv2"``).
    model_path : str | None
        Path to the ``.pth`` file.  Defaults to
        ``models/{model_name}_best.pth``.
    device : torch.device
        Target device.
    **model_kwargs
        Extra arguments forwarded to the model constructor (e.g.
        ``pretrained=False`` for transfer‑learning models during inference).

    Returns
    -------
    nn.Module
        Model with weights loaded, set to eval mode, on *device*.

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist.
    """
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, f"{model_name}_best.pth")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}\n"
            "Please train the model first."
        )

    # For transfer‑learning models, avoid re‑downloading ImageNet weights
    if "pretrained" not in model_kwargs and model_name in ("resnet18", "mobilenetv2"):
        model_kwargs["pretrained"] = False

    model = get_model(model_name, **model_kwargs)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


# ────────────────────────────────────────────────────────────────────
# Single‑image prediction
# ────────────────────────────────────────────────────────────────────
def predict_single_image(
    image_path,
    model,
    device=DEVICE,
    transform=None,
):
    """Run inference on a single image.

    Parameters
    ----------
    image_path : str
        Path to the cell image.
    model : nn.Module
        Trained model in eval mode.
    device : torch.device
    transform : Compose | None
        If *None*, uses the default eval transform.

    Returns
    -------
    dict
        * ``prediction`` (str) — "Parasitized" or "Uninfected"
        * ``confidence`` (float) — probability of the predicted class
        * ``probability_parasitized`` (float) — P(Parasitized)
        * ``probability_uninfected`` (float) — P(Uninfected)
        * ``raw_logit`` (float)
    """
    if transform is None:
        transform = get_transforms(is_training=False)

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(input_tensor)
        prob = torch.sigmoid(logit).item()

    predicted_label = 1 if prob >= 0.5 else 0
    prediction = CLASS_NAMES[predicted_label]
    confidence = prob if predicted_label == 1 else 1 - prob

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "probability_parasitized": round(prob, 4),
        "probability_uninfected": round(1 - prob, 4),
        "raw_logit": round(logit.item(), 4),
    }


# ────────────────────────────────────────────────────────────────────
# Prediction + Grad‑CAM
# ────────────────────────────────────────────────────────────────────
def predict_with_gradcam(
    image_path,
    model,
    model_name,
    device=DEVICE,
    transform=None,
    save_dir=None,
):
    """Predict and generate a Grad‑CAM overlay.

    Parameters
    ----------
    image_path : str
    model : nn.Module
    model_name : str
    device : torch.device
    transform : Compose | None
    save_dir : str | None
        Directory to save the overlay.  Defaults to
        ``results/gradcam_outputs/{model_name}/``.

    Returns
    -------
    dict
        Same as :func:`predict_single_image` plus:
        * ``gradcam_path`` (str) — path to saved overlay image.
    """
    from src.gradcam import GradCAM, get_target_layer, overlay_heatmap, _denormalize

    if transform is None:
        transform = get_transforms(is_training=False)
    if save_dir is None:
        from src.config import GRADCAM_DIR
        save_dir = os.path.join(GRADCAM_DIR, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # ── prediction ────────────────────────────────────────────────
    result = predict_single_image(image_path, model, device, transform)

    # ── Grad‑CAM ──────────────────────────────────────────────────
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    target_layer = get_target_layer(model, model_name)
    cam = GradCAM(model, target_layer)
    heatmap = cam.generate_cam(input_tensor)
    cam.remove_hooks()

    original = _denormalize(input_tensor.squeeze(0))
    overlay_img = overlay_heatmap(original, heatmap)

    # ── save overlay ──────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad‑CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay_img)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(
        f"{result['prediction']} ({result['confidence']:.1%} confidence)",
        fontsize=13,
    )
    plt.tight_layout()

    basename = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(save_dir, f"{basename}_gradcam.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    result["gradcam_path"] = save_path
    return result


# ────────────────────────────────────────────────────────────────────
# Batch prediction
# ────────────────────────────────────────────────────────────────────
def batch_predict(
    image_paths,
    model,
    device=DEVICE,
    transform=None,
):
    """Run inference on multiple images.

    Parameters
    ----------
    image_paths : list[str]
    model : nn.Module
    device : torch.device
    transform : Compose | None

    Returns
    -------
    list[dict]
        One result dict per image (same schema as
        :func:`predict_single_image`).
    """
    results = []
    for path in image_paths:
        try:
            res = predict_single_image(path, model, device, transform)
            res["image_path"] = path
            res["error"] = None
        except Exception as exc:
            res = {
                "image_path": path,
                "prediction": None,
                "confidence": None,
                "error": str(exc),
            }
        results.append(res)
    return results

