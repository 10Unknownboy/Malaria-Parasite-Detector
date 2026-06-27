"""
Grad‑CAM — Gradient‑weighted Class Activation Mapping for model explainability.

Produces heatmaps showing which regions of a cell image the model attends to
when making a prediction.  Works with any architecture that has convolutional
layers (SimpleCNN, ResNet‑18, MobileNetV2).


WARNING: This is an educational prototype only — NOT for clinical use.
"""

import os
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    CLASS_NAMES,
    DEVICE,
    GRADCAM_DIR,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    create_dirs,
)


# ────────────────────────────────────────────────────────────────────
# Grad‑CAM core
# ────────────────────────────────────────────────────────────────────
class GradCAM:
    """Compute Grad‑CAM heatmaps for a target convolutional layer.

    Parameters
    ----------
    model : nn.Module
        Trained model (moved to the desired device beforehand).
    target_layer : nn.Module
        The layer whose activations / gradients are captured.

    Usage
    -----
    >>> cam = GradCAM(model, model.get_last_conv_layer())
    >>> heatmap = cam.generate_cam(input_tensor)  # (H, W) numpy array
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self._activations = None
        self._gradients = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._forward_hook)
        self._bwd_hook = target_layer.register_full_backward_hook(self._backward_hook)

    # ── hooks ─────────────────────────────────────────────────────
    def _forward_hook(self, module, input, output):
        self._activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    # ── main API ──────────────────────────────────────────────────
    def generate_cam(
        self,
        input_tensor,
        target_class=None,
    ):
        """Generate a Grad‑CAM heatmap for *input_tensor*.

        Parameters
        ----------
        input_tensor : Tensor, shape ``(1, C, H, W)``
            Single preprocessed image.
        target_class : int | None
            For binary models with 1 output, leave as *None* (uses the
            raw logit).  For multi‑class, specify the class index.

        Returns
        -------
        numpy.ndarray, shape ``(H, W)``
            Heatmap normalised to ``[0, 1]``.
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward
        output = self.model(input_tensor)

        # Compute gradient of target score w.r.t. activations
        if target_class is not None:
            score = output[:, target_class]
        else:
            score = output.squeeze()  # binary: single logit
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Grad‑CAM weights: global‑average‑pool the gradients
        gradients = self._gradients  # (1, C, h, w)
        activations = self._activations  # (1, C, h, w)

        if gradients is None or activations is None:
            raise RuntimeError("Hooks not called. Ensure backward() was executed.")
        
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(
            cam, size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear", align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    # ── cleanup ───────────────────────────────────────────────────
    def remove_hooks(self):
        """Remove forward and backward hooks (call when done)."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────
# Target‑layer auto‑detection
# ────────────────────────────────────────────────────────────────────
def get_target_layer(model, model_name):
    """Return the appropriate Grad‑CAM target layer for a known architecture.

    Parameters
    ----------
    model : nn.Module
    model_name : str
        One of ``"simple_cnn"``, ``"resnet18"``, ``"mobilenetv2"``.

    Returns
    -------
    nn.Module
    """
    name = model_name.lower().strip()
    if name == "simple_cnn":
        return model.features[12]  # last Conv2d in Block 4
    elif name == "resnet18":
        return model.backbone.layer4[-1]
    elif name == "mobilenetv2":
        return model.backbone.features[-1]
    else:
        raise ValueError(
            f"Cannot auto‑detect target layer for model '{model_name}'. "
            "Use model.get_last_conv_layer() directly."
        )


# ────────────────────────────────────────────────────────────────────
# Overlay helper
# ────────────────────────────────────────────────────────────────────
def overlay_heatmap(
    image,
    heatmap,
    alpha=0.4,
):
    """Overlay a Grad‑CAM heatmap on the original image.

    Parameters
    ----------
    image : ndarray, shape ``(H, W, 3)``, dtype float in ``[0, 1]``
    heatmap : ndarray, shape ``(H, W)``, values in ``[0, 1]``
    alpha : float
        Blending weight for the heatmap.

    Returns
    -------
    ndarray, shape ``(H, W, 3)``, dtype float in ``[0, 1]``
    """
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_coloured = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_coloured = cv2.cvtColor(heatmap_coloured, cv2.COLOR_BGR2RGB)
    heatmap_float = heatmap_coloured.astype(np.float32) / 255.0

    # Resize heatmap to match image if necessary
    if heatmap_float.shape[:2] != image.shape[:2]:
        heatmap_float = cv2.resize(
            heatmap_float, (image.shape[1], image.shape[0]),
        )

    overlay = (1 - alpha) * image + alpha * heatmap_float
    return np.clip(overlay, 0, 1)


# ────────────────────────────────────────────────────────────────────
# De‑normalise tensor → displayable numpy image
# ────────────────────────────────────────────────────────────────────
def _denormalize(tensor):
    """Convert a normalised ``(C, H, W)`` tensor to ``(H, W, 3)`` float image."""
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    return np.clip(img, 0, 1)


# ────────────────────────────────────────────────────────────────────
# Grid generation
# ────────────────────────────────────────────────────────────────────
def generate_gradcam_grid(
    model,
    model_name,
    dataset,
    device=DEVICE,
    num_samples=8,
):
    """Generate a grid of Original / Heatmap / Overlay for both classes.

    Selects ``num_samples // 2`` parasitized and ``num_samples // 2``
    uninfected samples from *dataset*.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    model_name : str
        For file naming and target‑layer detection.
    dataset : MalariaDataset
        Provides ``(image_tensor, label)`` items.
    device : torch.device
    num_samples : int
        Total samples to display.

    Returns
    -------
    list[str]
        Absolute paths of saved images.
    """
    create_dirs()
    save_dir = os.path.join(GRADCAM_DIR, model_name)
    os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)
    model.eval()

    target_layer = get_target_layer(model, model_name)
    cam = GradCAM(model, target_layer)

    # Select balanced samples
    labels = np.array(dataset.labels)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    np.random.seed(42)
    n_each = max(1, num_samples // 2)
    sel_pos = np.random.choice(pos_idx, min(n_each, len(pos_idx)), replace=False)
    sel_neg = np.random.choice(neg_idx, min(n_each, len(neg_idx)), replace=False)
    selected = list(sel_pos) + list(sel_neg)

    saved_paths = []

    # ── individual images ─────────────────────────────────────────
    for i, idx in enumerate(selected):
        img_tensor, label = dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)

        heatmap = cam.generate_cam(input_tensor)
        original = _denormalize(img_tensor)
        overlay_img = overlay_heatmap(original, heatmap)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        titles = ["Original", "Grad‑CAM Heatmap", "Overlay"]
        images = [original, heatmap, overlay_img]

        for ax, img, title in zip(axes, images, titles):
            if img.ndim == 2:
                ax.imshow(img, cmap="jet")
            else:
                ax.imshow(img)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        class_label = CLASS_NAMES[int(label)]
        fig.suptitle(
            f"{model_name} — {class_label} (sample {i + 1})",
            fontsize=12, y=1.02,
        )
        plt.tight_layout()
        path = os.path.join(save_dir, f"gradcam_{class_label}_{i + 1}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(path)

    # ── combined grid ─────────────────────────────────────────────
    n = len(selected)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(selected):
        img_tensor, label = dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)

        heatmap = cam.generate_cam(input_tensor)
        original = _denormalize(img_tensor)
        overlay_img = overlay_heatmap(original, heatmap)

        class_label = CLASS_NAMES[int(label)]

        axes[row, 0].imshow(original)
        axes[row, 0].set_title(f"Original ({class_label})", fontsize=9)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(heatmap, cmap="jet")
        axes[row, 1].set_title("Grad‑CAM", fontsize=9)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(overlay_img)
        axes[row, 2].set_title("Overlay", fontsize=9)
        axes[row, 2].axis("off")

    fig.suptitle(
        f"Grad‑CAM Visualisations — {model_name}",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    grid_path = os.path.join(save_dir, f"{model_name}_gradcam_grid.png")
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(grid_path)

    cam.remove_hooks()

    print(f"    Grad‑CAM outputs → {save_dir}/ ({len(saved_paths)} files)")
    return saved_paths






