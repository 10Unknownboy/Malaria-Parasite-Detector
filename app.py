"""
app.py — Streamlit web application for interactive malaria cell classification.

Launch with:
    streamlit run app.py

Features:
* Model selection sidebar (Simple CNN, ResNet‑18, MobileNet V2)
* Drag‑and‑drop image upload
* Prediction with confidence bar
* Grad‑CAM explainability overlay

WARNING: This is an educational prototype only — NOT for clinical use.
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import CLASS_NAMES, DEVICE, MODELS_DIR, create_dirs
from src.dataset import get_transforms


# ────────────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Malaria Parasite Detector",
    page_icon="🔬",
    layout="wide",
)

# ────────────────────────────────────────────────────────────────────
# Disclaimer banner
# ────────────────────────────────────────────────────────────────────
st.warning(
    "⚠️ **DISCLAIMER**: This is an **educational prototype** only.  "
    "It is **NOT** validated for clinical or diagnostic use.  "
    "Always consult qualified medical professionals for malaria diagnosis."
)

st.title("🔬 Malaria Parasite Detector")
st.markdown(
    "Upload a thin blood‑smear cell image to classify it as "
    "**Parasitized** or **Uninfected** using deep learning."
)


# ────────────────────────────────────────────────────────────────────
# Sidebar — model selection
# ────────────────────────────────────────────────────────────────────
MODEL_OPTIONS = {
    "Simple CNN": "simple_cnn",
    "ResNet-18": "resnet18",
    "MobileNet V2": "mobilenetv2",
}

st.sidebar.header("⚙️ Settings")
selected_display = st.sidebar.selectbox(
    "Choose a model",
    list(MODEL_OPTIONS.keys()),
    index=2,  # default: MobileNet V2
)
model_key = MODEL_OPTIONS[selected_display]
show_gradcam = st.sidebar.checkbox("Show Grad‑CAM overlay", value=True)

st.sidebar.markdown("---")
st.sidebar.info(
    f"**Device**: `{DEVICE}`\n\n"
    f"**Model**: `{model_key}`"
)


# ────────────────────────────────────────────────────────────────────
# Model loading (cached)
# ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def _load_model(model_name: str):
    """Load a trained model, returning (model, error_msg)."""
    from src.predict import load_model

    try:
        model = load_model(model_name, device=DEVICE)
        return model, None
    except FileNotFoundError:
        ckpt = os.path.join(MODELS_DIR, f"{model_name}_best.pth")
        return None, (
            f"Checkpoint not found: `{ckpt}`.\n\n"
            "Please train the model first by running the training notebook or script."
        )
    except Exception as exc:
        return None, f"Error loading model: {exc}"


model, load_error = _load_model(model_key)

if load_error:
    st.error(load_error)
    st.stop()


# ────────────────────────────────────────────────────────────────────
# File uploader
# ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a cell image (PNG, JPG, BMP)",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
)

if uploaded_file is not None:
    # ── display uploaded image ────────────────────────────────────
    image = Image.open(uploaded_file).convert("RGB")

    col_img, col_result = st.columns([1, 1])
    with col_img:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # ── run prediction ────────────────────────────────────────────
    transform = get_transforms(is_training=False)
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = model(input_tensor)
        prob = torch.sigmoid(logit).item()

    predicted_label = 1 if prob >= 0.5 else 0
    prediction = CLASS_NAMES[predicted_label]
    confidence = prob if predicted_label == 1 else 1 - prob

    # ── display result ────────────────────────────────────────────
    with col_result:
        st.subheader("Prediction")

        if predicted_label == 1:
            st.error(f"🦠 **{prediction}**")
        else:
            st.success(f"✅ **{prediction}**")

        st.metric("Confidence", f"{confidence:.1%}")

        st.markdown("**Class Probabilities**")
        st.progress(prob, text=f"Parasitized: {prob:.1%}")
        st.progress(1 - prob, text=f"Uninfected: {1 - prob:.1%}")

    # ── Grad‑CAM ──────────────────────────────────────────────────
    if show_gradcam:
        st.markdown("---")
        st.subheader("🔍 Grad‑CAM Explainability")

        try:
            from src.gradcam import GradCAM, get_target_layer, overlay_heatmap, _denormalize
            import numpy as np

            target_layer = get_target_layer(model, model_key)
            cam_obj = GradCAM(model, target_layer)

            # Need a fresh forward pass with grad enabled
            input_grad = transform(image).unsqueeze(0).to(DEVICE)
            heatmap = cam_obj.generate_cam(input_grad)
            cam_obj.remove_hooks()

            original_np = _denormalize(input_grad.squeeze(0))
            overlay_np = overlay_heatmap(original_np, heatmap)

            gc1, gc2, gc3 = st.columns(3)
            with gc1:
                st.image(original_np, caption="Original", use_container_width=True)
            with gc2:
                # Convert heatmap to coloured image for display
                import cv2
                hm_uint8 = np.uint8(255 * heatmap)
                hm_coloured = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
                hm_coloured = cv2.cvtColor(hm_coloured, cv2.COLOR_BGR2RGB)
                st.image(hm_coloured, caption="Grad‑CAM Heatmap", use_container_width=True)
            with gc3:
                st.image(
                    (overlay_np * 255).astype(np.uint8),
                    caption="Overlay", use_container_width=True,
                )

        except Exception as exc:
            st.warning(f"Grad‑CAM generation failed: {exc}")

    st.markdown("---")
    st.caption(
        "⚠️ This is an educational prototype — NOT for clinical use.  "
        "Consult qualified medical professionals for malaria diagnosis."
    )
else:
    st.info("👆 Upload a cell image to get started.")
