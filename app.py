"""
app.py — Advanced Streamlit dashboard for Malaria Cell Classification
"""

import os
import sys
import random
import json
import pandas as pd
import numpy as np
import torch
import streamlit as st
from PIL import Image

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import CLASS_NAMES, DEVICE, MODELS_DIR, DATA_DIR, RESULTS_DIR, REPORTS_DIR, create_dirs
from src.dataset import get_transforms

# ==========================================
# Paths Override for Downloaded Zip Layout
# ==========================================
if os.path.exists(os.path.join(MODELS_DIR, "results")):
    RESULTS_DIR = os.path.join(MODELS_DIR, "results")
if os.path.exists(os.path.join(MODELS_DIR, "reports")):
    REPORTS_DIR = os.path.join(MODELS_DIR, "reports")

st.set_page_config(page_title="Malaria Parasite Detector", page_icon="🔬", layout="wide")

# ==========================================
# Helpers & Data Loading
# ==========================================
MODEL_OPTIONS = {
    "Simple CNN": "simple_cnn",
    "ResNet-18": "resnet18",
    "MobileNet V2": "mobilenetv2",
}

@st.cache_data
def load_metadata():
    comp_df = pd.DataFrame()
    csv_path = os.path.join(MODELS_DIR, "model_comparison.csv")
    if os.path.exists(csv_path):
        comp_df = pd.read_csv(csv_path)
        comp_df.rename(columns={
            "model": "ModelKey", "accuracy": "Accuracy", "precision": "Precision",
            "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC", "specificity": "Specificity"
        }, inplace=True)
        # Create a display name column
        rev_map = {v: k for k, v in MODEL_OPTIONS.items()}
        comp_df["Model"] = comp_df["ModelKey"].map(lambda x: rev_map.get(x, x))
    
    config = {}
    if os.path.exists(os.path.join(MODELS_DIR, "model_config.json")):
        with open(os.path.join(MODELS_DIR, "model_config.json")) as f:
            config = json.load(f)
            
    best_report = {}
    if os.path.exists(os.path.join(MODELS_DIR, "best_model_report.json")):
        with open(os.path.join(MODELS_DIR, "best_model_report.json")) as f:
            best_report = json.load(f)
            
    return comp_df, config, best_report

comp_df, model_config, best_report = load_metadata()

@st.cache_data
def get_dataset_stats():
    para_dir = os.path.join(DATA_DIR, "Parasitized")
    uninf_dir = os.path.join(DATA_DIR, "Uninfected")
    
    p_count = len([f for f in os.listdir(para_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]) if os.path.exists(para_dir) else 0
    u_count = len([f for f in os.listdir(uninf_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]) if os.path.exists(uninf_dir) else 0
    
    total = p_count + u_count
    balance = (p_count / total * 100) if total > 0 else 0
    return total, p_count, u_count, balance

def get_model_metrics(m_key):
    if comp_df.empty:
        return {}
    row = comp_df[comp_df["ModelKey"] == m_key]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()

# ==========================================
# Sidebar
# ==========================================
st.sidebar.header("⚙️ Settings")
selected_display = st.sidebar.selectbox("Choose a model", list(MODEL_OPTIONS.keys()), index=2)
model_key = MODEL_OPTIONS[selected_display]
show_gradcam = st.sidebar.checkbox("Show Grad-CAM overlay", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")

if model_key in model_config:
    info = model_config[model_key]
    metric_info = get_model_metrics(model_key)
    acc = metric_info.get("Accuracy", 0) * 100
    
    params = info.get("trainable_parameters", 0)
    def format_params(p):
        if p > 1_000_000: return f"{p/1_000_000:.1f}M"
        if p > 1_000: return f"{p/1_000:.1f}K"
        return str(p)
        
    st.sidebar.markdown(f"**{selected_display}**\n\n"
                        f"- **Parameters**: {format_params(params)}\n"
                        f"- **Input**: {info.get('input_size', [3,128,128])[1]}x{info.get('input_size', [3,128,128])[2]}\n"
                        f"- **Accuracy**: {acc:.2f}%\n"
                        f"- **Epochs**: 20\n\n"
                        f"**Device**: `{DEVICE}`")

@st.cache_resource(show_spinner="Loading model…")
def _load_model(model_name: str):
    from src.predict import load_model
    try:
        model = load_model(model_name, device=DEVICE)
        return model, None
    except Exception as exc:
        return None, str(exc)

model, load_error = _load_model(model_key)
if load_error:
    st.sidebar.error(f"Error loading model: {load_error}")
    st.stop()

# ==========================================
# Main Layout
# ==========================================
st.title("🔬 Malaria Parasite Detector Dashboard")
st.warning("⚠️ **DISCLAIMER**: This is an educational prototype only. NOT validated for clinical use.")

tab1, tab2, tab3 = st.tabs(["🏠 Home", "📊 Model Insights", "🔬 Predictions"])

# ------------------------------------------
# TAB 1: HOME
# ------------------------------------------
with tab1:
    st.header("Project Overview")
    st.markdown("""
    **Project Title**: Malaria Parasite Detector from Blood Smear Images  
    **Dataset Source**: NIH/LHNCBC (National Library of Medicine) via Kaggle.  
    **Problem Statement**: Automate the binary classification of thin blood smear cell images as **Parasitized** or **Uninfected** to assist microscopy workflows in low-resource settings.  
    **Model Summary**: We trained a custom Simple CNN, ResNet-18, and MobileNetV2. MobileNetV2 achieved the best overall metrics and efficiency.
    """)
    
    st.subheader("Dataset Statistics")
    total_imgs, p_count, u_count, bal_pct = get_dataset_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Images", f"{total_imgs:,}")
    c2.metric("Parasitized", f"{p_count:,}")
    c3.metric("Uninfected", f"{u_count:,}")
    c4.metric("Balance (Parasitized)", f"{bal_pct:.1f}%")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("#### Class Distribution")
        chart_data = pd.DataFrame({"Class": ["Parasitized", "Uninfected"], "Count": [p_count, u_count]})
        st.bar_chart(chart_data.set_index("Class"), use_container_width=True)
        
    with col_g2:
        if not comp_df.empty:
            st.markdown("#### Model Comparison (Accuracy & F1)")
            st.bar_chart(comp_df.set_index("Model")[["Accuracy", "F1"]], use_container_width=True)
            
    st.subheader("Model Comparison Leaderboard")
    if not comp_df.empty:
        st.dataframe(
            comp_df[["Model", "Accuracy", "Precision", "Recall", "F1"]].style.format({
                "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}"
            }), 
            use_container_width=True, hide_index=True
        )
        
    st.subheader("Dataset Sample Gallery")
    gal_col1, gal_col2 = st.columns(2)
    with gal_col1:
        st.markdown("**Parasitized Samples**")
        para_dir = os.path.join(DATA_DIR, "Parasitized")
        if os.path.exists(para_dir):
            p_files = [f for f in os.listdir(para_dir) if f.lower().endswith(('.png','.jpg'))]
            p_samples = random.sample(p_files, min(5, len(p_files)))
            cols = st.columns(5)
            for i, p_file in enumerate(p_samples):
                cols[i].image(Image.open(os.path.join(para_dir, p_file)), use_container_width=True)
    with gal_col2:
        st.markdown("**Uninfected Samples**")
        uninf_dir = os.path.join(DATA_DIR, "Uninfected")
        if os.path.exists(uninf_dir):
            u_files = [f for f in os.listdir(uninf_dir) if f.lower().endswith(('.png','.jpg'))]
            u_samples = random.sample(u_files, min(5, len(u_files)))
            cols = st.columns(5)
            for i, u_file in enumerate(u_samples):
                cols[i].image(Image.open(os.path.join(uninf_dir, u_file)), use_container_width=True)
                
    st.markdown("---")
    
    with st.expander("📄 Dataset Insights"):
        insight_path = os.path.join(RESULTS_DIR, "data_insights.txt")
        if os.path.exists(insight_path):
            with open(insight_path, "r", encoding="utf-8") as f:
                st.text(f.read())
        else:
            st.info(f"No data_insights.txt found at {insight_path}")
            
    with st.expander("📄 Project Report"):
        report_path = os.path.join(REPORTS_DIR, "project_report.md")
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.info(f"No project_report.md found at {report_path}")

# ------------------------------------------
# TAB 2: MODEL INSIGHTS
# ------------------------------------------
with tab2:
    st.header(f"Insights for: {selected_display}")
    
    if best_report:
        try:
            rank = best_report.get("ranking", []).index(model_key) + 1
            total_models = len(best_report.get("ranking", []))
            st.markdown(f"### 🏆 Leaderboard Rank: #{rank} / #{total_models}")
        except ValueError:
            pass
        
    metrics = get_model_metrics(model_key)
    if metrics:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{metrics.get('Accuracy',0)*100:.2f}%")
        m2.metric("Precision", f"{metrics.get('Precision',0)*100:.2f}%")
        m3.metric("Recall (Sens.)", f"{metrics.get('Recall',0)*100:.2f}%")
        m4.metric("F1 Score", f"{metrics.get('F1',0)*100:.2f}%")
        m5.metric("ROC-AUC", f"{metrics.get('ROC-AUC',0)*100:.2f}%")
        
    st.markdown("---")
    
    def safe_image(path, caption):
        if os.path.exists(path):
            st.image(Image.open(path), caption=caption, use_container_width=True)
        else:
            st.info(f"Image not found: {os.path.basename(path)}")

    with st.expander("📈 Training & Performance Curves", expanded=False):
        c_perf1, c_perf2 = st.columns(2)
        with c_perf1:
            st.subheader("Training Curves")
            safe_image(os.path.join(RESULTS_DIR, "training_curves", f"{model_key}_curves.png"), "Training & Validation Performance")
        with c_perf2:
            st.subheader("ROC Analysis")
            safe_image(os.path.join(RESULTS_DIR, "roc_curves", f"{model_key}_roc.png"), "Receiver Operating Characteristic")

    with st.expander("🎯 Classification Evaluation", expanded=False):
        c_cm1, c_cm2 = st.columns(2)
        with c_cm1:
            st.subheader("Confusion Matrix")
            safe_image(os.path.join(RESULTS_DIR, "confusion_matrices", f"{model_key}_cm.png"), "Confusion Matrix Heatmap")
        with c_cm2:
            st.subheader("Classification Examples")
            safe_image(os.path.join(RESULTS_DIR, "sample_predictions", f"{model_key}_samples.png"), "Correct & Incorrect Predictions")

    with st.expander("🔍 Grad-CAM Explainability", expanded=False):
        st.markdown("Visual attention maps explaining model predictions.")
        safe_image(os.path.join(RESULTS_DIR, "gradcam_outputs", model_key, f"{model_key}_gradcam_grid.png"), "Grad-CAM Overlays")

    with st.expander("🛡️ Robustness Analysis", expanded=False):
        c_rob1, c_rob2 = st.columns(2)
        with c_rob1:
            safe_image(os.path.join(RESULTS_DIR, "robustness", f"{model_key}_blur.png"), "Performance under Gaussian Blur")
        with c_rob2:
            safe_image(os.path.join(RESULTS_DIR, "robustness", f"{model_key}_noise.png"), "Performance under Gaussian Noise")


# ------------------------------------------
# TAB 3: PREDICTIONS
# ------------------------------------------
with tab3:
    st.header("Predict Blood Smear Images")
    
    def run_prediction(image_pil):
        transform = get_transforms(is_training=False)
        input_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logit = model(input_tensor)
            prob = torch.sigmoid(logit).item()
        
        predicted_label = 1 if prob >= 0.5 else 0
        prediction_str = CLASS_NAMES[predicted_label]
        return predicted_label, prediction_str, prob
        
    # --- Mode 1: Manual Upload ---
    with st.expander("Option 1: Upload Custom Image", expanded=True):
        uploaded_file = st.file_uploader("Upload a cell image (PNG, JPG, BMP)", type=["png", "jpg", "jpeg", "bmp"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            col_img, col_result = st.columns([1, 1])
            
            with col_img:
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
            predicted_label, prediction_str, prob = run_prediction(image)
            confidence = prob if predicted_label == 1 else 1 - prob
            
            with col_result:
                if predicted_label == 1:
                    st.error(f"🦠 **{prediction_str}**")
                else:
                    st.success(f"✅ **{prediction_str}**")
                    
                st.metric("Confidence", f"{confidence:.1%}")
                st.markdown("**Probability Chart**")
                st.progress(prob, text=f"Parasitized: {prob:.1%}")
                st.progress(1 - prob, text=f"Uninfected: {1 - prob:.1%}")
                
            if show_gradcam:
                with st.spinner("Generating Grad-CAM..."):
                    try:
                        from src.gradcam import GradCAM, get_target_layer, overlay_heatmap, _denormalize
                        target_layer = get_target_layer(model, model_key)
                        cam_obj = GradCAM(model, target_layer)
                        
                        transform = get_transforms(is_training=False)
                        input_grad = transform(image).unsqueeze(0).to(DEVICE)
                        heatmap = cam_obj.generate_cam(input_grad)
                        cam_obj.remove_hooks()
                        
                        original_np = _denormalize(input_grad.squeeze(0))
                        overlay_np = overlay_heatmap(original_np, heatmap)
                        
                        st.markdown("#### Grad-CAM Overlay")
                        g1, g2 = st.columns(2)
                        g1.image(original_np, caption="Original Normalized", use_container_width=True)
                        g2.image((overlay_np * 255).astype(np.uint8), caption="Heatmap Overlay", use_container_width=True)
                    except Exception as e:
                        st.warning(f"Grad-CAM failed: {e}")

    # --- Mode 2: Random Dataset Prediction ---
    with st.expander("Option 2: Predict Random Dataset Images", expanded=True):
        st.markdown("Select 5 random Parasitized and 5 random Uninfected images to evaluate the current model on the fly.")
        if st.button("🚀 Predict Random Images"):
            with st.spinner("Sampling and running predictions..."):
                para_dir = os.path.join(DATA_DIR, "Parasitized")
                uninf_dir = os.path.join(DATA_DIR, "Uninfected")
                
                p_files = []
                if os.path.exists(para_dir):
                    p_files = [os.path.join(para_dir, f) for f in os.listdir(para_dir) if f.endswith('.png')]
                u_files = []
                if os.path.exists(uninf_dir):
                    u_files = [os.path.join(uninf_dir, f) for f in os.listdir(uninf_dir) if f.endswith('.png')]
                    
                if len(p_files) >= 5 and len(u_files) >= 5:
                    sampled_p = random.sample(p_files, 5)
                    sampled_u = random.sample(u_files, 5)
                    
                    all_samples = []
                    for p in sampled_p: all_samples.append((p, 1, "Parasitized"))
                    for u in sampled_u: all_samples.append((u, 0, "Uninfected"))
                    
                    random.shuffle(all_samples)
                    
                    correct_count = 0
                    cols = st.columns(5)
                    
                    for i, (fpath, true_label, true_str) in enumerate(all_samples):
                        img = Image.open(fpath).convert("RGB")
                        pred_label, pred_str, prob = run_prediction(img)
                        conf = prob if pred_label == 1 else 1 - prob
                        
                        if pred_label == true_label:
                            correct_count += 1
                            
                        # To wrap columns properly, we use i % 5, this creates 2 rows of 5
                        with cols[i % 5]:
                            st.image(img, use_container_width=True)
                            st.caption(f"**True**: {true_str}")
                            
                            if pred_label == true_label:
                                st.success(f"**Pred**: {pred_str}\n\nConf: {conf:.1%}")
                            else:
                                st.error(f"**Pred**: {pred_str}\n\nConf: {conf:.1%}")
                                
                    st.markdown("### Prediction Summary")
                    st.metric("Correct Predictions", f"{correct_count} / 10", f"{(correct_count/10)*100:.0f}% Accuracy")
                else:
                    st.error("Not enough images in the data directory to sample 5 of each class.")
