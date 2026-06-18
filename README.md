# 🦟 Malaria Parasite Detector from Blood Smear Images

> **⚠️ DISCLAIMER: This is an educational prototype only — NOT intended for clinical use or medical diagnosis.**

A deep learning pipeline for binary classification of thin blood smear cell images as **Parasitized** or **Uninfected** using PyTorch. The project trains three CNN architectures, evaluates them comprehensively, and provides local inference tools.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup](#setup)
  - [Local Setup (Windows)](#local-setup-windows)
  - [Google Colab Training](#google-colab-training)
- [Workflow](#workflow)
- [Model Architectures](#model-architectures)
- [Evaluation Metrics](#evaluation-metrics)
- [Local Inference](#local-inference)
- [Results & Outputs](#results--outputs)
- [Technical Details](#technical-details)
- [License](#license)

---

## 🔬 Project Overview

Malaria is a life-threatening disease caused by Plasmodium parasites transmitted through mosquito bites. Microscopic examination of blood smears is the gold standard for diagnosis but requires skilled technicians. This project demonstrates how AI can assist in automated screening.

### Business Use Cases
- Assistive screening tool for malaria microscopy in low-resource settings
- Educational prototype for cell-level medical image classification
- Quality-control support for microscopy image archives
- Demonstration of AI in early infectious disease screening

### Key Features
- **3 Model Architectures**: CNN from scratch, ResNet-18, MobileNetV2
- **Transfer Learning**: Fine-tuned ImageNet pretrained models
- **Mixed Precision Training (AMP)**: Faster GPU training on Colab
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, Specificity, F1, ROC-AUC
- **Robustness Testing**: Blur, noise, and reduced training data experiments
- **Grad-CAM Explainability**: Visual attention maps showing model focus areas
- **Automated Reports**: Model comparison tables, best model selection, project report

---

## 📊 Dataset

**Malaria Cell Images Dataset** — Source: NIH/LHNCBC (National Library of Medicine)

| Property | Value |
|----------|-------|
| Total Images | ~27,558 |
| Classes | 2 (Parasitized, Uninfected) |
| Balance | Approximately equal (50/50) |
| Format | PNG cell images |
| Source | Thin blood smear slides |

The dataset is publicly available on [Kaggle](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria).

---

## 📁 Project Structure

```
Malaria Parasite Detector/
├── data/                              # Dataset
│   ├── Parasitized/                   # ~13,779 parasitized cell images
│   └── Uninfected/                    # ~13,779 uninfected cell images
├── env/                               # Python virtual environment
├── src/                               # Source package
│   ├── __init__.py
│   ├── config.py                      # Hyperparameters & paths
│   ├── dataset.py                     # Data loading & augmentation
│   ├── models/
│   │   ├── __init__.py                # Model registry & factory
│   │   ├── simple_cnn.py              # CNN from scratch
│   │   ├── resnet18.py                # Transfer learning — ResNet-18
│   │   └── mobilenet.py               # Transfer learning — MobileNetV2
│   ├── train.py                       # Training loop with AMP
│   ├── evaluate.py                    # Metrics & visualization
│   ├── robustness.py                  # Robustness experiments
│   ├── gradcam.py                     # Grad-CAM explainability
│   ├── predict.py                     # Inference utilities
│   ├── sanity_checks.py              # Automated verification
│   └── report_generator.py           # Auto-generate project report
├── notebooks/
│   └── malaria_training.ipynb         # ⭐ Google Colab training notebook
├── models/                            # Exported model weights & metadata
│   ├── simple_cnn_best.pth
│   ├── resnet18_best.pth
│   ├── mobilenetv2_best.pth
│   ├── model_config.json
│   ├── class_mapping.json
│   ├── training_metrics.json
│   ├── best_model_report.json
│   ├── model_comparison.csv
│   └── *_history.csv
├── results/                           # Generated plots & reports
├── reports/
│   └── project_report.md             # Auto-generated submission report
├── app.py                             # Streamlit inference app
├── run_inference.py                   # CLI inference script
├── requirements.txt                   # Local dependencies
├── requirements_colab.txt             # Colab dependencies
└── README.md                          # This file
```

---

## 🛠️ Setup

### Local Setup (Windows)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/10Unknownboy/Malaria-Parasite-Detector.git
   cd "Malaria Parasite Detector"
   ```

2. **Activate the virtual environment**:
   ```bash
   # The venv is already created in env/
   env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset** (if not already present):
   - Download from [Kaggle](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
   - Extract into `data/` so you have `data/Parasitized/` and `data/Uninfected/`

5. **Place trained models**:
   - After training on Colab, download the exported files from Google Drive
   - Place all `.pth` and `.json` files into the `models/` folder

### Local Training (GPU recommended)

If you have a local NVIDIA GPU (or don't mind waiting for CPU training), you can run the exact same pipeline locally without using Google Colab:

1. **Activate the virtual environment**:
   ```bash
   env\Scripts\activate
   ```

2. **Run the local training orchestrator**:
   ```bash
   python run_training_local.py
   ```
   This script will automatically execute all sanity checks, data splitting, model training, evaluation, robustness testing, Grad-CAM generation, and report building. All results will be saved exactly as they would be in the Colab workflow.

### Google Colab Training

You can train the models on Google Colab using the provided notebook which handles downloading data, training, and exposing a web dashboard:

1. **Upload the notebook**:
   - Upload `notebooks/malaria_training.ipynb` to Google Colab.

2. **Enable GPU**:
   - Go to `Runtime` → `Change runtime type` → Select `T4 GPU`.

3. **Run the Notebook Cells**:
   - **Cell 1**: Clones the GitHub repository and installs dependencies.
   - **Cell 2**: Downloads the dataset via `kagglehub` (no API key required).
   - **Cell 3**: Runs the training pipeline (`python main.py`).
   - **Cell 4**: Packages all models, metrics, and Grad-CAM results into `malaria_models_and_results.zip` and automatically downloads it to your browser.
   - **Cell 5 (Streamlit Dashboard)**: Prompts you for a free [Ngrok Auth Token](https://dashboard.ngrok.com/get-started/your-authtoken), and gives you a public URL to view the Streamlit web dashboard running directly from Colab!

4. **Use Models Locally**:
   - Extract the downloaded `.zip` file on your local PC.
   - Paste the contents into your local `models/` folder. All saved `.pth` files, graphs, and JSON reports will be inside.

---

## 🔄 Workflow

```
┌─────────────────────────────────────────────────┐
│                 Google Colab (GPU)                │
│                                                   │
│  Dataset Download → Sanity Checks → EDA          │
│       ↓                                           │
│  Train CNN → Train ResNet-18 → Train MobileNetV2 │
│       ↓                                           │
│  Evaluate All → Compare → Select Best            │
│       ↓                                           │
│  Robustness Tests → Grad-CAM → Export All        │
│       ↓                                           │
│  Save to Google Drive                            │
└─────────────────────┬───────────────────────────┘
                      │ Download .pth + metadata
                      ▼
┌─────────────────────────────────────────────────┐
│               Local PC (CPU)                     │
│                                                   │
│  Load Models → CLI Inference / Streamlit App     │
│  Select: CNN | ResNet-18 | MobileNetV2           │
│  Output: Prediction + Confidence + Grad-CAM      │
└─────────────────────────────────────────────────┘
```

---

## 🧠 Model Architectures

### 1. Simple CNN (from scratch)
- 4 convolutional blocks: Conv2d → BatchNorm → ReLU → MaxPool
- Channel progression: 3 → 32 → 64 → 128 → 256
- Global Average Pooling → FC layers with Dropout
- **Purpose**: Baseline model to understand CNN fundamentals

### 2. ResNet-18 (Transfer Learning)
- Pre-trained on ImageNet (1000 classes)
- Final fully-connected layer replaced for binary classification
- Early layers optionally frozen
- **Purpose**: Demonstrate transfer learning with residual connections

### 3. MobileNetV2 (Transfer Learning)
- Pre-trained on ImageNet, lightweight architecture
- Classifier head replaced for binary output
- Depthwise separable convolutions
- **Purpose**: Efficient model suitable for mobile/edge deployment

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **Precision** | True positives / (True positives + False positives) |
| **Recall / Sensitivity** | True positives / (True positives + False negatives) — *Critical for malaria detection* |
| **Specificity** | True negatives / (True negatives + False positives) |
| **F1-Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under the ROC curve |
| **Confusion Matrix** | Detailed breakdown of predictions |

> **Note**: Recall/Sensitivity is especially important because missing a parasitized cell (false negative) can have serious clinical consequences.

---

## 🖥️ Local Inference

### CLI Script

```bash
# Activate venv
env\Scripts\activate

# Predict with ResNet-18 (default)
python run_inference.py --image path/to/cell_image.png

# Predict with specific model
python run_inference.py --image path/to/cell_image.png --model simple_cnn
python run_inference.py --image path/to/cell_image.png --model resnet18
python run_inference.py --image path/to/cell_image.png --model mobilenetv2

# With Grad-CAM visualization
python run_inference.py --image path/to/cell_image.png --model resnet18 --gradcam
```

### Streamlit Web App

```bash
# Activate venv
env\Scripts\activate

# Launch the app
streamlit run app.py
```

The app provides:
- Drag-and-drop image upload
- Model selection dropdown (all 3 models)
- Prediction with confidence score
- Grad-CAM attention visualization

---

## 📦 Results & Outputs

### 🏆 Final Training Results
After training all three architectures on Google Colab, **MobileNet V2** emerged as the best-performing model with the following metrics on the test set:
- **Accuracy**: 97.0%
- **Precision**: 97.3%
- **Recall**: 96.7%
- **F1-Score**: 96.9%
- **ROC-AUC**: 99.6%

The following files were generated and are available in the `models/` folder:

### Model Files (`models/`)
| File | Description |
|------|-------------|
| `*_best.pth` | Trained model weights for each architecture |
| `model_config.json` | Architecture details and parameter counts |
| `class_mapping.json` | Class label mapping |
| `training_metrics.json` | Final metrics for all models |
| `best_model_report.json` | Best model selection with reasoning |
| `model_comparison.csv` | Side-by-side model comparison table |
| `*_history.csv` | Epoch-wise training history per model |

### Result Plots (`results/`)
| Directory | Contents |
|-----------|----------|
| `confusion_matrices/` | Confusion matrix heatmaps |
| `roc_curves/` | ROC curve plots |
| `training_curves/` | Loss and accuracy curves |
| `gradcam_outputs/` | Grad-CAM visualizations per model |
| `robustness/` | Robustness experiment plots |
| `sample_predictions/` | Correctly/incorrectly classified examples |
| `final_results_summary.json` | Consolidated results |

### Reports (`reports/`)
| File | Description |
|------|-------------|
| `project_report.md` | Auto-generated submission-ready report |

---

## ⚙️ Technical Details

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch |
| Image Size | 128 × 128 |
| Batch Size | 64 |
| Learning Rate | 1e-3 |
| Optimizer | Adam (weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau |
| Loss Function | BCEWithLogitsLoss |
| Epochs | 20 (with early stopping) |
| Train/Val/Test Split | 70/15/15 (stratified) |
| Mixed Precision | AMP (on GPU) |
| Random Seed | 42 |
| Normalization | ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) |

### Training Augmentations
- Random Horizontal Flip
- Random Vertical Flip
- Random Rotation (±15°)
- Color Jitter (brightness=0.2, contrast=0.2, saturation=0.2)

---

## 📄 License

This project is for educational purposes only. The Malaria Cell Images dataset is provided by the NIH National Library of Medicine.

---

*Built as an educational prototype for Healthcare AI / Microscopy Image Analysis.*
