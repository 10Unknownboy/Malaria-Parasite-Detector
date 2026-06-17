"""
Configuration — Central hyperparameters, paths, and reproducibility settings.
"""

import os
import random
import numpy as np
import torch


# =============================================================================
# Reproducibility
# =============================================================================
SEED = 42


def set_seed(seed: int = SEED):
    """Set random seeds for full reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# =============================================================================
# Paths
# =============================================================================
# Project root is two levels up from this file (src/config.py → project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# Result subdirectories
CONFUSION_MATRIX_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")
ROC_CURVES_DIR = os.path.join(RESULTS_DIR, "roc_curves")
TRAINING_CURVES_DIR = os.path.join(RESULTS_DIR, "training_curves")
GRADCAM_DIR = os.path.join(RESULTS_DIR, "gradcam_outputs")
ROBUSTNESS_DIR = os.path.join(RESULTS_DIR, "robustness")
SAMPLE_PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "sample_predictions")

# Class directories
PARASITIZED_DIR = os.path.join(DATA_DIR, "Parasitized")
UNINFECTED_DIR = os.path.join(DATA_DIR, "Uninfected")


def create_dirs():
    """Create all output directories if they don't exist."""
    dirs = [
        MODELS_DIR, RESULTS_DIR, REPORTS_DIR,
        CONFUSION_MATRIX_DIR, ROC_CURVES_DIR, TRAINING_CURVES_DIR,
        GRADCAM_DIR, ROBUSTNESS_DIR, SAMPLE_PREDICTIONS_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# =============================================================================
# Data
# =============================================================================
IMAGE_SIZE = 128          # Resize all images to IMAGE_SIZE x IMAGE_SIZE
NUM_CLASSES = 1           # Binary classification (single sigmoid output)
CLASS_NAMES = ["Uninfected", "Parasitized"]
CLASS_MAPPING = {0: "Uninfected", 1: "Parasitized"}

# Dataset splits
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# =============================================================================
# Training Hyperparameters
# =============================================================================
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR = 0.5
WEIGHT_DECAY = 1e-4

# Mixed Precision Training
USE_AMP = True  # Automatic Mixed Precision — only effective on CUDA

# =============================================================================
# Model Names (registry keys)
# =============================================================================
MODEL_SIMPLE_CNN = "simple_cnn"
MODEL_RESNET18 = "resnet18"
MODEL_MOBILENETV2 = "mobilenetv2"
ALL_MODELS = [MODEL_SIMPLE_CNN, MODEL_RESNET18, MODEL_MOBILENETV2]

# =============================================================================
# Normalization (ImageNet statistics for transfer learning)
# =============================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================================================================
# Device
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
