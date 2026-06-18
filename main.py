#!/usr/bin/env python
"""
run_training_local.py — Orchestrates the full training pipeline locally.

Run this script if you have a local GPU and want to skip Google Colab.
It performs the exact same steps as the Colab notebook:
1. Sanity Checks
2. Data Splitting & DataLoaders
3. Model Initialization
4. Training (with AMP if on GPU)
5. Evaluation & Comparison
6. Robustness Testing
7. Grad-CAM Visualization
8. Report Generation

WARNING: This is an educational prototype only — NOT for clinical use.
"""

import os
import json
import torch

# Ensure project root is on the path
import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    set_seed, DATA_DIR, DEVICE, create_dirs, MODELS_DIR, 
    RESULTS_DIR, REPORTS_DIR
)
from src.sanity_checks import run_all_checks
from src.dataset import create_data_splits, get_dataloaders, print_class_balance
from src.models import get_all_models
from src.train import train_model
from src.evaluate import evaluate_model, compare_models, save_all_metadata
from src.robustness import run_all_robustness
from src.gradcam import generate_gradcam_grid
from src.report_generator import generate_report
from src.data_insights import generate_data_insights

def main():
    print(f"{'='*50}")
    print(" Starting Local Training Pipeline")
    print(f"{'='*50}")
    
    # 1. Setup & Checks
    set_seed()
    create_dirs()
    print(f"\n Device: {DEVICE}")
    if DEVICE.type == 'cpu':
        print("  WARNING: Running on CPU. Training will be very slow!")
        print("If you have an NVIDIA GPU, ensure CUDA is installed.")
        
    print("\n--- 1. Sanity Checks ---")
    run_all_checks(DATA_DIR)
    
    # 2. Data Preparation
    print("\n--- 2. Data Preparation ---")
    generate_data_insights(DATA_DIR)
    print_class_balance(DATA_DIR)
    train_loader, val_loader, test_loader, _, _, _ = get_dataloaders(DATA_DIR)
    
    # 3. Initialize Models
    print("\n--- 3. Initialize Models ---")
    models = get_all_models()
    for name in models.keys():
        print(f" Loaded architecture: {name}")
    
    # 4. Train Models
    print("\n--- 4. Train Models ---")
    all_histories = {}
    for name, model in models.items():
        print(f"\n[{name}] Starting training...")
        model = model.to(DEVICE)
        history = train_model(model, train_loader, val_loader, name, device=DEVICE)
        all_histories[name] = history
        
    # 5. Evaluate Models
    print("\n--- 5. Evaluate Models ---")
    all_metrics = {}
    for name, model in models.items():
        print(f"\n[{name}] Evaluating on test set...")
        model.eval()
        metrics = evaluate_model(model, test_loader, name, device=DEVICE)
        all_metrics[name] = metrics
        
    # 6. Compare and Select Best
    print("\n--- 6. Compare Models ---")
    compare_models(all_metrics)
    save_all_metadata(all_metrics, all_histories, models)
    
    # Load the best model name that was just saved
    with open(os.path.join(MODELS_DIR, "best_model_report.json"), "r") as f:
        best_report = json.load(f)
    best_model_name = best_report["best_model"]
    best_model = models[best_model_name]
    
    # 7. Robustness Testing
    print(f"\n--- 7. Robustness Testing ({best_model_name}) ---")
    run_all_robustness(best_model, best_model_name, test_loader, device=DEVICE)
    
    # 8. Grad-CAM
    print("\n--- 8. Grad-CAM Visualizations ---")
    for name, model in models.items():
        print(f"[{name}] Generating Grad-CAM grids...")
        generate_gradcam_grid(model, name, test_loader.dataset, device=DEVICE)
        
    # 9. Generate Report
    print("\n--- 9. Generate Final Report ---")
    generate_report(RESULTS_DIR, MODELS_DIR, REPORTS_DIR)
    
    print(f"\n{'='*50}")
    print(" Local Training Pipeline Complete!")
    print(f"All models saved to: {MODELS_DIR}")
    print(f"All reports saved to: {REPORTS_DIR}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
