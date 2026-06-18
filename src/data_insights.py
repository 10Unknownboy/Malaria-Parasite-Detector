"""
Data Insights — Generates a comprehensive text report about the dataset.
"""

import os
import random
import numpy as np
from PIL import Image

from src.config import DATA_DIR, RESULTS_DIR

def generate_data_insights(data_dir=DATA_DIR, output_file=None):
    """
    Analyzes the dataset directory to compute total counts, class balance, 
    and average image dimensions, then writes a formatted text report to disk.
    
    Parameters
    ----------
    data_dir : str
        The root path to the training dataset.
    output_file : str, optional
        Where to save the text report.
    """
    if output_file is None:
        output_file = os.path.join(RESULTS_DIR, "data_insights.txt")
        
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    classes = ["Parasitized", "Uninfected"]
    insights = []
    
    insights.append("=" * 50)
    insights.append(" MALARIA DATASET INSIGHTS")
    insights.append("=" * 50 + "\n")
    
    total_images = 0
    class_counts = {}
    
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        if os.path.exists(cls_path):
            files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            count = len(files)
            class_counts[cls] = count
            total_images += count
        else:
            class_counts[cls] = 0
            
    insights.append(f"Total Images: {total_images}")
    for cls, count in class_counts.items():
        percentage = (count / total_images * 100) if total_images > 0 else 0
        insights.append(f"  - {cls}: {count} ({percentage:.2f}%)")
        
    insights.append("\n" + "=" * 50)
    insights.append(" IMAGE DIMENSIONS (Sampled 500 images per class)")
    insights.append("=" * 50)
    
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        if not os.path.exists(cls_path): 
            continue
        
        files = [os.path.join(cls_path, f) for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        sampled_files = random.sample(files, min(500, len(files)))
        
        widths = []
        heights = []
        
        for f in sampled_files:
            try:
                with Image.open(f) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except Exception:
                continue
                
        if widths and heights:
            insights.append(f"\n[{cls}]")
            insights.append(f"  - Min Dimensions  : {min(widths)} x {min(heights)}")
            insights.append(f"  - Max Dimensions  : {max(widths)} x {max(heights)}")
            insights.append(f"  - Mean Dimensions : {int(np.mean(widths))} x {int(np.mean(heights))}")
            
    insights.append("\n" + "=" * 50)
    insights.append(" CONCLUSION")
    insights.append("=" * 50)
    if class_counts.get("Parasitized", 0) == class_counts.get("Uninfected", 0):
        insights.append("Dataset is perfectly balanced.")
    elif total_images > 0:
        ratio = min(class_counts.values()) / max(class_counts.values())
        if ratio > 0.9:
            insights.append("Dataset is well balanced.")
        else:
            insights.append("Dataset has a class imbalance. Consider using class weights or augmentation.")
            
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(insights))
        
    print(f"   Data insights saved to: {output_file}")





