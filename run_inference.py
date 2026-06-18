#!/usr/bin/env python
"""
run_inference.py — CLI script for single‑image malaria prediction.

Usage
-----
    python run_inference.py --image path/to/cell.png
    python run_inference.py --image path/to/cell.png --model mobilenetv2
    python run_inference.py --image path/to/cell.png --model resnet18 --gradcam

WARNING: This is an educational prototype only — NOT for clinical use.
"""

import argparse
import json
import os
import sys

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import DEVICE, MODELS_DIR, create_dirs
from src.predict import load_model, predict_single_image, predict_with_gradcam


def _load_metadata():
    """Load model_config.json and class_mapping.json if available."""
    config_path = os.path.join(MODELS_DIR, "model_config.json")
    mapping_path = os.path.join(MODELS_DIR, "class_mapping.json")

    model_config = None
    class_mapping = None

    if os.path.isfile(config_path):
        with open(config_path) as f:
            model_config = json.load(f)
    if os.path.isfile(mapping_path):
        with open(mapping_path) as f:
            class_mapping = json.load(f)

    return model_config, class_mapping


def main():
    parser = argparse.ArgumentParser(
        description="Malaria Parasite Detector — CLI Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python run_inference.py --image data/Parasitized/C100P61ThinF_IMG_20150918_144104_cell_162.png\n"
            "  python run_inference.py --image cell.png --model mobilenetv2 --gradcam\n"
            "\n"
            "  DISCLAIMER: Educational prototype — NOT for clinical use."
        ),
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the cell image to classify.",
    )
    parser.add_argument(
        "--model", type=str, default="resnet18",
        choices=["simple_cnn", "resnet18", "mobilenetv2"],
        help="Which trained model to use (default: resnet18).",
    )
    parser.add_argument(
        "--gradcam", action="store_true",
        help="Generate and save a Grad‑CAM overlay alongside the prediction.",
    )

    args = parser.parse_args()

    # ── validate input ────────────────────────────────────────────
    if not os.path.isfile(args.image):
        print(f"  Image file not found: {args.image}")
        sys.exit(1)

    # ── load metadata (informational) ─────────────────────────────
    model_config, class_mapping = _load_metadata()
    if model_config and args.model in model_config:
        info = model_config[args.model]
        print(f"\n    Model: {args.model} "
              f"({info.get('architecture', '?')}, "
              f"{info.get('trainable_parameters', '?'):,} params)")
    else:
        print(f"\n    Model: {args.model}")

    if class_mapping:
        print(f"    Classes: {class_mapping.get('class_names', ['Uninfected', 'Parasitized'])}")

    # ── load model ────────────────────────────────────────────────
    create_dirs()
    try:
        model = load_model(args.model, device=DEVICE)
    except FileNotFoundError as exc:
        print(f"\n  {exc}")
        sys.exit(1)

    print(f"    Device: {DEVICE}")

    # ── run prediction ────────────────────────────────────────────
    if args.gradcam:
        result = predict_with_gradcam(
            args.image, model, args.model, DEVICE,
        )
    else:
        result = predict_single_image(args.image, model, DEVICE)

    # ── display result ────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"    Prediction Result")
    print(f"{'=' * 50}")
    print(f"  Image       : {os.path.basename(args.image)}")
    print(f"  Prediction  : {result['prediction']}")
    print(f"  Confidence  : {result['confidence']:.1%}")
    print(f"  P(Parasitized) : {result['probability_parasitized']:.4f}")
    print(f"  P(Uninfected)  : {result['probability_uninfected']:.4f}")

    if "gradcam_path" in result:
        print(f"  Grad‑CAM    : {result['gradcam_path']}")

    print(f"{'=' * 50}")
    print("\n    DISCLAIMER: Educational prototype — NOT for clinical use.\n")


if __name__ == "__main__":
    main()



