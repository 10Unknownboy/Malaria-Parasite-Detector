"""
Sanity Checks — Pre‑training validation of data integrity and model plumbing.

Run :func:`run_all_checks` before any training loop to catch common issues
early (missing data, class imbalance, corrupted images, wrong output shapes).

WARNING: This is an educational prototype only — NOT for clinical use.
"""

import os
import random
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from src.config import CLASS_NAMES, DATA_DIR, IMAGE_SIZE, SEED


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────
_PASS = ""
_FAIL = ""
_WARN = ""

EXPECTED_TOTAL = 27_558  # NIH Malaria dataset reference count
BALANCE_RANGE = (0.45, 0.55)  # acceptable positive‑class fraction


def _collect_image_files(data_dir: str) -> Dict[str, List[str]]:
    """Walk class folders and return ``{class_name: [paths]}``."""
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    result: Dict[str, List[str]] = {}
    for class_name in ["Parasitized", "Uninfected"]:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            result[class_name] = []
            continue
        files = [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
        result[class_name] = files
    return result


# ────────────────────────────────────────────────────────────────────
# Individual checks
# ────────────────────────────────────────────────────────────────────
def _check_dataset_size(
    files_by_class: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Check 1: Total image count is within 5 % of expected."""
    total = sum(len(v) for v in files_by_class.values())
    lower = EXPECTED_TOTAL * 0.95
    upper = EXPECTED_TOTAL * 1.05
    passed = lower <= total <= upper

    detail = (
        f"Found {total:,} images (expected ~{EXPECTED_TOTAL:,}, "
        f"tolerance ±5 %: [{int(lower):,} – {int(upper):,}])"
    )
    return {"name": "Dataset Size", "passed": passed, "detail": detail}


def _check_class_balance(
    files_by_class: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Check 2: Positive‑class fraction is between 45 % and 55 %."""
    n_para = len(files_by_class.get("Parasitized", []))
    n_uninf = len(files_by_class.get("Uninfected", []))
    total = n_para + n_uninf
    if total == 0:
        return {
            "name": "Class Balance",
            "passed": False,
            "detail": "No images found!",
        }
    ratio = n_para / total
    passed = BALANCE_RANGE[0] <= ratio <= BALANCE_RANGE[1]
    detail = (
        f"Parasitized: {n_para:,} ({ratio:.1%})  |  "
        f"Uninfected: {n_uninf:,} ({1 - ratio:.1%})  |  "
        f"Acceptable: {BALANCE_RANGE[0]:.0%}–{BALANCE_RANGE[1]:.0%}"
    )
    return {"name": "Class Balance", "passed": passed, "detail": detail}


def _check_image_loading(
    files_by_class: Dict[str, List[str]],
    n_samples: int = 50,
) -> Dict[str, Any]:
    """Check 3: Randomly sample *n_samples* images and verify they open."""
    all_files = [f for fs in files_by_class.values() for f in fs]
    if len(all_files) == 0:
        return {
            "name": "Image Loading",
            "passed": False,
            "detail": "No images available to test.",
        }

    rng = random.Random(SEED)
    sample = rng.sample(all_files, min(n_samples, len(all_files)))

    failed: List[str] = []
    for path in sample:
        try:
            img = Image.open(path)
            img.verify()  # quick integrity check
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{os.path.basename(path)}: {exc}")

    passed = len(failed) == 0
    if passed:
        detail = f"Successfully loaded {len(sample)} random sample images."
    else:
        detail = (
            f"{len(failed)}/{len(sample)} images failed to load: "
            + "; ".join(failed[:5])
        )
    return {"name": "Image Loading", "passed": passed, "detail": detail}


def _check_model_outputs(
    models_dict: Optional[Dict[str, torch.nn.Module]],
) -> Dict[str, Any]:
    """Check 4: Feed a dummy batch through each model and verify output shape."""
    if models_dict is None or len(models_dict) == 0:
        return {
            "name": "Model Output Shape",
            "passed": True,
            "detail": "No models provided — skipped.",
        }

    dummy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    issues: List[str] = []

    for name, model in models_dict.items():
        try:
            model.eval()
            with torch.no_grad():
                out = model(dummy)
            expected_shape = (2, 1)
            if tuple(out.shape) != expected_shape:
                issues.append(
                    f"{name}: got shape {tuple(out.shape)}, "
                    f"expected {expected_shape}"
                )
        except Exception as exc:  # noqa: BLE001
            issues.append(f"{name}: forward pass failed — {exc}")

    passed = len(issues) == 0
    if passed:
        detail = (
            f"All {len(models_dict)} model(s) produce correct output shape "
            f"(2, 1) for dummy input (2, 3, {IMAGE_SIZE}, {IMAGE_SIZE})."
        )
    else:
        detail = "Issues: " + "; ".join(issues)

    return {"name": "Model Output Shape", "passed": passed, "detail": detail}


# ────────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────────
def run_all_checks(
    data_dir: str = DATA_DIR,
    models_dict: Optional[Dict[str, torch.nn.Module]] = None,
) -> Dict[str, bool]:
    """Execute all sanity checks and print a formatted report.

    Parameters
    ----------
    data_dir : str
        Root data directory containing ``Parasitized/`` and ``Uninfected/``.
    models_dict : dict[str, nn.Module] | None
        Optional dictionary of ``{model_name: model}`` for output‑shape
        verification.  If *None*, the model check is skipped.

    Returns
    -------
    dict[str, bool]
        ``{check_name: passed}`` for every check.
    """
    files_by_class = _collect_image_files(data_dir)

    checks = [
        _check_dataset_size(files_by_class),
        _check_class_balance(files_by_class),
        _check_image_loading(files_by_class),
        _check_model_outputs(models_dict),
    ]

    # ── formatted report ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("    Sanity Check Report")
    print("=" * 65)
    results: Dict[str, bool] = {}
    for chk in checks:
        icon = _PASS if chk["passed"] else _FAIL
        print(f"  {icon}  {chk['name']}")
        print(f"       {chk['detail']}")
        results[chk["name"]] = chk["passed"]
    print("=" * 65)

    all_ok = all(results.values())
    if all_ok:
        print(f"  {_PASS}  All checks passed — ready for training!\n")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"  {_FAIL}  {len(failed)} check(s) failed: {', '.join(failed)}\n")

    return results
