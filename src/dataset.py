"""
Dataset — Data loading, augmentation, splitting, and DataLoader creation.

Handles the Parasitized / Uninfected folder structure, applies per-split
transforms, and exposes ready-to-use DataLoaders for train / val / test.

WARNING: This is an educational prototype only — NOT for clinical use.
"""

import os
import glob

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from src.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DATA_DIR,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PARASITIZED_DIR,
    SEED,
    TEST_RATIO,
    TRAIN_RATIO,
    UNINFECTED_DIR,
    VAL_RATIO,
    set_seed,
)


# ────────────────────────────────────────────────────────────────────
# Transforms
# ────────────────────────────────────────────────────────────────────
def get_transforms(is_training=True):
    """Return an image transform pipeline.

    Parameters
    ----------
    is_training : bool
        If *True*, includes random augmentations (flip, rotation, colour
        jitter).  Otherwise only deterministic resize + normalisation.

    Returns
    -------
    torchvision.transforms.Compose
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────
class MalariaDataset(Dataset):
    """PyTorch Dataset for thin blood‑smear cell images.

    Each image is labelled:
        * **1** — Parasitized (contains *Plasmodium* parasite)
        * **0** — Uninfected (healthy red blood cell)

    Parameters
    ----------
    image_paths : list[str]
        Absolute paths to the image files.
    labels : list[int]
        Corresponding binary labels (0 or 1).
    transform : torchvision.transforms.Compose, optional
        Transform pipeline applied to each image on load.
    """

    def __init__(
        self,
        image_paths,
        labels,
        transform=None,
    ):
        assert len(image_paths) == len(labels), (
            f"Mismatch: {len(image_paths)} images vs {len(labels)} labels"
        )
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    # ── magic methods ──────────────────────────────────────────────
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

    # ── helpers ────────────────────────────────────────────────────
    def get_label_counts(self):
        """Return per-class sample counts."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return {CLASS_NAMES[int(u)]: int(c) for u, c in zip(unique, counts)}


# ────────────────────────────────────────────────────────────────────
# Collect all image paths + labels from disk
# ────────────────────────────────────────────────────────────────────
def _collect_image_paths(
    data_dir=DATA_DIR,
):
    """Walk the Parasitized / Uninfected folders and return paths + labels.

    Only files with common image extensions (.png, .jpg, .jpeg, .bmp, .tif)
    are included; Thumbs.db and other non‑image files are silently skipped.
    """
    parasitized_dir = os.path.join(data_dir, "Parasitized")
    uninfected_dir = os.path.join(data_dir, "Uninfected")

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_paths = []
    labels = []

    for img_dir, label in [(parasitized_dir, 1), (uninfected_dir, 0)]:
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(
                f"Expected class directory not found: {img_dir}"
            )
        for fname in os.listdir(img_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in valid_exts:
                image_paths.append(os.path.join(img_dir, fname))
                labels.append(label)

    return image_paths, labels


# ────────────────────────────────────────────────────────────────────
# Stratified train / val / test split
# ────────────────────────────────────────────────────────────────────
def create_data_splits(
    data_dir=DATA_DIR,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    test_ratio=TEST_RATIO,
    seed=SEED,
):
    """Split image paths into stratified train / val / test sets.

    Returns
    -------
    train_paths, train_labels,
    val_paths, val_labels,
    test_paths, test_labels
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Split ratios must sum to 1.0"
    )

    image_paths, labels = _collect_image_paths(data_dir)

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Second split: val vs test (relative ratio within the temp set)
    relative_test_ratio = test_ratio / val_test_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=relative_test_ratio,
        stratify=temp_labels,
        random_state=seed,
    )

    return (
        train_paths, train_labels,
        val_paths, val_labels,
        test_paths, test_labels,
    )


# ────────────────────────────────────────────────────────────────────
# DataLoader factory
# ────────────────────────────────────────────────────────────────────
def get_dataloaders(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    seed=SEED,
    num_workers=0,
):
    """Create train, validation, and test DataLoaders.

    Parameters
    ----------
    data_dir : str
        Root directory containing ``Parasitized/`` and ``Uninfected/``.
    batch_size : int
        Mini‑batch size.
    seed : int
        Random seed for reproducibility.
    num_workers : int
        Number of sub‑processes for data loading (0 = main process only,
        recommended on Windows).

    Returns
    -------
    train_loader, val_loader, test_loader,
    train_dataset, val_dataset, test_dataset
    """
    set_seed(seed)

    (
        train_paths, train_labels,
        val_paths, val_labels,
        test_paths, test_labels,
    ) = create_data_splits(data_dir=data_dir, seed=seed)

    train_transform = get_transforms(is_training=True)
    eval_transform = get_transforms(is_training=False)

    train_dataset = MalariaDataset(train_paths, train_labels, train_transform)
    val_dataset = MalariaDataset(val_paths, val_labels, eval_transform)
    test_dataset = MalariaDataset(test_paths, test_labels, eval_transform)

    # Reproducible worker seeding
    g = torch.Generator()
    g.manual_seed(seed)

    def _seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        generator=g,
    )

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return (
        train_loader, val_loader, test_loader,
        train_dataset, val_dataset, test_dataset,
    )


# ────────────────────────────────────────────────────────────────────
# Class balance analysis
# ────────────────────────────────────────────────────────────────────
def print_class_balance(data_dir=DATA_DIR):
    """Print a formatted class-balance summary and return counts.

    Returns
    -------
    dict
        ``{"Parasitized": <int>, "Uninfected": <int>}``
    """
    image_paths, labels = _collect_image_paths(data_dir)
    total = len(labels)
    n_parasitized = sum(labels)
    n_uninfected = total - n_parasitized

    print("\n" + "=" * 55)
    print("    Dataset Class Balance")
    print("=" * 55)
    print(f"  Total images       : {total:,}")
    print(f"  Parasitized  (1)   : {n_parasitized:,}  "
          f"({n_parasitized / total * 100:.1f}%)")
    print(f"  Uninfected   (0)   : {n_uninfected:,}  "
          f"({n_uninfected / total * 100:.1f}%)")
    print("=" * 55 + "\n")

    return {"Parasitized": n_parasitized, "Uninfected": n_uninfected}


def print_split_summary(
    train_labels,
    val_labels,
    test_labels,
):
    """Print a formatted per-split summary."""
    print("\n" + "=" * 55)
    print("    Data Split Summary")
    print("=" * 55)
    for name, lbls in [("Train", train_labels),
                        ("Val", val_labels),
                        ("Test", test_labels)]:
        total = len(lbls)
        pos = sum(lbls)
        neg = total - pos
        print(f"  {name:5s} — Total: {total:>6,}  |  "
              f"P: {pos:>5,} ({pos/total*100:.1f}%)  |  "
              f"U: {neg:>5,} ({neg/total*100:.1f}%)")
    print("=" * 55 + "\n")



