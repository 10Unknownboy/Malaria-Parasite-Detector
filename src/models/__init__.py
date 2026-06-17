"""
Models — Registry and factory for all malaria‑detection architectures.

Supported architectures
-----------------------
* ``simple_cnn``   — :class:`~src.models.simple_cnn.SimpleCNN`
* ``resnet18``     — :class:`~src.models.resnet18.ResNet18Model`
* ``mobilenetv2``  — :class:`~src.models.mobilenet.MobileNetV2Model`

Use :func:`get_model` to instantiate a model by name, or
:func:`get_all_models` to get a dictionary of all available models.

WARNING: This is an educational prototype only — NOT for clinical use.
"""

from typing import Dict

import torch.nn as nn

from src.config import MODEL_MOBILENETV2, MODEL_RESNET18, MODEL_SIMPLE_CNN
from src.models.simple_cnn import SimpleCNN
from src.models.resnet18 import ResNet18Model
from src.models.mobilenet import MobileNetV2Model


# ────────────────────────────────────────────────────────────────────
# Registry
# ────────────────────────────────────────────────────────────────────
_MODEL_REGISTRY: Dict[str, type] = {
    MODEL_SIMPLE_CNN: SimpleCNN,
    MODEL_RESNET18: ResNet18Model,
    MODEL_MOBILENETV2: MobileNetV2Model,
}


def get_model(name: str, **kwargs) -> nn.Module:
    """Instantiate and return a model by its registry key.

    Parameters
    ----------
    name : str
        One of ``"simple_cnn"``, ``"resnet18"``, ``"mobilenetv2"``.
    **kwargs
        Forwarded to the model constructor (e.g. ``pretrained``,
        ``freeze_backbone``).

    Returns
    -------
    nn.Module

    Raises
    ------
    ValueError
        If *name* is not in the registry.
    """
    name = name.lower().strip()
    if name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model '{name}'. Available models: {available}"
        )
    return _MODEL_REGISTRY[name](**kwargs)


def get_model_class(name: str) -> type:
    """Return the model **class** (not an instance) by its registry key.

    Useful when you need to instantiate the model later (e.g. for
    robustness tests that retrain from scratch).
    """
    name = name.lower().strip()
    if name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model '{name}'. Available models: {available}"
        )
    return _MODEL_REGISTRY[name]


def get_all_models(**kwargs) -> Dict[str, nn.Module]:
    """Return a dictionary mapping every registry key to a fresh model instance.

    Parameters
    ----------
    **kwargs
        Forwarded to each model constructor.

    Returns
    -------
    dict[str, nn.Module]
    """
    return {name: cls(**kwargs) for name, cls in _MODEL_REGISTRY.items()}


def count_parameters(model: nn.Module) -> int:
    """Count the total number of **trainable** parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "get_model",
    "get_model_class",
    "get_all_models",
    "count_parameters",
    "SimpleCNN",
    "ResNet18Model",
    "MobileNetV2Model",
]
