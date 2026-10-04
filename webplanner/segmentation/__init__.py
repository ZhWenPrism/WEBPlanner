"""Anatomy segmentation: networks, losses, training, sliding-window inference and post-processing.

Importing the package does not import torch: the torch-based classes are loaded on first attribute access.
:func:`postprocess` is NumPy/SciPy only and is imported eagerly.
"""
from __future__ import annotations

import importlib
from typing import Any

from .postprocess import postprocess

_EXPORTS = {
    "ResidualUNet3D": ".networks.residual_unet",
    "UNet3D": ".networks.unet3d",
    "build_swin_unetr": ".networks.swin",
    "build_network": ".networks",
    "DiceLoss": ".losses",
    "DiceCELoss": ".losses",
    "DeepSupervisionLoss": ".losses",
    "SegmentationTrainer": ".trainer",
    "TrainerConfig": ".trainer",
    "SlidingWindowPredictor": ".inference",
    "ensemble_predict": ".inference",
    "predict_volume": ".inference",
    "nnUNetTrainerWEBPlanner": ".nnunet_trainer",
}

__all__ = ["postprocess", *_EXPORTS]


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        value = getattr(importlib.import_module(_EXPORTS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
