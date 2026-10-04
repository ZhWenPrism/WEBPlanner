"""Segmentation networks. Importing this package does not import torch; the classes load on first access."""
from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

NETWORKS: Dict[str, Tuple[str, str]] = {
    "residual_unet": (".residual_unet", "ResidualUNet3D"),
    "unet3d": (".unet3d", "UNet3D"),
    "swin_unetr": (".swin", "build_swin_unetr"),
}
_EXPORTS = {"ResidualUNet3D": ".residual_unet", "UNet3D": ".unet3d", "build_swin_unetr": ".swin"}

__all__ = ["NETWORKS", "build_network", *_EXPORTS]


def build_network(name: str, **kw: Any):
    """Instantiate a network by name (``residual_unet``, ``unet3d``, ``swin_unetr``).

    Common keyword arguments: ``in_ch``, ``n_classes``, ``deep_supervision``; ``features`` for the U-Nets and
    ``img_size`` / ``feature_size`` for SwinUNETR.
    """
    key = name.lower()
    if key not in NETWORKS:
        raise ValueError(f"unknown network {name!r}; choose from {sorted(NETWORKS)}")
    module, attr = NETWORKS[key]
    return getattr(importlib.import_module(module, __name__), attr)(**kw)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        value = getattr(importlib.import_module(_EXPORTS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
