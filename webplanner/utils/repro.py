"""Seeding, environment records and logging."""
from __future__ import annotations

import importlib
import json
import logging
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

PACKAGES = ("numpy", "scipy", "scikit-learn:sklearn", "statsmodels", "scikit-image:skimage", "torch", "monai", "nnunetv2",
            "SimpleITK", "nibabel", "pydicom", "pandas", "yaml")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def environment() -> Dict[str, object]:
    """Library versions, hardware and time, written next to every experiment output."""
    versions: Dict[str, Optional[str]] = {}
    for item in PACKAGES:
        name, _, mod = item.partition(":")
        try:
            versions[name] = getattr(importlib.import_module(mod or name), "__version__", "unknown")
        except Exception:
            versions[name] = None
    env: Dict[str, object] = {"python": sys.version.split()[0], "platform": platform.platform(), "packages": versions,
                              "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        import torch

        env["cuda"] = torch.version.cuda
        env["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        env["gpus"] = []
    return env


def write_environment(out_dir: Union[str, Path]) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "environment.json"
    p.write_text(json.dumps(environment(), indent=2), encoding="utf-8")
    return p


def get_logger(name: str = "webplanner", log_file: Optional[Union[str, Path]] = None, level: int = logging.INFO) -> logging.Logger:
    if name != "webplanner" and name.startswith("webplanner."):
        get_logger("webplanner", log_file, level)          # children propagate to the package logger
        return logging.getLogger(name)
    logger = logging.getLogger(name)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        h = logging.StreamHandler()
        h.setFormatter(fmt)
        logger.addHandler(h)
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        logger.setLevel(level)
    return logger
