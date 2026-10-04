"""nnU-Net v2 trainer variant for WEBPlanner.

``nnUNetTrainerWEBPlanner`` keeps the nnU-Net defaults (Dice + cross-entropy with deep supervision, SGD with
Nesterov momentum, polynomial learning-rate decay, 1/3 foreground oversampling) and only shortens the schedule:
``num_epochs`` defaults to ``WEBPLANNER_EPOCHS`` (300) and can be overridden with the environment variable
``WEBPLANNER_NNUNET_EPOCHS``.

Usage
-----
nnU-Net finds trainers by class name inside the ``nnunetv2.training.nnUNetTrainer`` package, so the class has to be
visible there once::

    python -c "from webplanner.segmentation.nnunet_trainer import install_trainer; install_trainer()"

which writes a one-line stub module (``nnUNetTrainerWEBPlanner.py``) into nnU-Net's ``variants`` folder. Then::

    nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
    nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainerWEBPlanner      # folds 0..4
    nnUNetv2_predict -i imagesTs -o pred -d 501 -c 3d_fullres -tr nnUNetTrainerWEBPlanner -f 0 1 2 3 4

The patient-level folds are taken from ``splits_final.json`` written by
:func:`webplanner.data.nnunet_export.export_nnunet_dataset`. This module imports without nnU-Net; instantiating
the trainer then raises an :class:`ImportError` that explains how to install it.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

WEBPLANNER_EPOCHS = 300
_INSTALL_HINT = ("nnUNetTrainerWEBPlanner needs nnU-Net v2 (pip install nnunetv2) and its environment variables "
                 "nnUNet_raw, nnUNet_preprocessed and nnUNet_results; the built-in trainer "
                 "(webplanner.segmentation.trainer) does not")

try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer as _nnUNetTrainer
    NNUNET_AVAILABLE = True
    _IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _e:              # nnU-Net is optional
    _nnUNetTrainer = None
    NNUNET_AVAILABLE = False
    _IMPORT_ERROR = _e


def _num_epochs() -> int:
    return int(os.environ.get("WEBPLANNER_NNUNET_EPOCHS", WEBPLANNER_EPOCHS))


def require_nnunet() -> None:
    """Raise a helpful :class:`ImportError` when nnU-Net v2 is not installed."""
    if not NNUNET_AVAILABLE:
        raise ImportError(_INSTALL_HINT) from _IMPORT_ERROR


if NNUNET_AVAILABLE:

    class nnUNetTrainerWEBPlanner(_nnUNetTrainer):
        """nnU-Net default trainer with a shorter schedule (DC + CE loss and all other settings unchanged).

        ``__init__`` is not overridden because its signature differs between nnU-Net releases; the epoch count is
        applied in :meth:`initialize`, before the optimiser and the polynomial scheduler are configured.
        """

        def initialize(self):
            if not self.was_initialized:
                self.num_epochs = _num_epochs()
            return super().initialize()

else:

    class nnUNetTrainerWEBPlanner:  # type: ignore[no-redef]
        """Placeholder used when nnU-Net v2 is not installed."""

        def __init__(self, *args, **kwargs) -> None:
            require_nnunet()


def install_trainer(target_dir: Optional[Union[str, Path]] = None) -> Path:
    """Write a stub module exposing :class:`nnUNetTrainerWEBPlanner` inside nnU-Net's trainer package.

    ``target_dir`` defaults to ``nnunetv2/training/nnUNetTrainer/variants`` of the installed nnU-Net.
    Returns the path of the stub.
    """
    if target_dir is None:
        require_nnunet()
        import nnunetv2.training.nnUNetTrainer as pkg

        target_dir = Path(pkg.__file__).parent / "variants"
    target = Path(target_dir) / "nnUNetTrainerWEBPlanner.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("from webplanner.segmentation.nnunet_trainer import nnUNetTrainerWEBPlanner  # noqa: F401\n",
                      encoding="utf-8")
    return target
