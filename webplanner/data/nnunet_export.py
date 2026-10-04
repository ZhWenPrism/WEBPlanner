"""Export cases to the nnU-Net v2 raw dataset layout, with our patient-level folds as ``splits_final.json``.

Layout written under ``out_dir`` (normally ``$nnUNet_raw``)::

    DatasetXXX_Name/
        imagesTr/<case>_0000.nii.gz
        labelsTr/<case>.nii.gz
        dataset.json
        splits_final.json          (only when folds are given)

nnU-Net reads ``splits_final.json`` from ``$nnUNet_preprocessed/DatasetXXX_Name/``; pass ``preprocessed_dir`` to
write it there as well, or copy it after ``nnUNetv2_plan_and_preprocess``. Folds must be defined at patient level
(see ``webplanner.data.splits``); this function only checks that no case is in both parts of a fold.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..types import LABELS, Volume
from ..utils.io import save_volume, write_json

NNUNET_LABELS: Dict[str, int] = {name: value for value, name in sorted(LABELS.items())}


def _check_folds(folds: Sequence[Dict[str, Sequence[str]]], case_ids: Sequence[str]) -> List[Dict[str, List[str]]]:
    known = set(case_ids)
    out = []
    for i, fold in enumerate(folds):
        train, val = [str(c) for c in fold["train"]], [str(c) for c in fold["val"]]
        unknown = (set(train) | set(val)) - known
        if unknown:
            raise ValueError(f"fold {i}: unknown case ids {sorted(unknown)[:5]}")
        overlap = set(train) & set(val)
        if overlap:
            raise ValueError(f"fold {i}: cases in both train and val {sorted(overlap)[:5]}")
        out.append({"train": train, "val": val})
    return out


def export_nnunet_dataset(cases: Sequence[Tuple[str, Volume, Volume]], out_dir: Union[str, Path], dataset_id: int = 501,
                          dataset_name: str = "WEBPlanner", folds: Optional[Sequence[Dict[str, Sequence[str]]]] = None,
                          preprocessed_dir: Optional[Union[str, Path]] = None, file_ending: str = ".nii.gz") -> Path:
    """Write ``cases`` (``(case_id, image, label)``) in nnU-Net v2 raw format; returns the dataset folder.

    Images are stored as float32, labels as uint8 with the WEBPlanner label map; image and label must share shape
    and spacing. ``folds`` is a list of ``{"train": [...], "val": [...]}`` with case ids.
    """
    if not 1 <= int(dataset_id) <= 999:
        raise ValueError("nnU-Net dataset ids are 1 to 999")
    folder = f"Dataset{int(dataset_id):03d}_{dataset_name}"
    root = Path(out_dir) / folder
    images_dir, labels_dir = root / "imagesTr", root / "labelsTr"
    ids = [str(c[0]) for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case ids")
    valid = np.asarray(sorted(LABELS), dtype=np.int64)
    for case_id, image, label in cases:
        if image.shape != label.shape:
            raise ValueError(f"{case_id}: image {image.shape} and label {label.shape} differ")
        if not np.allclose(image.spacing, label.spacing):
            raise ValueError(f"{case_id}: image and label spacing differ")
        values = np.unique(label.array)
        if not np.isin(values, valid).all():
            raise ValueError(f"{case_id}: label values {values.tolist()} outside {valid.tolist()}")
        save_volume(Volume(image.array.astype(np.float32), image.spacing, image.origin, image.direction),
                    images_dir / f"{case_id}_0000{file_ending}")
        save_volume(Volume(label.array.astype(np.uint8), image.spacing, image.origin, image.direction),
                    labels_dir / f"{case_id}{file_ending}")
    write_json({"name": dataset_name,
                "description": "WEBPlanner four-class anatomy segmentation of 3D rotational angiography",
                "channel_names": {"0": "3DRA"}, "labels": NNUNET_LABELS, "numTraining": len(ids),
                "file_ending": file_ending}, root / "dataset.json")
    if folds is not None:
        splits = _check_folds(folds, ids)
        write_json(splits, root / "splits_final.json")
        if preprocessed_dir is not None:
            write_json(splits, Path(preprocessed_dir) / folder / "splits_final.json")
    return root
