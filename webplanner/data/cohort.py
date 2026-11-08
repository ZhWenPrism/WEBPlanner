"""Cohort manifest (``data/cohort.csv``) <-> :class:`CaseRecord` (see docs/DATA_INTERFACE.md)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..types import CaseRecord, DeviceSize, Volume
from ..utils.io import load_volume

REQUIRED = ("case_id", "patient_id", "centre")
MANIFEST_COLUMNS = REQUIRED + ("treatment", "location", "ruptured", "access_tortuosity", "implanted_width",
                               "implanted_height", "image", "label")


def _opt_bool(v) -> Optional[bool]:
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
        return None
    token = str(v).strip().lower()
    if token in ("1", "1.0", "true", "yes", "y"):
        return True
    if token in ("0", "0.0", "false", "no", "n"):
        return False
    raise ValueError("ruptured must be a boolean or 0/1")


def _opt_str(v) -> Optional[str]:
    return None if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "" else str(v)


def read_cohort(path: Union[str, Path]) -> List[CaseRecord]:
    path = Path(path)
    df = pd.read_csv(path, converters={name: str.strip for name in REQUIRED})
    missing = [c for c in REQUIRED if c not in df]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    for column in REQUIRED:
        empty = df[column].isna() | df[column].eq("")
        if empty.any():
            raise ValueError(f"{path}: missing {column} in {int(empty.sum())} rows")
    if df.case_id.duplicated().any():
        raise ValueError(f"{path}: duplicated case_id values")
    base = path.parent
    out = []
    for _, r in df.iterrows():
        width = _opt_str(r.get("implanted_width"))
        height = _opt_str(r.get("implanted_height"))
        if (width is None) != (height is None):
            raise ValueError(f"{path}: both implanted dimensions are required when either is present")
        imp = None
        if width is not None and height is not None:
            w, h = float(width), float(height)
            if not all(np.isfinite(v) and v > 0 for v in (w, h)):
                raise ValueError(f"{path}: implanted dimensions must be positive finite values")
            imp = DeviceSize(w, h)
        extra: Dict[str, object] = {k: r[k] for k in df.columns if k.startswith("vasoct_") and pd.notna(r[k])}
        rel = lambda k: str((base / r[k]).resolve()) if k in df and _opt_str(r.get(k)) else None
        out.append(CaseRecord(case_id=str(r["case_id"]), patient_id=str(r["patient_id"]), centre=str(r["centre"]),
                              treatment=_opt_str(r.get("treatment")) or "WEB", location=_opt_str(r.get("location")) or "other",
                              ruptured=_opt_bool(r.get("ruptured")), access_tortuosity=_opt_str(r.get("access_tortuosity")),
                              image_path=rel("image"), label_path=rel("label"), implanted=imp, extra=extra))
    return out


def write_cohort(records: Sequence[CaseRecord], path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base = path.parent.resolve()

    def export_path(value: Optional[str]) -> str:
        if not value:
            return ""
        resolved = Path(value).resolve()
        return str(resolved.relative_to(base) if resolved.is_relative_to(base) else resolved)

    rows = []
    for r in records:
        overlap = r.extra.keys() & set(MANIFEST_COLUMNS)
        if overlap:
            raise ValueError(f"extra metadata overlaps manifest columns: {sorted(overlap)}")
        rows.append({"case_id": r.case_id, "patient_id": r.patient_id, "centre": r.centre, "treatment": r.treatment,
                     "location": r.location, "ruptured": r.ruptured, "access_tortuosity": r.access_tortuosity,
                     "implanted_width": r.implanted.width if r.implanted else None,
                     "implanted_height": r.implanted.height if r.implanted else None,
                     "image": export_path(r.image_path), "label": export_path(r.label_path), **r.extra})
    pd.DataFrame(rows, columns=None if rows else MANIFEST_COLUMNS).to_csv(path, index=False)
    return path


def select(records: Sequence[CaseRecord], ids: Sequence[str]) -> List[CaseRecord]:
    keep = set(ids)
    return [r for r in records if r.case_id in keep]


def label_loader(label_dir: Optional[Union[str, Path]] = None):
    """Callable record -> label Volume, from ``record.label_path`` or ``<label_dir>/<case_id>.(npz|nii.gz)``."""
    def load(rec: CaseRecord) -> Volume:
        if label_dir is not None:
            for ext in (".npz", ".nii.gz", ".nii"):
                p = Path(label_dir) / f"{rec.case_id}{ext}"
                if p.exists():
                    return load_volume(p)
            raise FileNotFoundError(f"no label file for {rec.case_id} in {label_dir}")
        if not rec.label_path:
            raise FileNotFoundError(f"no label path for {rec.case_id}")
        return load_volume(rec.label_path)

    return load


def image_loader():
    def load(rec: CaseRecord) -> Optional[Volume]:
        return load_volume(rec.image_path) if rec.image_path else None

    return load
