"""Reading reconstructed 3D-RA volumes from DICOM.

Philips workstations export 3D-RA reconstructions either as a stack of single-frame images or as one multi-frame
(enhanced) object. Both are handled: slices are ordered along the slice normal, rescale slope and intercept are
applied and the spacing is taken from the pixel spacing and the slice positions.

Only technical attributes are read. Patient identifiers are never returned or logged.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from ..types import Volume

TECHNICAL_TAGS = ("Modality", "SeriesDescription", "Manufacturer", "ManufacturerModelName", "Rows", "Columns", "NumberOfFrames")


@dataclass
class SeriesInfo:
    series_uid: str
    description: str
    modality: str
    model: str
    n_files: int
    n_frames: int
    rows: int
    columns: int


def _read(path: Path, stop_before_pixels: bool = True):
    import pydicom

    return pydicom.dcmread(str(path), stop_before_pixels=stop_before_pixels, force=True)


def _image_series_uid(ds) -> str | None:
    uid = getattr(ds, "SeriesInstanceUID", None)
    if uid and getattr(ds, "Rows", 0) and getattr(ds, "Columns", 0):
        return str(uid)
    return None


def list_series(folder: Union[str, Path]) -> List[SeriesInfo]:
    """Technical summary of every series in a folder (recursive)."""
    groups: Dict[str, List[Path]] = defaultdict(list)
    first: Dict[str, object] = {}
    for p in sorted(Path(folder).rglob("*")):
        if not p.is_file():
            continue
        try:
            ds = _read(p)
        except Exception:
            continue
        uid = _image_series_uid(ds)
        if uid is not None:
            groups[uid].append(p)
            first.setdefault(uid, ds)
    out = []
    for uid, files in groups.items():
        ds = first[uid]
        out.append(SeriesInfo(uid, str(getattr(ds, "SeriesDescription", "")), str(getattr(ds, "Modality", "")),
                              str(getattr(ds, "ManufacturerModelName", "")), len(files), int(getattr(ds, "NumberOfFrames", 1) or 1),
                              int(getattr(ds, "Rows", 0)), int(getattr(ds, "Columns", 0))))
    return out


def find_3dra_series(folder: Union[str, Path], min_slices: int = 64, keywords=("3D", "3DRA", "XperCT", "Prop")) -> Optional[str]:
    """Series UID of the most likely 3D-RA reconstruction (a volume, not a rotational projection run)."""
    cands = []
    for s in list_series(folder):
        n = s.n_files if s.n_frames <= 1 else s.n_frames
        score = n + 1000 * any(k.lower() in s.description.lower() for k in keywords) + 500 * (s.rows == s.columns)
        if n >= min_slices:
            cands.append((score, s.series_uid))
    return max(cands)[1] if cands else None


def _rescale(ds, arr: np.ndarray) -> np.ndarray:
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    return arr.astype(np.float32) * slope + inter


def load_3dra_series(folder: Union[str, Path], series_uid: Optional[str] = None) -> Volume:
    """Load a reconstructed 3D-RA volume as a :class:`Volume` with arrays ``(z, y, x)``."""
    folder = Path(folder)
    series_uid = series_uid or find_3dra_series(folder)
    if series_uid is None:
        raise FileNotFoundError(f"no volumetric series found under {folder}")
    files = []
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            try:
                ds = _read(p)
            except Exception:
                continue
            if _image_series_uid(ds) == series_uid:
                files.append(p)
    if not files:
        raise FileNotFoundError("requested DICOM image series was not found")
    dss = [_read(p, stop_before_pixels=False) for p in files]
    if len(dss) == 1 and (int(getattr(dss[0], "NumberOfFrames", 1) or 1) > 1
                          or hasattr(dss[0], "PerFrameFunctionalGroupsSequence")):
        return _load_multiframe(dss[0])
    return _load_stack(dss)


def _direction(orientation) -> np.ndarray:
    iop = np.asarray(orientation, dtype=float)
    row, column = iop[:3], iop[3:]
    return np.column_stack((row, column, np.cross(row, column)))


def _load_stack(dss) -> Volume:
    direction = _direction(dss[0].ImageOrientationPatient)
    normal = direction[:, 2]
    pos = np.array([np.asarray(d.ImagePositionPatient, float) @ normal for d in dss])
    order = np.argsort(pos)
    dss = [dss[i] for i in order]
    pos = pos[order]
    arr = np.stack([_rescale(d, d.pixel_array) for d in dss])
    dz = (float(np.median(np.diff(pos))) if len(pos) > 1 else
          float(getattr(dss[0], "SpacingBetweenSlices", getattr(dss[0], "SliceThickness", 1.0))))
    dy, dx = (float(x) for x in dss[0].PixelSpacing)
    ox, oy, oz = (float(x) for x in dss[0].ImagePositionPatient)
    return Volume(arr, (abs(dz), dy, dx), (oz, oy, ox), tuple(direction.ravel()))


def _load_multiframe(ds) -> Volume:
    arr = _rescale(ds, ds.pixel_array)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError("expected a multi-frame volume")
    shared = next(iter(getattr(ds, "SharedFunctionalGroupsSequence", ())), None)
    orient = next(iter(getattr(shared, "PlaneOrientationSequence", ())), ds)
    measures = next(iter(getattr(shared, "PixelMeasuresSequence", ())), ds)
    direction = _direction(getattr(orient, "ImageOrientationPatient", (1, 0, 0, 0, 1, 0)))
    origin = getattr(ds, "ImagePositionPatient", (0., 0., 0.))
    dy, dx = (float(x) for x in getattr(measures, "PixelSpacing", getattr(ds, "PixelSpacing", (1., 1.))))
    dz = None
    try:
        positions = np.asarray([f.PlanePositionSequence[0].ImagePositionPatient
                                for f in ds.PerFrameFunctionalGroupsSequence], dtype=float)
    except (AttributeError, IndexError):
        positions = None
    if positions is not None:
        if positions.shape != (len(arr), 3):
            raise ValueError("DICOM frame positions must match the pixel frame count")
        offsets = positions @ direction[:, 2]
        order = np.argsort(offsets)
        arr = arr[order]
        origin = positions[order[0]]
        if len(order) > 1:
            dz = float(np.median(np.diff(offsets[order])))
    if dz is None:
        thickness = getattr(measures, "SliceThickness", getattr(ds, "SliceThickness", dy))
        dz = float(getattr(measures, "SpacingBetweenSlices", getattr(ds, "SpacingBetweenSlices", thickness)))
    return Volume(arr, (abs(dz), dy, dx), tuple(float(x) for x in reversed(origin)), tuple(direction.ravel()))
