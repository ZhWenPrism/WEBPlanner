"""Reading and writing volumes (NIfTI through SimpleITK or nibabel, NumPy archives as a fallback) and JSON records."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Union

import numpy as np

from ..types import Volume

PathLike = Union[str, Path]


def load_volume(path: PathLike) -> Volume:
    path = Path(path)
    if path.suffix == ".npz":
        z = np.load(path)
        return Volume(z["array"], tuple(z["spacing"]), tuple(z.get("origin", (0.0, 0.0, 0.0))))
    try:
        import SimpleITK as sitk

        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img)                       # (z, y, x)
        sx, sy, sz = img.GetSpacing()
        ox, oy, oz = img.GetOrigin()
        return Volume(arr, (sz, sy, sx), (oz, oy, ox), tuple(img.GetDirection()))
    except ImportError:
        import nibabel as nib

        img = nib.load(str(path))
        arr = np.asarray(img.dataobj).transpose(2, 1, 0)        # nibabel is (x, y, z)
        sx, sy, sz = img.header.get_zooms()[:3]
        return Volume(arr, (float(sz), float(sy), float(sx)))


def save_volume(vol: Volume, path: PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".npz":
        np.savez_compressed(path, array=vol.array, spacing=np.asarray(vol.spacing), origin=np.asarray(vol.origin))
        return path
    try:
        import SimpleITK as sitk

        img = sitk.GetImageFromArray(vol.array)
        img.SetSpacing(tuple(reversed(vol.spacing)))
        img.SetOrigin(tuple(reversed(vol.origin)))
        sitk.WriteImage(img, str(path), useCompression=True)
    except ImportError:
        import nibabel as nib

        aff = np.diag(list(reversed(vol.spacing)) + [1.0])
        nib.save(nib.Nifti1Image(vol.array.transpose(2, 1, 0), aff), str(path))
    return path


class _Encoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def write_json(obj: Any, path: PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, cls=_Encoder, ensure_ascii=False), encoding="utf-8")
    return path


def read_json(path: PathLike) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
