"""Volume preprocessing: isotropic resampling, aneurysm-centred cropping and intensity normalisation.

The intensity statistics are estimated on the training cases of a fold only (:class:`Preprocessor.fit`) and then
applied unchanged to validation, external and clinical-evaluation cases.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage as ndi

from ..types import Volume


def resample_isotropic(vol: Volume, spacing: float, order: int = 1) -> Volume:
    """Resample to an isotropic grid. Use ``order=0`` for label maps."""
    zoom = np.asarray(vol.spacing) / spacing
    if np.allclose(zoom, 1.0):
        return vol
    arr = ndi.zoom(vol.array, zoom, order=order, mode="nearest", prefilter=order > 1)
    return Volume(arr.astype(vol.array.dtype, copy=False), (spacing, spacing, spacing), vol.origin, vol.direction)


def crop_cube(vol: Volume, center_mm: Sequence[float], size_mm: float = 40.0, fill: Optional[float] = None) -> Tuple[Volume, np.ndarray]:
    """Cube of ``size_mm`` centred on ``center_mm`` (array frame), padded where it leaves the volume.

    Returns the crop and the index offset of its first voxel in the source volume.
    """
    sp = np.asarray(vol.spacing)
    half = np.round(size_mm / 2 / sp).astype(int)
    c = np.round(np.asarray(center_mm) / sp).astype(int)
    lo, hi = c - half, c + half
    fill = float(vol.array.min()) if fill is None else fill
    out = np.full(tuple(hi - lo), fill, dtype=vol.array.dtype)
    src_lo, src_hi = np.maximum(lo, 0), np.minimum(hi, vol.shape)
    dst_lo = src_lo - lo
    dst_hi = dst_lo + (src_hi - src_lo)
    if np.all(src_hi > src_lo):
        out[tuple(slice(a, b) for a, b in zip(dst_lo, dst_hi))] = vol.array[tuple(slice(a, b) for a, b in zip(src_lo, src_hi))]
    return Volume(out, vol.spacing, vol.origin, vol.direction), lo


def uncrop(arr: np.ndarray, offset: np.ndarray, shape: Sequence[int], fill: int = 0) -> np.ndarray:
    """Place a crop back into a volume of ``shape``."""
    out = np.full(tuple(shape), fill, dtype=arr.dtype)
    lo = np.asarray(offset)
    src_lo = np.maximum(-lo, 0)
    dst_lo = np.maximum(lo, 0)
    n = np.minimum(np.asarray(arr.shape) - src_lo, np.asarray(shape) - dst_lo)
    if np.all(n > 0):
        out[tuple(slice(a, a + k) for a, k in zip(dst_lo, n))] = arr[tuple(slice(a, a + k) for a, k in zip(src_lo, n))]
    return out


@dataclass
class Preprocessor:
    """Percentile clipping and z-scoring with statistics from foreground voxels of the training cases."""

    spacing: float = 0.30
    crop_mm: float = 40.0
    lower_percentile: float = 0.5
    upper_percentile: float = 99.5
    clip_low: Optional[float] = None
    clip_high: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    def fit(self, volumes: Iterable[Volume], foreground: Optional[Iterable[np.ndarray]] = None, max_voxels: int = 200_000,
            seed: int = 0) -> "Preprocessor":
        rng = np.random.default_rng(seed)
        samples = []
        fg_iter = iter(foreground) if foreground is not None else None
        for vol in volumes:
            a = vol.array.astype(np.float32)
            m = next(fg_iter) > 0 if fg_iter is not None else a > np.percentile(a, 90)
            v = a[m]
            if v.size > max_voxels:
                v = rng.choice(v, max_voxels, replace=False)
            samples.append(v)
        if not samples:
            raise ValueError("no training volume")
        allv = np.concatenate(samples)
        self.clip_low, self.clip_high = (float(x) for x in np.percentile(allv, [self.lower_percentile, self.upper_percentile]))
        c = np.clip(allv, self.clip_low, self.clip_high)
        self.mean, self.std = float(c.mean()), float(c.std() + 1e-8)
        return self

    def transform(self, vol: Volume) -> Volume:
        if self.mean is None:
            raise RuntimeError("Preprocessor.fit must be called on training data first")
        v = resample_isotropic(vol, self.spacing, order=1)
        a = (np.clip(v.array.astype(np.float32), self.clip_low, self.clip_high) - self.mean) / self.std
        return Volume(a.astype(np.float32), v.spacing, v.origin, v.direction)

    def transform_labels(self, lab: Volume) -> Volume:
        return resample_isotropic(lab, self.spacing, order=0)

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Preprocessor":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def aneurysm_seed(vol: Volume, threshold_percentile: float = 99.0) -> np.ndarray:
    """Rough aneurysm centre for the initial crop: the bright blob with the largest inscribed sphere.

    Used only when no manual seed is given; the segmentation network then refines the anatomy.
    """
    a = ndi.gaussian_filter(vol.array.astype(np.float32), 1.0)
    mask = a > np.percentile(a, threshold_percentile)
    dt = ndi.distance_transform_edt(mask, sampling=vol.spacing)
    return np.asarray(np.unravel_index(int(np.argmax(dt)), dt.shape)) * np.asarray(vol.spacing)
