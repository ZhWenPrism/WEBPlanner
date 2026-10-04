"""Overlap and surface-distance metrics for binary masks and label maps.

* Dice similarity coefficient (Dice, 1945).
* 95th-percentile Hausdorff distance: the larger of the two directed 95th percentiles of surface-to-surface
  distances (Huttenlocher et al., 1993; the convention of MONAI ``compute_hausdorff_distance``).
* Average symmetric surface distance: mean of the pooled distances in both directions (Heimann et al., 2009).

Surfaces are the mask voxels removed by one binary erosion with face connectivity; distances come from Euclidean
distance transforms in mm (``scipy.ndimage.distance_transform_edt`` with ``sampling=spacing``; Maurer et al., 2003).
An empty prediction or reference makes a metric undefined: it is returned as ``nan`` with a flag
(``"both_empty"``, ``"pred_empty"``, ``"ref_empty"``), never replaced by a default value.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage as ndi

from ..types import FOREGROUND_CLASSES, Volume

ClassSpec = Union[int, Tuple[int, ...]]
Spacing = Optional[Sequence[float]]


def _array(x: Union[np.ndarray, Volume]) -> np.ndarray:
    return x.array if isinstance(x, Volume) else np.asarray(x)


def _spacing(spacing: Spacing, ndim: int, *volumes: object) -> np.ndarray:
    if spacing is None:
        for v in volumes:
            if isinstance(v, Volume):
                spacing = v.spacing
                break
    sp = np.ones(ndim) if spacing is None else np.asarray(spacing, float)
    if sp.shape != (ndim,) or np.any(sp <= 0):
        raise ValueError(f"spacing must have {ndim} positive values, got {spacing}")
    return sp


def _pair(a: Union[np.ndarray, Volume], b: Union[np.ndarray, Volume]) -> Tuple[np.ndarray, np.ndarray]:
    x, y = _array(a).astype(bool), _array(b).astype(bool)
    if x.shape != y.shape:
        raise ValueError(f"masks differ in shape: {x.shape} vs {y.shape}")
    return x, y


def _class_mask(labels: np.ndarray, cls: ClassSpec) -> np.ndarray:
    return np.isin(labels, np.atleast_1d(np.asarray(cls)))


def empty_flag(pred_mask: np.ndarray, ref_mask: np.ndarray) -> str:
    """``""`` when both masks are non-empty, else ``"both_empty"``, ``"pred_empty"`` or ``"ref_empty"``."""
    p, r = bool(np.any(pred_mask)), bool(np.any(ref_mask))
    if p and r:
        return ""
    if not p and not r:
        return "both_empty"
    return "pred_empty" if not p else "ref_empty"


def dice(a: Union[np.ndarray, Volume], b: Union[np.ndarray, Volume]) -> float:
    """Dice coefficient ``2|A n B| / (|A| + |B|)`` (Dice, 1945); ``nan`` when both masks are empty."""
    x, y = _pair(a, b)
    s = int(x.sum()) + int(y.sum())
    return float("nan") if s == 0 else 2.0 * float(np.logical_and(x, y).sum()) / s


def dice_per_class(pred: Union[np.ndarray, Volume], ref: Union[np.ndarray, Volume],
                   classes: Iterable[ClassSpec] = FOREGROUND_CLASSES) -> Dict[ClassSpec, float]:
    """Dice of each class (an int label or a tuple of labels merged into one mask, e.g. ``(1, 4)`` for the sac)."""
    p, r = _array(pred), _array(ref)
    if p.shape != r.shape:
        raise ValueError(f"label maps differ in shape: {p.shape} vs {r.shape}")
    return {c: dice(_class_mask(p, c), _class_mask(r, c)) for c in classes}


def surface(mask: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """Surface voxels: the mask minus its binary erosion (voxels on the array border count as surface)."""
    m = np.asarray(mask, bool)
    if not m.any():
        return m.copy()
    st = ndi.generate_binary_structure(m.ndim, connectivity)
    return m & ~ndi.binary_erosion(m, structure=st, border_value=0)


def surface_distances(pred_mask: Union[np.ndarray, Volume], ref_mask: Union[np.ndarray, Volume],
                      spacing: Spacing = None) -> Tuple[np.ndarray, np.ndarray]:
    """Distances (mm) from each prediction surface voxel to the reference surface, and the reverse.

    Both arrays are empty when either mask is empty. The computation is restricted to the bounding box of the
    union of the masks (plus one voxel), which gives the same distances as the full grid.
    """
    sp = _spacing(spacing, _array(pred_mask).ndim, pred_mask, ref_mask)
    p, r = _pair(pred_mask, ref_mask)
    if not p.any() or not r.any():
        return np.empty(0), np.empty(0)
    idx = np.argwhere(p | r)
    lo = np.maximum(idx.min(axis=0) - 1, 0)
    hi = np.minimum(idx.max(axis=0) + 2, p.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    sp_, sr_ = surface(p[box]), surface(r[box])
    dt_r = ndi.distance_transform_edt(~sr_, sampling=sp)
    dt_p = ndi.distance_transform_edt(~sp_, sampling=sp)
    return dt_r[sp_], dt_p[sr_]


def hd95(pred_mask: Union[np.ndarray, Volume], ref_mask: Union[np.ndarray, Volume], spacing: Spacing = None,
         percentile: float = 95.0, return_flag: bool = False):
    """95th-percentile Hausdorff distance in mm: ``max(P95(d(P->R)), P95(d(R->P)))``.

    ``nan`` when either mask is empty; with ``return_flag=True`` returns ``(value, flag)``.
    """
    p, r = _pair(pred_mask, ref_mask)
    flag = empty_flag(p, r)
    value = float("nan")
    if not flag:
        d_pr, d_rp = surface_distances(pred_mask, ref_mask, spacing)
        value = float(max(np.percentile(d_pr, percentile), np.percentile(d_rp, percentile)))
    return (value, flag) if return_flag else value


def hausdorff(pred_mask: Union[np.ndarray, Volume], ref_mask: Union[np.ndarray, Volume], spacing: Spacing = None) -> float:
    """Maximum (100th-percentile) symmetric Hausdorff distance in mm; ``nan`` when either mask is empty."""
    return hd95(pred_mask, ref_mask, spacing, percentile=100.0)


def average_surface_distance(pred_mask: Union[np.ndarray, Volume], ref_mask: Union[np.ndarray, Volume],
                             spacing: Spacing = None, symmetric: bool = True, return_flag: bool = False):
    """Average surface distance in mm (Heimann et al., 2009).

    ``symmetric=True`` gives the ASSD, the mean of all distances pooled over both directions; ``False`` gives the
    directed mean from the prediction to the reference. ``nan`` when either mask is empty.
    """
    p, r = _pair(pred_mask, ref_mask)
    flag = empty_flag(p, r)
    value = float("nan")
    if not flag:
        d_pr, d_rp = surface_distances(pred_mask, ref_mask, spacing)
        value = float(np.concatenate([d_pr, d_rp]).mean() if symmetric else d_pr.mean())
    return (value, flag) if return_flag else value


def surface_metrics(pred_mask: Union[np.ndarray, Volume], ref_mask: Union[np.ndarray, Volume], spacing: Spacing = None,
                    percentile: float = 95.0) -> Dict[str, object]:
    """Dice, HD95, maximum Hausdorff distance and ASSD of one pair of masks, with the empty-mask flag."""
    p, r = _pair(pred_mask, ref_mask)
    flag = empty_flag(p, r)
    out: Dict[str, object] = {"dice": dice(p, r), "hd95": float("nan"), "hd": float("nan"), "assd": float("nan"),
                              "flag": flag, "n_pred": int(p.sum()), "n_ref": int(r.sum())}
    if not flag:
        d_pr, d_rp = surface_distances(pred_mask, ref_mask, spacing)
        out["hd95"] = float(max(np.percentile(d_pr, percentile), np.percentile(d_rp, percentile)))
        out["hd"] = float(max(d_pr.max(), d_rp.max()))
        out["assd"] = float(np.concatenate([d_pr, d_rp]).mean())
    return out


def per_class_metrics(pred_labels: Union[np.ndarray, Volume], ref_labels: Union[np.ndarray, Volume],
                      spacing: Spacing = None, classes: Iterable[ClassSpec] = FOREGROUND_CLASSES,
                      percentile: float = 95.0) -> Dict[ClassSpec, Dict[str, object]]:
    """Dice, HD95, HD and ASSD per class of two label maps.

    ``classes`` holds int labels or tuples of labels merged into one mask. ``spacing`` defaults to the spacing of
    a :class:`Volume` input, else 1. Each entry has ``dice, hd95, hd, assd, flag, n_pred, n_ref``.
    """
    p, r = _array(pred_labels), _array(ref_labels)
    if p.shape != r.shape:
        raise ValueError(f"label maps differ in shape: {p.shape} vs {r.shape}")
    sp = _spacing(spacing, p.ndim, pred_labels, ref_labels)
    return {c: surface_metrics(_class_mask(p, c), _class_mask(r, c), sp, percentile) for c in classes}
