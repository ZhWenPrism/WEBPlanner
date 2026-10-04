"""Anatomy-aware clean-up of predicted label maps (NumPy/SciPy only).

Steps, in order:

1. sac: keep the largest connected component of the sac mask (labels 1 and 4 together); other sac components become
   background;
2. vessels: remove connected components of the artery mask (labels 2 and 3) smaller than ``min_vessel_mm3``, unless
   they touch the sac;
3. neck landing zone: a connected component of label 4 is kept only if it is adjacent to the sac body (label 1) and
   touches an artery (the artery mask dilated by one voxel); otherwise its voxels are relabelled as sac, so the sac
   volume is unchanged and only the landing-zone assignment is removed.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
from scipy import ndimage as ndi

from ..types import BACKGROUND, BRANCH, NECK, PARENT, SAC, Volume


def _structure(connectivity: int) -> np.ndarray:
    if connectivity not in (1, 2, 3):
        raise ValueError("connectivity must be 1 (6-), 2 (18-) or 3 (26-neighbourhood)")
    return ndi.generate_binary_structure(3, connectivity)


def largest_component(mask: np.ndarray, connectivity: int = 3) -> np.ndarray:
    """Boolean mask of the largest connected component (empty input gives an empty mask)."""
    lab, n = ndi.label(mask, structure=_structure(connectivity))
    if n == 0:
        return np.zeros(mask.shape, bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(sizes.argmax())


def postprocess(labels: Union[np.ndarray, Volume], spacing: Optional[Sequence[float]] = None, min_vessel_mm3: float = 2.0,
                connectivity: int = 3, keep_largest_sac: bool = True, constrain_landing_zone: bool = True
                ) -> Union[np.ndarray, Volume]:
    """Clean a predicted label map; returns ``uint8`` labels (a :class:`Volume` if a Volume was given).

    Args:
        labels: label map ``(z, y, x)`` with values in {0, 1, 2, 3, 4}.
        spacing: voxel size in mm (taken from the Volume when omitted; 1 mm otherwise).
        min_vessel_mm3: vessel islands below this volume are removed (0 disables the step).
        connectivity: neighbourhood for connected components (3 = 26-connectivity).
        keep_largest_sac: run step 1.
        constrain_landing_zone: run step 3.
    """
    vol = labels if isinstance(labels, Volume) else None
    arr = np.asarray(vol.array if vol is not None else labels)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D label map, got shape {arr.shape}")
    if spacing is None:
        spacing = vol.spacing if vol is not None else (1.0, 1.0, 1.0)
    out = arr.astype(np.uint8, copy=True)
    structure = _structure(connectivity)
    voxel_mm3 = float(np.prod(spacing))

    sac = (out == SAC) | (out == NECK)
    if keep_largest_sac and sac.any():
        keep = largest_component(sac, connectivity)
        out[sac & ~keep] = BACKGROUND
        sac = keep

    if min_vessel_mm3 > 0:
        vessels = (out == PARENT) | (out == BRANCH)
        lab, n = ndi.label(vessels, structure=structure)
        if n:
            sizes = np.bincount(lab.ravel()) * voxel_mm3
            near_sac = np.unique(lab[ndi.binary_dilation(sac, structure) & vessels])
            small = np.flatnonzero(sizes < min_vessel_mm3)
            drop = np.setdiff1d(small[small > 0], near_sac)
            if drop.size:
                out[np.isin(lab, drop)] = BACKGROUND

    if constrain_landing_zone:
        neck = out == NECK
        if neck.any():
            body = out == SAC
            arteries = ndi.binary_dilation((out == PARENT) | (out == BRANCH), structure)
            lab, _ = ndi.label(neck, structure=structure)
            keep = np.unique(lab[neck & arteries])
            if body.any():          # a sac without a body label keeps the adjacency test vacuous
                keep = np.intersect1d(keep, np.unique(lab[neck & ndi.binary_dilation(body, structure)]))
            out[neck & ~np.isin(lab, keep)] = SAC
    return Volume(out, vol.spacing, vol.origin, vol.direction) if vol is not None else out
