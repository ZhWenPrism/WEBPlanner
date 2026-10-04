"""Neck plane from the landing zone and the vessels that leave the neck.

The neck frame is a principal component fit to the landing-zone voxels (label 4): the centre is their centroid,
the normal is the direction of least variance (the layer is thin and wide) oriented towards the dome, and ``e1`` is
the direction of largest in-plane extent. Vessel origins are the parent (label 2) and branch (label 3) components
that reach within ``radius_mm`` of the neck.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy import ndimage as ndi

from ..types import BRANCH, NECK, PARENT, SAC, NeckFrame, VesselOrigin, Volume, unit
from .centerline import terminal_segment

MIN_NECK_POINTS = 20
STRUCT26 = np.ones((3, 3, 3), bool)


def sac_mask(lab: np.ndarray) -> np.ndarray:
    """Sac voxels: labels {1, 4} (the landing zone is the base layer of the sac)."""
    return (lab == SAC) | (lab == NECK)


def neck_mask(labels: Volume, min_points: int = MIN_NECK_POINTS) -> Tuple[np.ndarray, str]:
    """Voxels that define the neck and their source.

    The landing zone (label 4) is used when it has at least ``min_points`` voxels; otherwise the sac-artery interface
    (sac voxels within one voxel, 26-neighbourhood, of label 2 or 3). Raises ``ValueError`` when neither has three
    voxels.
    """
    lab = np.asarray(labels.array)
    lz = lab == NECK
    n_lz = int(lz.sum())
    if n_lz >= min_points:
        return lz, "landing_zone"
    vessels = (lab == PARENT) | (lab == BRANCH)
    iface = sac_mask(lab) & ndi.binary_dilation(vessels, STRUCT26)
    n_if = int(iface.sum())
    if n_if >= min_points or (n_if >= 3 and n_if >= n_lz):
        return iface, "sac_artery_interface"
    if n_lz >= 3:
        return lz, "landing_zone"
    raise ValueError("no landing zone and no sac-artery interface: the neck cannot be located")


def fit_neck_frame(labels: Volume, min_points: int = MIN_NECK_POINTS) -> NeckFrame:
    """PCA neck frame (centre, normal towards the dome, in-plane axes ``e1``, ``e2 = normal x e1``).

    ``eigenvalues`` are the covariance eigenvalues in ascending order (the first belongs to the normal).
    """
    lab = np.asarray(labels.array)
    sp = np.asarray(labels.spacing)
    mask, _ = neck_mask(labels, min_points)
    pts = np.argwhere(mask) * sp
    center = pts.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov(pts, rowvar=False))
    normal = evecs[:, 0]

    to_dome = np.argwhere(sac_mask(lab)).mean(axis=0) * sp - center
    if abs(normal @ to_dome) < 1e-6:                     # flat sac: point away from the arteries instead
        to_dome = center - np.argwhere((lab == PARENT) | (lab == BRANCH)).mean(axis=0) * sp
    if normal @ to_dome < 0:
        normal = -normal
    e1 = evecs[:, 2] - (evecs[:, 2] @ normal) * normal
    e1 = unit(e1)
    e2 = np.cross(normal, e1)
    return NeckFrame(center=center, normal=normal, e1=e1, e2=e2, eigenvalues=evals, n_points=len(pts))


def vessel_origins(labels: Volume, frame: NeckFrame, radius_mm: float = 1.5, window_mm: float = 4.0,
                   junction_mm: float = 3.0, min_voxels: int = 5, method: str = "centerline") -> List[VesselOrigin]:
    """Parent and branch arteries that leave the neck.

    A vessel is a 26-connected component of label 2 or 3 with at least ``min_voxels`` voxels within ``radius_mm`` of
    the neck mask; ``point`` is the centroid of those voxels. A branch component that does not reach the neck but
    joins the parent artery within ``junction_mm`` of the neck (a daughter leaving a bifurcation apex that is
    labelled as parent) is also kept, with ``point`` at the centroid of its contact voxels; ``junction_mm=0``
    disables this. ``direction`` points away from the neck along the vessel: the terminal centreline tangent over
    ``window_mm`` (``method="centerline"``) or the principal axis of the vessel voxels within ``window_mm`` of the
    origin (``method="pca"``); ``radius_mm`` comes from the same segment (see ``terminal_segment``). The parent is
    the largest label-2 component; branches are ``branch_1``, ``branch_2``, ... by decreasing component size.
    """
    if method not in ("centerline", "pca"):
        raise ValueError(f"unknown method {method!r}")
    lab = np.asarray(labels.array)
    sp = np.asarray(labels.spacing)
    nmask, _ = neck_mask(labels)
    dist = ndi.distance_transform_edt(~nmask, sampling=sp)
    near = dist <= radius_mm
    seg_method = "skeleton" if method == "centerline" else "pca"

    out: List[VesselOrigin] = []
    contact = np.zeros(lab.shape, bool)
    for cls in (PARENT, BRANCH):
        comp, n = ndi.label(lab == cls, structure=STRUCT26)
        if n == 0:
            continue
        n_near = np.bincount(comp[near], minlength=n + 1)
        n_contact = np.bincount(comp[contact], minlength=n + 1)
        size = np.bincount(comp.ravel(), minlength=n + 1)
        keep = [i for i in range(1, n + 1) if n_near[i] >= min_voxels or n_contact[i] >= min_voxels]
        keep.sort(key=lambda i: (-size[i], i))
        if cls == PARENT:
            keep = keep[:1]
        for k, i in enumerate(keep, start=1):
            m = comp == i
            seed = m & near if n_near[i] >= min_voxels else m & contact
            point = np.argwhere(seed).mean(axis=0) * sp
            seg = terminal_segment(point, m, sp, length_mm=window_mm, method=seg_method)
            name = "parent" if cls == PARENT else f"branch_{k}"
            out.append(VesselOrigin(name=name, point=point, direction=seg.direction, radius_mm=float(seg.radius_mm)))
            if cls == PARENT and junction_mm > 0:
                contact = ndi.binary_dilation(m, STRUCT26) & (dist <= junction_mm)
    return out

