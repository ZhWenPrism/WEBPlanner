"""Sac morphometry in the neck frame.

All measurements use the sac mask (labels {1, 4}) in local coordinates (t along the normal, u along ``e1``, v along
``e2``). Definitions
--------------------
* Ostium plane: the lower face of the landing-zone layer, ``t_base`` = 1st percentile of the landing-zone ``t``
  minus half a voxel. Heights are measured from it (``h = t - t_base``); the frame centre, the landing-zone
  centroid, sits about half a layer above it.
* Voxel-centre extents are converted to physical extents by adding one voxel length along the measurement
  direction (half a voxel at each end); the voxel length along a unit direction d is ``sqrt(sum((d_i s_i)^2))``.
* Widths are measured parallel to the neck plane in height slabs of two voxels: the extent of a slab along an
  in-plane direction is the 0.3 to 99.7 percentile range of the projected coordinates, and a width is the maximum
  over slabs. ``width_1`` uses ``e1``, ``width_2`` uses ``e2``, ``max_width`` the largest of 36 directions over
  180 degrees.
* ``height`` is the 99.7th percentile of ``h`` over sac voxels with ``h >= 0``.
* ``smallest_height`` approximates the smaller of the heights read on two orthogonal working projections, each
  measured from the neck line at the neck centre: the projection along ``e2`` shows, at u = 0, the highest sac voxel
  of the band ``|u| <= band_mm`` (all v), and the projection along ``e1`` the band ``|v| <= band_mm``. It is the
  smaller of the two band heights (99.7th percentile), capped at ``height``.
* ``neck_width`` is the largest diameter of the landing zone projected on the neck plane (maximum distance between
  convex-hull vertices, plus one voxel as above).
* ``solidity`` is digital: sac voxels divided by the lattice points inside the convex hull of the sac voxel
  centres. A digitised convex sac scores exactly 1 and the measure is free of the voxel-size bias of a continuous
  volume / hull-volume ratio.
* ``branch_incorporated``: at least ``min_contact`` sac voxels touch a branch artery (label 3, 6-neighbourhood) more
  than ``incorporation_mm`` above the top of the landing-zone layer, or a branch origin point lies inside the neck
  footprint cylinder (in-plane radius <= neck_width / 2 + 0.5 mm) at that height. Contacts within the landing-zone
  layer are the normal relation of daughter vessels to a bifurcation neck and are not counted.
* ``sac_axis_extent`` (h_sac) is the distance from the frame centre to the dome along the normal ray: the first 0.5
  crossing of the trilinearly interpolated sac mask, marched in 0.05 mm steps from the centre (the convention of
  ``device.deployment``). The axial length of the sac from the ostium plane is ``h_sac - ostium_offset``.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, Delaunay, QhullError

from ..types import BRANCH, Morphometry, NeckFrame, VesselOrigin, Volume
from .neck import neck_mask, sac_mask

STRUCT6 = ndi.generate_binary_structure(3, 1)


def voxel_length(direction: np.ndarray, spacing: Sequence[float]) -> float:
    """Length of one voxel along a unit direction, ``sqrt(sum((d_i s_i)^2))`` (the spacing for isotropic voxels)."""
    return float(np.linalg.norm(np.asarray(direction, float) * np.asarray(spacing, float)))


def ostium_offset(labels: Volume, frame: NeckFrame) -> float:
    """Signed position (mm, along the normal) of the ostium plane relative to the frame centre (normally < 0)."""
    sp = np.asarray(labels.spacing)
    nmask, _ = neck_mask(labels)
    t = (np.argwhere(nmask) * sp - frame.center) @ frame.normal
    return float(np.percentile(t, 1.0)) - 0.5 * voxel_length(frame.normal, sp)


def slab_extents(h: np.ndarray, uv: np.ndarray, directions: np.ndarray, slab_mm: float, q: float = 0.3,
                 min_count: int = 8) -> np.ndarray:
    """Maximum over height slabs of the ``q`` to ``100 - q`` percentile range along each in-plane direction.

    ``uv`` is ``(n, 2)``, ``directions`` is ``(k, 2)`` (unit vectors in the (u, v) plane). Returns ``(k,)``.
    """
    proj = uv @ directions.T
    slab = np.floor(h / slab_mm).astype(int)
    order = np.argsort(slab, kind="stable")
    bounds = np.flatnonzero(np.diff(slab[order])) + 1
    best = np.zeros(len(directions))
    for idx in np.split(order, bounds):
        if len(idx) < min_count:
            continue
        lo, hi = np.percentile(proj[idx], [q, 100.0 - q], axis=0)
        best = np.maximum(best, hi - lo)
    return best


def max_diameter_2d(xy: np.ndarray) -> float:
    """Largest pairwise distance of 2D points (on the convex hull vertices)."""
    if len(xy) < 2:
        return 0.0
    try:
        xy = xy[ConvexHull(xy).vertices]
    except (QhullError, ValueError):
        pass
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    return float(d.max())


def digital_solidity(mask: np.ndarray, spacing: Sequence[float]) -> float:
    """Voxels of ``mask`` divided by the lattice points inside the convex hull of its voxel centres.

    Returns 1.0 for degenerate (fewer than 4 or coplanar) masks.
    """
    sp = np.asarray(spacing, float)
    n = int(mask.sum())
    if n < 4:
        return 1.0
    shell = mask & ~ndi.binary_erosion(mask, STRUCT6)
    try:
        hull = ConvexHull(np.argwhere(shell) * sp)
        tri = Delaunay(hull.points[hull.vertices])
    except (QhullError, ValueError):
        return 1.0
    lo = np.argwhere(mask).min(axis=0)
    hi = np.argwhere(mask).max(axis=0) + 1
    box = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    cand = (np.argwhere(~box) + lo) * sp
    extra = int((tri.find_simplex(cand) >= 0).sum()) if len(cand) else 0
    return n / (n + extra)


def axis_extent(sac: np.ndarray, spacing: Sequence[float], center: np.ndarray, normal: np.ndarray,
                step_mm: float = 0.05) -> Tuple[float, float]:
    """(t_bottom, t_top) of the sac along the line ``center + t * normal`` (0.5 crossings, trilinear sampling)."""
    sp = np.asarray(spacing, float)
    t_max = float(np.linalg.norm(np.asarray(sac.shape) * sp))
    t = np.arange(-t_max, t_max + step_mm, step_mm)
    pts = (center[None, :] + t[:, None] * normal[None, :]) / sp
    val = ndi.map_coordinates(sac.astype(np.float32), pts.T, order=1, mode="constant", cval=0.0)
    inside = val >= 0.5
    if not inside.any():
        return 0.0, 0.0
    i0 = int(np.argmin(np.abs(t)))
    if not inside[i0]:                      # centre outside the sac: start at the first inside sample above it
        above = np.flatnonzero(inside & (t >= 0))
        i0 = int(above[0]) if len(above) else int(np.flatnonzero(inside)[-1])
    out = np.flatnonzero(~inside)
    up = out[out > i0]
    dn = out[out < i0]
    i_up = int(up[0]) if len(up) else len(t) - 1
    i_dn = int(dn[-1]) if len(dn) else 0

    def crossing(i_in: int, i_out: int) -> float:
        a, b = val[i_in], val[i_out]
        f = (a - 0.5) / (a - b) if a != b else 0.5
        return float(t[i_in] + f * (t[i_out] - t[i_in]))

    return crossing(i_dn + 1, i_dn), crossing(i_up - 1, i_up)


def measure(labels: Volume, frame: NeckFrame, origins: Optional[List[VesselOrigin]] = None,
            irregular_threshold: float = 0.90, slab_voxels: float = 2.0, n_directions: int = 36,
            band_mm: Optional[float] = None, incorporation_mm: float = 0.5, min_contact: int = 3) -> Morphometry:
    """Sac morphometry (see the module docstring for the definitions).

    ``band_mm`` (half-width of the projection bands for ``smallest_height``) defaults to max(0.5 mm, 2 voxels).
    """
    lab = np.asarray(labels.array)
    sp = np.asarray(labels.spacing)
    sac = sac_mask(lab)
    if not sac.any():
        raise ValueError("empty sac mask")
    nmask, _ = neck_mask(labels)
    loc = frame.to_local(np.argwhere(sac) * sp)
    t_base = ostium_offset(labels, frame)
    h, uv = loc[:, 0] - t_base, loc[:, 1:]
    s_n = voxel_length(frame.normal, sp)

    # widths parallel to the neck plane: e1, e2, then n_directions over 180 degrees
    theta = np.concatenate([[0.0, np.pi / 2], np.arange(n_directions) * np.pi / n_directions])
    dirs2 = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    dirs3 = np.cos(theta)[:, None] * frame.e1 + np.sin(theta)[:, None] * frame.e2
    above = h >= 0
    ext = slab_extents(h[above], uv[above], dirs2, slab_voxels * s_n)
    ext = ext + np.array([voxel_length(d, sp) for d in dirs3])
    width_1, width_2 = float(ext[0]), float(ext[1])
    mean_width = 0.5 * (width_1 + width_2)
    max_width = float(ext.max())

    # heights
    height = float(np.percentile(h[above], 99.7)) + 0.5 * s_n
    band = band_mm if band_mm is not None else max(0.5, 2.0 * float(np.mean(sp)))
    band_heights = []
    for coord in (uv[:, 0], uv[:, 1]):
        sel = above & (np.abs(coord) <= band)
        if sel.sum() >= 5:
            band_heights.append(float(np.percentile(h[sel], 99.7)) + 0.5 * s_n)
    smallest_height = min(band_heights + [height])

    # neck
    lz_loc = frame.to_local(np.argwhere(nmask) * sp)
    neck_width = max_diameter_2d(lz_loc[:, 1:]) + 0.5 * (voxel_length(frame.e1, sp) + voxel_length(frame.e2, sp))
    lz_top = float(np.percentile(lz_loc[:, 0], 99.0)) - t_base + 0.5 * s_n

    # volume and shape
    volume = float(sac.sum() * np.prod(sp))
    solidity = digital_solidity(sac, sp)

    # branch incorporation
    h_inc = lz_top + incorporation_mm
    touch = sac & ndi.binary_dilation(lab == BRANCH, STRUCT6)
    incorporated = False
    if touch.any():
        h_touch = (np.argwhere(touch) * sp - frame.center) @ frame.normal - t_base
        incorporated = int((h_touch > h_inc).sum()) >= min_contact
    for o in origins or []:
        if o.name.startswith("branch"):
            t_o, u_o, v_o = frame.to_local(o.point)
            if t_o - t_base > h_inc and np.hypot(u_o, v_o) <= 0.5 * neck_width + 0.5:
                incorporated = True

    _, t_up = axis_extent(sac, sp, frame.center, frame.normal)
    ratio = mean_width / neck_width if neck_width > 0 else float("nan")
    return Morphometry(mean_width=mean_width, max_width=max_width, width_1=width_1, width_2=width_2, height=height,
                       neck_width=neck_width, width_neck_ratio=ratio,
                       volume_mm3=volume, solidity=float(solidity), irregular=bool(solidity < irregular_threshold),
                       branch_incorporated=incorporated, sac_axis_extent=float(t_up),
                       smallest_height=float(smallest_height))
