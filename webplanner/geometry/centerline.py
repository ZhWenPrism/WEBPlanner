"""Vessel centrelines near the neck and terminal vessel directions.

A vessel mask is thinned to a one-voxel skeleton (``skimage.morphology.skeletonize``, Lee's 3D algorithm), the
skeleton voxels become a 26-neighbour graph with edge weights in mm, and the centreline is the shortest path from
the skeleton node nearest to the vessel origin to the geodesically farthest node. The terminal direction is the
principal axis of the centreline points between ``skip_mm`` and ``skip_mm + length_mm`` of arc length from the
origin; ``skip_mm`` (default: the local vessel radius) steps over the junction, where the skeleton bends into the
bifurcation. When the skeleton is shorter than 1 mm the direction is the principal axis of the vessel voxels within
``length_mm`` of the origin (PCA fallback).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
from scipy import ndimage as ndi

from ..types import NeckFrame, Volume, unit

STRUCT26 = np.ones((3, 3, 3), bool)
# the 13 "forward" offsets of the 26-neighbourhood (each undirected neighbour pair is visited once)
_OFFSETS = np.array([o for o in np.ndindex(3, 3, 3) if o > (1, 1, 1)]) - 1


@dataclass
class TerminalSegment:
    direction: np.ndarray    # unit vector pointing away from the neck along the vessel
    points: np.ndarray       # (n, 3) mm, centreline points of the fitted window ordered from the neck outwards
    radius_mm: float
    method: str              # "skeleton" or "pca"


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """One-voxel skeleton of a 3D mask; distance-transform ridge when scikit-image cannot thin in 3D."""
    mask = np.asarray(mask, bool)
    try:
        from skimage.morphology import skeletonize

        return np.asarray(skeletonize(mask), bool)
    except (ImportError, ValueError, TypeError):
        edt = ndi.distance_transform_edt(mask)
        return mask & (edt >= ndi.maximum_filter(edt, footprint=STRUCT26) - 1e-6) & (edt > 0)


def skeleton_graph(skel: np.ndarray, spacing: Sequence[float]) -> Tuple[nx.Graph, np.ndarray]:
    """26-neighbour graph of skeleton voxels. Returns the graph (nodes ``0..m-1``) and their indices ``(m, 3)``."""
    coords = np.argwhere(skel)
    lookup = np.full(skel.shape, -1, np.int64)
    lookup[tuple(coords.T)] = np.arange(len(coords))
    g = nx.Graph()
    g.add_nodes_from(range(len(coords)))
    shape = np.asarray(skel.shape)
    sp = np.asarray(spacing, float)
    for off in _OFFSETS:
        nb = coords + off
        ok = np.all((nb >= 0) & (nb < shape), axis=1)
        j = np.full(len(coords), -1, np.int64)
        j[ok] = lookup[tuple(nb[ok].T)]
        i = np.flatnonzero(j >= 0)
        w = float(np.linalg.norm(off * sp))
        g.add_weighted_edges_from(zip(i.tolist(), j[i].tolist(), [w] * len(i)))
    return g, coords


def centerline_path(mask: np.ndarray, spacing: Sequence[float], start_mm: np.ndarray,
                    skel: Optional[np.ndarray] = None) -> np.ndarray:
    """Ordered centreline (mm) from the skeleton node nearest ``start_mm`` to the geodesically farthest node.

    Returns an empty ``(0, 3)`` array when the mask has no skeleton.
    """
    sp = np.asarray(spacing, float)
    skel = skeletonize_mask(mask) if skel is None else skel
    g, coords = skeleton_graph(skel, sp)
    if len(coords) == 0:
        return np.zeros((0, 3))
    pts = coords * sp
    order = np.argsort(np.linalg.norm(pts - np.asarray(start_mm, float), axis=1))
    # start in the nearest skeleton piece that is not a tiny fragment
    start = int(order[0])
    for cand in order[:50]:
        if len(nx.node_connected_component(g, int(cand))) >= 3:
            start = int(cand)
            break
    dist, paths = nx.single_source_dijkstra(g, start)
    far = max(dist, key=dist.get)
    return pts[np.asarray(paths[far])]


def pca_direction(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Principal axis of ``points`` oriented away from ``reference``."""
    points = np.asarray(points, float)
    c = points.mean(axis=0)
    if len(points) < 2:
        return unit(c - reference) if np.linalg.norm(c - reference) > 0 else np.array([1.0, 0.0, 0.0])
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    d = vt[0]
    return d if d @ (c - np.asarray(reference, float)) >= 0 else -d


def terminal_segment(point: np.ndarray, mask: np.ndarray, spacing: Sequence[float], length_mm: float = 4.0,
                     skip_mm: Optional[float] = None, reach_mm: Optional[float] = None,
                     method: str = "skeleton") -> TerminalSegment:
    """Terminal centreline segment of the vessel in ``mask`` that starts at ``point`` (mm).

    The mask is restricted to the 26-connected piece nearest to ``point`` inside a ball of ``reach_mm`` (default
    ``length_mm + 6``). ``skip_mm`` defaults to the vessel radius at the start of the centreline, clipped to
    [0.5, 3] mm. The radius is the equivalent-circle radius ``sqrt(V / (pi L))`` of the vessel voxels in the slab
    spanned by the window (skeleton method; the median distance-transform value on the centreline when the window is
    shorter than 1 mm) or the 90th-percentile distance-transform value of the window voxels divided by
    ``1 - sqrt(0.1)``, its expected ratio for a circular cross-section (PCA method).
    """
    sp = np.asarray(spacing, float)
    point = np.asarray(point, float)
    reach = float(reach_mm) if reach_mm is not None else length_mm + 6.0
    m, lo = _local_piece(np.asarray(mask, bool), sp, point, reach)
    if not m.any():
        raise ValueError("empty vessel mask")
    edt = ndi.distance_transform_edt(m, sampling=sp)
    origin = point - lo * sp                               # point in crop coordinates (mm)
    half_vox = 0.5 * float(np.mean(sp))

    if method == "skeleton":
        path = centerline_path(m, sp, origin)
        if len(path) >= 3:
            arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
            r0 = float(edt[tuple(np.round(path[0] / sp).astype(int))])
            skip = float(np.clip(r0, 0.5, 3.0)) if skip_mm is None else float(skip_mm)
            if arc[-1] >= 1.0:
                a0 = min(skip, max(arc[-1] - length_mm, 0.0))
                sel = (arc >= a0) & (arc <= a0 + length_mm)
                win = path[sel] if sel.sum() >= 3 else path
                d = pca_direction(win, win[0])
                d = d if d @ (win[-1] - win[0]) >= 0 else -d
                r_edt = float(np.median(edt[tuple(np.round(win / sp).astype(int).T)]))
                r = _slab_radius(m, sp, win, d, r_edt)
                return TerminalSegment(d, win + lo * sp, max(r, half_vox), "skeleton")
    elif method != "pca":
        raise ValueError(f"unknown method {method!r}")

    # PCA fallback: vessel voxels in the distance window around the origin
    idx = np.argwhere(m)
    pts = idx * sp
    dist = np.linalg.norm(pts - origin, axis=1)
    skip = 0.0 if skip_mm is None else float(skip_mm)
    sel = (dist >= skip) & (dist <= skip + length_mm)
    if sel.sum() < 5:
        sel = np.ones(len(pts), bool)
    d = pca_direction(pts[sel], origin)
    r = float(np.percentile(edt[tuple(idx[sel].T)], 90)) / (1.0 - np.sqrt(0.1))
    proj = (pts[sel] - origin) @ d
    win = origin + np.outer(np.linspace(proj.min(), proj.max(), 5), d)
    return TerminalSegment(d, win + lo * sp, max(r, half_vox), "pca")


def direction_near(point: Optional[np.ndarray], mask: Union[np.ndarray, Volume], frame: NeckFrame,
                   length_mm: float = 4.0, towards_neck: bool = False,
                   spacing: Optional[Sequence[float]] = None) -> np.ndarray:
    """Unit tangent of the vessel centreline over ``length_mm`` next to the neck.

    Oriented away from the neck (daughter take-off direction) or, with ``towards_neck=True``, towards it (terminal
    flow direction of the parent artery). ``point`` defaults to the neck centre; ``mask`` is a boolean array (give
    ``spacing``) or a :class:`Volume`. Falls back to PCA when the skeleton is unusable.
    """
    if isinstance(mask, Volume):
        spacing, mask = mask.spacing, np.asarray(mask.array, bool)
    if spacing is None:
        raise ValueError("spacing is required when mask is an array")
    p = frame.center if point is None else np.asarray(point, float)
    d = terminal_segment(p, mask, spacing, length_mm=length_mm).direction
    return -d if towards_neck else d


def _slab_radius(m: np.ndarray, sp: np.ndarray, win: np.ndarray, d: np.ndarray, r_guess: float) -> float:
    """Equivalent-circle radius of the vessel voxels between the window ends (within 2 radii of the window axis)."""
    c = win.mean(axis=0)
    s_win = (win - c) @ d
    s0, s1 = float(s_win.min()), float(s_win.max())
    if s1 - s0 < 1.0:
        return r_guess
    rel = np.argwhere(m) * sp - c
    s = rel @ d
    perp = np.linalg.norm(rel - np.outer(s, d), axis=1)
    n = int(((s >= s0) & (s <= s1) & (perp <= 2.0 * r_guess + float(sp.max()))).sum())
    return float(np.sqrt(n * np.prod(sp) / (np.pi * (s1 - s0))))


def _local_piece(mask: np.ndarray, sp: np.ndarray, point: np.ndarray, reach: float) -> Tuple[np.ndarray, np.ndarray]:
    """Crop ``mask`` to the ball of radius ``reach`` around ``point``; keep the piece nearest to the point."""
    c = point / sp
    half = np.ceil(reach / sp).astype(int) + 1
    lo = np.maximum(np.floor(c).astype(int) - half, 0)
    hi = np.minimum(np.ceil(c).astype(int) + half + 1, mask.shape)
    sub = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    grid = np.ogrid[tuple(slice(0, n) for n in sub.shape)]
    d2 = sum(((g + l) * s - p) ** 2 for g, l, s, p in zip(grid, lo, sp, point))
    sub = sub & (d2 <= reach ** 2)
    comp, n = ndi.label(sub, structure=STRUCT26)
    if n > 1:
        idx = np.argwhere(sub)
        nearest = idx[np.argmin(np.linalg.norm(idx * sp + lo * sp - point, axis=1))]
        sub = comp == comp[tuple(nearest)]
    if sub.any():
        sub = ndi.binary_fill_holes(sub)
    return sub, lo
