"""Deployment-stage rehearsal: schematic fluoroscopic appearance of a planned WEB at the manufacturer's stages.

Stage geometry (along the device axis, ``t`` in mm from the neck centre; the deployed device spans
``[t_base, t_top]`` as in :mod:`webplanner.device.deployment`):

* ``constrained``: the device is collapsed in the microcatheter (radius ``constrained_radius_mm``, length
  ``constrained_length_ratio * H'``) with its distal marker at the final distal position;
* ``partial_expansion``: the distal ``partial_fraction`` of the device is unsheathed and has its final conformed shape,
  tapered into the catheter over its lowest part; the rest is still collapsed below it;
* ``full_expansion``: the conformed device at its final position, still attached to the delivery wire;
* ``detached``: the conformed device alone.

The unsheathing sequence (the distal part opens in place while the catheter is withdrawn) and every constant tagged
ASSUMED in :class:`FluoroConfig` are schematic choices for a visual rehearsal, not device specifications. Images are
parallel projections along ``view_dir`` of a slab around the device: the line integral (mm) of the vessel lumen
(labels > 0, faint), the device bodies and the delivery wire, each times a display weight, plus the two radiopaque
markers rendered analytically as balls. Rows run from the dome side (top) to the neck side; columns run along
``right = view_dir x up``. The weights are display choices, not attenuation coefficients.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
from scipy import ndimage as ndi

from ..types import DeploymentResult, NeckFrame, Volume, unit
from .deployment import (DeploymentConfig, DeviceProfile, SacField, VirtualDeployer, device_span, interp_profile,
                         orthonormal_frame)

STAGES = ("constrained", "partial_expansion", "full_expansion", "detached")


@dataclass
class FluoroConfig:
    constrained_radius_mm: float = 0.25        # ASSUMED collapsed braid radius inside the microcatheter
    constrained_length_ratio: float = 2.0      # ASSUMED collapsed length / deployed height
    partial_fraction: float = 0.5              # ASSUMED unsheathed fraction at the partial-expansion stage
    partial_taper_fraction: float = 0.4        # ASSUMED part of the unsheathed segment that tapers into the catheter
    marker_radius_mm: float = 0.25             # ASSUMED
    proximal_marker_offset_mm: float = 0.25    # ASSUMED marker centre above the device base
    distal_marker_offset_mm: float = 0.2       # ASSUMED marker centre below the device top
    wire_radius_mm: float = 0.15               # ASSUMED delivery wire
    wire_length_mm: float = 4.0                # shown length of the delivery wire
    lumen_weight: float = 0.05                 # display weights per mm of path (ASSUMED)
    braid_weight: float = 0.3
    constrained_weight: float = 1.0
    wire_weight: float = 0.6
    marker_weight: float = 20.0
    pixel_mm: float = 0.1
    step_mm: float = 0.1
    margin_mm: float = 1.5
    chunk_rows: int = 24


@dataclass
class Body:
    """Device segment: radii ``(n_levels, n_angles)`` on ``linspace(t0, t0 + height, n_levels)``."""

    t0: float
    height: float
    radii: np.ndarray
    weight: float


@dataclass
class StageGeometry:
    name: str
    bodies: List[Body]
    markers: Dict[str, float]          # axial positions of the marker centres (mm from the neck centre)


@dataclass
class ProjectionView:
    """Parallel projection: image centre, in-plane unit axes, viewing direction and sampling."""

    center: np.ndarray
    up: np.ndarray
    right: np.ndarray
    view_dir: np.ndarray
    pixel_mm: float
    n_pix: int
    half_depth_mm: float
    step_mm: float

    @property
    def offsets(self) -> np.ndarray:
        return (np.arange(self.n_pix) - (self.n_pix - 1) / 2.0) * self.pixel_mm

    @property
    def depths(self) -> np.ndarray:
        n = int(math.ceil(self.half_depth_mm / self.step_mm))
        return self.step_mm * np.arange(-n, n + 1)

    def to_pixel(self, pts_mm: np.ndarray) -> np.ndarray:
        """Array-frame points (mm) -> ``(row, col)`` pixel coordinates (float)."""
        d = np.atleast_2d(np.asarray(pts_mm, float)) - self.center
        c0 = (self.n_pix - 1) / 2.0
        return np.stack([c0 - d @ self.up / self.pixel_mm, c0 + d @ self.right / self.pixel_mm], axis=-1)


def _cylinder(t0: float, length: float, radius: float, weight: float) -> Body:
    return Body(float(t0), float(length), np.full((2, 1), float(radius)), float(weight))


def stage_geometry(profile: DeviceProfile, config: Optional[FluoroConfig] = None) -> Dict[str, StageGeometry]:
    """Device bodies, delivery wire and marker positions at each stage."""
    cfg = config or FluoroConfig()
    H, tb, tt = profile.height, profile.t_base, profile.t_top
    rc, po, do = cfg.constrained_radius_mm, cfg.proximal_marker_offset_mm, cfg.distal_marker_offset_mm
    Lc = cfg.constrained_length_ratio * H
    full = Body(tb, H, profile.r_device, cfg.braid_weight)

    def wire(t_end: float) -> Body:
        return _cylinder(t_end - cfg.wire_length_mm, cfg.wire_length_mm, cfg.wire_radius_mm, cfg.wire_weight)

    out: Dict[str, StageGeometry] = {}

    t_c = tt - Lc
    out["constrained"] = StageGeometry("constrained", [_cylinder(t_c, Lc, rc, cfg.constrained_weight), wire(t_c)],
                                       {"proximal": t_c + po, "distal": tt - do})

    f = float(np.clip(cfg.partial_fraction, 0.05, 1.0))
    z_cut = (1.0 - f) * H
    z_new = np.linspace(z_cut, H, max(8, int(round(f * profile.z.size))))
    r = interp_profile(profile.r_device, H, z_new[:, None], profile.angles[None, :])
    s = np.clip((z_new - z_cut) / max(cfg.partial_taper_fraction * f * H, 1e-6), 0.0, 1.0)
    w = (0.5 - 0.5 * np.cos(np.pi * s))[:, None]
    r = rc + (np.maximum(r, rc) - rc) * w
    t_cut = tb + z_cut
    Lp = (1.0 - f) * Lc
    bodies = [Body(t_cut, f * H, r, cfg.braid_weight)]
    if Lp > 0:
        bodies.append(_cylinder(t_cut - Lp, Lp, rc, cfg.constrained_weight))
    bodies.append(wire(t_cut - Lp))
    out["partial_expansion"] = StageGeometry("partial_expansion", bodies, {"proximal": t_cut - Lp + po, "distal": tt - do})

    markers = {"proximal": tb + po, "distal": tt - do}
    out["full_expansion"] = StageGeometry("full_expansion", [full, wire(tb)], dict(markers))
    out["detached"] = StageGeometry("detached", [full], dict(markers))
    return out


def projection_view(frame: NeckFrame, stages: Iterable[StageGeometry], view_dir: Optional[Sequence[float]] = None,
                    config: Optional[FluoroConfig] = None) -> ProjectionView:
    """Common view for all stages: centred on the axis, square field of view enclosing every body and marker."""
    cfg = config or FluoroConfig()
    fr = orthonormal_frame(frame)
    v = fr.e2 if view_dir is None else unit(view_dir)
    up = fr.normal - (fr.normal @ v) * v
    if np.linalg.norm(up) < 1e-6:
        up = fr.e1 - (fr.e1 @ v) * v
    up = unit(up)
    right = np.cross(v, up)
    stages = list(stages)
    t_lo = min(min(b.t0 for b in s.bodies) for s in stages)
    t_hi = max(max(b.t0 + b.height for b in s.bodies) for s in stages)
    t_lo = min(t_lo, min(min(s.markers.values()) for s in stages) - cfg.marker_radius_mm)
    t_hi = max(t_hi, max(max(s.markers.values()) for s in stages) + cfg.marker_radius_mm)
    r_max = max(float(np.max(b.radii)) for s in stages for b in s.bodies)
    side = math.hypot(t_hi - t_lo, 2.0 * r_max) + 2.0 * cfg.margin_mm
    center = fr.center + 0.5 * (t_lo + t_hi) * fr.normal
    n_pix = int(math.ceil(side / cfg.pixel_mm))
    return ProjectionView(center=center, up=up, right=right, view_dir=v, pixel_mm=cfg.pixel_mm, n_pix=n_pix,
                          half_depth_mm=side / 2.0, step_mm=cfg.step_mm)


def _lumen_sampler(vol: Volume, view: ProjectionView):
    """Occupancy of labels > 0 cropped to the sampled slab, returned as a point sampler (mm -> [0, 1])."""
    sp = np.asarray(vol.spacing, float)
    reach = math.sqrt(2.0) * (view.n_pix * view.pixel_mm / 2.0) + view.half_depth_mm
    c = view.center / sp
    lo = np.clip(np.floor(c - reach / sp).astype(int) - 1, 0, np.asarray(vol.shape))
    hi = np.clip(np.ceil(c + reach / sp).astype(int) + 2, 0, np.asarray(vol.shape))
    occ = (np.asarray(vol.array)[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] != 0).astype(np.float32)

    def sample(pts_mm: np.ndarray) -> np.ndarray:
        if occ.size == 0:
            return np.zeros(len(pts_mm))
        return ndi.map_coordinates(occ, (pts_mm / sp - lo).T, order=1, mode="constant", cval=0.0)

    return sample


def _marker_image(view: ProjectionView, frame: NeckFrame, markers: Dict[str, float], cfg: FluoroConfig) -> np.ndarray:
    img = np.zeros((view.n_pix, view.n_pix))
    idx = np.arange(view.n_pix)
    R = cfg.marker_radius_mm
    for t in markers.values():
        row, col = view.to_pixel(frame.to_array(np.array([t, 0.0, 0.0])))[0]
        d2 = ((idx[:, None] - row) ** 2 + (idx[None, :] - col) ** 2) * view.pixel_mm ** 2
        img += cfg.marker_weight * 2.0 * np.sqrt(np.maximum(R * R - d2, 0.0))
    return img


def render_stages(labels: Union[Volume, SacField], frame: Optional[NeckFrame], deployment: DeploymentResult,
                  view_dir: Optional[Sequence[float]] = None, *, deployer: Optional[VirtualDeployer] = None,
                  config: Optional[FluoroConfig] = None, stages: Sequence[str] = STAGES) -> Dict[str, np.ndarray]:
    """2D parallel-projection images ``(n_pix, n_pix)`` of the lumen, device and markers at each stage.

    ``deployment`` must come from the same deployment configuration as ``deployer`` (default configuration if None);
    the conformed profile is recomputed from the sac field. ``view_dir`` defaults to ``frame.e2`` (lateral view, device
    axis vertical).
    """
    cfg = config or FluoroConfig()
    dep = deployer or VirtualDeployer()
    fld = dep.as_field(labels, frame)
    if fld.labels is None:
        raise ValueError("the SacField carries no label volume for the lumen")
    prof = dep.profile(fld, deployment.size)
    if abs(prof.height - deployment.deployed_height) > 1e-6:
        raise ValueError("deployment was produced with a different deployment configuration: pass the same deployer")
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stages {sorted(unknown)}; choose from {STAGES}")
    geoms = stage_geometry(prof, cfg)
    fr = fld.frame
    view = projection_view(fr, geoms.values(), view_dir, cfg)
    lumen = _lumen_sampler(fld.labels, view)

    r_body = {s: [float(np.max(b.radii)) for b in geoms[s].bodies] for s in stages}
    off, dep_s = view.offsets, view.depths
    n = view.n_pix
    images = {s: np.zeros((n, n)) for s in stages}
    lum = np.zeros((n, n))
    for r0 in range(0, n, cfg.chunk_rows):
        rows = off[::-1][r0:r0 + cfg.chunk_rows]                      # row 0 is the +up side
        pts = (view.center + rows[:, None, None, None] * view.up + off[None, :, None, None] * view.right
               + dep_s[None, None, :, None] * view.view_dir)            # (rows, cols, depth, 3)
        shp = pts.shape[:3]
        flat = pts.reshape(-1, 3)
        lum[r0:r0 + rows.size] = lumen(flat).reshape(shp).sum(axis=-1) * view.step_mm
        loc = fr.to_local(flat)
        t, rho, phi = loc[:, 0], np.hypot(loc[:, 1], loc[:, 2]), np.arctan2(loc[:, 2], loc[:, 1])
        for s in stages:
            acc = np.zeros(flat.shape[0])
            for b, rb in zip(geoms[s].bodies, r_body[s]):
                zr = t - b.t0
                sel = np.flatnonzero((zr >= 0) & (zr <= b.height) & (rho <= rb))
                if sel.size:
                    occ = rho[sel] <= interp_profile(b.radii, b.height, zr[sel], phi[sel])
                    acc[sel] += b.weight * occ
            images[s][r0:r0 + rows.size] = acc.reshape(shp).sum(axis=-1) * view.step_mm
    return {s: (cfg.lumen_weight * lum + images[s] + _marker_image(view, fr, geoms[s].markers, cfg)).astype(np.float32)
            for s in stages}


def marker_positions(deployment: DeploymentResult, frame: NeckFrame, deploy_config: Optional[DeploymentConfig] = None,
                     config: Optional[FluoroConfig] = None) -> Dict[str, np.ndarray]:
    """Proximal and distal marker centres (mm, array frame) of the deployed device."""
    cfg = config or FluoroConfig()
    fr = orthonormal_frame(frame)
    t_base, t_top = device_span(deployment, deploy_config)
    return {"proximal": fr.to_array(np.array([t_base + cfg.proximal_marker_offset_mm, 0.0, 0.0])),
            "distal": fr.to_array(np.array([t_top - cfg.distal_marker_offset_mm, 0.0, 0.0]))}


__all__ = ["STAGES", "FluoroConfig", "Body", "StageGeometry", "ProjectionView", "stage_geometry", "projection_view",
           "render_stages", "marker_positions"]
