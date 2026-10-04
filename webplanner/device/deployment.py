"""Virtual deployment of WEB sizes in the segmented sac (manuscript equation 1).

Model
-----
Every catalogue size ``d`` (nominal width ``W_d``, height ``H_d``) is placed on the neck landing zone, centred on
the device axis (``NeckFrame.normal``, base at the neck plane), and conformed to the sac wall. Positions along the
axis are ``t`` (mm from the neck centre, ``t = 0`` is the neck plane); in-plane directions are angles ``phi`` from
``e1`` towards ``e2``.

* Sac wall: the ``iso_level`` (0.5) surface of a Gaussian-smoothed binary sac mask (labels {1, 4}). The wall radius
  ``r_s(t, phi)`` is the first exit of a ray that leaves the axis at height ``t`` (trilinear sampling); it is 0 where
  the axis itself lies outside the sac.
* Sac extent: the axis is inside the sac on ``[t_entry, h_sac]`` (the run of the axis that contains the neck centre,
  or the first run that starts within ``axis_search_mm`` above it). ``h_sac`` is the sac extent along the device axis
  from the neck centre; ``t_entry`` is the proximal sac boundary on the axis.
* Nominal device: a superellipse barrel ``r_n(z) = W/2 (1 - |2z/H - 1|^p)^(1/p)`` for ``0 <= z <= H`` (``p = 5`` for
  SL, ``p = 2`` for SLS).
* Conformation: ``r_d = min(r_n, max(r_s - tol, r_min))``.
* Compression at the equator (the level of maximal ``r_n``): ``c = 1 - r_bar / (W/2)``, ``r_bar`` the mean of
  ``r_d`` over angles.
* Height: ``H_free = H_d + k W_d c`` and ``H' = min(H_free, max(h_sac, f_min H_d))``. The profile depends on ``H'``,
  so the two steps are iterated (``max_iter`` passes, stopping when ``H'`` changes by less than ``height_tol_mm``);
  the reported ``c`` is the one that produced ``H_free``, so ``H_free = H_d + k W_d c`` holds exactly.
* Dome clearance: ``g = h_sac - H'`` (manuscript definition: the deployed, capped height, not the free height). It is
  0 for a device capped by the dome and negative only when the floor ``f_min H_d`` applies.
* Placement: the device occupies ``t in [t_base, t_base + H']`` with ``t_base = min(base_offset, h_sac - H')``; a
  device that is still taller than the sac after the floor is pushed proximally.
* Protrusion: ``max(0, t_ref - t_base)`` with ``t_ref = min(0, t_entry)``: the axial length of the device below the
  proximal sac boundary, or below the neck plane where the sac does not extend below it.
* Apposition: fraction of (level, angle) samples inside the sac (``r_s > 0``) where ``r_n >= r_s - tol``.
* Neck footprint: per angle, the largest ``r_d`` over the lower half of the device (``z <= H'/2``). Neck coverage is
  the fraction of landing-zone voxel centres, projected on the neck plane, inside the footprint; neck margin is the
  minimum over angle bins (that contain landing-zone voxels) of the footprint radius minus the largest landing-zone
  radius in that bin.
* Deployed width: ``2 * mean_phi(max_z r_d)``.

The device model is geometric (no braid mechanics). Constants tagged ASSUMED in :class:`DeploymentConfig` are
modelling choices, not measured device properties; ``k`` is fitted on patients with post-deployment cone-beam CT
(:mod:`webplanner.device.elongation`).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, fields
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage as ndi

from ..types import NECK, SAC, DeploymentResult, DeviceSize, NeckFrame, Volume, unit
from .catalogue import Catalogue

TWO_PI = 2.0 * math.pi
FAMILY_EXPONENT = {"SLS": 2.0}          # other families use DeploymentConfig.profile_exponent


@dataclass
class DeploymentConfig:
    """Parameters of the virtual deployment. Lengths in mm."""

    k: float = 0.45                          # elongation coefficient (fitted on post-deployment cone-beam CT)
    n_angles: int = 96
    n_levels: int = 60
    wall_tolerance_mm: float = 0.05          # ASSUMED gap between the conformed device and the sac wall
    min_radius_mm: float = 0.3               # ASSUMED smallest radius of a conformed braid section
    base_offset_mm: float = -0.15            # ASSUMED seating of the device base relative to the neck plane
    max_iter: int = 4
    height_tol_mm: float = 1e-4
    profile_exponent: float = 5.0            # ASSUMED superellipse exponent of the SL barrel
    min_height_fraction: float = 0.8         # ASSUMED: the dome cannot shorten a device below this fraction of H_d
    proximal_recess_radius_mm: float = 0.7   # ASSUMED proximal marker recess (occupancy mask only)
    proximal_recess_depth_mm: float = 0.5    # ASSUMED
    distal_recess_radius_mm: float = 0.8     # ASSUMED distal marker recess (occupancy mask only)
    distal_recess_depth_mm: float = 0.45     # ASSUMED
    smoothing_sigma_vox: float = 1.0         # Gaussian smoothing of the binary sac mask
    iso_level: float = 0.5
    ray_length_mm: float = 8.0               # must exceed the largest nominal radius
    ray_step_mm: float = 0.05
    level_step_mm: float = 0.05              # spacing of the cached wall-radius table along the axis
    axis_search_mm: float = 2.0              # search window for the sac boundary around the neck centre
    landing_layer_mm: float = 0.6            # fallback landing zone (lowest sac layer) when label 4 is absent


@dataclass
class SacField:
    """Smoothed sac occupancy and cached wall radii for one case and one neck frame (reused by all sizes)."""

    data: np.ndarray                   # smoothed occupancy on a crop of the input grid
    offset: np.ndarray                 # input-grid index of ``data[0, 0, 0]``
    spacing: np.ndarray
    shape: Tuple[int, int, int]        # input grid
    frame: NeckFrame                   # orthonormalised copy of the input frame
    iso_level: float
    t_entry: float                     # proximal sac boundary on the axis (mm from the neck centre)
    h_sac: float                       # sac extent along the axis from the neck centre (first exit)
    angles: np.ndarray                 # (n_angles,) from e1 towards e2
    table_t: np.ndarray                # (n_t,) axial positions of the cached wall radii
    table_r: np.ndarray                # (n_t, n_angles) wall radius r_s(t, phi)
    lz_uv: np.ndarray                  # (m, 2) landing-zone voxel centres projected on the neck plane
    lz_rmax: np.ndarray                # (n_angles,) largest landing-zone radius per angle bin (-inf if empty)
    lz_source: str                     # "landing_zone" | "lowest_sac_layer"
    ray_length_mm: float
    ray_step_mm: float
    labels: Optional[Volume] = None

    def sample(self, pts_mm: np.ndarray) -> np.ndarray:
        """Trilinear occupancy at array-frame points (mm), 0 outside the crop."""
        pts = np.asarray(pts_mm, float).reshape(-1, 3)
        idx = pts / self.spacing - self.offset
        return ndi.map_coordinates(self.data, idx.T, order=1, mode="constant", cval=0.0)

    def wall_radius(self, t: np.ndarray) -> np.ndarray:
        """Cached ``r_s(t, phi)`` for axial positions ``t`` (mm), shape ``(len(t), n_angles)``; 0 outside the sac."""
        t = np.atleast_1d(np.asarray(t, float))
        out = np.zeros((t.size, self.angles.size))
        ok = (t >= self.t_entry) & (t <= self.h_sac)
        if ok.any():
            dt = self.table_t[1] - self.table_t[0]
            pos = (t[ok] - self.table_t[0]) / dt
            i0 = np.clip(np.floor(pos).astype(int), 0, self.table_t.size - 2)
            f = np.clip(pos - i0, 0.0, 1.0)[:, None]
            out[ok] = (1.0 - f) * self.table_r[i0] + f * self.table_r[i0 + 1]
        return out


@dataclass
class DeviceProfile:
    """Conformed device of one size: radii on ``n_levels`` axial levels times ``n_angles`` angles."""

    size: DeviceSize
    t_base: float              # axial position of the device base (mm from the neck centre)
    height: float              # deployed height H'
    free_height: float         # H_d + k W_d c
    compression: float
    z: np.ndarray              # (n_levels,) levels above the base, 0..H'
    angles: np.ndarray
    r_nominal: np.ndarray      # (n_levels,)
    r_wall: np.ndarray         # (n_levels, n_angles)
    r_device: np.ndarray       # (n_levels, n_angles)
    iterations: int

    @property
    def t_top(self) -> float:
        return self.t_base + self.height


# --------------------------------------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------------------------------------
def nominal_profile(width: float, height: float, z: np.ndarray, exponent: float = 5.0) -> np.ndarray:
    """Unconstrained radius of a superellipse barrel at heights ``z`` above the base (0 outside ``[0, height]``)."""
    u = np.clip(np.asarray(z, float) / height, 0.0, 1.0)
    return (width / 2.0) * np.clip(1.0 - np.abs(2.0 * u - 1.0) ** exponent, 0.0, 1.0) ** (1.0 / exponent)


def orthonormal_frame(frame: NeckFrame) -> NeckFrame:
    """Copy of ``frame`` with a unit normal, ``e1`` orthogonalised against it and ``e2`` completing the basis
    (the sign of the given ``e2`` is kept)."""
    n = unit(frame.normal)
    e1 = unit(np.asarray(frame.e1, float) - (np.asarray(frame.e1, float) @ n) * n)
    e2 = np.cross(n, e1)
    if np.asarray(frame.e2, float) @ e2 < 0:
        e2 = -e2
    return NeckFrame(center=np.asarray(frame.center, float).copy(), normal=n, e1=e1, e2=e2,
                     eigenvalues=np.asarray(frame.eigenvalues, float).copy(), n_points=frame.n_points)


def _first_exit(f: np.ndarray, rho: np.ndarray, iso: float) -> np.ndarray:
    """Radius of the first ``f < iso`` crossing along the last axis (linear interpolation); 0 if ``f[0] < iso``,
    ``rho[-1]`` if the ray never exits."""
    inside = f >= iso
    outside = ~inside
    outside[..., 0] = False
    has = outside.any(axis=-1)
    i = np.where(has, np.argmax(outside, axis=-1), f.shape[-1] - 1)
    i = np.maximum(i, 1)
    fa = np.take_along_axis(f, (i - 1)[..., None], axis=-1)[..., 0]
    fb = np.take_along_axis(f, i[..., None], axis=-1)[..., 0]
    den = np.where(fa > fb, fa - fb, 1.0)
    frac = np.clip((fa - iso) / den, 0.0, 1.0)
    r = rho[i - 1] + frac * (rho[i] - rho[i - 1])
    r = np.where(has, r, rho[-1])
    return np.where(inside[..., 0], r, 0.0)


def _crossing(ts: np.ndarray, f: np.ndarray, ia: int, ib: int, iso: float) -> float:
    fa, fb = f[ia], f[ib]
    if fa == fb:
        return float(ts[ia])
    return float(ts[ia] + (fa - iso) / (fa - fb) * (ts[ib] - ts[ia]))


def sac_radius(field: SacField, heights_mm: Sequence[float], angles: Sequence[float],
               ray_length_mm: Optional[float] = None, ray_step_mm: Optional[float] = None,
               chunk: int = 16) -> np.ndarray:
    """Wall radius by ray casting from the device axis.

    ``heights_mm`` are axial positions ``t`` (mm from the neck centre); rays leave the axis in the neck-plane
    direction ``cos(phi) e1 + sin(phi) e2``. Returns ``(len(heights), len(angles))`` radii (mm): the first exit from
    the sac along the ray, 0 where the axis point is outside the sac, ``ray_length_mm`` where the ray never exits.
    """
    fr = field.frame
    t = np.atleast_1d(np.asarray(heights_mm, float))
    ang = np.atleast_1d(np.asarray(angles, float))
    L = field.ray_length_mm if ray_length_mm is None else float(ray_length_mm)
    step = field.ray_step_mm if ray_step_mm is None else float(ray_step_mm)
    rho = step * np.arange(int(round(L / step)) + 1)
    dirs = np.cos(ang)[:, None] * fr.e1 + np.sin(ang)[:, None] * fr.e2               # (A, 3)
    out = np.empty((t.size, ang.size))
    for s in range(0, t.size, chunk):
        tc = t[s:s + chunk]
        pts = (fr.center + tc[:, None, None, None] * fr.normal
               + rho[None, None, :, None] * dirs[None, :, None, :])                    # (T, A, R, 3)
        f = field.sample(pts.reshape(-1, 3)).reshape(tc.size, ang.size, rho.size)
        out[s:s + chunk] = _first_exit(f, rho, field.iso_level)
    return out


def axis_extent(field: SacField) -> float:
    """``h_sac``: sac extent along the device axis from the neck centre (mm)."""
    return float(field.h_sac)


def _axial_run(field: SacField, search_mm: float, step: float) -> Tuple[float, float]:
    """(t_entry, h_sac) of the axis run inside the sac (see module docstring)."""
    fr = field.frame
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
    corners = (field.offset + corners * (np.asarray(field.data.shape) - 1)) * field.spacing
    tmax = float(np.linalg.norm(corners - fr.center, axis=1).max()) + step
    ns, nt = int(round(search_mm / step)), int(math.ceil(tmax / step))
    ts = step * np.arange(-ns, nt + 1)
    f = field.sample(fr.center + ts[:, None] * fr.normal)
    inside = f >= field.iso_level
    i0 = ns
    if inside[i0]:
        below = np.flatnonzero(~inside[:i0])
        s = int(below[-1]) + 1 if below.size else 0
    else:
        above = np.flatnonzero(inside[i0:i0 + ns + 1])
        if not above.size:
            raise ValueError(f"the device axis does not enter the sac within {search_mm} mm of the neck centre")
        s = i0 + int(above[0])
    after = np.flatnonzero(~inside[s:])
    e = s + int(after[0]) if after.size else None
    t_entry = _crossing(ts, f, s - 1, s, field.iso_level) if s > 0 else float(ts[0])
    h_sac = _crossing(ts, f, e - 1, e, field.iso_level) if e is not None else float(ts[-1])
    return t_entry, h_sac


def interp_profile(radii: np.ndarray, height: float, z: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Bilinear lookup of a radius table ``(n_levels, n_angles)`` defined on ``linspace(0, height, n_levels)`` and
    angles ``2 pi j / n_angles`` (periodic). ``z`` is clipped to ``[0, height]``."""
    radii = np.asarray(radii, float)
    nl, na = radii.shape
    if nl < 2:
        radii = np.vstack([radii, radii])
        nl = 2
    zf = np.clip(np.asarray(z, float) / max(height, 1e-9) * (nl - 1), 0.0, nl - 1)
    i0 = np.minimum(np.floor(zf).astype(int), nl - 2)
    fz = zf - i0
    pf = np.mod(np.asarray(phi, float), TWO_PI) / TWO_PI * na
    j = np.floor(pf)
    fp = pf - j
    j0 = j.astype(int) % na
    j1 = (j0 + 1) % na
    lo = (1 - fp) * radii[i0, j0] + fp * radii[i0, j1]
    hi = (1 - fp) * radii[i0 + 1, j0] + fp * radii[i0 + 1, j1]
    return (1 - fz) * lo + fz * hi


def device_span(deployment: DeploymentResult, config: Optional[DeploymentConfig] = None) -> Tuple[float, float]:
    """Axial extent ``(t_base, t_top)`` of a deployed device (mm from the neck centre)."""
    cfg = config or DeploymentConfig()
    t_base = min(cfg.base_offset_mm, deployment.h_sac - deployment.deployed_height)
    return float(t_base), float(t_base + deployment.deployed_height)


def device_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Dice overlap of two device occupancy masks (1.0 when both are empty)."""
    a, b = np.asarray(mask_a, bool), np.asarray(mask_b, bool)
    if a.shape != b.shape:
        raise ValueError(f"mask shapes differ: {a.shape} vs {b.shape}")
    den = int(a.sum()) + int(b.sum())
    return 1.0 if den == 0 else 2.0 * int(np.logical_and(a, b).sum()) / den


# --------------------------------------------------------------------------------------------------------------------
# deployer
# --------------------------------------------------------------------------------------------------------------------
class VirtualDeployer:
    """Deploys catalogue sizes in a segmented sac. ``VirtualDeployer(k=0.5)`` overrides single config fields."""

    def __init__(self, config: Optional[DeploymentConfig] = None, **overrides: float) -> None:
        cfg = config or DeploymentConfig()
        if overrides:
            names = {f.name for f in fields(DeploymentConfig)}
            unknown = set(overrides) - names
            if unknown:
                raise TypeError(f"unknown deployment parameters: {sorted(unknown)}")
            cfg = DeploymentConfig(**{**{f.name: getattr(cfg, f.name) for f in fields(cfg)}, **overrides})
        if cfg.n_levels < 2 or cfg.n_angles < 3:
            raise ValueError("n_levels must be >= 2 and n_angles >= 3")
        self.config = cfg

    # ------------------------------------------------------------------ preparation
    def prepare(self, labels: Volume, frame: NeckFrame) -> SacField:
        """Smoothed sac field, axial extent, cached wall radii and projected landing zone for one case."""
        cfg = self.config
        lab = np.asarray(labels.array)
        sp = np.asarray(labels.spacing, float)
        sac = (lab == SAC) | (lab == NECK)
        if not sac.any():
            raise ValueError("the label map has no sac voxels (labels 1 or 4)")
        fr = orthonormal_frame(frame)

        pad = int(math.ceil(4.0 * cfg.smoothing_sigma_vox)) + 2
        nz = np.argwhere(sac)
        lo = np.maximum(nz.min(axis=0) - pad, 0)
        hi = np.minimum(nz.max(axis=0) + pad + 1, lab.shape)
        crop = sac[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
        data = ndi.gaussian_filter(crop, cfg.smoothing_sigma_vox) if cfg.smoothing_sigma_vox > 0 else crop

        angles = np.linspace(0.0, TWO_PI, cfg.n_angles, endpoint=False)
        fld = SacField(data=data, offset=lo.astype(float), spacing=sp, shape=tuple(lab.shape), frame=fr,
                       iso_level=cfg.iso_level, t_entry=0.0, h_sac=0.0, angles=angles, table_t=np.zeros(2),
                       table_r=np.zeros((2, cfg.n_angles)), lz_uv=np.zeros((0, 2)), lz_rmax=np.full(cfg.n_angles, -np.inf),
                       lz_source="landing_zone", ray_length_mm=cfg.ray_length_mm, ray_step_mm=cfg.ray_step_mm,
                       labels=labels)
        fld.t_entry, fld.h_sac = _axial_run(fld, cfg.axis_search_mm, cfg.ray_step_mm)

        n_t = max(2, int(math.ceil((fld.h_sac - fld.t_entry) / cfg.level_step_mm)) + 1)
        fld.table_t = np.linspace(fld.t_entry, fld.h_sac, n_t)
        eps = min(0.25 * cfg.level_step_mm, 0.25 * (fld.h_sac - fld.t_entry))
        fld.table_r = sac_radius(fld, np.clip(fld.table_t, fld.t_entry + eps, fld.h_sac - eps), angles)

        lz = lab == NECK
        if lz.any():
            pts = np.argwhere(lz) * sp
        else:
            warnings.warn("no landing-zone voxels (label 4): using the lowest sac layer for neck coverage")
            pts = np.argwhere(sac) * sp
            t = fr.to_local(pts)[:, 0]
            pts = pts[t <= fld.t_entry + cfg.landing_layer_mm]
            fld.lz_source = "lowest_sac_layer"
        uv = fr.to_local(pts)[:, 1:]
        fld.lz_uv = uv
        r = np.hypot(uv[:, 0], uv[:, 1])
        bins = np.rint(np.mod(np.arctan2(uv[:, 1], uv[:, 0]), TWO_PI) / (TWO_PI / cfg.n_angles)).astype(int) % cfg.n_angles
        np.maximum.at(fld.lz_rmax, bins, r)
        return fld

    # ------------------------------------------------------------------ device model
    def _exponent(self, size: DeviceSize) -> float:
        return FAMILY_EXPONENT.get(size.family.upper(), self.config.profile_exponent)

    def profile(self, field: SacField, size: DeviceSize) -> DeviceProfile:
        """Conformed device and heights for one size (iterated height cap, see module docstring)."""
        cfg = self.config
        W, H = float(size.width), float(size.height)
        if W / 2 + cfg.wall_tolerance_mm >= field.ray_length_mm:
            raise ValueError(f"ray_length_mm={field.ray_length_mm} is too short for a {W:g} mm device")
        p = self._exponent(size)
        cap = max(field.h_sac, cfg.min_height_fraction * H)
        Hd, c, H_free, it = H, 0.0, H, 0
        for it in range(1, cfg.max_iter + 1):
            z = np.linspace(0.0, Hd, cfg.n_levels)
            t_base = min(cfg.base_offset_mm, field.h_sac - Hd)
            rs = field.wall_radius(t_base + z)
            rn = nominal_profile(W, Hd, z, p)
            rd = np.minimum(rn[:, None], np.maximum(rs - cfg.wall_tolerance_mm, cfg.min_radius_mm))
            c = float(1.0 - rd[int(np.argmax(rn))].mean() / (W / 2))
            H_free = H + cfg.k * W * c
            H_new = min(H_free, cap)
            done = abs(H_new - Hd) < cfg.height_tol_mm
            Hd = H_new
            if done:
                break
        z = np.linspace(0.0, Hd, cfg.n_levels)
        t_base = min(cfg.base_offset_mm, field.h_sac - Hd)
        rs = field.wall_radius(t_base + z)
        rn = nominal_profile(W, Hd, z, p)
        rd = np.minimum(rn[:, None], np.maximum(rs - cfg.wall_tolerance_mm, cfg.min_radius_mm))
        return DeviceProfile(size=size, t_base=float(t_base), height=float(Hd), free_height=float(H_free),
                             compression=c, z=z, angles=field.angles, r_nominal=rn, r_wall=rs, r_device=rd,
                             iterations=it)

    def as_field(self, labels: Union[Volume, SacField], frame: Optional[NeckFrame]) -> SacField:
        """A prepared field: ``labels`` itself if it is a :class:`SacField` (checked against ``frame`` when given),
        otherwise ``prepare(labels, frame)``."""
        if isinstance(labels, SacField):
            if frame is not None:
                f = orthonormal_frame(frame)
                if not (np.allclose(f.center, labels.frame.center) and np.allclose(f.normal, labels.frame.normal)):
                    raise ValueError("frame differs from the frame the SacField was prepared with")
            return labels
        if frame is None:
            raise ValueError("a NeckFrame is required with a label Volume")
        return self.prepare(labels, frame)

    def deploy(self, labels: Union[Volume, SacField], frame: Optional[NeckFrame], size: DeviceSize,
               return_mask: bool = False) -> DeploymentResult:
        """Deploy one size. ``labels`` is a label :class:`Volume` or a prepared :class:`SacField` (``frame`` may then
        be None)."""
        cfg = self.config
        fld = self.as_field(labels, frame)
        prof = self.profile(fld, size)
        rn, rs, rd, Hd = prof.r_nominal, prof.r_wall, prof.r_device, prof.height

        inside = rs > 0
        apposed = (rn[:, None] >= rs - cfg.wall_tolerance_mm) & inside
        apposition = float(apposed.sum() / inside.sum()) if inside.any() else 0.0

        footprint = rd[prof.z <= Hd / 2 + 1e-9].max(axis=0)
        if fld.lz_uv.shape[0]:
            r = np.hypot(fld.lz_uv[:, 0], fld.lz_uv[:, 1])
            phi = np.arctan2(fld.lz_uv[:, 1], fld.lz_uv[:, 0])
            fp_at = np.interp(phi, fld.angles, footprint, period=TWO_PI)
            coverage = float(np.mean(r <= fp_at + 1e-9))
            has = np.isfinite(fld.lz_rmax)
            margin = float(np.min(footprint[has] - fld.lz_rmax[has]))
        else:
            coverage, margin = 0.0, float(-footprint.max())

        t_ref = min(0.0, fld.t_entry)
        res = DeploymentResult(size=size, compression=prof.compression, deployed_width=float(2.0 * rd.max(axis=0).mean()),
                               deployed_height=Hd, free_height=prof.free_height, dome_clearance=float(fld.h_sac - Hd),
                               apposition=apposition, neck_coverage=coverage, neck_margin=margin,
                               protrusion=float(max(0.0, t_ref - prof.t_base)), h_sac=float(fld.h_sac))
        if return_mask:
            res.mask = self.voxelise(fld, prof)
        return res

    def voxelise(self, field: SacField, prof: DeviceProfile) -> np.ndarray:
        """Device occupancy (voxel centres) on the input grid, with the proximal and distal marker recesses."""
        cfg = self.config
        fr, sp = field.frame, field.spacing
        R = float(prof.r_device.max()) + sp.max()
        corners = np.array([[t, u, v] for t in (prof.t_base, prof.t_top) for u in (-R, R) for v in (-R, R)])
        idx = fr.to_array(corners) / sp
        lo = np.clip(np.floor(idx.min(axis=0)).astype(int) - 1, 0, np.asarray(field.shape) - 1)
        hi = np.clip(np.ceil(idx.max(axis=0)).astype(int) + 2, 0, np.asarray(field.shape))
        mask = np.zeros(field.shape, bool)
        if np.any(hi <= lo):
            return mask
        grid = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij"), axis=-1)
        loc = fr.to_local(grid.reshape(-1, 3) * sp)
        zr = loc[:, 0] - prof.t_base
        rho = np.hypot(loc[:, 1], loc[:, 2])
        inside = (zr >= 0) & (zr <= prof.height)
        if inside.any():
            sel = np.flatnonzero(inside)
            rlim = interp_profile(prof.r_device, prof.height, zr[sel], np.arctan2(loc[sel, 2], loc[sel, 1]))
            inside[sel] = rho[sel] <= rlim
        inside &= ~((zr < cfg.proximal_recess_depth_mm) & (rho < cfg.proximal_recess_radius_mm))
        inside &= ~((prof.height - zr < cfg.distal_recess_depth_mm) & (rho < cfg.distal_recess_radius_mm))
        mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = inside.reshape(grid.shape[:3])
        return mask


def _sizes(catalogue: Union[Catalogue, Iterable[DeviceSize], None]) -> List[DeviceSize]:
    if catalogue is None:
        return Catalogue.default().sizes
    if isinstance(catalogue, Catalogue):
        return catalogue.sizes
    return list(catalogue)


def deploy_catalogue(labels: Union[Volume, SacField], frame: Optional[NeckFrame],
                     catalogue: Union[Catalogue, Iterable[DeviceSize], None] = None,
                     deployer: Optional[VirtualDeployer] = None, return_masks: bool = False) -> List[DeploymentResult]:
    """Deploy every catalogue size (catalogue order) on one prepared sac field."""
    dep = deployer or VirtualDeployer()
    fld = dep.as_field(labels, frame)
    return [dep.deploy(fld, None, s, return_mask=return_masks) for s in _sizes(catalogue)]


__all__ = ["DeploymentConfig", "SacField", "DeviceProfile", "VirtualDeployer", "deploy_catalogue", "nominal_profile",
           "sac_radius", "axis_extent", "interp_profile", "device_span", "device_dice", "orthonormal_frame"]
