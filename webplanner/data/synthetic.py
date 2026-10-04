"""Synthetic bifurcation-aneurysm phantoms.

The phantoms reproduce the geometry the planner works with (a parent artery ending in a bifurcation, two daughter
vessels and a sac whose base forms the neck landing zone) with known ground truth. They are used by the unit and
integration tests and by ``webplanner demo``; they contain no patient data.

Construction (all lengths in mm, array frame)
---------------------------------------------
* the parent artery runs along ``d_parent`` and ends at the bifurcation point ``B``;
* the aneurysm axis ``a`` is tilted from ``d_parent`` by the entry angle;
* the neck centre is ``N = B + offset * a``; the sac is an ellipsoid with semi-axes ``(w1/2, w2/2, c)`` along
  ``(e1, e2, a)`` whose centre lies ``t0`` above the neck plane, with ``t0`` chosen so that the cross-section at the
  neck plane has the requested neck radius; the sac occupies ``t >= 0``;
* daughter vessels leave ``B`` at take-off angles measured from the aneurysm axis;
* an optional bleb makes the sac irregular.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi

from ..types import BACKGROUND, BRANCH, NECK, PARENT, SAC, CaseRecord, DeviceSize, Volume, unit


@dataclass
class PhantomSpec:
    shape: Tuple[int, int, int] = (96, 96, 96)
    spacing: float = 0.30
    width_1: float = 6.0                 # sac width along e1 (mm)
    width_2: float = 5.4                 # sac width along e2 (mm)
    height: float = 5.5                  # sac height along the axis from the neck plane (mm)
    neck_width: float = 4.2              # neck diameter (mm)
    entry_angle_deg: float = 25.0        # parent direction vs aneurysm axis
    takeoff_deg: Tuple[float, float] = (100.0, 105.0)   # daughter directions vs aneurysm axis
    parent_radius: float = 1.4
    branch_radii: Tuple[float, float] = (1.1, 1.0)
    bleb_radius: float = 0.0             # >0 adds a bleb on the dome (irregular sac)
    bleb_polar_deg: float = 35.0
    noise_sd: float = 0.03
    blur_sigma_vox: float = 0.6
    azimuth_deg: float = 0.0             # rotation of the whole construction about z
    seed: int = 0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class Phantom:
    image: Volume
    labels: Volume
    spec: PhantomSpec
    truth: Dict[str, object] = field(default_factory=dict)


def _segment_distance(P: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab), 0.0, 1.0)
    proj = a + t[..., None] * ab
    return np.linalg.norm(P - proj, axis=-1), t


def _rotz(v: np.ndarray, deg: float) -> np.ndarray:
    th = np.deg2rad(deg)
    R = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])   # rotation in the (y, x) plane, z fixed
    return R @ v


def _orthonormal(a: np.ndarray, hint: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    e1 = hint - (hint @ a) * a
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.array([0.0, 1.0, 0.0]) - a[1] * a
    e1 = unit(e1)
    return e1, np.cross(a, e1)


def make_phantom(spec: Optional[PhantomSpec] = None, **overrides) -> Phantom:
    spec = spec or PhantomSpec()
    if overrides:
        spec = PhantomSpec(**{**spec.to_dict(), **overrides})
    rng = np.random.default_rng(spec.seed)
    shape, s = spec.shape, spec.spacing
    ext = np.asarray(shape) * s
    P = np.stack(np.meshgrid(*[np.arange(n) * s for n in shape], indexing="ij"), axis=-1)

    a = _rotz(unit([1.0, 0.0, 0.0]), spec.azimuth_deg)                       # aneurysm axis (array z is "up")
    tilt = np.deg2rad(spec.entry_angle_deg)
    side = _rotz(np.array([0.0, 1.0, 0.0]), spec.azimuth_deg)
    d_parent = unit(np.cos(tilt) * a + np.sin(tilt) * side)                  # direction of flow at the bifurcation
    B = np.array([ext[0] * 0.36, ext[1] * 0.5, ext[2] * 0.5])
    e1, e2 = _orthonormal(a, side)

    # daughters: take-off angle from the aneurysm axis, on opposite sides of the axis (+/- e2)
    daughters = []
    for k, (ang, r) in enumerate(zip(spec.takeoff_deg, spec.branch_radii)):
        th = np.deg2rad(ang)
        sgn = 1.0 if k == 0 else -1.0
        daughters.append((unit(np.cos(th) * a + np.sin(th) * sgn * e2), r))

    # sac: truncated ellipsoid above the neck plane
    a1, a2 = spec.width_1 / 2, spec.width_2 / 2
    rn = spec.neck_width / 2
    am = np.sqrt(a1 * a2)
    if rn >= am:
        raise ValueError("neck must be narrower than the sac")
    # height = t0 + c and rn = am * sqrt(1 - (t0/c)^2)  ->  t0 = c * q with q = sqrt(1 - (rn/am)^2)
    q = np.sqrt(1 - (rn / am) ** 2)
    c = spec.height / (1 + q)
    t0 = c * q
    offset = 0.35 * spec.parent_radius
    N = B + offset * a
    d = P - N
    t, u, v = d @ a, d @ e1, d @ e2
    sac = ((u / a1) ** 2 + (v / a2) ** 2 + ((t - t0) / c) ** 2 <= 1.0) & (t >= 0)
    if spec.bleb_radius > 0:
        pol = np.deg2rad(spec.bleb_polar_deg)
        surf_dir = unit(np.cos(pol) * a + np.sin(pol) * e1)
        # point on the ellipsoid surface along surf_dir from the ellipsoid centre
        lam = 1.0 / np.sqrt((surf_dir @ e1 / a1) ** 2 + (surf_dir @ e2 / a2) ** 2 + (surf_dir @ a / c) ** 2)
        tip = N + t0 * a + lam * surf_dir
        sac |= np.linalg.norm(P - tip, axis=-1) <= spec.bleb_radius

    # vessels
    parent_d, _ = _segment_distance(P, B - 18.0 * d_parent, B)
    parent = parent_d <= spec.parent_radius
    junction = (np.linalg.norm(P - B, axis=-1) <= max(rn, spec.parent_radius) * 1.05) & (t < 0.0)   # bifurcation apex
    parent |= junction
    branch = np.zeros(shape, bool)
    for dvec, r in daughters:
        bd, _ = _segment_distance(P, B, B + 16.0 * dvec)
        branch |= bd <= r
    branch &= ~parent & (t < 0.6)

    lab = np.zeros(shape, np.uint8)
    lab[parent] = PARENT
    lab[branch] = BRANCH
    lab[sac] = SAC
    # landing zone: the base layer of the sac over the neck disk (about 2-3 voxels thick)
    rho = np.sqrt(u ** 2 + v ** 2)
    neck = sac & (t < max(2.5 * s, 0.6)) & (rho <= rn)
    lab[neck] = NECK

    lumen = (lab != BACKGROUND).astype(np.float32)
    img = ndi.gaussian_filter(lumen, spec.blur_sigma_vox) if spec.blur_sigma_vox > 0 else lumen
    img = 0.05 + 0.95 * img + rng.normal(0.0, spec.noise_sd, shape).astype(np.float32)
    img = (img * 1000.0).astype(np.float32)

    vox = s ** 3
    truth = {
        "neck_center": N, "normal": a, "e1": e1, "e2": e2, "bifurcation": B, "d_parent": d_parent,
        "daughter_dirs": [dv for dv, _ in daughters],
        "width_1": spec.width_1, "width_2": spec.width_2, "mean_width": (spec.width_1 + spec.width_2) / 2,
        "max_width": max(spec.width_1, spec.width_2), "height": spec.height, "neck_width": spec.neck_width,
        "entry_angle_deg": spec.entry_angle_deg, "takeoff_deg": tuple(spec.takeoff_deg),
        "daughter_angle_diff_deg": abs(spec.takeoff_deg[0] - spec.takeoff_deg[1]),
        "irregular": spec.bleb_radius > 0, "sac_volume_mm3": float((lab == SAC).sum() + (lab == NECK).sum()) * vox,
        "ellipsoid": {"a1": a1, "a2": a2, "c": c, "t0": t0},
    }
    sp = (s, s, s)
    return Phantom(Volume(img, sp), Volume(lab, sp), spec, truth)


def random_spec(rng: np.random.Generator, shape: Tuple[int, int, int] = (96, 96, 96), spacing: float = 0.30,
                seed: Optional[int] = None) -> PhantomSpec:
    """Plausible WEB-range anatomy (widths 3.5-9 mm, width-to-neck 1.1-2.0, height 3-8.5 mm)."""
    w1 = rng.uniform(3.5, 9.0)
    w2 = w1 * rng.uniform(0.82, 1.0)
    wn = rng.uniform(1.1, 2.0)
    neck = min((w1 + w2) / 2 / wn, 0.95 * np.sqrt(w1 * w2) / 1.0)
    height = float(np.clip(rng.normal(0.95 * (w1 + w2) / 2, 0.9), 3.0, 8.5))
    t1 = rng.uniform(85, 125)
    t2 = t1 + rng.choice([-1, 1]) * rng.uniform(0, 60)
    return PhantomSpec(shape=shape, spacing=spacing, width_1=float(w1), width_2=float(w2), height=height,
                       neck_width=float(neck), entry_angle_deg=float(rng.uniform(0, 80)), takeoff_deg=(float(t1), float(t2)),
                       parent_radius=float(rng.uniform(1.1, 1.7)), branch_radii=(float(rng.uniform(0.8, 1.3)), float(rng.uniform(0.7, 1.2))),
                       bleb_radius=float(rng.uniform(0.6, 1.2)) if rng.random() < 0.3 else 0.0,
                       bleb_polar_deg=float(rng.uniform(15, 60)), azimuth_deg=float(rng.uniform(0, 360)),
                       seed=int(rng.integers(1 << 31)) if seed is None else seed)


# --------------------------------------------------------------------------------------------------------------------
# synthetic operator decisions (labels for testing the ranking model)
# --------------------------------------------------------------------------------------------------------------------
def operator_choice(mean_width: float, smallest_height: float, neck_width: float, irregular: bool, entry_angle_deg: float,
                    catalogue, rng: np.random.Generator) -> DeviceSize:
    """A stochastic operator policy used only to create labels for tests and the demo.

    Starts from the chart size and oversizes the width by one step in most cases (two for wide or angled sacs) and
    lowers the height for the expected elongation, mimicking published practice reviews. It is not a model of any
    real operator.
    """
    from ..device.sizing_chart import chart_size

    base = chart_size(mean_width, smallest_height, neck_width, catalogue).size
    p_up = 0.55 + 0.15 * irregular + 0.1 * (entry_angle_deg > 60)
    steps = int(rng.random() < p_up) + int(rng.random() < 0.15 * (mean_width > 7.5))
    w = catalogue.step_width(base.width, steps)
    h = base.height - (1.0 if steps and rng.random() < 0.5 else 0.0)
    return DeviceSize(w, catalogue.nearest_height(w, max(h, min(catalogue.heights[w]))))


def synthetic_cases(n: int, centres: Sequence[str] = ("Centre 1", "Centre 2", "Centre 3"), seed: int = 0,
                    weights: Optional[Sequence[float]] = None) -> List[Tuple[CaseRecord, PhantomSpec]]:
    """Case records with phantom specifications; volumes are generated lazily with :func:`make_phantom`."""
    rng = np.random.default_rng(seed)
    w = np.asarray(weights if weights is not None else np.ones(len(centres)), float)
    out = []
    for i in range(n):
        spec = random_spec(rng)
        centre = str(rng.choice(list(centres), p=w / w.sum()))
        rec = CaseRecord(case_id=f"syn{i:04d}", patient_id=f"P{i:04d}", centre=centre, treatment="WEB",
                         location=str(rng.choice(["MCA", "AComA", "basilar tip", "ICA terminus"], p=[0.45, 0.25, 0.2, 0.1])),
                         ruptured=bool(rng.random() < 0.25),
                         access_tortuosity=str(rng.choice(["minimal", "moderate", "severe"], p=[0.6, 0.3, 0.1])))
        out.append((rec, spec))
    return out
