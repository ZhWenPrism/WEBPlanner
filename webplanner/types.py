"""Shared data structures.

Conventions used throughout the package
---------------------------------------
* Arrays are indexed ``(z, y, x)``.
* ``spacing`` is the voxel size in mm in the same order ``(sz, sy, sx)``.
* Geometry is computed in the *array frame*: physical position in mm = voxel index * spacing.
  The scanner origin and direction cosines are kept in :class:`Volume` for export only.
* Device sizes are nominal (unconstrained) width x height in mm.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------------------------------------------
# label map of the four-class anatomy segmentation
# --------------------------------------------------------------------------------------------------------------------
BACKGROUND, SAC, PARENT, BRANCH, NECK = 0, 1, 2, 3, 4
LABELS: Dict[int, str] = {BACKGROUND: "background", SAC: "aneurysm_sac", PARENT: "parent_artery",
                          BRANCH: "branch_arteries", NECK: "neck_landing_zone"}
FOREGROUND_CLASSES: Tuple[int, ...] = (SAC, PARENT, BRANCH, NECK)


@dataclass
class Volume:
    """A 3D image or label map with its voxel spacing."""

    array: np.ndarray
    spacing: Tuple[float, float, float]
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: Tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if self.array.ndim != 3:
            raise ValueError(f"expected a 3D array, got shape {self.array.shape}")
        self.spacing = tuple(float(s) for s in self.spacing)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.array.shape)

    @property
    def extent_mm(self) -> np.ndarray:
        return np.asarray(self.shape) * np.asarray(self.spacing)

    def index_to_mm(self, idx: np.ndarray) -> np.ndarray:
        return np.asarray(idx, float) * np.asarray(self.spacing)

    def mm_to_index(self, pts: np.ndarray) -> np.ndarray:
        return np.asarray(pts, float) / np.asarray(self.spacing)


# --------------------------------------------------------------------------------------------------------------------
# devices
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, order=True)
class DeviceSize:
    """Nominal WEB size. ``family`` is kept for completeness (the planner ranks one family, SL)."""

    width: float
    height: float
    family: str = "SL"

    @property
    def label(self) -> str:
        return f"{self.width:g}x{self.height:g}"

    def as_tuple(self) -> Tuple[str, float, float]:
        return (self.family, float(self.width), float(self.height))


# --------------------------------------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class NeckFrame:
    """Neck plane and the local device frame.

    ``center`` is the landing-zone centroid (mm, array frame); ``normal`` points from the neck into the dome;
    ``e1``/``e2`` span the neck plane (``e1`` along the largest in-plane extent of the landing zone).
    """

    center: np.ndarray
    normal: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    eigenvalues: np.ndarray = field(default_factory=lambda: np.zeros(3))
    n_points: int = 0

    def to_local(self, pts_mm: np.ndarray) -> np.ndarray:
        """Array-frame mm -> local (t along the normal, u along e1, v along e2)."""
        d = np.asarray(pts_mm, float) - self.center
        return np.stack([d @ self.normal, d @ self.e1, d @ self.e2], axis=-1)

    def to_array(self, local: np.ndarray) -> np.ndarray:
        local = np.asarray(local, float)
        return (self.center + local[..., :1] * self.normal + local[..., 1:2] * self.e1 + local[..., 2:3] * self.e2)


@dataclass
class VesselOrigin:
    name: str              # "parent", "branch_1", ...
    point: np.ndarray      # mm, array frame (centroid of the vessel voxels adjacent to the landing zone)
    direction: np.ndarray  # unit vector pointing away from the neck along the vessel
    radius_mm: float


@dataclass
class Morphometry:
    mean_width: float            # mean of two orthogonal widths parallel to the neck plane
    max_width: float
    width_1: float
    width_2: float
    height: float                # sac extent along the neck normal (99.7th percentile)
    neck_width: float            # largest diameter of the landing zone
    width_neck_ratio: float
    volume_mm3: float
    solidity: float              # sac volume / convex-hull volume
    irregular: bool
    branch_incorporated: bool
    sac_axis_extent: float       # h_sac: sac extent along the device axis from the neck centre
    smallest_height: float       # smallest of the two projected heights, used by the sizing chart

    def to_dict(self) -> Dict[str, float]:
        return {k: (float(v) if not isinstance(v, bool) else bool(v)) for k, v in asdict(self).items()}


@dataclass
class PlanningFactors:
    entry_angle_deg: float               # terminal parent artery direction vs aneurysm axis
    daughter_angle_diff_deg: float       # difference between the take-off angles of the two daughter vessels
    solidity: float
    ruptured: Optional[bool] = None
    access_tortuosity: Optional[str] = None   # "minimal" | "moderate" | "severe" (from the records)
    significant_angulation: bool = False
    asymmetric_daughters: bool = False
    simple_case: bool = False
    simple_case_failures: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class AnatomyResult:
    frame: NeckFrame
    morphometry: Morphometry
    factors: PlanningFactors
    origins: List[VesselOrigin] = field(default_factory=list)


# --------------------------------------------------------------------------------------------------------------------
# virtual deployment
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class DeploymentResult:
    size: DeviceSize
    compression: float           # c = 1 - r_bar / (W/2)
    deployed_width: float
    deployed_height: float       # H' (capped by the sac along the axis)
    free_height: float           # H_d + k W_d c (uncapped)
    dome_clearance: float        # g = h_sac - H'
    apposition: float            # fraction of the device surface within tolerance of the sac wall
    neck_coverage: float         # fraction of landing-zone points under the device footprint
    neck_margin: float           # smallest footprint radius minus landing-zone radius (mm), per angle
    protrusion: float            # device extent below the neck plane (mm), >0 means protrusion
    h_sac: float
    mask: Optional[np.ndarray] = None   # optional device occupancy on the input grid

    def features(self) -> Dict[str, float]:
        return {"compression": self.compression, "height_ratio": self.deployed_height / max(self.h_sac, 1e-6),
                "dome_clearance": self.dome_clearance, "apposition": self.apposition, "neck_coverage": self.neck_coverage,
                "neck_margin": self.neck_margin, "protrusion": self.protrusion}


# --------------------------------------------------------------------------------------------------------------------
# cases and reports
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class CaseRecord:
    """One patient with one treated index aneurysm."""

    case_id: str
    patient_id: str
    centre: str
    treatment: str = "WEB"                  # "WEB" | "FD" | "other"
    location: str = "MCA"                   # MCA | AComA | basilar tip | ICA terminus | other
    ruptured: Optional[bool] = None
    access_tortuosity: Optional[str] = None
    image_path: Optional[str] = None
    label_path: Optional[str] = None
    implanted: Optional[DeviceSize] = None  # reference standard for the sizing analyses
    extra: Dict[str, object] = field(default_factory=dict)


@dataclass
class Recommendation:
    size: DeviceSize
    probability: float
    deployment: Optional[DeploymentResult] = None


@dataclass
class PlanReport:
    case_id: str
    top: List[Recommendation]
    chart_size: Optional[DeviceSize]
    review_flag: bool
    tau: float
    anatomy: Optional[AnatomyResult] = None
    notes: List[str] = field(default_factory=list)

    @property
    def top1(self) -> Recommendation:
        return self.top[0]

    def summary(self) -> Dict[str, object]:
        a = self.anatomy
        return {
            "case_id": self.case_id,
            "top": [{"size": r.size.label, "probability": round(float(r.probability), 4)} for r in self.top],
            "chart_size": self.chart_size.label if self.chart_size else None,
            "review_flag": bool(self.review_flag),
            "tau": float(self.tau),
            "morphometry": a.morphometry.to_dict() if a else None,
            "planning_factors": a.factors.to_dict() if a else None,
            "notes": list(self.notes),
        }


def unit(v: Sequence[float]) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero vector")
    return v / n
