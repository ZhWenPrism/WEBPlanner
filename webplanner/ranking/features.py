"""Size-specific feature vectors phi(x, d) and the feasible set C(x) (manuscript equation 2).

For one case x and every catalogue size d, :class:`FeatureBuilder` returns one row of features. In a conditional
logit, terms that are constant within a case cancel, so case-level variables (planning factors, morphometry,
location, rupture status, the image-based sac score) enter only through interactions with size-specific terms.

Feasible set
------------
``C(x) = {d : W_d >= N_max}`` with ``N_max`` the largest neck width (``Morphometry.neck_width``). When no size is
wide enough, all sizes of the maximal width form the feasible set (``CandidateSet.fallback`` is then True).

Feature names (``FeatureBuilder.names``; default locations, ``interactions=True``, ``image_score=True``)
--------------------------------------------------------------------------------------------------------
Deviation from the chart size (dw = signed width steps, dh = height difference in mm, both from the chart size to d):

* ``dw_steps``, ``abs_dw_steps``                         linear and symmetric cost of moving away from the chart width
* ``dw_m1``, ``dw_p1``                                   indicators of dw = -1 and +1 (reference dw = 0): routine one-step
                                                         oversizing is not a linear function of dw. Indicators for
                                                         |dw| >= 2 are omitted because, with the two linear terms, they
                                                         are nearly collinear (tail sizes rarely carry probability)
* ``dh_mm``, ``abs_dh_mm``, ``dh_m1``, ``dh_p1``        height deviation, its magnitude, indicators of dh = -1, +1 mm

Virtual deployment of d (``DeploymentResult.features()`` plus two quadratic terms and a hinge):

* ``compression``, ``compression_sq``                    c and c^2 so that a preferred compression range can be learned
* ``height_ratio``, ``height_ratio_sq``                  H'/h_sac and its square (preferred filling of the sac height)
* ``dome_clearance``, ``apposition``, ``neck_coverage``  g, wall apposition and neck coverage
* ``neck_margin``, ``neck_margin_neg``                   footprint margin over the neck and its negative part
                                                         ``max(0, -margin)`` (a footprint that fails to cover the neck)
* ``protrusion``                                         device extent below the neck plane

Interactions of case-level variables with size-specific terms (case variables centred and scaled by fixed constants,
see ``CASE_REFERENCE``, so no statistic is estimated from the data here):

* ``dw_x_irregular``, ``dw_x_entry_angle``, ``dw_x_daughter_angle_diff``, ``dw_x_solidity``, ``dw_x_mean_width``,
  ``dw_x_ruptured``                                      planning factors and size that change the preferred oversizing
* ``dw_x_loc_<location>`` for every location except the first (reference)   location-specific oversizing
* ``comp_x_irregular``, ``comp_x_solidity``              tolerated compression differs for irregular, non-convex sacs
* ``hratio_x_irregular``, ``dh_x_irregular``, ``dh_x_entry_angle``   height choice for irregular or angled sacs
* ``dw_x_image_score``                                   the image-based sac score enters only through the width step

Within a case every feature is a function of the size (W, H) only, so some columns are nearly collinear (for
example compression and width step both grow with W, hence ``comp_x_*`` and ``dw_x_*``). The L2 penalty resolves
this for prediction; individual coefficients should not be read as independent effects.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from ..device.catalogue import Catalogue
from ..device.sizing_chart import chart_size
from ..types import AnatomyResult, CaseRecord, DeploymentResult, DeviceSize

LOCATIONS = ("MCA", "AComA", "basilar tip", "ICA terminus", "other")

# fixed centring and scaling of case-level variables inside interactions: (centre, scale)
CASE_REFERENCE: Dict[str, tuple] = {
    "entry_angle": (45.0, 30.0),            # degrees
    "daughter_angle_diff": (20.0, 20.0),    # degrees
    "solidity": (0.9, 0.1),
    "mean_width": (6.5, 1.5),               # mm
}

CHART_NAMES = ["dw_steps", "abs_dw_steps", "dw_m1", "dw_p1", "dh_mm", "abs_dh_mm", "dh_m1", "dh_p1"]
DEPLOYMENT_NAMES = ["compression", "compression_sq", "height_ratio", "height_ratio_sq", "dome_clearance", "apposition",
                    "neck_coverage", "neck_margin", "neck_margin_neg", "protrusion"]
CASE_DW_NAMES = ["dw_x_irregular", "dw_x_entry_angle", "dw_x_daughter_angle_diff", "dw_x_solidity", "dw_x_mean_width",
                 "dw_x_ruptured"]
OTHER_INTERACTION_NAMES = ["comp_x_irregular", "comp_x_solidity", "hratio_x_irregular", "dh_x_irregular",
                           "dh_x_entry_angle"]
IMAGE_NAMES = ["dw_x_image_score"]


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s.strip())


@dataclass
class CandidateSet:
    """Feature rows of all catalogue sizes for one case."""

    case_id: str
    sizes: List[DeviceSize]
    X: np.ndarray                      # (n_sizes, p)
    feasible: np.ndarray               # (n_sizes,) bool, C(x)
    names: List[str]
    chosen_index: Optional[int] = None
    chart_index: Optional[int] = None
    patient_id: Optional[str] = None
    centre: Optional[str] = None
    fallback: bool = False             # True when no width covers the neck and the widest sizes were used
    info: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, float)
        self.feasible = np.asarray(self.feasible, bool)
        if self.X.shape != (len(self.sizes), len(self.names)):
            raise ValueError(f"X has shape {self.X.shape}, expected {(len(self.sizes), len(self.names))}")
        if self.feasible.shape != (len(self.sizes),):
            raise ValueError("feasible mask must have one entry per size")
        if not self.feasible.any():
            raise ValueError(f"case {self.case_id}: empty feasible set")

    @property
    def n_sizes(self) -> int:
        return len(self.sizes)

    def index_of(self, size: DeviceSize) -> int:
        for i, s in enumerate(self.sizes):
            if abs(s.width - size.width) < 1e-6 and abs(s.height - size.height) < 1e-6:
                return i
        raise KeyError(f"{size.label} is not a candidate size of case {self.case_id}")

    @property
    def chosen(self) -> Optional[DeviceSize]:
        return None if self.chosen_index is None else self.sizes[self.chosen_index]

    def select(self, names: Sequence[str]) -> "CandidateSet":
        """Copy restricted to a subset of feature columns (for ablations)."""
        cols = [self.names.index(n) for n in names]
        return CandidateSet(self.case_id, list(self.sizes), self.X[:, cols].copy(), self.feasible.copy(), list(names),
                            self.chosen_index, self.chart_index, self.patient_id, self.centre, self.fallback, dict(self.info))

    def to_dict(self) -> Dict[str, Any]:
        return {"case_id": self.case_id, "sizes": [[s.family, s.width, s.height] for s in self.sizes],
                "X": self.X.tolist(), "feasible": self.feasible.tolist(), "names": list(self.names),
                "chosen_index": self.chosen_index, "chart_index": self.chart_index, "patient_id": self.patient_id,
                "centre": self.centre, "fallback": bool(self.fallback), "info": dict(self.info)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CandidateSet":
        sizes = [DeviceSize(float(w), float(h), str(f)) for f, w, h in d["sizes"]]
        return cls(str(d["case_id"]), sizes, np.asarray(d["X"], float), np.asarray(d["feasible"], bool), list(d["names"]),
                   d.get("chosen_index"), d.get("chart_index"), d.get("patient_id"), d.get("centre"),
                   bool(d.get("fallback", False)), dict(d.get("info", {})))


def feasible_mask(sizes: Sequence[DeviceSize], neck_width: float) -> tuple:
    """``(mask, fallback)`` for C(x) = {d : W_d >= N_max}; all sizes of the maximal width if none qualify."""
    w = np.array([s.width for s in sizes], float)
    mask = w >= float(neck_width) - 1e-9
    if mask.any():
        return mask, False
    return np.isclose(w, w.max()), True


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def case_variables(anatomy: Union[AnatomyResult, Mapping[str, Any], Any], record: Optional[CaseRecord] = None) -> Dict[str, float]:
    """Case-level variables from an :class:`AnatomyResult`, a mapping with ``morphometry``/``factors`` entries
    (dataclasses or mappings), or a flat mapping with the field names of ``Morphometry`` and ``PlanningFactors``."""
    morph = _get(anatomy, "morphometry")
    fac = _get(anatomy, "factors")
    if morph is None and fac is None:
        morph = fac = anatomy

    def pick(key: str, *sources: Any, required: bool = True) -> Any:
        for src in sources:
            v = _get(src, key)
            if v is not None:
                return v
        if required:
            raise KeyError(f"case variable '{key}' not found in the anatomy input")
        return None

    ruptured = pick("ruptured", fac, required=False)
    if ruptured is None and record is not None:
        ruptured = record.ruptured
    return {
        "mean_width": float(pick("mean_width", morph)),
        "neck_width": float(pick("neck_width", morph)),
        "smallest_height": float(pick("smallest_height", morph, required=False) or np.nan),
        "irregular": float(bool(pick("irregular", morph))),
        "solidity": float(pick("solidity", fac, morph)),
        "entry_angle_deg": float(pick("entry_angle_deg", fac)),
        "daughter_angle_diff_deg": float(pick("daughter_angle_diff_deg", fac)),
        "ruptured": float(bool(ruptured)) if ruptured is not None else 0.0,   # unknown rupture status coded 0
        "ruptured_known": float(ruptured is not None),
    }


class FeatureBuilder:
    """Builds :class:`CandidateSet` objects with a stable, documented feature order (see module docstring)."""

    def __init__(self, catalogue: Optional[Catalogue] = None, locations: Sequence[str] = LOCATIONS,
                 interactions: bool = True, image_score: bool = True) -> None:
        self.catalogue = catalogue or Catalogue.default()
        self.locations = tuple(locations)
        if not self.locations:
            raise ValueError("at least one location is required")
        self.interactions = bool(interactions)
        self.image_score = bool(image_score)
        self.sizes = self.catalogue.sizes

    # ---------------------------------------------------------------- names
    @property
    def location_names(self) -> List[str]:
        return [f"dw_x_loc_{_slug(loc)}" for loc in self.locations[1:]]

    @property
    def groups(self) -> Dict[str, List[str]]:
        g: Dict[str, List[str]] = {"chart": list(CHART_NAMES), "deployment": list(DEPLOYMENT_NAMES)}
        if self.interactions:
            g["interactions"] = CASE_DW_NAMES + self.location_names + OTHER_INTERACTION_NAMES
        if self.image_score:
            g["image"] = list(IMAGE_NAMES)
        return g

    @property
    def names(self) -> List[str]:
        return [n for v in self.groups.values() for n in v]

    def _location(self, loc: Optional[str]) -> str:
        key = (loc or "").strip().lower()
        for l in self.locations:
            if l.lower() == key:
                return l
        if any(l.lower() == "other" for l in self.locations):
            return next(l for l in self.locations if l.lower() == "other")
        raise ValueError(f"location '{loc}' is not one of {self.locations}")

    # ---------------------------------------------------------------- build
    def build(self, record: CaseRecord, anatomy: Union[AnatomyResult, Mapping[str, Any], Any],
              deployments: Sequence[DeploymentResult], chart: Union[DeviceSize, Any, None] = None,
              image_score: float = 0.0) -> CandidateSet:
        """Feature rows for all catalogue sizes. ``deployments`` must follow the catalogue order. ``chart`` is the
        sizing-chart size (a ``DeviceSize`` or a ``ChartDecision``); when None it is computed from the morphometry."""
        cv = case_variables(anatomy, record)
        if chart is None:
            chart = chart_size(cv["mean_width"], cv["smallest_height"], cv["neck_width"], self.catalogue).size
        chart = getattr(chart, "size", chart)
        cat, sizes = self.catalogue, self.sizes
        if len(deployments) != len(sizes):
            raise ValueError(f"case {record.case_id}: {len(deployments)} deployments for {len(sizes)} catalogue sizes")
        for s, dep in zip(sizes, deployments):
            if abs(dep.size.width - s.width) > 1e-6 or abs(dep.size.height - s.height) > 1e-6:
                raise ValueError(f"case {record.case_id}: deployments are not in catalogue order ({dep.size.label} vs {s.label})")

        def z(key: str, value: float) -> float:
            c, s = CASE_REFERENCE[key]
            return (value - c) / s

        irr, rup = cv["irregular"], cv["ruptured"]
        ang = z("entry_angle", cv["entry_angle_deg"])
        dau = z("daughter_angle_diff", cv["daughter_angle_diff_deg"])
        sol = z("solidity", cv["solidity"])
        mw = z("mean_width", cv["mean_width"])
        loc = self._location(record.location)
        loc_ind = [float(loc == l) for l in self.locations[1:]]

        rows = []
        for s, dep in zip(sizes, deployments):
            dw, dh = cat.steps_between(chart, s)
            f = dep.features()
            bad = [k for k, v in f.items() if not np.isfinite(v)]
            if bad:
                raise ValueError(f"case {record.case_id}, size {s.label}: non-finite deployment features {bad}")
            c, hr = f["compression"], f["height_ratio"]
            row = [dw, abs(dw), dw == -1, dw == 1, dh, abs(dh), abs(dh + 1) < 1e-6, abs(dh - 1) < 1e-6,
                   c, c * c, hr, hr * hr, f["dome_clearance"], f["apposition"], f["neck_coverage"], f["neck_margin"],
                   max(0.0, -f["neck_margin"]), f["protrusion"]]
            if self.interactions:
                row += [dw * irr, dw * ang, dw * dau, dw * sol, dw * mw, dw * rup]
                row += [dw * li for li in loc_ind]
                row += [c * irr, c * sol, hr * irr, dh * irr, dh * ang]
            if self.image_score:
                row += [dw * float(image_score)]
            rows.append(row)
        X = np.asarray(rows, float)

        feasible, fallback = feasible_mask(sizes, cv["neck_width"])
        chosen = None
        if record.implanted is not None:
            if cat.contains(record.implanted):
                chosen = cat.index(record.implanted)
            else:
                warnings.warn(f"case {record.case_id}: implanted size {record.implanted.label} is not in the catalogue; "
                              "chosen_index left empty", stacklevel=2)
        info = {k: v for k, v in cv.items()}
        info["image_score"] = float(image_score)
        return CandidateSet(record.case_id, list(sizes), X, feasible, self.names, chosen, cat.index(chart),
                            record.patient_id, record.centre, fallback, info)

    def build_many(self, records: Sequence[CaseRecord], anatomies: Sequence[Any],
                   deployments: Sequence[Sequence[DeploymentResult]], charts: Optional[Sequence[Any]] = None,
                   image_scores: Optional[Sequence[float]] = None) -> List[CandidateSet]:
        n = len(records)
        if not (len(anatomies) == len(deployments) == n):
            raise ValueError("records, anatomies and deployments must have the same length")
        charts = list(charts) if charts is not None else [None] * n
        scores = list(image_scores) if image_scores is not None else [0.0] * n
        return [self.build(r, a, d, c, float(s)) for r, a, d, c, s in zip(records, anatomies, deployments, charts, scores)]
