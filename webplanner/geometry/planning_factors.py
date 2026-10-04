"""Planning factors: inflow angulation, daughter-vessel symmetry and the manufacturer's simple-case profile.

Angles are measured against the aneurysm axis (the neck normal, pointing into the dome):

* entry angle: between the terminal flow direction of the parent artery (towards the neck) and the axis; 0 means
  flow straight into the sac;
* take-off angle of a daughter vessel: between its direction (away from the neck) and the axis;
* daughter angle difference: ``|theta_1 - theta_2|`` of the two largest daughter vessels.

The simple-case characteristics checked are those listed by the manufacturer: width 3 to 10 mm, width-to-neck ratio
1 to 1.5, symmetrical daughter vessels, no significant angulation at the neck, unruptured, minimal access
tortuosity. Unknown rupture status or tortuosity does not fail the profile but is recorded as a note.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi

from ..types import BRANCH, PARENT, Morphometry, NeckFrame, PlanningFactors, VesselOrigin, Volume
from .centerline import terminal_segment


@dataclass
class Thresholds:
    significant_angulation_deg: float = 60.0     # entry angle above this is significant angulation
    asymmetry_deg: float = 45.0                  # daughter angle difference above this is asymmetric
    width_range: Tuple[float, float] = (3.0, 10.0)
    width_neck_range: Tuple[float, float] = (1.0, 1.5)
    irregular_solidity: float = 0.90             # passed to morphometry.measure by geometry.analyze


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two vectors in degrees (NaN for a zero or non-finite vector)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if not (np.isfinite(na) and np.isfinite(nb)) or na == 0 or nb == 0:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1.0, 1.0))))


def entry_angle(parent_direction_towards_neck: np.ndarray, normal: np.ndarray) -> float:
    """Angle (degrees) between the terminal parent-artery flow direction and the aneurysm axis."""
    return angle_deg(parent_direction_towards_neck, normal)


def takeoff_angles(daughter_directions: Sequence[np.ndarray], normal: np.ndarray) -> List[float]:
    """Take-off angle (degrees) of each daughter direction (pointing away from the neck) from the aneurysm axis."""
    return [angle_deg(d, normal) for d in daughter_directions]


def daughter_angle_difference(angles: Sequence[float]) -> Tuple[float, Optional[str]]:
    """``|theta_1 - theta_2|`` of the first two angles; 0 and a note when fewer than two daughters are available."""
    vals = [a for a in angles if np.isfinite(a)]
    if len(vals) < 2:
        return 0.0, f"{len(vals)} daughter vessel(s) found: daughter symmetry not assessable (difference set to 0)"
    return float(abs(vals[0] - vals[1])), None


def is_simple_case(mean_width: float, width_neck_ratio: float, daughter_angle_diff_deg: float, entry_angle_deg: float,
                   ruptured: Optional[bool] = None, tortuosity: Optional[str] = None,
                   thresholds: Optional[Thresholds] = None,
                   notes: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
    """Check the manufacturer's simple-case characteristics. Returns ``(simple, failed criteria)``.

    Missing information (NaN angles, unknown rupture status or tortuosity) does not fail a criterion; a note is
    appended to ``notes`` when a list is given.
    """
    th = thresholds or Thresholds()
    notes = notes if notes is not None else []
    failed: List[str] = []
    lo, hi = th.width_range
    if not lo <= mean_width <= hi:
        failed.append(f"width {mean_width:.1f} mm outside {lo:g}-{hi:g} mm")
    lo, hi = th.width_neck_range
    if not np.isfinite(width_neck_ratio):
        notes.append("width-to-neck ratio not available")
    elif not lo <= width_neck_ratio <= hi:
        failed.append(f"width-to-neck ratio {width_neck_ratio:.2f} outside {lo:g}-{hi:g}")
    if not np.isfinite(daughter_angle_diff_deg):
        notes.append("daughter angle difference not available")
    elif daughter_angle_diff_deg > th.asymmetry_deg:
        failed.append(f"asymmetric daughter vessels ({daughter_angle_diff_deg:.0f} deg > {th.asymmetry_deg:g} deg)")
    if not np.isfinite(entry_angle_deg):
        notes.append("entry angle not available")
    elif entry_angle_deg > th.significant_angulation_deg:
        failed.append(f"significant angulation at the neck ({entry_angle_deg:.0f} deg > "
                      f"{th.significant_angulation_deg:g} deg)")
    if ruptured is None:
        notes.append("rupture status unknown (not counted as a failure)")
    elif ruptured:
        failed.append("ruptured")
    if tortuosity is None:
        notes.append("access tortuosity unknown (not counted as a failure)")
    elif str(tortuosity).lower() != "minimal":
        failed.append(f"access tortuosity {tortuosity}")
    return not failed, failed


def _direction(o: VesselOrigin, labels: Volume, cls: int) -> np.ndarray:
    """Origin direction, recomputed from the label map when it is missing or not finite."""
    d = np.asarray(o.direction, float)
    if d.shape == (3,) and np.all(np.isfinite(d)) and np.linalg.norm(d) > 0:
        return d
    comp, n = ndi.label(np.asarray(labels.array) == cls, structure=np.ones((3, 3, 3), bool))
    if n == 0:
        return np.full(3, np.nan)
    pts = np.argwhere(comp > 0)
    nearest = pts[np.argmin(np.linalg.norm(pts * np.asarray(labels.spacing) - np.asarray(o.point), axis=1))]
    return terminal_segment(o.point, comp == comp[tuple(nearest)], labels.spacing).direction


def assess(labels: Volume, frame: NeckFrame, origins: Sequence[VesselOrigin], morph: Morphometry,
           ruptured: Optional[bool] = None, tortuosity: Optional[str] = None,
           thresholds: Optional[Thresholds] = None) -> PlanningFactors:
    """Planning factors from the neck frame, the vessel origins and the morphometry.

    The parent is the origin named ``"parent"``; the daughters are the first two ``branch_*`` origins (largest
    first). Notes (missing vessels, unknown inputs) are stored in ``factors.notes``.
    """
    th = thresholds or Thresholds()
    notes: List[str] = []
    parent = [o for o in origins if o.name == "parent"]
    daughters = [o for o in origins if o.name.startswith("branch")]

    if parent:
        ent = entry_angle(-_direction(parent[0], labels, PARENT), frame.normal)
    else:
        ent = float("nan")
        notes.append("no parent artery at the neck: entry angle not available")
    angles = takeoff_angles([_direction(o, labels, BRANCH) for o in daughters[:2]], frame.normal)
    diff, note = daughter_angle_difference(angles)
    if note:
        notes.append(note)
    if len(daughters) > 2:
        notes.append(f"{len(daughters)} branch vessels at the neck: the two largest were used")

    simple, failed = is_simple_case(morph.mean_width, morph.width_neck_ratio, diff, ent, ruptured, tortuosity, th,
                                    notes)
    factors = PlanningFactors(entry_angle_deg=ent, daughter_angle_diff_deg=diff, solidity=morph.solidity,
                              ruptured=ruptured, access_tortuosity=tortuosity,
                              significant_angulation=bool(np.isfinite(ent) and ent > th.significant_angulation_deg),
                              asymmetric_daughters=bool(diff > th.asymmetry_deg), simple_case=simple,
                              simple_case_failures=failed)
    factors.notes = notes
    return factors
