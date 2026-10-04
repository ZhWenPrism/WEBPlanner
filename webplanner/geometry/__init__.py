"""Neck frame, morphometry and planning factors from the four-class label map."""
from __future__ import annotations

from typing import Optional

from ..types import AnatomyResult, Volume
from .centerline import TerminalSegment, centerline_path, direction_near, terminal_segment
from .morphometry import measure, ostium_offset
from .neck import fit_neck_frame, neck_mask, sac_mask, vessel_origins
from .planning_factors import Thresholds, assess, entry_angle, is_simple_case, takeoff_angles


def analyze(labels: Volume, ruptured: Optional[bool] = None, tortuosity: Optional[str] = None,
            thresholds: Optional[Thresholds] = None) -> AnatomyResult:
    """Neck frame, vessel origins, morphometry and planning factors of one label map."""
    th = thresholds or Thresholds()
    frame = fit_neck_frame(labels)
    origins = vessel_origins(labels, frame)
    morph = measure(labels, frame, origins, irregular_threshold=th.irregular_solidity)
    factors = assess(labels, frame, origins, morph, ruptured=ruptured, tortuosity=tortuosity, thresholds=th)
    return AnatomyResult(frame=frame, morphometry=morph, factors=factors, origins=origins)


__all__ = ["analyze", "fit_neck_frame", "vessel_origins", "neck_mask", "sac_mask", "measure", "ostium_offset",
           "assess", "is_simple_case", "entry_angle", "takeoff_angles", "Thresholds", "direction_near",
           "terminal_segment", "centerline_path", "TerminalSegment"]
