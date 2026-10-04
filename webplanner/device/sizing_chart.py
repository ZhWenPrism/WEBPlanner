"""The manufacturer's sizing chart, applied as in routine practice (the comparator).

Device width is read from the average of two orthogonal aneurysm widths, device height from the smallest aneurysm
height, and a wider device is chosen when the chart width is narrower than the largest neck width.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..types import DeviceSize, Morphometry
from .catalogue import Catalogue, lookup


@dataclass
class ChartDecision:
    size: DeviceSize
    flags: List[str] = field(default_factory=list)


def chart_size(mean_width: float, smallest_height: float, max_neck: float, catalogue: Optional[Catalogue] = None) -> ChartDecision:
    cat = catalogue or Catalogue.default()
    flags: List[str] = []
    w, f = lookup(cat.width_bins, mean_width)
    if f:
        flags.append(f"width {f}")
    if w < max_neck - 1e-9:
        flags.append("upsized to the neck")
        while w < max_neck - 1e-9 and w < cat.widths.max():
            w = cat.step_width(w, 1)
    h, f = lookup(cat.height_bins, smallest_height)
    if f:
        flags.append(f"height {f}")
    if h not in cat.heights[w]:
        flags.append("empty cell: nearest available height")
        h = cat.nearest_height(w, h)
    return ChartDecision(DeviceSize(float(w), float(h)), flags)


def chart_from_morphometry(m: Morphometry, catalogue: Optional[Catalogue] = None) -> ChartDecision:
    return chart_size(m.mean_width, m.smallest_height, m.neck_width, catalogue)
