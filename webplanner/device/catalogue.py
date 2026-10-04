"""WEB device catalogue and the manufacturer's selection table.

The default table (``web17``) is a transcription of the WEB 17 SL selection table: device width is read from the
average aneurysm width (two orthogonal projections) in columns, device height from the smallest aneurysm height in
rows, and only the heights available for the chosen width are valid. The table can be replaced with a YAML file
(``configs/device/*.yaml``) so that the planner uses the chart in force at a given centre.

This file is not manufacturer documentation. Check every value against the instructions for use before any
clinical or research use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

from ..types import DeviceSize

Bin = Tuple[float, float, float]   # (lower bound, upper bound, device value), bounds in mm, inclusive at 0.1 mm

WEB17 = {
    "name": "WEB 17 SL",
    "width_bins": [(3.0, 3.4, 4.0), (3.5, 3.8, 4.5), (3.9, 4.3, 5.0), (4.4, 5.0, 6.0), (5.1, 6.0, 7.0), (6.1, 7.0, 8.0),
                   (7.1, 8.0, 9.0), (8.1, 9.0, 10.0), (9.1, 10.0, 11.0)],
    "height_bins": [(2.6, 3.6, 2.0), (3.7, 4.9, 3.0), (5.0, 5.9, 4.0), (6.0, 6.9, 5.0), (7.0, 7.9, 6.0), (8.0, 8.9, 7.0),
                    (9.0, 9.9, 8.0), (10.0, 10.9, 9.0)],
    "heights": {4.0: [2.0, 3.0], 4.5: [2.0, 3.0], 5.0: [2.0, 3.0], 6.0: [2.0, 3.0, 4.0], 7.0: [2.0, 3.0, 4.0, 5.0],
                8.0: [3.0, 4.0, 5.0, 6.0], 9.0: [4.0, 5.0, 6.0, 7.0], 10.0: [5.0, 6.0, 7.0, 8.0], 11.0: [6.0, 7.0, 8.0, 9.0]},
    "microcatheter": [(7.0, "VIA 17"), (9.0, "VIA 27"), (11.0, "VIA 33")],
}


@dataclass
class Catalogue:
    """Device sizes and the selection table of one WEB generation."""

    name: str
    width_bins: List[Bin]
    height_bins: List[Bin]
    heights: Dict[float, List[float]]
    microcatheter: List[Tuple[float, str]] = field(default_factory=list)

    # ---------------------------------------------------------------- construction
    @classmethod
    def default(cls) -> "Catalogue":
        return cls.from_dict(WEB17)

    @classmethod
    def from_dict(cls, d: dict) -> "Catalogue":
        return cls(name=d["name"], width_bins=[tuple(map(float, b)) for b in d["width_bins"]],
                   height_bins=[tuple(map(float, b)) for b in d["height_bins"]],
                   heights={float(k): sorted(float(h) for h in v) for k, v in d["heights"].items()},
                   microcatheter=[(float(w), str(n)) for w, n in d.get("microcatheter", [])])

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Catalogue":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh))

    def to_dict(self) -> dict:
        return {"name": self.name, "width_bins": [list(b) for b in self.width_bins],
                "height_bins": [list(b) for b in self.height_bins],
                "heights": {k: list(v) for k, v in self.heights.items()},
                "microcatheter": [list(m) for m in self.microcatheter]}

    # ---------------------------------------------------------------- sizes
    @property
    def widths(self) -> np.ndarray:
        return np.array(sorted(self.heights))

    @property
    def sizes(self) -> List[DeviceSize]:
        return [DeviceSize(w, h) for w in sorted(self.heights) for h in self.heights[w]]

    def __len__(self) -> int:
        return len(self.sizes)

    def index(self, size: DeviceSize) -> int:
        return self.sizes.index(self.snap(size))

    def contains(self, size: DeviceSize) -> bool:
        return any(abs(size.width - s.width) < 1e-6 and abs(size.height - s.height) < 1e-6 for s in self.sizes)

    def snap(self, size: DeviceSize) -> DeviceSize:
        """Nearest catalogue size (width first, then the nearest available height)."""
        w = float(self.widths[np.argmin(np.abs(self.widths - size.width))])
        return DeviceSize(w, self.nearest_height(w, size.height))

    def width_step(self, width: float) -> int:
        return int(np.argmin(np.abs(self.widths - width)))

    def step_width(self, width: float, steps: int) -> float:
        i = int(np.clip(self.width_step(width) + steps, 0, len(self.widths) - 1))
        return float(self.widths[i])

    def nearest_height(self, width: float, height: float) -> float:
        hs = np.asarray(self.heights[float(width)])
        # ties go to the lower height (WEB is undersized in height)
        return float(hs[np.argmin(np.abs(hs - height) + 1e-3 * (hs > height))])

    def microcatheter_for(self, width: float) -> Optional[str]:
        for wmax, name in self.microcatheter:
            if width <= wmax + 1e-9:
                return name
        return self.microcatheter[-1][1] if self.microcatheter else None

    # ---------------------------------------------------------------- distances between sizes
    def steps_between(self, a: DeviceSize, b: DeviceSize) -> Tuple[int, float]:
        """(width steps, height difference in mm) from ``a`` to ``b``."""
        return self.width_step(b.width) - self.width_step(a.width), float(b.height - a.height)

    def within_one_size(self, a: DeviceSize, b: DeviceSize) -> bool:
        """Width within one catalogue step and height within 1 mm."""
        dw, dh = self.steps_between(a, b)
        return abs(dw) <= 1 and abs(dh) <= 1.0 + 1e-6

    def clinically_acceptable(self, plan: DeviceSize, implanted: DeviceSize) -> bool:
        """Prespecified acceptable plan: the implanted size, or within one size of it and not narrower."""
        return self.within_one_size(plan, implanted) and plan.width >= implanted.width - 1e-6


def lookup(bins: Sequence[Bin], x: float) -> Tuple[float, str]:
    """Read a table column/row. Values are rounded to 0.1 mm as on the printed chart."""
    x = round(float(x), 1)
    if x < bins[0][0]:
        return bins[0][2], "below table"
    if x > bins[-1][1]:
        return bins[-1][2], "above table"
    for lo, hi, d in bins:
        if lo - 1e-9 <= x <= hi + 1e-9:
            return d, ""
    return bins[-1][2], "gap"
