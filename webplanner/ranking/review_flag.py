"""Expert-review flag: cases whose top-1 probability is below tau.

``tau`` is the ``quantile`` (default 0.2) of the top-1 probabilities obtained in development cross-validation
(out-of-fold predictions), so that about 20% of development cases are flagged. It is fixed before the model is
applied to external data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, Union

import numpy as np

from ..utils.io import read_json, write_json


def fit_tau(top1_probs: Sequence[float], quantile: float = 0.2) -> float:
    """``numpy.quantile`` (linear interpolation) of the development top-1 probabilities."""
    p = np.asarray(top1_probs, float)
    if p.size == 0 or not np.isfinite(p).all():
        raise ValueError("top-1 probabilities must be a non-empty finite array")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    return float(np.quantile(p, quantile))


def flag(p_top1: Union[float, Sequence[float], np.ndarray], tau: float) -> Union[bool, np.ndarray]:
    """True when the top-1 probability is strictly below ``tau`` (scalar in, bool out; array in, bool array out)."""
    p = np.asarray(p_top1, float)
    out = p < float(tau)
    return bool(out) if out.ndim == 0 else out


@dataclass
class ReviewFlag:
    tau: float
    quantile: float = 0.2
    n_development: int = 0
    source: str = "development cross-validation (out-of-fold top-1 probabilities)"

    @classmethod
    def fit(cls, top1_probs: Sequence[float], quantile: float = 0.2) -> "ReviewFlag":
        return cls(fit_tau(top1_probs, quantile), float(quantile), int(len(top1_probs)))

    def __call__(self, p_top1: Union[float, Sequence[float], np.ndarray]) -> Union[bool, np.ndarray]:
        return flag(p_top1, self.tau)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReviewFlag":
        return cls(float(d["tau"]), float(d.get("quantile", 0.2)), int(d.get("n_development", 0)),
                   str(d.get("source", cls.source)))

    def save(self, path: Union[str, Path]) -> Path:
        return write_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ReviewFlag":
        return cls.from_dict(read_json(path))
