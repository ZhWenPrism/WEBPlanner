"""Elongation coefficient ``k`` of equation 1, fitted on devices measured on post-deployment cone-beam CT.

Equation 1 gives the free height ``H' = H_d + k W_d c``. With observed deployed heights ``H_obs`` the coefficient is the
no-intercept least-squares slope of ``y = H_obs - H_d`` on ``x = W_d c``:

    k = sum(x y) / sum(x^2)

The confidence interval is a percentile bootstrap over patients: patients (``groups``; each observation is its own
patient by default) are resampled with replacement and ``k`` is refitted on the pooled observations.

Use devices whose height was not limited by the dome (``H_obs`` below the sac extent along the axis); a capped height
carries no information on ``k``. The compression ``c`` usually comes from the virtual deployment, which itself
depends weakly on ``k`` through the deployed height; one refit after updating ``c`` is enough in practice.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class ElongationFit:
    k: float
    ci_low: float
    ci_high: float
    rmse: float          # residual root mean square error of H_obs (mm)
    n: int               # observations
    n_groups: int = 0    # patients used by the bootstrap
    alpha: float = 0.05

    @property
    def ci(self) -> Tuple[float, float]:
        return (self.ci_low, self.ci_high)


def predict_height(nominal_width, nominal_height, compression, k: float = 0.45) -> np.ndarray:
    """Free deployed height ``H_d + k W_d c`` (mm), elementwise."""
    return np.asarray(nominal_height, float) + k * np.asarray(nominal_width, float) * np.asarray(compression, float)


def fit_elongation(nominal_width: Sequence[float], nominal_height: Sequence[float], compression: Sequence[float],
                   observed_height: Sequence[float], n_boot: int = 2000, seed: int = 0,
                   groups: Optional[Sequence[object]] = None, alpha: float = 0.05) -> ElongationFit:
    """Least-squares ``k`` (no intercept) with a patient-level percentile bootstrap CI."""
    W = np.asarray(nominal_width, float).ravel()
    H = np.asarray(nominal_height, float).ravel()
    c = np.asarray(compression, float).ravel()
    Hobs = np.asarray(observed_height, float).ravel()
    n = W.size
    if not (H.size == c.size == Hobs.size == n):
        raise ValueError("nominal_width, nominal_height, compression and observed_height must have the same length")
    ok = np.isfinite(W) & np.isfinite(H) & np.isfinite(c) & np.isfinite(Hobs)
    if not ok.all():
        raise ValueError(f"{int((~ok).sum())} of {n} observations are not finite")
    x, y = W * c, Hobs - H
    sxx = float(x @ x)
    if sxx <= 0:
        raise ValueError("all W_d * c are zero: k is not identifiable")
    k = float(x @ y) / sxx
    rmse = float(np.sqrt(np.mean((y - k * x) ** 2)))

    g = np.arange(n) if groups is None else np.asarray(groups)
    if g.shape[0] != n:
        raise ValueError("groups must have one entry per observation")
    _, gi = np.unique(g, return_inverse=True)
    G = int(gi.max()) + 1
    sxy_g = np.bincount(gi, weights=x * y, minlength=G)
    sxx_g = np.bincount(gi, weights=x * x, minlength=G)
    lo = hi = float("nan")
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, G, size=(n_boot, G))
        den = sxx_g[idx].sum(axis=1)
        valid = den > 0
        kb = sxy_g[idx].sum(axis=1)[valid] / den[valid]
        if kb.size:
            lo, hi = (float(v) for v in np.quantile(kb, [alpha / 2, 1 - alpha / 2]))
    return ElongationFit(k=k, ci_low=lo, ci_high=hi, rmse=rmse, n=n, n_groups=G, alpha=alpha)


__all__ = ["ElongationFit", "fit_elongation", "predict_height"]
