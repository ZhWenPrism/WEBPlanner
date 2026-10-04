"""Discrimination: AUROC with DeLong confidence intervals and the DeLong test for correlated AUROCs.

The covariance of one or more AUROCs computed on the same cases follows DeLong, DeLong and Clarke-Pearson (1988),
computed with the fast midrank algorithm of Sun and Xu (2014). Ties are handled by midranks (an AUROC tie counts as
one half), so the estimate equals ``sklearn.metrics.roc_auc_score``.
"""
from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np
from scipy import stats

from .proportions import _z

ArrayLike = Union[Sequence[float], np.ndarray]


def _binary(y_true: ArrayLike) -> np.ndarray:
    y = np.asarray(y_true)
    if y.ndim != 1:
        raise ValueError(f"y_true must be one-dimensional, got shape {y.shape}")
    if y.dtype != bool:
        f = y.astype(float)
        if np.isnan(f).any() or not np.isin(f, (0.0, 1.0)).all():
            raise ValueError("y_true must be binary (0/1 or bool) without missing values")
        y = f.astype(bool)
    if y.all() or not y.any():
        raise ValueError("y_true must contain both classes")
    return y


def _scores(y: np.ndarray, scores: Sequence[ArrayLike]) -> np.ndarray:
    s = np.vstack([np.asarray(x, float) for x in scores])
    if s.shape[1] != y.size:
        raise ValueError(f"scores must have length {y.size}, got {s.shape[1]}")
    if not np.isfinite(s).all():
        raise ValueError("scores contain missing or infinite values")
    return s


def delong_cov(y_true: ArrayLike, *scores: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """AUROCs of each score and their DeLong covariance matrix (DeLong et al., 1988; Sun and Xu, 2014).

    Returns ``(aucs, cov)`` with ``aucs`` of shape ``(k,)`` and ``cov`` of shape ``(k, k)``.
    """
    if not scores:
        raise ValueError("give at least one score vector")
    y = _binary(y_true)
    s = _scores(y, scores)
    pos, neg = s[:, y], s[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    tx = stats.rankdata(pos, axis=1)
    ty = stats.rankdata(neg, axis=1)
    tz = stats.rankdata(np.hstack([pos, neg]), axis=1)
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n            # structural components of the positives
    v10 = 1.0 - (tz[:, m:] - ty) / m      # structural components of the negatives
    sx = np.atleast_2d(np.cov(v01)) if m > 1 else np.zeros((s.shape[0], s.shape[0]))
    sy = np.atleast_2d(np.cov(v10)) if n > 1 else np.zeros((s.shape[0], s.shape[0]))
    return aucs, sx / m + sy / n


def auroc_delong(y_true: ArrayLike, score: ArrayLike, alpha: float = 0.05) -> Tuple[float, float, float]:
    """AUROC with its DeLong Wald interval, clipped to [0, 1]: ``(auc, lo, hi)`` (DeLong et al., 1988)."""
    aucs, cov = delong_cov(y_true, score)
    auc, se = float(aucs[0]), float(np.sqrt(max(cov[0, 0], 0.0)))
    z = _z(alpha)
    return auc, float(max(auc - z * se, 0.0)), float(min(auc + z * se, 1.0))


def auroc_se(y_true: ArrayLike, score: ArrayLike) -> float:
    """DeLong standard error of the AUROC."""
    _, cov = delong_cov(y_true, score)
    return float(np.sqrt(max(cov[0, 0], 0.0)))


def delong_test(y_true: ArrayLike, s1: ArrayLike, s2: ArrayLike) -> Tuple[float, float, float, float]:
    """DeLong test of two correlated AUROCs on the same cases (DeLong et al., 1988): ``(auc1, auc2, z, p)``.

    Identical AUROCs with zero variance of the difference (for example identical scores) give ``z = 0, p = 1``.
    """
    aucs, cov = delong_cov(y_true, s1, s2)
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    if var <= 1e-15:
        if abs(diff) <= 1e-12:
            return float(aucs[0]), float(aucs[1]), 0.0, 1.0
        return float(aucs[0]), float(aucs[1]), float(np.sign(diff) * np.inf), 0.0
    z = diff / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(z), float(2.0 * stats.norm.sf(abs(z)))


def auroc_difference_ci(y_true: ArrayLike, s1: ArrayLike, s2: ArrayLike, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Difference ``AUC1 - AUC2`` with its DeLong Wald interval."""
    aucs, cov = delong_cov(y_true, s1, s2)
    diff = float(aucs[0] - aucs[1])
    se = float(np.sqrt(max(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1], 0.0)))
    z = _z(alpha)
    return diff, diff - z * se, diff + z * se
