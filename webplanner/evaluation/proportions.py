"""Proportions, paired binary comparisons and the case-level bootstrap.

* Wilson score interval for one proportion (Wilson, 1927).
* Hybrid score interval for the difference between two paired proportions (Newcombe, 1998, method 10).
* Exact McNemar test: two-sided binomial test on the discordant pairs (McNemar, 1947).
* Paired bootstrap of a difference that resamples clusters (cases) with all their rows, percentile interval
  (Efron and Tibshirani, 1993) and bootstrap p value with the +1 correction (Davison and Hinkley, 1997).
* Formatting helpers in the journal style ("21/25 (84.0%; 95% CI 65.3% to 93.6%)"). Negative numbers keep the
  ASCII hyphen; the caller replaces it with a typographic minus sign when needed.
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats

ArrayLike = Union[Sequence[float], np.ndarray]


def _z(alpha: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return float(stats.norm.ppf(1.0 - alpha / 2.0))


def ci_label(alpha: float = 0.05) -> str:
    """'95% CI' for alpha = 0.05."""
    return f"{100.0 * (1.0 - alpha):g}% CI"


def as_bool_vector(x: ArrayLike, name: str = "x") -> np.ndarray:
    """Boolean vector from bool or 0/1 input; missing values raise."""
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if arr.dtype == bool:
        return arr
    try:
        f = arr.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be boolean or 0/1") from exc
    if np.isnan(f).any():
        raise ValueError(f"{name} contains {int(np.isnan(f).sum())} missing values; remove them explicitly first")
    if not np.isin(f, (0.0, 1.0)).all():
        raise ValueError(f"{name} must be boolean or 0/1, found values {np.unique(f)[:10]}")
    return f.astype(bool)


# --------------------------------------------------------------------------------------------------------------------
# one proportion
# --------------------------------------------------------------------------------------------------------------------
class Proportion(NamedTuple):
    k: int
    n: int
    p: float
    ci_low: float
    ci_high: float
    alpha: float = 0.05

    def fmt(self, digits: int = 1) -> str:
        return fmt_prop(self.k, self.n, alpha=self.alpha, digits=digits)


def wilson_ci(k: Union[int, ArrayLike], n: Union[int, ArrayLike], alpha: float = 0.05):
    """Wilson score interval for ``k`` successes in ``n`` trials (Wilson, 1927).

    Returns ``(lo, hi)`` as floats (or arrays for array input). ``n = 0`` gives ``(nan, nan)``.
    """
    k_arr = np.asarray(k, float)
    n_arr = np.asarray(n, float)
    if np.any(n_arr < 0) or np.any(k_arr < 0) or np.any(k_arr > n_arr):
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    z = _z(alpha)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k_arr / n_arr
        denom = 1.0 + z ** 2 / n_arr
        centre = (p + z ** 2 / (2.0 * n_arr)) / denom
        half = z * np.sqrt(p * (1.0 - p) / n_arr + z ** 2 / (4.0 * n_arr ** 2)) / denom
    lo = np.where(k_arr == 0, 0.0, np.clip(centre - half, 0.0, 1.0))
    hi = np.where(k_arr == n_arr, 1.0, np.clip(centre + half, 0.0, 1.0))
    lo = np.where(n_arr > 0, lo, np.nan)
    hi = np.where(n_arr > 0, hi, np.nan)
    if lo.ndim == 0:
        return float(lo), float(hi)
    return lo, hi


def proportion(k: int, n: int, alpha: float = 0.05) -> Proportion:
    """``k/n`` with its Wilson interval."""
    lo, hi = wilson_ci(k, n, alpha)
    return Proportion(int(k), int(n), float(k) / n if n > 0 else float("nan"), lo, hi, alpha)


def proportion_of(x: ArrayLike, alpha: float = 0.05, name: str = "x") -> Proportion:
    """Proportion of ``True`` in a boolean vector, with its Wilson interval."""
    b = as_bool_vector(x, name)
    return proportion(int(b.sum()), int(b.size), alpha)


# --------------------------------------------------------------------------------------------------------------------
# paired proportions
# --------------------------------------------------------------------------------------------------------------------
def paired_counts(x1: ArrayLike, x2: ArrayLike) -> Tuple[int, int, int, int]:
    """Cells of the paired 2x2 table: (both, first only, second only, neither)."""
    a1, a2 = as_bool_vector(x1, "x1"), as_bool_vector(x2, "x2")
    if a1.shape != a2.shape:
        raise ValueError(f"paired vectors differ in length: {a1.size} vs {a2.size}")
    return (int((a1 & a2).sum()), int((a1 & ~a2).sum()), int((~a1 & a2).sum()), int((~a1 & ~a2).sum()))


def discordant_counts(x1: ArrayLike, x2: ArrayLike) -> Tuple[int, int]:
    """(b, c): pairs positive on the first measure only and on the second measure only."""
    _, b, c, _ = paired_counts(x1, x2)
    return b, c


def _check_count(v: object, name: str) -> int:
    f = float(v)
    if f < 0 or f != int(f):
        raise ValueError(f"{name} must be a non-negative integer count, got {v}")
    return int(f)


def newcombe_paired_diff_ci(a: Union[int, ArrayLike], b: Union[int, ArrayLike], c: Optional[int] = None,
                            d: Optional[int] = None, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Difference ``p1 - p2`` of two paired proportions with Newcombe's hybrid score interval (Newcombe, 1998,
    method 10, without continuity correction).

    Call with the four cells of the paired table (``a`` both positive, ``b`` first only, ``c`` second only, ``d``
    neither) or with two paired boolean vectors ``(x1, x2)``. Returns ``(difference, lo, hi)``.
    """
    if c is None and d is None:
        a, b, c, d = paired_counts(a, b)
    elif c is None or d is None:
        raise ValueError("give either four cell counts or two paired boolean vectors")
    a, b, c, d = (_check_count(v, nm) for v, nm in zip((a, b, c, d), "abcd"))
    n = a + b + c + d
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p1, p2 = (a + b) / n, (a + c) / n
    theta = p1 - p2
    l1, u1 = wilson_ci(a + b, n, alpha)
    l2, u2 = wilson_ci(a + c, n, alpha)
    margins = float(a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / np.sqrt(margins) if margins > 0 else 0.0
    delta = np.sqrt(max((p1 - l1) ** 2 - 2.0 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0.0))
    eps = np.sqrt(max((u1 - p1) ** 2 - 2.0 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0.0))
    return float(theta), float(max(theta - delta, -1.0)), float(min(theta + eps, 1.0))


def mcnemar_exact(b: Union[int, ArrayLike], c: Union[int, ArrayLike]) -> float:
    """Two-sided exact McNemar p value (McNemar, 1947): ``min(1, 2 P(X <= min(b, c)))``, ``X ~ Bin(b + c, 1/2)``.

    ``b`` and ``c`` are the discordant counts, or two paired boolean vectors from which they are computed.
    No discordant pairs gives ``p = 1``.
    """
    if np.ndim(b) == 0 and np.ndim(c) == 0:
        nb, nc = _check_count(b, "b"), _check_count(c, "c")
    else:
        nb, nc = discordant_counts(b, c)
    n = nb + nc
    if n == 0:
        return 1.0
    return float(min(1.0, 2.0 * stats.binom.cdf(min(nb, nc), n, 0.5)))


# --------------------------------------------------------------------------------------------------------------------
# cluster bootstrap
# --------------------------------------------------------------------------------------------------------------------
class BootstrapResult(NamedTuple):
    estimate: float          # stat(x_ai) - stat(x_un) on the observed data
    ci_low: float            # percentile interval
    ci_high: float
    p: float                 # two-sided bootstrap p value for a zero difference, (1 + count) / (B + 1)
    p_min: float             # smallest attainable p value, 2 / (B + 1)
    n_boot: int
    n_units: int             # resampled units (cases when groups are given, else pairs)
    n_obs: int
    distribution: np.ndarray


def bootstrap_p(boot: np.ndarray, null: float = 0.0) -> Tuple[float, float]:
    """Two-sided percentile bootstrap p value with the +1 correction (Davison and Hinkley, 1997).

    Returns ``(p, p_min)`` where ``p_min = 2 / (B + 1)`` is the smallest attainable value.
    """
    boot = np.asarray(boot, float)
    boot = boot[np.isfinite(boot)]
    nb = boot.size
    if nb == 0:
        return float("nan"), float("nan")
    lo = (1.0 + np.sum(boot <= null)) / (nb + 1.0)
    hi = (1.0 + np.sum(boot >= null)) / (nb + 1.0)
    return float(min(1.0, 2.0 * min(lo, hi))), 2.0 / (nb + 1.0)


def cluster_codes(groups: ArrayLike, n: int) -> Tuple[np.ndarray, int]:
    """Integer codes ``0..G-1`` for a cluster label vector of length ``n``."""
    g = np.asarray(groups)
    if g.shape != (n,):
        raise ValueError(f"groups must have length {n}, got shape {g.shape}")
    _, codes = np.unique(g.astype(str) if g.dtype == object else g, return_inverse=True)
    return codes.astype(int), int(codes.max()) + 1


def resample_counts(n_units: int, n_boot: int, seed: int) -> np.ndarray:
    """``(n_boot, n_units)`` matrix of how often each unit is drawn in each bootstrap resample."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_units, size=(n_boot, n_units))
    w = np.zeros((n_boot, n_units))
    np.add.at(w, (np.repeat(np.arange(n_boot), n_units), idx.ravel()), 1.0)
    return w


def paired_bootstrap_diff(x_un: ArrayLike, x_ai: ArrayLike, groups: Optional[ArrayLike] = None,
                          stat: Callable[[np.ndarray], float] = np.mean, n_boot: int = 4000, seed: int = 0,
                          alpha: float = 0.05) -> BootstrapResult:
    """Bootstrap interval of ``stat(x_ai) - stat(x_un)`` for paired observations (Efron and Tibshirani, 1993).

    Element ``i`` of ``x_un`` and ``x_ai`` is one pair (for example one reader on one case, unaided and assisted).
    With ``groups`` (the case of each pair) whole cases are resampled with replacement so that all reads of a case
    stay together; otherwise pairs are resampled.
    """
    u = np.asarray(x_un, float)
    a = np.asarray(x_ai, float)
    if u.ndim != 1 or u.shape != a.shape:
        raise ValueError(f"x_un and x_ai must be paired 1D arrays, got {u.shape} and {a.shape}")
    if u.size == 0:
        raise ValueError("no paired observations")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    n = u.size
    codes, n_units = cluster_codes(groups, n) if groups is not None else (np.arange(n), n)
    estimate = float(stat(a) - stat(u))
    w = resample_counts(n_units, n_boot, seed)
    if stat is np.mean:
        size = np.bincount(codes, minlength=n_units).astype(float)
        diff_sum = np.bincount(codes, weights=a - u, minlength=n_units)
        boot = (w @ diff_sum) / (w @ size)
    else:
        order = np.argsort(codes, kind="stable")
        bounds = np.concatenate([[0], np.cumsum(np.bincount(codes, minlength=n_units))])
        members = [order[bounds[g]:bounds[g + 1]] for g in range(n_units)]
        boot = np.empty(n_boot)
        for i in range(n_boot):
            rows = np.concatenate([np.repeat(members[g], int(c)) for g, c in enumerate(w[i]) if c > 0])
            boot[i] = stat(a[rows]) - stat(u[rows])
    finite = boot[np.isfinite(boot)]
    lo, hi = (np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0]) if finite.size else (np.nan, np.nan))
    p, p_min = bootstrap_p(boot)
    return BootstrapResult(estimate, float(lo), float(hi), p, p_min, int(n_boot), int(n_units), int(n), boot)


# --------------------------------------------------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------------------------------------------------
def fmt_pct(x: float, digits: int = 1) -> str:
    return "NA" if not np.isfinite(x) else f"{100.0 * x:.{digits}f}%"


def fmt_prop(k: int, n: int, alpha: float = 0.05, digits: int = 1) -> str:
    """``"21/25 (84.0%; 95% CI 65.3% to 93.6%)"``."""
    if n == 0:
        return f"{k}/{n} (not estimable)"
    lo, hi = wilson_ci(k, n, alpha)
    return f"{k}/{n} ({fmt_pct(k / n, digits)}; {ci_label(alpha)} {fmt_pct(lo, digits)} to {fmt_pct(hi, digits)})"


def fmt_ci(estimate: float, lo: float, hi: float, digits: int = 2, pct: bool = False, alpha: float = 0.05,
           unit: str = "") -> str:
    """``"0.84 (95% CI 0.65 to 0.94)"``; ``pct=True`` prints proportions as percentages."""
    if pct:
        f = lambda v: fmt_pct(v, digits)  # noqa: E731
    else:
        f = lambda v: ("NA" if not np.isfinite(v) else f"{v:.{digits}f}{unit}")  # noqa: E731
    return f"{f(estimate)} ({ci_label(alpha)} {f(lo)} to {f(hi)})"


def fmt_p(p: float, digits: int = 3, threshold: float = 0.001) -> str:
    """``"p = 0.012"`` or ``"p < 0.001"``."""
    if not np.isfinite(p):
        return "p = NA"
    if p < threshold:
        return f"p < {threshold:g}"
    return f"p = {p:.{digits}f}"
