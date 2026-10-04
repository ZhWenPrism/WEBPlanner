"""Agreement between methods or readers.

* ``icc_2_1``: two-way random-effects, absolute-agreement, single-measure intraclass correlation, ICC(2,1) of
  Shrout and Fleiss (1979) = ICC(A,1) of McGraw and Wong (1996), with the F-based confidence interval (McGraw and
  Wong, 1996, Table 7) and the F test of ICC = 0.
* ``bland_altman``: bias and 95% limits of agreement (Bland and Altman, 1986) with confidence intervals of the bias
  and of the limits, approximate (Bland and Altman, 1999) or exact (Carkeet, 2015).
* ``cohen_kappa``: Cohen's kappa (Cohen, 1960), unweighted or with linear / quadratic agreement weights (Cohen,
  1968), large-sample standard error of Fleiss, Cohen and Everitt (1969) or a subject bootstrap.
* ``fleiss_kappa``: Fleiss' kappa for several raters (Fleiss, 1971); standard error under the null hypothesis
  (Fleiss, Nee and Landis, 1979) for the z test and the non-null linearisation variance (Gwet, 2008) for the
  confidence interval.
"""
from __future__ import annotations

import warnings
from typing import NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats

from .proportions import _z, cluster_codes, resample_counts

ArrayLike = Union[Sequence, np.ndarray]


# --------------------------------------------------------------------------------------------------------------------
# intraclass correlation
# --------------------------------------------------------------------------------------------------------------------
class ICCResult(NamedTuple):
    icc: float
    ci_low: float
    ci_high: float
    F: float          # MSR / MSE, test of ICC = 0
    df1: int
    df2: int
    p: float
    n: int            # subjects
    k: int            # raters / methods


def two_way_anova(y: np.ndarray) -> Tuple[float, float, float]:
    """Mean squares (rows = subjects, columns = raters, error) of a complete two-way layout without replication."""
    n, k = y.shape
    grand = y.mean()
    ss_r = k * np.sum((y.mean(axis=1) - grand) ** 2)
    ss_c = n * np.sum((y.mean(axis=0) - grand) ** 2)
    ss_e = max(np.sum((y - grand) ** 2) - ss_r - ss_c, 0.0)
    return ss_r / (n - 1), ss_c / (k - 1), ss_e / ((n - 1) * (k - 1))


def icc_2_1(ratings: ArrayLike, alpha: float = 0.05, nan_policy: str = "raise") -> ICCResult:
    """ICC(2,1) / ICC(A,1) for an ``(n_subjects, k_raters)`` matrix (Shrout and Fleiss, 1979; McGraw and Wong, 1996).

    ``ICC = (MSR - MSE) / (MSR + (k - 1) MSE + k (MSC - MSE) / n)``. The interval uses the Satterthwaite degrees of
    freedom of McGraw and Wong (1996). ``nan_policy="omit"`` drops subjects with any missing rating (with a warning
    that gives the numbers before and after); the default raises.
    """
    y = np.asarray(ratings, float)
    if y.ndim != 2:
        raise ValueError(f"ratings must be (n_subjects, k_raters), got shape {y.shape}")
    bad = np.isnan(y).any(axis=1)
    if bad.any():
        if nan_policy != "omit":
            raise ValueError(f"{int(bad.sum())} subjects have missing ratings; use nan_policy='omit' to drop them")
        warnings.warn(f"icc_2_1: dropped {int(bad.sum())} of {y.shape[0]} subjects with missing ratings "
                      f"(n = {int((~bad).sum())} remain)", stacklevel=2)
        y = y[~bad]
    n, k = y.shape
    if n < 2 or k < 2:
        raise ValueError(f"need at least 2 subjects and 2 raters, got {n} x {k}")
    msr, msc, mse = two_way_anova(y)
    df1, df2 = n - 1, (n - 1) * (k - 1)
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    icc = (msr - mse) / denom if denom > 0 else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        F = msr / mse if mse > 0 else (float("inf") if msr > 0 else float("nan"))
    p = float(stats.f.sf(F, df1, df2)) if np.isfinite(F) else (0.0 if F == np.inf else float("nan"))
    if not np.isfinite(icc):
        return ICCResult(float("nan"), float("nan"), float("nan"), F, df1, df2, p, n, k)
    if icc >= 1.0 - 1e-12:
        return ICCResult(1.0, 1.0, 1.0, F, df1, df2, p, n, k)
    a = k * icc / (n * (1.0 - icc))
    b = 1.0 + k * icc * (n - 1) / (n * (1.0 - icc))
    num = (a * msc + b * mse) ** 2
    den = (a * msc) ** 2 / (k - 1) + (b * mse) ** 2 / ((n - 1) * (k - 1))
    v = num / den if den > 0 else float("nan")
    lo = hi = float("nan")
    if np.isfinite(v) and v > 0:
        f_up = stats.f.ppf(1.0 - alpha / 2.0, n - 1, v)
        f_lo = stats.f.ppf(1.0 - alpha / 2.0, v, n - 1)
        c = k * msc + (k * n - k - n) * mse
        lo = n * (msr - f_up * mse) / (f_up * c + n * msr)
        hi = n * (f_lo * msr - mse) / (c + n * f_lo * msr)
    return ICCResult(float(icc), float(lo), float(hi), float(F), df1, df2, p, n, k)


# --------------------------------------------------------------------------------------------------------------------
# Bland-Altman
# --------------------------------------------------------------------------------------------------------------------
class BlandAltmanResult(NamedTuple):
    bias: float              # mean of a - b
    sd: float                # SD of a - b
    loa_low: float           # bias - z sd
    loa_high: float          # bias + z sd
    bias_ci_low: float
    bias_ci_high: float
    loa_low_ci_low: float
    loa_low_ci_high: float
    loa_high_ci_low: float
    loa_high_ci_high: float
    n: int
    method: str


def bland_altman(a: ArrayLike, b: ArrayLike, alpha: float = 0.05, loa: float = 0.95,
                 method: str = "approx") -> BlandAltmanResult:
    """Bias and limits of agreement of ``a - b`` (Bland and Altman, 1986).

    The bias CI uses the t distribution. The CIs of the limits are ``loa +/- t s sqrt(1/n + z^2 / (2 (n - 1)))``
    for ``method="approx"`` (Bland and Altman, 1999) or the exact non-central t limits for ``method="exact"``
    (Carkeet, 2015). Missing pairs raise.
    """
    x = np.asarray(a, float)
    y = np.asarray(b, float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"a and b must be paired 1D arrays, got {x.shape} and {y.shape}")
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("missing values in a or b; remove incomplete pairs explicitly first")
    n = x.size
    if n < 3:
        raise ValueError("need at least 3 pairs")
    d = x - y
    bias, sd = float(d.mean()), float(d.std(ddof=1))
    z = float(stats.norm.ppf(0.5 + loa / 2.0))
    t = float(stats.t.ppf(1.0 - alpha / 2.0, n - 1))
    lo_lim, hi_lim = bias - z * sd, bias + z * sd
    se_bias = sd / np.sqrt(n)
    if method == "approx":
        se_loa = sd * np.sqrt(1.0 / n + z ** 2 / (2.0 * (n - 1)))
        hi_ci = (hi_lim - t * se_loa, hi_lim + t * se_loa)
        lo_ci = (lo_lim - t * se_loa, lo_lim + t * se_loa)
    elif method == "exact":
        nc = z * np.sqrt(n)
        q_lo = stats.nct.ppf(alpha / 2.0, n - 1, nc) / np.sqrt(n)
        q_hi = stats.nct.ppf(1.0 - alpha / 2.0, n - 1, nc) / np.sqrt(n)
        hi_ci = (bias + q_lo * sd, bias + q_hi * sd)
        lo_ci = (bias - q_hi * sd, bias - q_lo * sd)
    else:
        raise ValueError("method must be 'approx' or 'exact'")
    return BlandAltmanResult(bias, sd, float(lo_lim), float(hi_lim), float(bias - t * se_bias),
                             float(bias + t * se_bias), float(lo_ci[0]), float(lo_ci[1]), float(hi_ci[0]),
                             float(hi_ci[1]), int(n), method)


# --------------------------------------------------------------------------------------------------------------------
# Cohen's kappa
# --------------------------------------------------------------------------------------------------------------------
class KappaResult(NamedTuple):
    kappa: float
    se: float
    ci_low: float
    ci_high: float
    n: int
    weights: str
    ci_method: str


def agreement_weights(n_categories: int, weights: Optional[str] = None) -> np.ndarray:
    """Agreement weights on category positions: identity, ``1 - |i-j|/(K-1)`` or ``1 - (i-j)^2/(K-1)^2``."""
    K = n_categories
    if weights is None or weights == "none":
        return np.eye(K)
    if K < 2:
        return np.ones((K, K))
    i = np.arange(K)
    d = np.abs(i[:, None] - i[None, :]) / (K - 1)
    if weights == "linear":
        return 1.0 - d
    if weights == "quadratic":
        return 1.0 - d ** 2
    raise ValueError("weights must be None, 'linear' or 'quadratic'")


def kappa_from_table(table: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    """Weighted kappa and its large-sample variance (Fleiss, Cohen and Everitt, 1969) from a K x K count table."""
    table = np.asarray(table, float)
    n = table.sum()
    p = table / n
    pr, pc = p.sum(axis=1), p.sum(axis=0)
    po = float(np.sum(w * p))
    pe = float(np.sum(w * np.outer(pr, pc)))
    if 1.0 - pe <= 1e-12:
        return float("nan"), float("nan")
    kappa = (po - pe) / (1.0 - pe)
    w_row = w @ pc            # w_bar_i. = sum_j p_.j w_ij
    w_col = pr @ w            # w_bar_.j = sum_i p_i. w_ij
    term = np.sum(p * (w - (w_row[:, None] + w_col[None, :]) * (1.0 - kappa)) ** 2)
    var = (term - (kappa - pe * (1.0 - kappa)) ** 2) / (n * (1.0 - pe) ** 2)
    return float(kappa), float(max(var, 0.0))


def _codes(a: ArrayLike, b: ArrayLike, labels: Optional[Sequence]) -> Tuple[np.ndarray, np.ndarray, list]:
    x, y = np.asarray(a), np.asarray(b)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"a and b must be paired 1D label vectors, got {x.shape} and {y.shape}")
    for v, nm in ((x, "a"), (y, "b")):
        if v.dtype.kind == "f" and np.isnan(v).any():
            raise ValueError(f"{nm} contains missing labels")
        if v.dtype == object and any(e is None for e in v):
            raise ValueError(f"{nm} contains missing labels")
    cats = list(labels) if labels is not None else list(np.unique(np.concatenate([x, y])))
    index = {c: i for i, c in enumerate(cats)}
    try:
        cx = np.array([index[v] for v in x.tolist()])
        cy = np.array([index[v] for v in y.tolist()])
    except KeyError as exc:
        raise ValueError(f"label {exc.args[0]!r} is not in labels") from exc
    return cx, cy, cats


def cohen_kappa(a: ArrayLike, b: ArrayLike, weights: Optional[str] = None, labels: Optional[Sequence] = None,
                alpha: float = 0.05, ci: str = "asymptotic", n_boot: int = 2000, seed: int = 0) -> KappaResult:
    """Cohen's kappa between two raters (Cohen, 1960; weighted, Cohen, 1968).

    ``weights``: ``None``, ``"linear"`` or ``"quadratic"``, applied to category positions in ``labels`` (default:
    sorted observed labels), as in ``sklearn.metrics.cohen_kappa_score``. ``ci="asymptotic"`` uses the
    Fleiss-Cohen-Everitt (1969) standard error; ``ci="bootstrap"`` the percentile interval over resampled subjects.
    """
    cx, cy, cats = _codes(a, b, labels)
    K, n = len(cats), cx.size
    if n == 0:
        raise ValueError("no ratings")
    w = agreement_weights(K, weights)
    table = np.bincount(cx * K + cy, minlength=K * K).reshape(K, K)
    kappa, var = kappa_from_table(table, w)
    se = float(np.sqrt(var))
    if ci == "asymptotic":
        z = _z(alpha)
        lo, hi = kappa - z * se, kappa + z * se
    elif ci == "bootstrap":
        rng = np.random.default_rng(seed)
        boot = np.empty(n_boot)
        for i in range(n_boot):
            s = rng.integers(0, n, n)
            boot[i] = kappa_from_table(np.bincount(cx[s] * K + cy[s], minlength=K * K).reshape(K, K), w)[0]
        boot = boot[np.isfinite(boot)]
        lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0]) if boot.size else (np.nan, np.nan)
        se = float(boot.std(ddof=1)) if boot.size > 1 else float("nan")
    else:
        raise ValueError("ci must be 'asymptotic' or 'bootstrap'")
    return KappaResult(kappa, se, float(max(lo, -1.0)), float(min(hi, 1.0)), int(n), weights or "none", ci)


# --------------------------------------------------------------------------------------------------------------------
# Fleiss' kappa
# --------------------------------------------------------------------------------------------------------------------
class FleissResult(NamedTuple):
    kappa: float
    se: float            # non-null standard error (Gwet, 2008), used for the CI
    ci_low: float
    ci_high: float
    se0: float           # standard error under kappa = 0 (Fleiss, Nee and Landis, 1979), used for z and p
    z: float
    p: float
    n_subjects: int
    n_raters: int
    n_categories: int


def ratings_to_table(ratings: ArrayLike, categories: Optional[Sequence] = None) -> Tuple[np.ndarray, list]:
    """Subjects x raters label matrix -> subjects x categories count table (like ``statsmodels`` aggregate_raters)."""
    r = np.asarray(ratings, dtype=object)
    if r.ndim != 2:
        raise ValueError(f"ratings must be (n_subjects, n_raters), got shape {r.shape}")
    flat = r.ravel().tolist()
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in flat):
        raise ValueError("ratings contain missing values; Fleiss' kappa needs every subject rated by every rater")
    cats = list(categories) if categories is not None else sorted(set(flat), key=lambda v: (str(type(v)), v))
    index = {c: i for i, c in enumerate(cats)}
    try:
        codes = np.array([index[v] for v in flat]).reshape(r.shape)
    except KeyError as exc:
        raise ValueError(f"rating {exc.args[0]!r} is not in categories") from exc
    table = np.zeros((r.shape[0], len(cats)), dtype=int)
    for j in range(r.shape[1]):
        np.add.at(table, (np.arange(r.shape[0]), codes[:, j]), 1)
    return table, cats


def fleiss_components(table: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Per-subject observed agreement ``P_i`` and category shares ``n_ij / m`` of a count table with ``m`` raters."""
    t = np.asarray(table, float)
    m_i = t.sum(axis=1)
    if t.ndim != 2 or t.shape[0] == 0:
        raise ValueError("table must be (n_subjects, n_categories) with at least one subject")
    if not np.allclose(m_i, m_i[0]):
        raise ValueError("every subject must be rated by the same number of raters")
    m = int(round(m_i[0]))
    if m < 2:
        raise ValueError("need at least 2 raters per subject")
    p_i = (np.sum(t ** 2, axis=1) - m) / (m * (m - 1))
    return p_i, t / m, m


def fleiss_kappa(data: ArrayLike, kind: str = "table", categories: Optional[Sequence] = None,
                 alpha: float = 0.05) -> FleissResult:
    """Fleiss' kappa (Fleiss, 1971) for a subjects x categories count table (``kind="table"``, as in
    ``statsmodels.stats.inter_rater.fleiss_kappa``) or a subjects x raters label matrix (``kind="ratings"``).

    ``se0`` is the null standard error (Fleiss, Nee and Landis, 1979) used for the z test; ``se`` is the non-null
    linearisation standard error (Gwet, 2008) used for the Wald interval.
    """
    if kind == "ratings":
        table, _ = ratings_to_table(data, categories)
    elif kind == "table":
        table = np.asarray(data, float)
    else:
        raise ValueError("kind must be 'table' or 'ratings'")
    p_i, share, m = fleiss_components(table)
    N, K = table.shape
    pj = share.mean(axis=0)
    pbar, pe = float(p_i.mean()), float(np.sum(pj ** 2))
    if 1.0 - pe <= 1e-12:
        nan = float("nan")
        return FleissResult(nan, nan, nan, nan, nan, nan, nan, N, m, K)
    kappa = (pbar - pe) / (1.0 - pe)
    pq = pj * (1.0 - pj)
    s = float(pq.sum())
    var0 = 2.0 * (s ** 2 - float(np.sum(pq * (1.0 - 2.0 * pj)))) / (N * m * (m - 1) * s ** 2)
    se0 = float(np.sqrt(max(var0, 0.0)))
    k_i = (p_i - pe) / (1.0 - pe)
    pe_i = share @ pj
    k_star = k_i - 2.0 * (1.0 - kappa) * (pe_i - pe) / (1.0 - pe)
    se = float(np.sqrt(np.sum((k_star - kappa) ** 2) / (N * (N - 1)))) if N > 1 else float("nan")
    z_crit = _z(alpha)
    z = kappa / se0 if se0 > 0 else float("nan")
    p = float(2.0 * stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan")
    return FleissResult(float(kappa), se, float(max(kappa - z_crit * se, -1.0)), float(min(kappa + z_crit * se, 1.0)),
                        se0, float(z), p, int(N), int(m), int(K))


def fleiss_kappa_bootstrap(table: ArrayLike, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
                           groups: Optional[ArrayLike] = None) -> Tuple[float, float, float, np.ndarray]:
    """Percentile bootstrap interval of Fleiss' kappa over resampled subjects: ``(kappa, lo, hi, distribution)``."""
    p_i, share, _ = fleiss_components(np.asarray(table, float))
    n = p_i.size
    codes, n_units = cluster_codes(groups, n) if groups is not None else (np.arange(n), n)
    w = resample_counts(n_units, n_boot, seed)
    boot = fleiss_from_weights(w, p_i, share, codes, n_units)
    kappa = fleiss_from_weights(np.ones((1, n_units)), p_i, share, codes, n_units)[0]
    finite = boot[np.isfinite(boot)]
    lo, hi = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0]) if finite.size else (np.nan, np.nan)
    return float(kappa), float(lo), float(hi), boot


def fleiss_from_weights(w: np.ndarray, p_i: np.ndarray, share: np.ndarray, codes: np.ndarray,
                        n_units: int) -> np.ndarray:
    """Fleiss' kappa for each row of a resampling-count matrix ``w`` (rows = resamples, columns = units)."""
    size = np.bincount(codes, minlength=n_units).astype(float)
    p_sum = np.bincount(codes, weights=p_i, minlength=n_units)
    s_sum = np.stack([np.bincount(codes, weights=share[:, j], minlength=n_units) for j in range(share.shape[1])], 1)
    tot = w @ size
    pbar = (w @ p_sum) / tot
    pj = (w @ s_sum) / tot[:, None]
    pe = np.sum(pj ** 2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = (pbar - pe) / (1.0 - pe)
    return np.where(1.0 - pe > 1e-12, k, np.nan)
