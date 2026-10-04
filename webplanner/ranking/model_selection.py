"""Patient-grouped cross-validation for the conditional logit.

All splits are computed from patient identifiers before any model is fitted, and every split is checked so that no
patient contributes cases to both the training and the test side. The penalty ``l2`` is chosen on the same splits for
every grid value; ties are broken towards the larger penalty.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .conditional_logit import ChosenLike, ConditionalLogit, resolve_chosen
from .features import CandidateSet

DEFAULT_GRID = np.logspace(-3, 2, 11)
EPS = 1e-12


def check_no_overlap(groups: Sequence[Any], train: np.ndarray, test: np.ndarray) -> None:
    """Raise ``AssertionError`` when a patient (group) has cases on both sides of a split."""
    g = np.asarray(groups, dtype=object)
    shared = set(g[train].tolist()) & set(g[test].tolist())
    if shared:
        raise AssertionError(f"patients in both train and test: {sorted(map(str, shared))[:5]}")


def group_kfold(groups: Sequence[Any], n_splits: int = 5, seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Patient-grouped K-fold with shuffled group order.

    Groups are permuted with ``seed``, ordered by size (largest first, stable) and each is assigned to the fold with
    the fewest cases, as in scikit-learn's ``GroupKFold`` but reproducibly shuffled and independent of its version.
    """
    g = np.asarray(groups, dtype=object)
    uniq, inv, counts = np.unique(g.astype(str), return_inverse=True, return_counts=True)
    if n_splits < 2 or n_splits > len(uniq):
        raise ValueError(f"n_splits={n_splits} needs 2 <= n_splits <= number of groups ({len(uniq)})")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    order = perm[np.argsort(-counts[perm], kind="stable")]
    fold_of_group = np.empty(len(uniq), int)
    load = np.zeros(n_splits, int)
    for gi in order:
        k = int(np.argmin(load))
        fold_of_group[gi] = k
        load[k] += counts[gi]
    fold = fold_of_group[inv]
    splits = []
    for k in range(n_splits):
        test = np.where(fold == k)[0]
        train = np.where(fold != k)[0]
        check_no_overlap(g, train, test)
        splits.append((train, test))
    return splits


def ranking_metrics(proba: Sequence[np.ndarray], chosen: np.ndarray) -> Dict[str, float]:
    """Mean log loss (probabilities clipped at 1e-12), top-1 and top-3 accuracy."""
    chosen = np.asarray(chosen, int)
    p_c = np.array([p[c] for p, c in zip(proba, chosen)])
    rank = np.array([int(np.sum(p > p[c])) for p, c in zip(proba, chosen)])
    return {"n": int(len(chosen)), "log_loss": float(np.mean(-np.log(np.maximum(p_c, EPS)))),
            "top1": float(np.mean(rank == 0)), "top3": float(np.mean(rank < 3))}


def _subset(sets: Sequence[CandidateSet], idx: np.ndarray) -> List[CandidateSet]:
    return [sets[i] for i in idx]


def _select(table: List[Dict[str, float]], metric: str) -> int:
    if metric == "log_loss":
        key = [(r["log_loss"], -r["l2"]) for r in table]
    elif metric == "top1":
        key = [(-r["top1"], r["log_loss"], -r["l2"]) for r in table]
    else:
        raise ValueError("metric must be 'log_loss' or 'top1'")
    return int(min(range(len(table)), key=lambda i: key[i]))


def select_l2(sets: Sequence[CandidateSet], chosen: ChosenLike, groups: Sequence[Any], grid: Sequence[float] = DEFAULT_GRID,
              n_splits: int = 5, seed: int = 0, metric: str = "log_loss",
              model_kw: Optional[Dict[str, Any]] = None) -> Tuple[float, List[Dict[str, float]]]:
    """Choose ``l2`` by patient-grouped CV. Returns ``(best_l2, table)``; the table holds pooled out-of-fold log loss,
    top-1 and top-3 accuracy and the across-fold SD of the log loss for each grid value."""
    idx = resolve_chosen(sets, chosen)
    grid = [float(v) for v in grid]
    splits = group_kfold(groups, n_splits, seed)
    proba: List[List[Optional[np.ndarray]]] = [[None] * len(sets) for _ in grid]
    fold_ll = np.zeros((len(grid), n_splits))
    order = np.argsort(grid)[::-1]                        # strong to weak penalty, warm-started
    for k, (tr, te) in enumerate(splits):
        model = ConditionalLogit(**{**(model_kw or {}), "warm_start": True})
        for gi in order:
            model.l2 = grid[gi]
            model.fit(_subset(sets, tr), idx[tr])
            pk = model.predict_proba_many(_subset(sets, te))
            for i, p in zip(te, pk):
                proba[gi][i] = p
            fold_ll[gi, k] = ranking_metrics(pk, idx[te])["log_loss"]
    table = []
    for gi, l2 in enumerate(grid):
        m = ranking_metrics(proba[gi], idx)
        table.append({"l2": l2, "log_loss": m["log_loss"], "log_loss_sd": float(fold_ll[gi].std(ddof=1)),
                      "top1": m["top1"], "top3": m["top3"]})
    return grid[_select(table, metric)], table


def cross_val_predict(sets: Sequence[CandidateSet], chosen: ChosenLike, groups: Sequence[Any], l2: float = 1.0,
                      n_splits: int = 5, seed: int = 0, model_kw: Optional[Dict[str, Any]] = None,
                      return_folds: bool = False):
    """Out-of-fold probability arrays (one per case, aligned with ``sets``) from patient-grouped folds."""
    idx = resolve_chosen(sets, chosen)
    splits = group_kfold(groups, n_splits, seed)
    proba: List[Optional[np.ndarray]] = [None] * len(sets)
    fold = np.full(len(sets), -1)
    for k, (tr, te) in enumerate(splits):
        check_no_overlap(groups, tr, te)
        model = ConditionalLogit(**{**(model_kw or {}), "l2": l2}).fit(_subset(sets, tr), idx[tr])
        for i in te:
            proba[i] = model.predict_proba(sets[i])
            fold[i] = k
    assert (fold >= 0).all()
    return (proba, fold) if return_folds else proba


@dataclass
class CVResult:
    """Out-of-fold predictions with the penalty chosen inside each outer fold."""

    proba: List[np.ndarray]
    fold: np.ndarray                                       # outer fold index or centre name per case
    l2: Dict[Any, float] = field(default_factory=dict)
    tables: Dict[Any, List[Dict[str, float]]] = field(default_factory=dict)

    def top1_probs(self) -> np.ndarray:
        return np.array([float(p.max()) for p in self.proba])

    def metrics(self, chosen: np.ndarray) -> Dict[str, float]:
        return ranking_metrics(self.proba, chosen)


def nested_cv(sets: Sequence[CandidateSet], chosen: ChosenLike, groups: Sequence[Any], grid: Sequence[float] = DEFAULT_GRID,
              n_outer: int = 5, n_inner: int = 5, seed: int = 0, metric: str = "log_loss",
              model_kw: Optional[Dict[str, Any]] = None) -> CVResult:
    """Outer patient-grouped folds for prediction; ``l2`` selected by inner grouped CV on each outer training set."""
    idx = resolve_chosen(sets, chosen)
    g = np.asarray(groups, dtype=object)
    outer = group_kfold(g, n_outer, seed)
    proba: List[Optional[np.ndarray]] = [None] * len(sets)
    fold = np.full(len(sets), -1)
    res = CVResult([], fold)
    for k, (tr, te) in enumerate(outer):
        best, table = select_l2(_subset(sets, tr), idx[tr], g[tr], grid, min(n_inner, len(set(g[tr].tolist()))),
                                seed + 1 + k, metric, model_kw)
        model = ConditionalLogit(**{**(model_kw or {}), "l2": best}).fit(_subset(sets, tr), idx[tr])
        for i in te:
            proba[i] = model.predict_proba(sets[i])
            fold[i] = k
        res.l2[k], res.tables[k] = best, table
    assert (fold >= 0).all()
    res.proba = proba
    return res


def leave_one_centre_out(sets: Sequence[CandidateSet], chosen: ChosenLike, centres: Sequence[Any], groups: Sequence[Any],
                         grid: Sequence[float] = DEFAULT_GRID, n_inner: int = 5, seed: int = 0, metric: str = "log_loss",
                         model_kw: Optional[Dict[str, Any]] = None) -> CVResult:
    """Each centre in turn is the test set; ``l2`` is chosen by grouped CV on the other centres only."""
    idx = resolve_chosen(sets, chosen)
    c = np.asarray(centres, dtype=object)
    g = np.asarray(groups, dtype=object)
    if len(c) != len(sets) or len(g) != len(sets):
        raise ValueError("centres and groups must have one entry per candidate set")
    names = sorted(set(c.tolist()), key=str)
    if len(names) < 2:
        raise ValueError("leave-one-centre-out needs at least two centres")
    proba: List[Optional[np.ndarray]] = [None] * len(sets)
    count = np.zeros(len(sets), int)
    res = CVResult([], c.copy())
    for centre in names:
        te = np.where(c == centre)[0]
        tr = np.where(c != centre)[0]
        check_no_overlap(g, tr, te)
        best, table = select_l2(_subset(sets, tr), idx[tr], g[tr], grid, min(n_inner, len(set(g[tr].tolist()))), seed,
                                metric, model_kw)
        model = ConditionalLogit(**{**(model_kw or {}), "l2": best}).fit(_subset(sets, tr), idx[tr])
        for i in te:
            proba[i] = model.predict_proba(sets[i])
            count[i] += 1
        res.l2[centre], res.tables[centre] = best, table
    assert (count == 1).all()
    res.proba = proba
    return res
