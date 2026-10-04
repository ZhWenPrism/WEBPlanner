"""Penalised conditional logit over the feasible sizes of each case (manuscript equation 2).

    P(d | x) = exp(beta^T phi(x, d)) / sum_{d' in C(x)} exp(beta^T phi(x, d'))

The objective is the negative log-likelihood plus ``(l2 / 2) * ||beta||^2`` on the standardised feature scale (no
intercept: a constant cancels within a case, so every coefficient is penalised). Standardisation statistics are
computed on the feasible candidate rows of the training cases only. Infeasible sizes get utility ``-inf`` and
probability 0.

Implanted sizes outside C(x)
----------------------------
Real records can contain an implanted size narrower than the measured neck (measurement disagreement). With
``infeasible_chosen="add"`` (default) such a size is added to the feasible set of that training case, a warning gives
the number of affected cases, and ``n_added_chosen_`` records it; no case is dropped. ``"raise"`` stops instead.
Prediction always uses the feasible set of the candidate set as built.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import minimize

from ..types import DeviceSize
from ..utils.io import read_json, write_json
from .features import CandidateSet

ChosenLike = Optional[Sequence[Union[int, DeviceSize, None]]]


def resolve_chosen(sets: Sequence[CandidateSet], chosen: ChosenLike = None) -> np.ndarray:
    """Chosen indices from ints, ``DeviceSize`` objects, or (``chosen=None``) ``CandidateSet.chosen_index``."""
    if chosen is None:
        chosen = [cs.chosen_index for cs in sets]
    if len(chosen) != len(sets):
        raise ValueError(f"{len(chosen)} chosen entries for {len(sets)} candidate sets")
    out = np.empty(len(sets), int)
    for i, (cs, c) in enumerate(zip(sets, chosen)):
        if c is None:
            raise ValueError(f"case {cs.case_id}: no chosen (implanted) size")
        idx = cs.index_of(c) if isinstance(c, DeviceSize) else int(c)
        if not 0 <= idx < cs.n_sizes:
            raise ValueError(f"case {cs.case_id}: chosen index {idx} out of range")
        out[i] = idx
    return out


def pack(sets: Sequence[CandidateSet]) -> Tuple[np.ndarray, np.ndarray]:
    """Pad candidate sets to ``X (n, m, p)`` and ``mask (n, m)``; padded rows are zero and masked out."""
    if not sets:
        raise ValueError("no candidate sets")
    names = sets[0].names
    for cs in sets:
        if cs.names != names:
            raise ValueError(f"case {cs.case_id}: feature names differ from the first candidate set")
    n, m, p = len(sets), max(cs.n_sizes for cs in sets), len(names)
    X = np.zeros((n, m, p))
    mask = np.zeros((n, m), bool)
    for i, cs in enumerate(sets):
        X[i, :cs.n_sizes] = cs.X
        mask[i, :cs.n_sizes] = cs.feasible
    return X, mask


def penalised_nll(beta: np.ndarray, X: np.ndarray, mask: np.ndarray, chosen: np.ndarray, l2: float) -> Tuple[float, np.ndarray]:
    """``(f, grad)`` with ``f = -sum_i log P(chosen_i | x_i) + (l2 / 2) ||beta||^2``; log-sum-exp stabilised."""
    rows = np.arange(X.shape[0])
    U = np.where(mask, X @ beta, -np.inf)
    umax = U.max(axis=1, keepdims=True)
    E = np.exp(U - umax)
    S = E.sum(axis=1, keepdims=True)
    log_z = umax[:, 0] + np.log(S[:, 0])
    P = E / S
    f = -float(np.sum(U[rows, chosen] - log_z)) + 0.5 * l2 * float(beta @ beta)
    g = -(X[rows, chosen] - np.einsum("nm,nmp->np", P, X)).sum(axis=0) + l2 * beta
    return f, g


def _softmax_masked(u: np.ndarray, mask: np.ndarray) -> np.ndarray:
    u = np.where(mask, u, -np.inf)
    e = np.exp(u - u.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class ConditionalLogit:
    """L2-penalised conditional logit fitted with L-BFGS-B and an analytic gradient."""

    def __init__(self, l2: float = 1.0, standardize: bool = True, max_iter: int = 500, tol: float = 1e-8,
                 infeasible_chosen: str = "add", warm_start: bool = False) -> None:
        if infeasible_chosen not in ("add", "raise"):
            raise ValueError("infeasible_chosen must be 'add' or 'raise'")
        self.l2 = float(l2)
        self.standardize = bool(standardize)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.infeasible_chosen = infeasible_chosen
        self.warm_start = bool(warm_start)
        self.names_: Optional[List[str]] = None
        self.beta_: Optional[np.ndarray] = None

    # ---------------------------------------------------------------- data
    def _training_mask(self, sets: Sequence[CandidateSet], mask: np.ndarray, chosen: np.ndarray, warn: bool = True) -> Tuple[np.ndarray, int]:
        rows = np.arange(len(sets))
        bad = np.where(~mask[rows, chosen])[0]
        if len(bad) and self.infeasible_chosen == "raise":
            raise ValueError(f"chosen size outside the feasible set in {len(bad)} cases, e.g. {sets[bad[0]].case_id}")
        if len(bad):
            mask = mask.copy()
            mask[bad, chosen[bad]] = True
            if warn:
                ids = ", ".join(sets[i].case_id for i in bad[:5])
                warnings.warn(f"{len(bad)} chosen sizes were outside the feasible set and were added to it ({ids})",
                              stacklevel=3)
        return mask, len(bad)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    # ---------------------------------------------------------------- fit
    def fit(self, sets: Sequence[CandidateSet], chosen: ChosenLike = None) -> "ConditionalLogit":
        """Fit on candidate sets; ``chosen`` as in :func:`resolve_chosen`."""
        idx = resolve_chosen(sets, chosen)
        X, mask = pack(sets)
        mask, n_added = self._training_mask(sets, mask, idx)
        p = X.shape[2]
        if self.standardize:
            rows = X[mask]
            mean, scale = rows.mean(axis=0), rows.std(axis=0)
            scale = np.where(scale > 1e-12, scale, 1.0)
        else:
            mean, scale = np.zeros(p), np.ones(p)
        self.mean_, self.scale_ = mean, scale
        Xs = np.where(mask[..., None], self._transform(X), 0.0)
        n = len(sets)

        def fun(b: np.ndarray) -> Tuple[float, np.ndarray]:
            f, g = penalised_nll(b, Xs, mask, idx, self.l2)
            return f / n, g / n

        x0 = np.zeros(p)
        if self.warm_start and self.beta_ is not None and self.names_ == list(sets[0].names):
            x0 = np.asarray(self.beta_, float).copy()
        res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": self.tol})
        self.beta_ = np.asarray(res.x, float)
        self.names_ = list(sets[0].names)
        self.n_cases_ = n
        self.n_iter_ = int(res.nit)
        self.converged_ = bool(res.success)
        self.n_added_chosen_ = int(n_added)
        self.objective_ = float(res.fun) * n
        if not res.success:
            warnings.warn(f"ConditionalLogit did not converge: {res.message}", stacklevel=2)
        return self

    # ---------------------------------------------------------------- prediction
    def _check(self, cs: CandidateSet) -> None:
        if self.beta_ is None:
            raise RuntimeError("model is not fitted")
        if list(cs.names) != self.names_:
            raise ValueError(f"case {cs.case_id}: feature names do not match the fitted model")

    def utility(self, cs: CandidateSet) -> np.ndarray:
        """Linear utilities ``beta^T phi`` (standardised scale) for all sizes, ``-inf`` for infeasible sizes."""
        self._check(cs)
        return np.where(cs.feasible, self._transform(cs.X) @ self.beta_, -np.inf)

    def predict_proba(self, cs: CandidateSet) -> np.ndarray:
        """Probabilities over all sizes of ``cs`` (0 outside the feasible set; sums to 1)."""
        self._check(cs)
        return _softmax_masked(self._transform(cs.X) @ self.beta_, cs.feasible)

    def predict_proba_many(self, sets: Sequence[CandidateSet]) -> List[np.ndarray]:
        return [self.predict_proba(cs) for cs in sets]

    def rank(self, cs: CandidateSet, k: int = 3) -> List[Tuple[DeviceSize, float]]:
        """The ``k`` most probable feasible sizes (ties in catalogue order)."""
        p = self.predict_proba(cs)
        order = [i for i in np.argsort(-p, kind="stable") if cs.feasible[i]][:k]
        return [(cs.sizes[i], float(p[i])) for i in order]

    def log_likelihood(self, sets: Sequence[CandidateSet], chosen: ChosenLike = None) -> float:
        """Unpenalised log-likelihood, with the same handling of infeasible chosen sizes as :meth:`fit`."""
        idx = resolve_chosen(sets, chosen)
        X, mask = pack(sets)
        for cs in sets:
            self._check(cs)
        mask, _ = self._training_mask(sets, mask, idx, warn=False)
        Xs = np.where(mask[..., None], self._transform(X), 0.0)
        f, _ = penalised_nll(self.beta_, Xs, mask, idx, 0.0)
        return -f

    # ---------------------------------------------------------------- coefficients
    @property
    def beta_raw_(self) -> np.ndarray:
        """Coefficients on the raw feature scale (the mean shift cancels within a case)."""
        return self.beta_ / self.scale_

    @property
    def coef_(self) -> Dict[str, list]:
        """Coefficient table as columns: ``name``, ``standardized``, ``raw``, ``mean``, ``scale``."""
        if self.beta_ is None:
            raise RuntimeError("model is not fitted")
        return {"name": list(self.names_), "standardized": self.beta_.tolist(), "raw": self.beta_raw_.tolist(),
                "mean": self.mean_.tolist(), "scale": self.scale_.tolist()}

    # ---------------------------------------------------------------- IO
    def to_dict(self) -> Dict[str, Any]:
        if self.beta_ is None:
            raise RuntimeError("model is not fitted")
        return {"model": "ConditionalLogit", "l2": self.l2, "standardize": self.standardize, "max_iter": self.max_iter,
                "tol": self.tol, "infeasible_chosen": self.infeasible_chosen, "names": list(self.names_),
                "beta": self.beta_.tolist(), "mean": self.mean_.tolist(), "scale": self.scale_.tolist(),
                "n_cases": self.n_cases_, "n_iter": self.n_iter_, "converged": self.converged_,
                "n_added_chosen": self.n_added_chosen_}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConditionalLogit":
        m = cls(l2=d["l2"], standardize=d["standardize"], max_iter=d["max_iter"], tol=d["tol"],
                infeasible_chosen=d.get("infeasible_chosen", "add"))
        m.names_ = list(d["names"])
        m.beta_ = np.asarray(d["beta"], float)
        m.mean_ = np.asarray(d["mean"], float)
        m.scale_ = np.asarray(d["scale"], float)
        m.n_cases_ = int(d.get("n_cases", 0))
        m.n_iter_ = int(d.get("n_iter", 0))
        m.converged_ = bool(d.get("converged", True))
        m.n_added_chosen_ = int(d.get("n_added_chosen", 0))
        return m

    def save(self, path: Union[str, Path]) -> Path:
        return write_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ConditionalLogit":
        return cls.from_dict(read_json(path))
