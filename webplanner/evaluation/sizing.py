"""Agreement of planned WEB sizes with the implanted size (the reference standard).

Per case: exact top-1 match, implanted size among the top 3, within one size (``Catalogue.within_one_size``),
clinically acceptable plan (``Catalogue.clinically_acceptable``), and the direction of a top-1 miss (narrower / wider
in catalogue width steps, taller / shorter in height). Proportions carry Wilson intervals (Wilson, 1927); paired
method comparisons use Newcombe's interval (Newcombe, 1998, method 10) and the exact McNemar test (McNemar, 1947).
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..device.catalogue import Catalogue
from ..types import DeviceSize
from .proportions import (Proportion, ci_label, fmt_ci, fmt_p, fmt_prop, mcnemar_exact, newcombe_paired_diff_ci,
                          paired_counts, proportion)

SizeLike = Union[DeviceSize, str, Tuple[float, float], Sequence[float]]
OUTCOMES = ("exact_top1", "top3", "within_one_size", "clinically_acceptable", "exact_width", "exact_height",
            "narrower", "wider", "taller", "shorter")


def as_size(x: SizeLike) -> DeviceSize:
    """DeviceSize from a DeviceSize, a ``"WxH"`` label or a ``(width, height)`` pair."""
    if isinstance(x, DeviceSize):
        return x
    if isinstance(x, str):
        try:
            w, h = x.lower().replace(" ", "").split("x")
            return DeviceSize(float(w), float(h))
        except ValueError as exc:
            raise ValueError(f"cannot parse size label {x!r}; expected 'WxH'") from exc
    if x is None:
        raise ValueError("missing size")
    seq = list(x)
    if len(seq) != 2:
        raise ValueError(f"cannot interpret {x!r} as a (width, height) size")
    return DeviceSize(float(seq[0]), float(seq[1]))


def _is_single(x: object) -> bool:
    if isinstance(x, (DeviceSize, str)):
        return True
    try:
        seq = list(x)  # type: ignore[arg-type]
    except TypeError:
        return False
    return len(seq) == 2 and all(isinstance(v, (int, float, np.integer, np.floating)) for v in seq)


def _ranked_list(x: object) -> List[DeviceSize]:
    if _is_single(x):
        return [as_size(x)]  # type: ignore[arg-type]
    out = [as_size(v) for v in x]  # type: ignore[union-attr]
    if not out:
        raise ValueError("empty ranked list")
    return out


def _same(a: DeviceSize, b: DeviceSize) -> bool:
    return abs(a.width - b.width) < 1e-6 and abs(a.height - b.height) < 1e-6


def sizing_cases(pred: Sequence[object], implanted: Sequence[SizeLike], catalogue: Optional[Catalogue] = None,
                 case_ids: Optional[Sequence[object]] = None, k: int = 3) -> pd.DataFrame:
    """Per-case agreement table.

    ``pred`` holds one top-1 size per case or one ranked list per case (best first). Columns: ``top1, implanted,
    exact_top1, top3`` (ranked lists only), ``within_one_size, clinically_acceptable, exact_width, exact_height,
    width_steps, height_diff_mm, narrower, wider, taller, shorter``. Steps and differences are plan minus implanted.
    """
    cat = catalogue or Catalogue.default()
    if len(pred) != len(implanted):
        raise ValueError(f"pred and implanted differ in length: {len(pred)} vs {len(implanted)}")
    if case_ids is not None and len(case_ids) != len(pred):
        raise ValueError("case_ids must match pred in length")
    missing = [i for i, v in enumerate(implanted) if v is None]
    if missing:
        raise ValueError(f"implanted size missing for {len(missing)} cases (positions {missing[:10]}); "
                         "remove them explicitly before the analysis")
    lists = [_ranked_list(p) for p in pred]
    ranked = any(not _is_single(p) for p in pred)
    rows = []
    for i, (lst, imp_) in enumerate(zip(lists, implanted)):
        imp = as_size(imp_)
        top1 = lst[0]
        dw, dh = cat.steps_between(imp, top1)
        row: Dict[str, object] = {"top1": top1.label, "implanted": imp.label, "exact_top1": _same(top1, imp)}
        if ranked:
            row[f"top{k}"] = any(_same(s, imp) for s in lst[:k])
        row.update({"within_one_size": cat.within_one_size(top1, imp),
                    "clinically_acceptable": cat.clinically_acceptable(top1, imp),
                    "exact_width": dw == 0, "exact_height": abs(dh) < 1e-6, "width_steps": int(dw),
                    "height_diff_mm": float(dh), "narrower": dw < 0, "wider": dw > 0, "taller": dh > 1e-6,
                    "shorter": dh < -1e-6})
        rows.append(row)
    df = pd.DataFrame(rows)
    if case_ids is not None:
        df.insert(0, "case_id", list(case_ids))
    return df


def sizing_agreement(pred: Sequence[object], implanted: Sequence[SizeLike], catalogue: Optional[Catalogue] = None,
                     alpha: float = 0.05, k: int = 3) -> Dict[str, object]:
    """Agreement proportions with Wilson intervals.

    Returns ``{"n": n, "exact_top1": Proportion, "top3": Proportion (ranked lists only), "within_one_size",
    "clinically_acceptable", "exact_width", "exact_height", "narrower", "wider", "taller", "shorter"}``; every
    proportion has denominator ``n`` (all cases). ``top3`` is ``top{k}`` for another ``k``.
    """
    df = sizing_cases(pred, implanted, catalogue, k=k)
    out: Dict[str, object] = {"n": int(len(df))}
    for col in [c if c != "top3" else f"top{k}" for c in OUTCOMES]:
        if col in df:
            out[col] = proportion(int(df[col].sum()), int(len(df)), alpha)
    return out


def compare_methods(ai_correct: Sequence[bool], chart_correct: Sequence[bool], alpha: float = 0.05) -> Dict[str, object]:
    """Paired comparison of two methods on the same cases (difference = AI minus chart).

    Newcombe's hybrid score interval (Newcombe, 1998, method 10) and the exact McNemar p value. ``b`` counts cases
    where only the AI was correct, ``c`` cases where only the chart was correct.
    """
    a, b, c, d = paired_counts(ai_correct, chart_correct)
    n = a + b + c + d
    diff, lo, hi = newcombe_paired_diff_ci(a, b, c, d, alpha)
    return {"n": n, "ai": proportion(a + b, n, alpha), "chart": proportion(a + c, n, alpha), "difference": diff,
            "ci_low": lo, "ci_high": hi, "both_correct": a, "b": b, "c": c, "neither_correct": d,
            "p_mcnemar": mcnemar_exact(b, c), "text": f"{fmt_ci(diff, lo, hi, digits=1, pct=True, alpha=alpha)}, "
                                                       f"{fmt_p(mcnemar_exact(b, c))}"}


def _bool_column(s: pd.Series, name: str) -> pd.Series:
    v = s.astype(float)
    if not v.dropna().isin([0.0, 1.0]).all():
        raise ValueError(f"column {name!r} must be boolean or 0/1")
    return v


def subgroup_table(df: pd.DataFrame, by: Union[str, Sequence[str]], outcomes: Sequence[str] = ("exact_top1",),
                   paired: Optional[Tuple[str, str]] = None, alpha: float = 0.05, overall: bool = True) -> pd.DataFrame:
    """Proportions with Wilson intervals per subgroup, optionally with a paired comparison of two outcome columns.

    ``by`` is one or more grouping columns (missing levels are reported as ``"missing"``). ``outcomes`` are boolean
    columns. ``paired=(col_a, col_b)`` adds, per subgroup, the difference ``a - b`` with Newcombe's interval and the
    exact McNemar p value. Rows with a missing outcome are excluded from that outcome with a warning that gives the
    counts. Subgroup analyses are descriptive unless prespecified.
    """
    by_cols = [by] if isinstance(by, str) else list(by)
    cols = by_cols + list(outcomes) + (list(paired) if paired else [])
    absent = [c for c in cols if c not in df.columns]
    if absent:
        raise KeyError(f"subgroup_table: columns not found: {absent}")
    data = df.copy()
    for c in by_cols:
        data[c] = data[c].astype(object).where(data[c].notna(), "missing").astype(str)
    for c in set(outcomes) | set(paired or ()):
        data[c] = _bool_column(data[c], c)
        n_miss = int(data[c].isna().sum())
        if n_miss:
            warnings.warn(f"subgroup_table: {n_miss} of {len(data)} rows have missing {c!r} and are excluded "
                          f"from it (n = {len(data) - n_miss})", stacklevel=2)
    groups: List[Tuple[object, pd.DataFrame]] = []
    if overall:
        groups.append(("overall", data))
    groups.extend(data.groupby(by_cols if len(by_cols) > 1 else by_cols[0], sort=True, dropna=False))
    rows = []
    label = " x ".join(by_cols)
    for level, g in groups:
        lev = " / ".join(map(str, level)) if isinstance(level, tuple) else str(level)
        for c in outcomes:
            v = g[c].dropna()
            pr = proportion(int(v.sum()), int(v.size), alpha)
            rows.append({"by": label, "level": lev, "outcome": c, "k": pr.k, "n": pr.n, "estimate": pr.p,
                         "ci_low": pr.ci_low, "ci_high": pr.ci_high, "p": np.nan,
                         "text": fmt_prop(pr.k, pr.n, alpha)})
        if paired:
            pa, pb = paired
            v = g[[pa, pb]].dropna()
            if len(v):
                res = compare_methods(v[pa].astype(bool).to_numpy(), v[pb].astype(bool).to_numpy(), alpha)
                rows.append({"by": label, "level": lev, "outcome": f"{pa} - {pb}", "k": np.nan, "n": res["n"],
                             "estimate": res["difference"], "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                             "p": res["p_mcnemar"], "text": res["text"]})
    return pd.DataFrame(rows, columns=["by", "level", "outcome", "k", "n", "estimate", "ci_low", "ci_high", "p", "text"])


def agreement_table(results: Dict[str, object]) -> pd.DataFrame:
    """Tidy table of a :func:`sizing_agreement` result."""
    rows = []
    for key, v in results.items():
        if isinstance(v, Proportion):
            rows.append({"outcome": key, "k": v.k, "n": v.n, "estimate": v.p, "ci_low": v.ci_low,
                         "ci_high": v.ci_high, "text": v.fmt(), "ci": ci_label(v.alpha)})
    return pd.DataFrame(rows)
