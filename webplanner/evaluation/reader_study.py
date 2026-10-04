"""Multi-reader study: every reader plans every case unaided and then AI-assisted.

Analyses of the statistical analysis plan
-----------------------------------------
* Exact agreement with the implanted size: GEE logistic model, ``correct ~ assisted + reader`` (reader as a fixed
  effect), exchangeable working correlation within case, robust sandwich standard errors (Liang and Zeger, 1986;
  ``statsmodels`` GEE). The effect of interest is the assistance odds ratio.
* Reader experience x assistance: ``correct ~ reader + assisted + assisted:senior`` in the same GEE model (the
  experience main effect is a reader-level constant and is absorbed by the reader fixed effects).
* Planning time: linear mixed model of log minutes, ``log(t) ~ assisted + reader`` with a random intercept for
  case (Laird and Ware, 1982; REML); time ratio = ``exp(coef)``.
* Inter-reader agreement: Fleiss' kappa per session (Fleiss, 1971) and a bootstrap test of the difference that
  resamples cases (1000 resamples by default; the smallest attainable p value is reported).
* Paired differences (assisted minus unaided) with percentile intervals from a bootstrap that resamples cases and
  keeps all reads of a case together (4000 resamples by default; Efron and Tibshirani, 1993).
* Descriptive counts: reads corrected by assistance, correct choices changed to incorrect, and adoption of incorrect
  AI recommendations.

Research use only; the outputs need review by a qualified statistician and clinician.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .agreement import FleissResult, fleiss_components, fleiss_from_weights, fleiss_kappa
from .proportions import (_z, bootstrap_p, mcnemar_exact, paired_bootstrap_diff, proportion, resample_counts)

SESSIONS = ("unaided", "assisted")
EXPERIENCE_LEVELS = ("junior", "senior")
REQUIRED_COLUMNS = ("case_id", "reader_id", "experience", "session", "correct")
BINARY_COLUMNS = ("correct", "acceptable", "ai_top1_correct")
NUMERIC_COLUMNS = ("chosen_width", "chosen_height", "minutes", "tlx", "confidence", "ai_top1_width", "ai_top1_height")
TABLE_COLUMNS = ["analysis", "outcome", "group", "estimate_type", "estimate", "ci_low", "ci_high", "p", "n", "method",
                 "note"]


def _binary(s: pd.Series, name: str) -> pd.Series:
    if s.dtype == bool:
        return s.astype(float)
    v = pd.to_numeric(s.map(lambda x: float(x) if isinstance(x, (bool, np.bool_)) else x), errors="coerce")
    bad = (v.notna() & ~v.isin([0.0, 1.0])) | (v.isna() & s.notna())
    if bad.any():
        raise ValueError(f"column {name!r} must be boolean or 0/1, found {sorted(map(str, s[bad].unique()))[:10]}")
    return v.astype(float)


class ReaderStudy:
    """Reads of a paired multi-reader study (one row per case x reader x session).

    Required columns: ``case_id, reader_id, experience`` (``"junior"``/``"senior"``), ``session``
    (``"unaided"``/``"assisted"``), ``correct``. Optional: ``chosen_width, chosen_height, acceptable, minutes, tlx,
    confidence``, and per case ``ai_top1_correct`` (plus ``ai_top1_width, ai_top1_height`` for adoption of the AI's
    size). Each method checks the columns it needs.
    """

    def __init__(self, reads: pd.DataFrame, alpha: float = 0.05):
        if not isinstance(reads, pd.DataFrame):
            raise TypeError(f"reads must be a pandas DataFrame, got {type(reads).__name__}")
        missing = [c for c in REQUIRED_COLUMNS if c not in reads.columns]
        if missing:
            raise KeyError(f"reads is missing required columns {missing}; required: {list(REQUIRED_COLUMNS)}")
        if len(reads) == 0:
            raise ValueError("reads is empty")
        _z(alpha)
        self.alpha = alpha
        d = reads.copy().reset_index(drop=True)
        for c in ("case_id", "reader_id"):
            if d[c].isna().any():
                raise ValueError(f"{int(d[c].isna().sum())} rows have a missing {c}")
        for c, allowed in (("session", SESSIONS), ("experience", EXPERIENCE_LEVELS)):
            d[c] = d[c].astype(str).str.strip().str.lower()
            bad = sorted(set(d[c]) - set(allowed))
            if bad:
                raise ValueError(f"column {c!r} has values {bad}; allowed: {list(allowed)}")
        n_exp = d.groupby("reader_id")["experience"].nunique()
        if (n_exp > 1).any():
            raise ValueError(f"readers with more than one experience level: {list(n_exp[n_exp > 1].index)}")
        dup = d.duplicated(["case_id", "reader_id", "session"], keep=False)
        if dup.any():
            ex = d.loc[dup, ["case_id", "reader_id", "session"]].head(5).to_dict("records")
            raise ValueError(f"{int(dup.sum())} rows duplicate a (case_id, reader_id, session) combination, e.g. {ex}")
        for c in BINARY_COLUMNS:
            if c in d.columns:
                d[c] = _binary(d[c], c)
        if d["correct"].isna().any():
            raise ValueError(f"{int(d['correct'].isna().sum())} rows have a missing 'correct'; remove them explicitly")
        for c in NUMERIC_COLUMNS:
            if c in d.columns:
                v = pd.to_numeric(d[c], errors="coerce")
                if (v.isna() & d[c].notna()).any():
                    raise ValueError(f"column {c!r} must be numeric")
                d[c] = v.astype(float)
        if "ai_top1_correct" in d.columns:
            n_ai = d.groupby("case_id")["ai_top1_correct"].nunique(dropna=True)
            if (n_ai > 1).any():
                raise ValueError(f"ai_top1_correct differs within case for {list(n_ai[n_ai > 1].index)[:10]}")
        d["assisted"] = (d["session"] == "assisted").astype(int)
        d["senior"] = (d["experience"] == "senior").astype(int)
        self.reads = d

    # ------------------------------------------------------------------------------------------------ helpers
    @property
    def readers(self) -> List[object]:
        return sorted(self.reads["reader_id"].unique().tolist(), key=str)

    @property
    def n_cases(self) -> int:
        return int(self.reads["case_id"].nunique())

    def __repr__(self) -> str:
        return (f"ReaderStudy(reads={len(self.reads)}, cases={self.n_cases}, readers={len(self.readers)}, "
                f"sessions={sorted(self.reads['session'].unique())})")

    def _require(self, *cols: str) -> None:
        missing = [c for c in cols if c not in self.reads.columns]
        if missing:
            raise KeyError(f"this analysis needs columns {missing}, which are not in reads")

    def _frame(self, cols: Sequence[str], what: str) -> pd.DataFrame:
        self._require(*cols)
        d = self.reads
        keep = d[list(cols)].notna().all(axis=1)
        if not keep.all():
            warnings.warn(f"{what}: {int((~keep).sum())} of {len(d)} reads have missing {list(cols)} and are "
                          f"excluded (n = {int(keep.sum())})", stacklevel=3)
        d = d[keep].copy()
        if set(d["session"]) != set(SESSIONS):
            raise ValueError(f"{what}: both sessions {SESSIONS} are needed, found {sorted(set(d['session']))}")
        return d

    def _paired(self, outcome: str, what: str) -> pd.DataFrame:
        d = self._frame([outcome], what)
        wide = d.pivot(index=["case_id", "reader_id"], columns="session", values=outcome).reset_index()
        for s in SESSIONS:
            if s not in wide:
                wide[s] = np.nan
        ok = wide[list(SESSIONS)].notna().all(axis=1)
        if not ok.all():
            warnings.warn(f"{what}: {int((~ok).sum())} of {len(wide)} case-reader pairs lack one session and are "
                          f"excluded (n = {int(ok.sum())} pairs)", stacklevel=3)
        wide = wide[ok].copy()
        exp = self.reads.drop_duplicates("reader_id").set_index("reader_id")["experience"]
        wide["experience"] = wide["reader_id"].map(exp)
        return wide

    def _is_binary(self, outcome: str) -> bool:
        return outcome in BINARY_COLUMNS

    # ------------------------------------------------------------------------------------------------ descriptive
    def accuracy_by_session(self, outcome: str = "correct", by: Optional[str] = None) -> pd.DataFrame:
        """Proportion per session (and per ``by`` level) with Wilson intervals over reads (descriptive: reads of a
        case are correlated, so these intervals are too narrow for inference)."""
        if not self._is_binary(outcome):
            raise ValueError(f"{outcome!r} is not a binary outcome")
        keys = ["session"] + ([by] if by else [])
        d = self._frame([outcome] + ([by] if by else []), "accuracy_by_session")
        rows = []
        for key, g in d.groupby(keys, sort=False):
            key = key if isinstance(key, tuple) else (key,)
            pr = proportion(int(g[outcome].sum()), int(len(g)), self.alpha)
            rows.append({**dict(zip(keys, key)), "k": pr.k, "n": pr.n, "proportion": pr.p, "ci_low": pr.ci_low,
                         "ci_high": pr.ci_high, "text": pr.fmt()})
        out = pd.DataFrame(rows)
        out["session"] = pd.Categorical(out["session"], SESSIONS, ordered=True)
        return out.sort_values(keys).reset_index(drop=True)

    # ------------------------------------------------------------------------------------------------ GEE
    def _gee(self, formula: str, outcome: str, cov_type: str, what: str):
        import statsmodels.api as sm
        import statsmodels.formula.api as smf

        if not self._is_binary(outcome):
            raise ValueError(f"{outcome!r} is not a binary outcome")
        d = self._frame([outcome], what)
        d["y"] = d[outcome].astype(float)
        model = smf.gee(formula, groups="case_id", data=d, family=sm.families.Binomial(),
                        cov_struct=sm.cov_struct.Exchangeable())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = model.fit(cov_type=cov_type)
        return res, d, sorted({str(w.message) for w in caught})

    def _gee_common(self, res, d: pd.DataFrame, msgs: List[str], cov_type: str) -> Dict[str, object]:
        dep = getattr(res.model.cov_struct, "dep_params", np.nan)
        return {"n_reads": int(len(d)), "n_cases": int(d["case_id"].nunique()),
                "n_readers": int(d["reader_id"].nunique()), "within_case_correlation": float(np.squeeze(dep)),
                "cov_type": cov_type, "converged": bool(getattr(res, "converged", True)), "warnings": msgs,
                "model": res}

    def _linear(self, res, weights: Dict[str, float]):
        names = list(res.params.index)
        L = np.zeros(len(names))
        for k, v in weights.items():
            L[names.index(k)] = v
        est = float(L @ res.params.to_numpy())
        se = float(np.sqrt(L @ res.cov_params().to_numpy() @ L))
        z = _z(self.alpha)
        p = float(2.0 * stats.norm.sf(abs(est / se))) if se > 0 else np.nan
        return est, se, est - z * se, est + z * se, p

    def accuracy_gee(self, outcome: str = "correct", cov_type: str = "robust") -> Dict[str, object]:
        """Assistance odds ratio from ``outcome ~ assisted + C(reader_id)``, GEE logistic, exchangeable correlation
        within case (Liang and Zeger, 1986). ``cov_type``: ``"robust"`` (sandwich), ``"bias_reduced"``
        (Mancl and DeRouen, 2001; for few cases) or ``"naive"``."""
        res, d, msgs = self._gee("y ~ assisted + C(reader_id)", outcome, cov_type, "accuracy_gee")
        est, se, lo, hi, p = self._linear(res, {"assisted": 1.0})
        rates = d.groupby("session")["y"].mean()
        return {"analysis": "accuracy_gee", "outcome": outcome, "odds_ratio": float(np.exp(est)),
                "ci_low": float(np.exp(lo)), "ci_high": float(np.exp(hi)), "p": p, "coef": est, "se": se,
                "unaided_rate": float(rates.get("unaided", np.nan)), "assisted_rate": float(rates.get("assisted", np.nan)),
                **self._gee_common(res, d, msgs, cov_type)}

    def interaction_gee(self, outcome: str = "correct", cov_type: str = "robust") -> Dict[str, object]:
        """Experience x assistance: ``outcome ~ C(reader_id) + assisted + assisted:senior`` (GEE as in
        :meth:`accuracy_gee`). Returns the ratio of odds ratios (senior / junior) and the assistance odds ratio in
        each experience group."""
        levels = set(self.reads["experience"])
        if levels != set(EXPERIENCE_LEVELS):
            raise ValueError(f"interaction_gee needs junior and senior readers, found {sorted(levels)}")
        res, d, msgs = self._gee("y ~ C(reader_id) + assisted + assisted:senior", outcome, cov_type, "interaction_gee")
        b_i, se_i, lo_i, hi_i, p_i = self._linear(res, {"assisted:senior": 1.0})
        b_j, se_j, lo_j, hi_j, p_j = self._linear(res, {"assisted": 1.0})
        b_s, se_s, lo_s, hi_s, p_s = self._linear(res, {"assisted": 1.0, "assisted:senior": 1.0})
        return {"analysis": "interaction_gee", "outcome": outcome, "ratio_of_odds_ratios": float(np.exp(b_i)),
                "ci_low": float(np.exp(lo_i)), "ci_high": float(np.exp(hi_i)), "p_interaction": p_i,
                "or_junior": float(np.exp(b_j)), "or_junior_ci": (float(np.exp(lo_j)), float(np.exp(hi_j))),
                "p_junior": p_j, "or_senior": float(np.exp(b_s)),
                "or_senior_ci": (float(np.exp(lo_s)), float(np.exp(hi_s))), "p_senior": p_s,
                **self._gee_common(res, d, msgs, cov_type)}

    # ------------------------------------------------------------------------------------------------ mixed models
    def mixed_model(self, outcome: str, log: bool = False) -> Dict[str, object]:
        """Linear mixed model ``y ~ assisted + C(reader_id)`` with a random intercept for case (Laird and Ware, 1982;
        REML, ``statsmodels`` MixedLM). With ``log=True`` the outcome is log-transformed and the estimate is the
        ratio ``exp(coef)``; otherwise it is the mean difference (assisted minus unaided)."""
        import statsmodels.formula.api as smf

        d = self._frame([outcome], "mixed_model")
        y = d[outcome].astype(float)
        if log:
            if (y <= 0).any():
                raise ValueError(f"{outcome!r} must be positive for a log model ({int((y <= 0).sum())} values <= 0)")
            y = np.log(y)
        d["y"] = y
        model = smf.mixedlm("y ~ assisted + C(reader_id)", d, groups=d["case_id"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = model.fit(reml=True)
        b, se = float(res.params["assisted"]), float(res.bse["assisted"])
        lo, hi = (float(v) for v in res.conf_int(self.alpha).loc["assisted"])
        out: Dict[str, object] = {"analysis": "mixed_model", "outcome": outcome, "log": log, "coef": b, "se": se,
                                  "p": float(res.pvalues["assisted"])}
        if log:
            out.update(estimate_type="ratio", estimate=float(np.exp(b)), ci_low=float(np.exp(lo)),
                       ci_high=float(np.exp(hi)))
            gm = d.groupby("session")["y"].mean().map(np.exp)
            out.update(unaided_geometric_mean=float(gm.get("unaided", np.nan)),
                       assisted_geometric_mean=float(gm.get("assisted", np.nan)))
        else:
            out.update(estimate_type="difference", estimate=b, ci_low=lo, ci_high=hi)
        out.update(n_reads=int(len(d)), n_cases=int(d["case_id"].nunique()),
                   case_variance=float(np.asarray(res.cov_re)[0, 0]), residual_variance=float(res.scale),
                   converged=bool(res.converged), warnings=sorted({str(w.message) for w in caught}), model=res)
        return out

    def time_mixed_model(self, outcome: str = "minutes") -> Dict[str, object]:
        """Planning time ratio (assisted / unaided) from the log-time mixed model of :meth:`mixed_model`."""
        out = self.mixed_model(outcome, log=True)
        out.update(analysis="time_mixed_model", time_ratio=out["estimate"])
        return out

    # ------------------------------------------------------------------------------------------------ Fleiss' kappa
    def size_labels(self, by: str = "size") -> pd.Series:
        """Category of each read: chosen size ``"WxH"`` (``by="size"``), width or height."""
        if by == "size":
            self._require("chosen_width", "chosen_height")
            w, h = self.reads["chosen_width"], self.reads["chosen_height"]
            lab = [f"{a:g}x{b:g}" if np.isfinite(a) and np.isfinite(b) else None for a, b in zip(w, h)]
        elif by in ("width", "height"):
            col = f"chosen_{by}"
            self._require(col)
            lab = [f"{a:g}" if np.isfinite(a) else None for a in self.reads[col]]
        else:
            raise ValueError("by must be 'size', 'width' or 'height'")
        return pd.Series(lab, index=self.reads.index, name=f"chosen_{by}_label", dtype=object)

    def _count_tables(self, by: str, both: bool, what: str) -> Dict[str, pd.DataFrame]:
        d = self.reads[["case_id", "reader_id", "session"]].copy()
        d["label"] = self.size_labels(by)
        readers = self.readers
        cats = sorted(d["label"].dropna().unique().tolist())
        wides, complete = {}, {}
        for s in SESSIONS:
            w = d[d["session"] == s].pivot(index="case_id", columns="reader_id", values="label")
            w = w.reindex(columns=readers)
            wides[s] = w
            complete[s] = set(w.index[w.notna().all(axis=1)])
        all_cases = set(d["case_id"])
        out = {}
        for i, s in enumerate(SESSIONS):
            keep = complete[s] & complete[SESSIONS[0]] & complete[SESSIONS[1]] if both else complete[s]
            if len(keep) < len(all_cases) and not (both and i > 0):
                warnings.warn(f"{what}: {len(all_cases) - len(keep)} of {len(all_cases)} cases are not read by all "
                              f"{len(readers)} readers{' in both sessions' if both else f' ({s})'} and are excluded "
                              f"(n = {len(keep)} cases)", stacklevel=3)
            w = wides[s].loc[sorted(keep, key=str)]
            out[s] = pd.DataFrame({c: (w == c).sum(axis=1) for c in cats}, index=w.index)
        return out

    def fleiss_by_session(self, by: str = "size") -> Dict[str, FleissResult]:
        """Fleiss' kappa among readers in each session (Fleiss, 1971); categories are the chosen size labels."""
        tables = self._count_tables(by, both=False, what="fleiss_by_session")
        return {s: fleiss_kappa(t.to_numpy(), kind="table", alpha=self.alpha) for s, t in tables.items() if len(t)}

    def kappa_difference_bootstrap(self, by: str = "size", n_boot: int = 1000, seed: int = 0) -> Dict[str, object]:
        """Difference of Fleiss' kappa (assisted minus unaided) with a percentile interval and two-sided p value from
        a bootstrap that resamples cases (both sessions of a case together). ``p_min = 2 / (B + 1)`` is the smallest
        attainable p value; ``p_at_floor`` marks results to report as ``p < p_min``."""
        tables = self._count_tables(by, both=True, what="kappa_difference_bootstrap")
        t_un, t_as = (tables[s].to_numpy(float) for s in SESSIONS)
        n = t_un.shape[0]
        if n < 2:
            raise ValueError("need at least 2 cases read by all readers in both sessions")
        comps = [fleiss_components(t) for t in (t_un, t_as)]
        codes = np.arange(n)
        w = resample_counts(n, n_boot, seed)
        boot = [fleiss_from_weights(w, c[0], c[1], codes, n) for c in comps]
        k_un, k_as = (fleiss_kappa(t, kind="table").kappa for t in (t_un, t_as))
        diff = boot[1] - boot[0]
        ok = np.isfinite(diff)
        q = [self.alpha / 2.0, 1.0 - self.alpha / 2.0]
        lo, hi = np.quantile(diff[ok], q) if ok.any() else (np.nan, np.nan)
        p, p_min = bootstrap_p(diff)
        ci = {s: tuple(float(v) for v in np.quantile(b[np.isfinite(b)], q)) for s, b in zip(SESSIONS, boot)}
        return {"analysis": "kappa_difference_bootstrap", "categories": by, "kappa_unaided": float(k_un),
                "kappa_assisted": float(k_as), "kappa_unaided_ci": ci["unaided"], "kappa_assisted_ci": ci["assisted"],
                "difference": float(k_as - k_un), "ci_low": float(lo), "ci_high": float(hi), "p": p, "p_min": p_min,
                "p_at_floor": bool(p <= p_min + 1e-12), "n_cases": int(n), "n_boot": int(n_boot),
                "n_valid": int(ok.sum()), "seed": seed}

    # ------------------------------------------------------------------------------------------------ paired outcomes
    def paired_summary(self, outcome: str = "correct", n_boot: int = 4000, seed: int = 0,
                       experience: Optional[str] = None) -> Dict[str, object]:
        """Paired difference (assisted minus unaided) of the mean of ``outcome`` over case-reader pairs, with a
        percentile interval and p value from a bootstrap that resamples cases with all their reads."""
        wide = self._paired(outcome, "paired_summary")
        if experience is not None:
            if experience not in EXPERIENCE_LEVELS:
                raise ValueError(f"experience must be one of {EXPERIENCE_LEVELS}")
            wide = wide[wide["experience"] == experience]
        if wide.empty:
            raise ValueError("no complete case-reader pairs")
        res = paired_bootstrap_diff(wide["unaided"].to_numpy(float), wide["assisted"].to_numpy(float),
                                    groups=wide["case_id"].to_numpy(), n_boot=n_boot, seed=seed, alpha=self.alpha)
        out: Dict[str, object] = {"analysis": "paired_summary", "outcome": outcome, "experience": experience or "all",
                                  "n_pairs": int(len(wide)), "n_cases": res.n_units,
                                  "unaided": float(wide["unaided"].mean()), "assisted": float(wide["assisted"].mean()),
                                  "difference": res.estimate, "ci_low": res.ci_low, "ci_high": res.ci_high,
                                  "p_boot": res.p, "p_min": res.p_min, "n_boot": int(n_boot), "seed": seed}
        if self._is_binary(outcome):
            out["unaided_prop"] = proportion(int(wide["unaided"].sum()), int(len(wide)), self.alpha)
            out["assisted_prop"] = proportion(int(wide["assisted"].sum()), int(len(wide)), self.alpha)
        return out

    def corrections(self, outcome: str = "correct") -> Dict[str, object]:
        """Reads corrected by assistance (unaided wrong, assisted right) and correct choices changed to incorrect.
        The McNemar p value over reads ignores clustering within case and is descriptive."""
        if not self._is_binary(outcome):
            raise ValueError(f"{outcome!r} is not a binary outcome")
        wide = self._paired(outcome, "corrections")
        u, a = wide["unaided"].astype(bool), wide["assisted"].astype(bool)
        wide = wide.assign(corrected=~u & a, harmed=u & ~a)
        by_reader = (wide.groupby(["reader_id", "experience"])
                     .agg(n_pairs=("corrected", "size"), corrected=("corrected", "sum"), harmed=("harmed", "sum"))
                     .reset_index())
        corrected, harmed = int(wide["corrected"].sum()), int(wide["harmed"].sum())
        return {"analysis": "corrections", "outcome": outcome, "n_pairs": int(len(wide)), "corrected": corrected,
                "harmed": harmed, "both_correct": int((u & a).sum()), "both_incorrect": int((~u & ~a).sum()),
                "net": corrected - harmed, "p_mcnemar_reads": mcnemar_exact(harmed, corrected),
                "text": f"assistance corrected {corrected} reads and changed {harmed} correct choices to incorrect",
                "by_reader": by_reader}

    def adoption_of_incorrect_ai(self) -> Dict[str, object]:
        """Assisted reads on cases whose AI top-1 size was incorrect: how often readers chose the AI's size
        (needs ``ai_top1_width, ai_top1_height, chosen_width, chosen_height``), how often they were incorrect, and how
        many correct unaided choices became incorrect. Descriptive (reads are clustered within case)."""
        self._require("ai_top1_correct")
        d = self.reads
        ai = d.groupby("case_id")["ai_top1_correct"].first()
        bad_cases = set(ai[ai == 0].index)
        n_unknown = int(ai.isna().sum())
        sub = d[d["case_id"].isin(bad_cases)]
        A = sub[sub["session"] == "assisted"]
        out: Dict[str, object] = {"analysis": "adoption_of_incorrect_ai", "n_cases_ai_incorrect": len(bad_cases),
                                  "n_cases_ai_unknown": n_unknown, "n_reads": int(len(A)),
                                  "reader_incorrect": proportion(int((A["correct"] == 0).sum()), int(len(A)), self.alpha)}
        U = sub[sub["session"] == "unaided"].set_index(["case_id", "reader_id"])
        Ai = A.set_index(["case_id", "reader_id"])
        both = Ai.join(U, how="inner", lsuffix="_as", rsuffix="_un")
        out["correct_to_incorrect"] = int(((both["correct_un"] == 1) & (both["correct_as"] == 0)).sum())
        cols = ("ai_top1_width", "ai_top1_height", "chosen_width", "chosen_height")
        if all(c in d.columns for c in cols):
            def same(frame: pd.DataFrame, sfx: str = "") -> pd.Series:
                return ((frame[f"chosen_width{sfx}"] - frame[f"ai_top1_width{sfx}"]).abs() < 1e-6) & \
                       ((frame[f"chosen_height{sfx}"] - frame[f"ai_top1_height{sfx}"]).abs() < 1e-6)
            adopted = same(A)
            out["adopted"] = proportion(int(adopted.sum()), int(len(A)), self.alpha)
            out["switched_to_ai"] = int((same(both, "_as") & ~same(both, "_un")).sum())
            out["text"] = (f"readers adopted {int(adopted.sum())} of {len(A)} incorrect AI recommendations "
                           f"({len(bad_cases)} cases)")
        else:
            out["adopted"] = None
            out["switched_to_ai"] = None
            out["text"] = f"AI size columns {list(cols)} not available; adoption not computed"
        return out

    # ------------------------------------------------------------------------------------------------ table
    def to_table(self, n_boot: int = 4000, n_boot_kappa: int = 1000, seed: int = 0, by: str = "size") -> pd.DataFrame:
        """All analyses whose columns are present, as a tidy table (``TABLE_COLUMNS``). A failed model fit is
        reported in ``note`` rather than dropped."""
        rows: List[Dict[str, object]] = []
        cols = set(self.reads.columns)

        def add(**kw: object) -> None:
            rows.append({c: kw.get(c, np.nan if c not in ("note", "method", "group") else "") for c in TABLE_COLUMNS})

        def guarded(name: str, outcome: str, fn) -> None:
            try:
                fn()
            except Exception as exc:  # reported, not swallowed
                add(analysis=name, outcome=outcome, note=f"failed: {type(exc).__name__}: {exc}")

        binary = [c for c in ("correct", "acceptable") if c in cols]
        continuous = [c for c in ("minutes", "tlx", "confidence") if c in cols]
        for o in binary:
            def desc(o: str = o) -> None:
                for _, r in self.accuracy_by_session(o).iterrows():
                    add(analysis="descriptive", outcome=o, group=str(r["session"]), estimate_type="proportion",
                        estimate=r["proportion"], ci_low=r["ci_low"], ci_high=r["ci_high"], n=r["n"],
                        method="Wilson over reads (descriptive)")
            guarded("descriptive", o, desc)

            def gee(o: str = o) -> None:
                g = self.accuracy_gee(o)
                add(analysis="accuracy_gee", outcome=o, group="assisted vs unaided", estimate_type="odds ratio",
                    estimate=g["odds_ratio"], ci_low=g["ci_low"], ci_high=g["ci_high"], p=g["p"], n=g["n_reads"],
                    method="GEE logistic, exchangeable within case, reader fixed effect, robust SE",
                    note="; ".join(g["warnings"]))
            guarded("accuracy_gee", o, gee)
        if "correct" in cols and set(self.reads["experience"]) == set(EXPERIENCE_LEVELS):
            def inter() -> None:
                g = self.interaction_gee("correct")
                m = "GEE logistic with assisted:senior, reader fixed effect, robust SE"
                add(analysis="interaction_gee", outcome="correct", group="senior / junior",
                    estimate_type="ratio of odds ratios", estimate=g["ratio_of_odds_ratios"], ci_low=g["ci_low"],
                    ci_high=g["ci_high"], p=g["p_interaction"], n=g["n_reads"], method=m)
                for lev in EXPERIENCE_LEVELS:
                    add(analysis="interaction_gee", outcome="correct", group=lev, estimate_type="odds ratio",
                        estimate=g[f"or_{lev}"], ci_low=g[f"or_{lev}_ci"][0], ci_high=g[f"or_{lev}_ci"][1],
                        p=g[f"p_{lev}"], n=g["n_reads"], method=m)
            guarded("interaction_gee", "correct", inter)
        for o in binary + continuous:
            def paired(o: str = o) -> None:
                s = self.paired_summary(o, n_boot=n_boot, seed=seed)
                add(analysis="paired_summary", outcome=o, group="assisted - unaided", estimate_type="difference",
                    estimate=s["difference"], ci_low=s["ci_low"], ci_high=s["ci_high"], p=s["p_boot"], n=s["n_pairs"],
                    method=f"case bootstrap ({n_boot} resamples), percentile", note=f"p_min = {s['p_min']:.2g}")
            guarded("paired_summary", o, paired)
        for o in continuous:
            def lmm(o: str = o) -> None:
                m = self.time_mixed_model(o) if o == "minutes" else self.mixed_model(o, log=False)
                add(analysis="time_mixed_model" if o == "minutes" else "mixed_model", outcome=o,
                    group="assisted vs unaided", estimate_type="time ratio" if o == "minutes" else "difference",
                    estimate=m["estimate"], ci_low=m["ci_low"], ci_high=m["ci_high"], p=m["p"], n=m["n_reads"],
                    method="LMM, random intercept for case, reader fixed effect" + (", log scale" if o == "minutes" else ""),
                    note="; ".join(m["warnings"]))
            guarded("mixed_model", o, lmm)
        need = {"size": ("chosen_width", "chosen_height"), "width": ("chosen_width",), "height": ("chosen_height",)}[by]
        if all(c in cols for c in need):
            def kap() -> None:
                for s, f in self.fleiss_by_session(by).items():
                    add(analysis="fleiss_kappa", outcome=f"chosen {by}", group=s, estimate_type="kappa",
                        estimate=f.kappa, ci_low=f.ci_low, ci_high=f.ci_high, p=f.p, n=f.n_subjects,
                        method="Fleiss kappa, Wald CI (Gwet SE), z test (null SE)")
                k = self.kappa_difference_bootstrap(by, n_boot=n_boot_kappa, seed=seed)
                add(analysis="kappa_difference_bootstrap", outcome=f"chosen {by}", group="assisted - unaided",
                    estimate_type="difference", estimate=k["difference"], ci_low=k["ci_low"], ci_high=k["ci_high"],
                    p=k["p"], n=k["n_cases"], method=f"case bootstrap ({n_boot_kappa} resamples)",
                    note=f"p_min = {k['p_min']:.2g}" + ("; report as p < p_min" if k["p_at_floor"] else ""))
            guarded("fleiss_kappa", f"chosen {by}", kap)

        def corr() -> None:
            c = self.corrections("correct")
            for key in ("corrected", "harmed"):
                add(analysis="corrections", outcome="correct", group=key, estimate_type="count", estimate=c[key],
                    p=c["p_mcnemar_reads"], n=c["n_pairs"], method="paired reads; exact McNemar over reads (descriptive)")
        guarded("corrections", "correct", corr)
        if "ai_top1_correct" in cols:
            def adopt() -> None:
                a = self.adoption_of_incorrect_ai()
                pr = a["adopted"] if a["adopted"] is not None else a["reader_incorrect"]
                add(analysis="adoption_of_incorrect_ai", outcome="adopted" if a["adopted"] is not None else "incorrect",
                    group="assisted reads, AI top-1 incorrect", estimate_type="proportion", estimate=pr.p,
                    ci_low=pr.ci_low, ci_high=pr.ci_high, n=pr.n, method="Wilson over reads (descriptive)",
                    note=a["text"])
            guarded("adoption_of_incorrect_ai", "adopted", adopt)
        return pd.DataFrame(rows, columns=TABLE_COLUMNS)
