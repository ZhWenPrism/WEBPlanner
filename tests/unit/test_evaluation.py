"""Unit tests of webplanner.evaluation (synthetic data only)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from webplanner.device.catalogue import Catalogue
from webplanner.evaluation import (ReaderStudy, auroc_delong, average_surface_distance, bland_altman, cohen_kappa,
                                   compare_methods, delong_cov, delong_test, dice, dice_per_class, fleiss_kappa,
                                   fmt_prop, hd95, icc_2_1, mcnemar_exact, newcombe_paired_diff_ci,
                                   paired_bootstrap_diff, per_class_metrics, sizing_agreement, sizing_cases,
                                   subgroup_table, wilson_ci)
from webplanner.evaluation.agreement import fleiss_kappa_bootstrap, ratings_to_table
from webplanner.types import DeviceSize, Volume

SHROUT_FLEISS = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], float)


# ---------------------------------------------------------------------------------------------------- proportions
def test_wilson_known_value_and_format():
    lo, hi = wilson_ci(21, 25)
    assert round(100 * lo, 1) == 65.3 and round(100 * hi, 1) == 93.6
    assert fmt_prop(21, 25) == "21/25 (84.0%; 95% CI 65.3% to 93.6%)"
    from statsmodels.stats.proportion import proportion_confint
    for k, n in [(0, 10), (3, 17), (10, 10), (55, 120)]:
        assert np.allclose(wilson_ci(k, n), proportion_confint(k, n, method="wilson"), atol=1e-10)
    assert np.isnan(wilson_ci(0, 0)[0])
    with pytest.raises(ValueError):
        wilson_ci(5, 4)


def test_mcnemar_exact():
    p = mcnemar_exact(1, 10)
    assert p == pytest.approx(2 * stats.binom.cdf(1, 11, 0.5))
    x1 = np.array([1] * 1 + [0] * 10 + [1] * 5, bool)
    x2 = np.array([0] * 1 + [1] * 10 + [1] * 5, bool)
    assert mcnemar_exact(x1, x2) == pytest.approx(p)
    from statsmodels.stats.contingency_tables import mcnemar
    assert p == pytest.approx(mcnemar([[5, 1], [10, 3]], exact=True).pvalue)
    assert mcnemar_exact(0, 0) == 1.0


def test_newcombe_paired_interval():
    d, lo, hi = newcombe_paired_diff_ci(12, 9, 2, 1)
    assert d == pytest.approx(7 / 24)
    assert lo < d < hi and -1 <= lo and hi <= 1
    x1 = np.r_[np.ones(12), np.ones(9), np.zeros(2), np.zeros(1)].astype(bool)
    x2 = np.r_[np.ones(12), np.zeros(9), np.ones(2), np.zeros(1)].astype(bool)
    assert np.allclose(newcombe_paired_diff_ci(x1, x2), (d, lo, hi))
    rng = np.random.default_rng(3)
    probs, n, hits = np.array([0.5, 0.2, 0.1, 0.2]), 40, 0
    for _ in range(800):
        _, l_, h_ = newcombe_paired_diff_ci(*rng.multinomial(n, probs))
        hits += l_ <= 0.1 <= h_
    assert 0.92 <= hits / 800 <= 0.98


def test_paired_bootstrap_reproducible_and_grouped():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(30), 4)
    un = rng.random(120) < 0.5
    ai = rng.random(120) < 0.8
    r1 = paired_bootstrap_diff(un, ai, groups=groups, n_boot=500, seed=7)
    r2 = paired_bootstrap_diff(un, ai, groups=groups, n_boot=500, seed=7)
    assert r1.estimate == pytest.approx(ai.mean() - un.mean())
    assert (r1.ci_low, r1.ci_high, r1.p) == (r2.ci_low, r2.ci_high, r2.p)
    assert r1.n_units == 30 and r1.ci_low < r1.estimate < r1.ci_high
    assert r1.p_min == pytest.approx(2 / 501)
    r3 = paired_bootstrap_diff(un, ai, groups=groups, stat=lambda v: float(np.mean(v)), n_boot=500, seed=7)
    assert np.allclose(r3.distribution, r1.distribution)


# ---------------------------------------------------------------------------------------------------- agreement
def test_icc_reference_values():
    perfect = np.tile(np.arange(8.0)[:, None], (1, 3))
    assert icc_2_1(perfect).icc == pytest.approx(1.0)
    r = icc_2_1(SHROUT_FLEISS)
    assert r.icc == pytest.approx(0.29, abs=0.005)
    assert r.F == pytest.approx(11.03, abs=0.01) and (r.df1, r.df2) == (5, 15)
    assert r.ci_low < r.icc < r.ci_high
    # mean squares against an independent two-way ANOVA (statsmodels)
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    long = pd.DataFrame([(i, j, SHROUT_FLEISS[i, j]) for i in range(6) for j in range(4)], columns=["s", "r", "y"])
    tab = anova_lm(smf.ols("y ~ C(s) + C(r)", long).fit())
    msr, msc, mse = tab.loc["C(s)", "mean_sq"], tab.loc["C(r)", "mean_sq"], tab.loc["Residual", "mean_sq"]
    assert r.icc == pytest.approx((msr - mse) / (msr + 3 * mse + 4 * (msc - mse) / 6))
    with pytest.raises(ValueError):
        icc_2_1(np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]]))


def test_bland_altman():
    rng = np.random.default_rng(1)
    a = rng.normal(6, 1.5, 200)
    b = a - 0.3 + rng.normal(0, 0.4, 200)
    for method in ("approx", "exact"):
        r = bland_altman(a, b, method=method)
        assert r.bias == pytest.approx(np.mean(a - b))
        assert r.loa_high == pytest.approx(r.bias + stats.norm.ppf(0.975) * r.sd)
        assert r.bias_ci_low < r.bias < r.bias_ci_high
        assert r.loa_low_ci_low < r.loa_low < r.loa_low_ci_high
        assert r.loa_high_ci_low < r.loa_high < r.loa_high_ci_high
    ra, re = bland_altman(a, b), bland_altman(a, b, method="exact")
    assert re.loa_high_ci_high == pytest.approx(ra.loa_high_ci_high, abs=0.03)


def test_cohen_kappa_matches_sklearn_and_statsmodels():
    from sklearn.metrics import cohen_kappa_score
    from statsmodels.stats.inter_rater import cohens_kappa
    rng = np.random.default_rng(2)
    a = rng.integers(0, 4, 150)
    b = np.where(rng.random(150) < 0.6, a, rng.integers(0, 4, 150))
    table = np.bincount(a * 4 + b, minlength=16).reshape(4, 4)
    for w in (None, "linear", "quadratic"):
        r = cohen_kappa(a, b, weights=w)
        assert r.kappa == pytest.approx(cohen_kappa_score(a, b, weights=w))
        ref = cohens_kappa(table, wt=w) if w else cohens_kappa(table)
        assert r.se == pytest.approx(ref.std_kappa, rel=1e-8)
        assert r.ci_low < r.kappa < r.ci_high
    labels = np.array(["4x2", "5x3", "6x4", "7x5"])
    rs = cohen_kappa(labels[a], labels[b], weights="linear")
    assert rs.kappa == pytest.approx(cohen_kappa_score(a, b, weights="linear"))
    rb = cohen_kappa(a, b, ci="bootstrap", n_boot=300, seed=0)
    assert rb.ci_low < rb.kappa < rb.ci_high


def test_fleiss_kappa_matches_statsmodels():
    from statsmodels.stats.inter_rater import aggregate_raters
    from statsmodels.stats.inter_rater import fleiss_kappa as sm_fleiss
    rng = np.random.default_rng(4)
    truth = rng.integers(0, 5, 60)
    ratings = np.column_stack([np.where(rng.random(60) < 0.6, truth, rng.integers(0, 5, 60)) for _ in range(4)])
    table, _ = aggregate_raters(ratings)
    r = fleiss_kappa(table)
    assert r.kappa == pytest.approx(sm_fleiss(table, method="fleiss"))
    assert fleiss_kappa(ratings, kind="ratings").kappa == pytest.approx(r.kappa)
    assert r.ci_low < r.kappa < r.ci_high and r.p < 0.05
    k, lo, hi, boot = fleiss_kappa_bootstrap(table, n_boot=400, seed=0)
    assert k == pytest.approx(r.kappa)
    assert np.nanstd(boot) == pytest.approx(r.se, rel=0.3)   # Gwet SE close to the bootstrap SE
    t2, cats = ratings_to_table(np.array([["a", "b"], ["a", "a"]]))
    assert cats == ["a", "b"] and t2.tolist() == [[1, 1], [2, 0]]


# ---------------------------------------------------------------------------------------------------- discrimination
def _slow_delong(y, s1, s2):
    """Placement-value DeLong covariance by explicit pairwise comparison (DeLong et al., 1988)."""
    out = []
    for s in (s1, s2):
        x, z = s[y == 1], s[y == 0]
        psi = (x[:, None] > z[None, :]) + 0.5 * (x[:, None] == z[None, :])
        out.append((psi.mean(), psi.mean(axis=1), psi.mean(axis=0)))
    aucs = np.array([o[0] for o in out])
    v10 = np.vstack([o[1] for o in out])
    v01 = np.vstack([o[2] for o in out])
    return aucs, np.cov(v10) / v10.shape[1] + np.cov(v01) / v01.shape[1]


def test_auroc_delong_and_test():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(5)
    widths = []
    for n in (60, 600):
        y = (rng.random(n) < 0.4).astype(int)
        s = np.round(y + rng.normal(0, 1.0, n), 1)       # rounded: ties
        auc, lo, hi = auroc_delong(y, s)
        assert auc == pytest.approx(roc_auc_score(y, s))
        assert 0 <= lo < auc < hi <= 1
        widths.append(hi - lo)
    assert widths[1] < widths[0]
    s2 = np.round(s + rng.normal(0, 1.0, s.size), 1)
    aucs, cov = delong_cov(y, s, s2)
    ref_aucs, ref_cov = _slow_delong(y, s, s2)
    assert np.allclose(aucs, ref_aucs) and np.allclose(cov, ref_cov)
    a1, a2, z, p = delong_test(y, s, s)
    assert a1 == a2 and z == 0 and p == 1.0
    a1, a2, z, p = delong_test(y, s, s2)
    assert a1 > a2 and 0 <= p < 0.05
    with pytest.raises(ValueError):
        auroc_delong(np.ones(5), np.arange(5.0))


# ---------------------------------------------------------------------------------------------------- segmentation
def _sphere(shape=(40, 40, 40), centre=(20, 20, 20), r=9):
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return (z - centre[0]) ** 2 + (y - centre[1]) ** 2 + (x - centre[2]) ** 2 <= r ** 2


def test_dice_and_surface_distances():
    s = _sphere()
    assert dice(s, s) == 1.0
    shifted = np.roll(s, 2, axis=0)
    spacing = (0.5, 0.5, 0.5)
    assert hd95(s, shifted, spacing) == pytest.approx(2 * 0.5, abs=0.25)
    assert hd95(s, s, spacing) == 0.0
    assert 0 < average_surface_distance(s, shifted, spacing) < 2 * 0.5
    # anisotropic spacing against a brute-force KD-tree computation on the surface voxels
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree
    aniso = (1.5, 0.5, 0.7)
    surf = [np.argwhere(m & ~ndi.binary_erosion(m)) * np.asarray(aniso) for m in (s, shifted)]
    d_ab, d_ba = cKDTree(surf[1]).query(surf[0])[0], cKDTree(surf[0]).query(surf[1])[0]
    assert hd95(s, shifted, aniso) == pytest.approx(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95)))
    assert average_surface_distance(s, shifted, aniso) == pytest.approx(np.r_[d_ab, d_ba].mean())
    empty = np.zeros_like(s)
    v, flag = hd95(s, empty, spacing, return_flag=True)
    assert np.isnan(v) and flag == "ref_empty"
    assert np.isnan(dice(empty, empty))


def test_per_class_metrics():
    lab = np.zeros((40, 40, 40), int)
    s = _sphere()
    lab[s] = 1
    lab[s & (np.arange(40)[:, None, None] < 14)] = 4
    lab[2:6, 18:22, 18:22] = 2
    pred = np.roll(lab, 1, axis=2)
    pred[pred == 2] = 0
    res = per_class_metrics(Volume(pred, (0.4, 0.4, 0.4)), Volume(lab, (0.4, 0.4, 0.4)), classes=(1, 2, 3, 4, (1, 4)))
    assert res[(1, 4)]["hd95"] == pytest.approx(0.4, abs=1e-6)
    assert res[2]["flag"] == "pred_empty" and np.isnan(res[2]["hd95"]) and res[2]["dice"] == 0.0
    assert res[3]["flag"] == "both_empty" and np.isnan(res[3]["dice"])
    assert dice_per_class(pred, lab, (1,))[1] == pytest.approx(res[1]["dice"])


# ---------------------------------------------------------------------------------------------------- sizing
def test_sizing_agreement_hand_made():
    cat = Catalogue.default()
    implanted = [DeviceSize(7, 4), DeviceSize(7, 4), DeviceSize(8, 5), DeviceSize(6, 3), DeviceSize(9, 6)]
    ranked = [
        [DeviceSize(7, 4), DeviceSize(8, 4), DeviceSize(7, 3)],    # exact
        [DeviceSize(8, 4), DeviceSize(7, 4), DeviceSize(8, 5)],    # wider by one, implanted at rank 2
        [DeviceSize(7, 5), DeviceSize(8, 4), DeviceSize(9, 5)],    # narrower by one, not in top 3
        ["6x4", "6x3", "7x3"],                                     # taller by 1 mm, implanted at rank 2
        [(11, 8), (10, 7), (11, 7)],                               # wider by two steps, taller by 2 mm
    ]
    r = sizing_agreement(ranked, implanted, cat)
    assert r["n"] == 5
    assert r["exact_top1"].k == 1 and r["top3"].k == 3
    assert r["within_one_size"].k == 4
    assert r["clinically_acceptable"].k == 3      # exact, wider by one, taller by 1 mm; narrower is not acceptable
    assert (r["narrower"].k, r["wider"].k, r["taller"].k, r["shorter"].k) == (1, 2, 2, 0)
    lo, hi = wilson_ci(3, 5)
    assert (r["top3"].ci_low, r["top3"].ci_high) == pytest.approx((lo, hi))
    top1_only = sizing_agreement([lst[0] for lst in ranked], implanted, cat)
    assert "top3" not in top1_only and top1_only["exact_top1"].k == 1
    df = sizing_cases(ranked, implanted, cat, case_ids=list("abcde"))
    assert df["width_steps"].tolist() == [0, 1, -1, 0, 2]


def test_compare_methods_and_subgroups():
    ai = np.array([1, 1, 1, 1, 0, 1, 1, 0, 1, 1], bool)
    chart = np.array([1, 0, 0, 1, 0, 0, 1, 1, 0, 1], bool)
    res = compare_methods(ai, chart)
    assert res["difference"] == pytest.approx(0.3)
    assert (res["b"], res["c"]) == (4, 1)
    assert res["p_mcnemar"] == pytest.approx(mcnemar_exact(4, 1))
    assert res["ci_low"] < 0.3 < res["ci_high"]
    df = pd.DataFrame({"ai": ai, "chart": chart, "loc": ["MCA"] * 6 + ["AComA"] * 3 + [None]})
    tab = subgroup_table(df, "loc", outcomes=["ai", "chart"], paired=("ai", "chart"))
    assert set(tab["level"]) == {"overall", "MCA", "AComA", "missing"}
    row = tab[(tab["level"] == "MCA") & (tab["outcome"] == "ai")].iloc[0]
    assert (row["k"], row["n"]) == (5, 6)


# ---------------------------------------------------------------------------------------------------- reader study
def simulate_reads(n_cases: int = 30, seed: int = 0) -> pd.DataFrame:
    """4 readers x n cases x 2 sessions with a true assistance effect on accuracy and time."""
    rng = np.random.default_rng(seed)
    cat = Catalogue.default()
    sizes = cat.sizes
    readers = [("R1", "junior"), ("R2", "junior"), ("R3", "senior"), ("R4", "senior")]
    rows = []
    for c in range(n_cases):
        imp = sizes[rng.integers(len(sizes))]
        near = [s for s in sizes if cat.within_one_size(s, imp) and s != imp]
        ai_ok = rng.random() < 0.8
        ai = imp if ai_ok else near[rng.integers(len(near))]
        u_case, t_case = rng.normal(0, 0.6), rng.normal(0, 0.3)
        for rid, exp in readers:
            for session in ("unaided", "assisted"):
                lin = -0.4 + 0.5 * (exp == "senior") + 1.6 * (session == "assisted") + u_case
                ok = rng.random() < 1 / (1 + np.exp(-lin))
                chosen = imp if ok else near[rng.integers(len(near))]
                if session == "assisted" and not ai_ok and rng.random() < 0.4:
                    chosen, ok = ai, False
                minutes = float(np.exp(np.log(6) - 0.4 * (session == "assisted") + t_case + rng.normal(0, 0.25)))
                rows.append({"case_id": f"C{c:03d}", "reader_id": rid, "experience": exp, "session": session,
                             "chosen_width": chosen.width, "chosen_height": chosen.height, "correct": bool(ok),
                             "acceptable": bool(cat.clinically_acceptable(chosen, imp)), "minutes": minutes,
                             "tlx": float(rng.normal(50 - 8 * (session == "assisted"), 10)),
                             "confidence": float(rng.integers(1, 6)), "ai_top1_correct": ai_ok,
                             "ai_top1_width": ai.width, "ai_top1_height": ai.height})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def study() -> ReaderStudy:
    return ReaderStudy(simulate_reads())


def test_reader_study_models(study):
    g = study.accuracy_gee()
    assert g["odds_ratio"] > 1 and g["ci_low"] < g["odds_ratio"] < g["ci_high"]
    assert g["n_reads"] == 240 and g["n_cases"] == 30
    i = study.interaction_gee()
    assert np.isfinite(i["ratio_of_odds_ratios"]) and i["ci_low"] < i["ratio_of_odds_ratios"] < i["ci_high"]
    assert i["or_junior"] == pytest.approx(g["odds_ratio"], rel=1.0)
    t = study.time_mixed_model()
    assert t["time_ratio"] < 1 and t["ci_high"] < 1
    assert t["time_ratio"] == pytest.approx(np.exp(-0.4), abs=0.1)


def test_reader_study_agreement_and_bootstrap(study):
    f = study.fleiss_by_session()
    assert set(f) == {"unaided", "assisted"} and f["assisted"].kappa > f["unaided"].kappa
    k1 = study.kappa_difference_bootstrap(n_boot=200, seed=3)
    k2 = study.kappa_difference_bootstrap(n_boot=200, seed=3)
    assert k1 == k2 and k1["p_min"] == pytest.approx(2 / 201)
    assert k1["difference"] == pytest.approx(f["assisted"].kappa - f["unaided"].kappa)
    s1 = study.paired_summary("correct", n_boot=500, seed=1)
    s2 = study.paired_summary("correct", n_boot=500, seed=1)
    assert s1 == {k: v for k, v in s2.items()}
    assert s1["difference"] > 0 and s1["ci_low"] < s1["difference"] < s1["ci_high"] and s1["n_cases"] == 30
    c = study.corrections()
    wide = study.reads.pivot(index=["case_id", "reader_id"], columns="session", values="correct")
    assert c["corrected"] == int(((wide["unaided"] == 0) & (wide["assisted"] == 1)).sum())
    assert c["harmed"] == int(((wide["unaided"] == 1) & (wide["assisted"] == 0)).sum())
    a = study.adoption_of_incorrect_ai()
    assert a["adopted"] is not None and 0 <= a["adopted"].k <= a["n_reads"]


def test_reader_study_table_and_validation(study):
    tab = study.to_table(n_boot=300, n_boot_kappa=200, seed=0)
    assert not tab["note"].str.startswith("failed").any(), tab.loc[tab["note"].str.startswith("failed"), "note"].tolist()
    assert {"accuracy_gee", "time_mixed_model", "kappa_difference_bootstrap", "paired_summary",
            "corrections", "adoption_of_incorrect_ai"} <= set(tab["analysis"])
    reads = simulate_reads(n_cases=5)
    with pytest.raises(KeyError, match="correct"):
        ReaderStudy(reads.drop(columns="correct"))
    with pytest.raises(ValueError, match="session"):
        ReaderStudy(reads.assign(session="first"))
    with pytest.raises(ValueError, match="duplicate"):
        ReaderStudy(pd.concat([reads, reads.iloc[:1]]))
    with pytest.raises(KeyError, match="minutes"):
        ReaderStudy(reads.drop(columns="minutes")).time_mixed_model()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="excluded"):
            ReaderStudy(reads.iloc[1:]).paired_summary(n_boot=50)
