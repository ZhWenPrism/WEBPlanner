"""Unit tests for webplanner.ranking on simulated candidate sets (no volumes) and small phantoms."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pytest

from webplanner.data.synthetic import make_phantom
from webplanner.device.catalogue import Catalogue
from webplanner.device.sizing_chart import chart_size
from webplanner.ranking import (CandidateSet, ConditionalLogit, FeatureBuilder, ReviewFlag, SacImageScorer,
                                check_no_overlap, cross_val_predict, fit_tau, flag, group_kfold, leave_one_centre_out,
                                nested_cv, pack, penalised_nll, ranking_metrics, select_l2)
from webplanner.types import (AnatomyResult, CaseRecord, DeploymentResult, DeviceSize, Morphometry, NeckFrame,
                              PlanningFactors)

CAT = Catalogue.default()
BUILDER = FeatureBuilder(CAT)
CENTRES = ("Centre 1", "Centre 2", "Centre 3")


# --------------------------------------------------------------------------------------------------------------------
# simulated cases
# --------------------------------------------------------------------------------------------------------------------
def _deploy(size: DeviceSize, mw: float, h_sac: float, neck: float, rng: np.random.Generator, k: float = 0.45) -> DeploymentResult:
    W, H = size.width, size.height
    c = max(0.0, 1.0 - mw / W)
    free = H + k * W * c
    hd = min(free, h_sac)
    # noise stands in for sac irregularity so that outputs are not exact linear functions of W and H
    return DeploymentResult(size=size, compression=c, deployed_width=min(W, mw), deployed_height=hd, free_height=free,
                            dome_clearance=h_sac - hd + rng.normal(0, 0.2),
                            apposition=float(np.exp(-((W - mw) / 1.5) ** 2) + rng.normal(0, 0.03)),
                            neck_coverage=min(1.0, (W / neck) ** 2), neck_margin=(W - neck) / 2 + abs(rng.normal(0, 0.1)),
                            protrusion=max(0.0, free - h_sac) * 0.5 + abs(rng.normal(0, 0.15)), h_sac=h_sac)


def _case(i: int, n: int, rng: np.random.Generator, neck: Optional[float] = None):
    mw = rng.uniform(3.6, 9.5)
    sh = float(np.clip(rng.normal(0.9 * mw, 0.8), 2.8, 10.5))
    neck = mw / rng.uniform(1.1, 2.0) if neck is None else neck
    irregular = bool(rng.random() < 0.3)
    solidity = rng.uniform(0.75, 0.9) if irregular else rng.uniform(0.88, 0.99)
    ruptured = bool(rng.random() < 0.25)
    morph = Morphometry(mean_width=mw, max_width=1.05 * mw, width_1=1.05 * mw, width_2=0.95 * mw, height=1.05 * sh,
                        neck_width=neck, width_neck_ratio=mw / neck, volume_mm3=0.5 * mw ** 2 * sh, solidity=solidity,
                        irregular=irregular, branch_incorporated=False, sac_axis_extent=1.02 * sh, smallest_height=sh)
    fac = PlanningFactors(entry_angle_deg=rng.uniform(0, 80), daughter_angle_diff_deg=rng.uniform(0, 60),
                          solidity=solidity, ruptured=ruptured)
    patient = f"P{i // 2:04d}" if i < n // 4 else f"P{i:04d}"          # first quarter: two cases per patient
    rec = CaseRecord(case_id=f"c{i:04d}", patient_id=patient, centre=CENTRES[int(patient[1:]) % 3],   # one centre per patient
                     location=str(rng.choice(["MCA", "AComA", "basilar tip", "ICA terminus"], p=[0.45, 0.25, 0.2, 0.1])),
                     ruptured=ruptured)
    deps = [_deploy(s, mw, 1.02 * sh, neck, rng) for s in CAT.sizes]
    chart = chart_size(mw, sh, neck, CAT).size
    return rec, {"morphometry": morph, "factors": fac}, deps, chart


def _beta(names: List[str], values: Dict[str, float]) -> np.ndarray:
    b = np.zeros(len(names))
    for k, v in values.items():
        b[names.index(k)] = v
    return b


def _feasible_scale(sets: List[CandidateSet]) -> np.ndarray:
    X, mask = pack(sets)
    sd = X[mask].std(axis=0)
    return np.where(sd > 1e-12, sd, 1.0)


def _simulate(n: int, seed: int, beta_std: Optional[Dict[str, float]] = None, beta_raw: Optional[Dict[str, float]] = None,
              subset: Optional[List[str]] = None):
    """Candidate sets whose chosen size is drawn from a known conditional logit over C(x) (Gumbel-max)."""
    rng = np.random.default_rng(seed)
    cases = [_case(i, n, rng) for i in range(n)]
    sets = [BUILDER.build(r, a, d, c) for r, a, d, c in cases]
    if subset is not None:
        sets = [cs.select(subset) for cs in sets]
    names = sets[0].names
    if beta_std is not None:
        b_std = _beta(names, beta_std)
        b_raw = b_std / _feasible_scale(sets)
    else:
        b_raw = _beta(names, beta_raw or {})
        b_std = b_raw * _feasible_scale(sets)
    for cs in sets:
        u = cs.X @ b_raw + rng.gumbel(size=cs.n_sizes)
        cs.chosen_index = int(np.argmax(np.where(cs.feasible, u, -np.inf)))
    return sets, b_std


TRUE_STD = {"abs_dw_steps": -1.2, "dw_p1": 0.8, "abs_dh_mm": -1.0, "dh_m1": 0.4, "compression_sq": -0.6,
            "apposition": 0.7, "dw_x_irregular": 0.5, "protrusion": -0.4}


@pytest.fixture(scope="module")
def dev():
    sets, b = _simulate(300, seed=1, beta_std=TRUE_STD)
    return sets, b


def _groups(sets):
    return [cs.patient_id for cs in sets]


# --------------------------------------------------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------------------------------------------------
def test_feature_names_stable_and_paths_agree():
    names = BUILDER.names
    assert len(names) == len(set(names)) == 34
    assert names[:2] == ["dw_steps", "abs_dw_steps"] and names[-1] == "dw_x_image_score"
    assert "dw_x_loc_basilar_tip" in names and "dw_x_loc_MCA" not in names          # MCA is the reference location
    assert len(FeatureBuilder(CAT, interactions=False, image_score=False).names) == 18
    rng = np.random.default_rng(3)
    rec, anat, deps, chart = _case(0, 10, rng)
    frame = NeckFrame(np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0]))
    a = BUILDER.build(rec, anat, deps, chart, image_score=0.7)
    b = BUILDER.build(rec, AnatomyResult(frame, anat["morphometry"], anat["factors"]), deps, None, image_score=0.7)
    np.testing.assert_array_equal(a.X, b.X)                                              # chart recomputed when None
    assert a.sizes[a.chart_index] == chart
    dw = a.X[:, names.index("dw_steps")]
    np.testing.assert_allclose(a.X[:, names.index("dw_x_image_score")], 0.7 * dw)
    assert a.X[a.chart_index, names.index("dw_steps")] == 0 and a.X[a.chart_index, names.index("dh_mm")] == 0
    round_trip = CandidateSet.from_dict(a.to_dict())
    np.testing.assert_array_equal(round_trip.X, a.X)
    with pytest.raises(ValueError):
        BUILDER.build(rec, anat, deps[::-1], chart)


# --------------------------------------------------------------------------------------------------------------------
# (1) gradient check
# --------------------------------------------------------------------------------------------------------------------
def test_gradient_matches_finite_differences():
    sets, _ = _simulate(40, seed=2, beta_std=TRUE_STD)
    X, mask = pack(sets)
    X = X / _feasible_scale(sets)
    chosen = np.array([cs.chosen_index for cs in sets])
    rng = np.random.default_rng(0)
    beta = rng.normal(0, 0.3, X.shape[2])
    f, g = penalised_nll(beta, X, mask, chosen, l2=0.7)
    h = 1e-6
    num = np.array([(penalised_nll(beta + h * e, X, mask, chosen, 0.7)[0] - penalised_nll(beta - h * e, X, mask, chosen, 0.7)[0])
                    / (2 * h) for e in np.eye(len(beta))])
    assert np.isfinite(f)
    np.testing.assert_allclose(g, num, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------------------------------------------------
# (2) recovery of the true coefficients, (3) probabilities, (4) fallback
# --------------------------------------------------------------------------------------------------------------------
# Within a case every feature is a function of (W, H) only, so the full design has weakly identified directions
# (e.g. compression and width step both grow with W). Coefficient recovery is therefore checked on a subset of
# columns that vary independently; the full design is exercised by the other tests.
RECOVERY_SUBSET = ["abs_dw_steps", "dw_p1", "abs_dh_mm", "dh_m1", "apposition", "protrusion", "dw_x_irregular",
                   "dw_x_entry_angle", "dw_x_ruptured", "dh_x_irregular"]


def test_fit_recovers_true_beta_direction():
    true = {k: v for k, v in TRUE_STD.items() if k in RECOVERY_SUBSET}
    sets, b_true = _simulate(1000, seed=4, beta_std=true, subset=RECOVERY_SUBSET)
    m = ConditionalLogit(l2=1e-3).fit(sets)
    assert m.converged_
    cos = float(m.beta_ @ b_true / (np.linalg.norm(m.beta_) * np.linalg.norm(b_true)))
    assert cos > 0.9, cos
    assert m.coef_["name"] == sets[0].names
    np.testing.assert_allclose(np.asarray(m.coef_["raw"]), m.beta_ / m.scale_)


def test_probabilities_sum_to_one_and_zero_outside_feasible(dev):
    sets, _ = dev
    m = ConditionalLogit(l2=1.0).fit(sets)
    for cs in sets[:50]:
        p = m.predict_proba(cs)
        assert p.shape == (cs.n_sizes,)
        assert abs(p[cs.feasible].sum() - 1.0) < 1e-12 and np.all(p[~cs.feasible] == 0)
        top = m.rank(cs, k=3)
        assert len(top) == 3 and all(cs.feasible[cs.index_of(s)] for s, _ in top)
        assert top[0][1] >= top[1][1] >= top[2][1]
    assert np.isfinite(m.log_likelihood(sets))


def test_feasible_set_fallback_when_neck_exceeds_all_widths(dev):
    sets, _ = dev
    m = ConditionalLogit(l2=1.0).fit(sets)
    rec, anat, deps, _ = _case(0, 1, np.random.default_rng(5), neck=12.5)
    cs = BUILDER.build(rec, anat, deps)
    widths = np.array([s.width for s in cs.sizes])
    assert cs.fallback and np.array_equal(cs.feasible, widths == widths.max())
    narrow = BUILDER.build(rec, {**anat, "morphometry": {**anat["morphometry"].to_dict(), "neck_width": 6.5}}, deps)
    assert not narrow.fallback and np.array_equal(narrow.feasible, widths >= 6.5)
    p = m.predict_proba(cs)
    assert abs(p.sum() - 1.0) < 1e-12 and np.all(p[widths < widths.max()] == 0)


def test_infeasible_chosen_is_added_with_warning(dev):
    sets, _ = dev
    sub = [CandidateSet.from_dict(cs.to_dict()) for cs in sets[:60]]
    target = next(cs for cs in sub if not cs.feasible.all())
    target.chosen_index = int(np.where(~target.feasible)[0][0])
    with pytest.warns(UserWarning, match="outside the feasible set"):
        m = ConditionalLogit(l2=1.0).fit(sub)
    assert m.n_added_chosen_ == 1
    with pytest.raises(ValueError):
        ConditionalLogit(l2=1.0, infeasible_chosen="raise").fit(sub)


# --------------------------------------------------------------------------------------------------------------------
# (5) better than the chart on deployment-driven choices
# --------------------------------------------------------------------------------------------------------------------
def test_model_beats_chart_on_deployment_driven_choices():
    k1, k2 = 120.0, 40.0                      # preferred compression about 0.15 and H'/h_sac about 0.95
    raw = {"compression": 0.3 * k1, "compression_sq": -k1, "height_ratio": 1.9 * k2, "height_ratio_sq": -k2,
           "protrusion": -3.0, "apposition": 2.0}
    sets, _ = _simulate(900, seed=6, beta_raw=raw)
    tr, te = group_kfold(_groups(sets), n_splits=3, seed=0)[0]
    m = ConditionalLogit(l2=0.1).fit([sets[i] for i in tr])
    test = [sets[i] for i in te]
    chosen = np.array([cs.chosen_index for cs in test])
    model_top1 = ranking_metrics(m.predict_proba_many(test), chosen)["top1"]
    chart_top1 = float(np.mean([cs.chart_index == cs.chosen_index for cs in test]))
    assert model_top1 > chart_top1 + 0.1, (model_top1, chart_top1)


# --------------------------------------------------------------------------------------------------------------------
# (6) model selection, (7) leave one centre out
# --------------------------------------------------------------------------------------------------------------------
def test_select_l2_and_cross_val_predict(dev):
    sets, _ = dev
    groups = _groups(sets)
    grid = np.logspace(-3, 2, 6)
    best, table = select_l2(sets, None, groups, grid=grid, n_splits=4, seed=0)
    assert any(best == g for g in grid) and len(table) == len(grid)
    best_top1, _ = select_l2(sets, None, groups, grid=grid, n_splits=4, seed=0, metric="top1")
    assert any(best_top1 == g for g in grid)
    for tr, te in group_kfold(groups, 5, seed=3):
        assert not set(np.asarray(groups)[tr]) & set(np.asarray(groups)[te])
        assert len(np.intersect1d(tr, te)) == 0
    proba, fold = cross_val_predict(sets, None, groups, l2=best, n_splits=5, seed=0, return_folds=True)
    assert len(proba) == len(sets) and all(abs(p.sum() - 1) < 1e-12 for p in proba)
    g = np.asarray(groups)
    for pid in set(groups):                                             # each patient sits in exactly one fold
        assert len(set(fold[g == pid])) == 1
    with pytest.raises(AssertionError):
        check_no_overlap(groups, np.array([0, 1]), np.array([1, 2]))


def test_leave_one_centre_out_predicts_every_case_once(dev):
    sets, _ = dev
    centres = [cs.centre for cs in sets]
    res = leave_one_centre_out(sets, None, centres, _groups(sets), grid=[0.1, 1.0, 10.0], n_inner=3, seed=0)
    assert len(res.proba) == len(sets) and all(p is not None and abs(p.sum() - 1) < 1e-12 for p in res.proba)
    assert list(res.fold) == centres and set(res.l2) == set(centres)
    assert all(v in (0.1, 1.0, 10.0) for v in res.l2.values())


def test_nested_cv_runs(dev):
    sets, _ = dev
    res = nested_cv(sets[:150], None, _groups(sets[:150]), grid=[0.1, 10.0], n_outer=3, n_inner=3, seed=0)
    assert set(res.fold.tolist()) == {0, 1, 2} and len(res.top1_probs()) == 150


# --------------------------------------------------------------------------------------------------------------------
# (8) review flag, (9) JSON round trips
# --------------------------------------------------------------------------------------------------------------------
def test_tau_is_quantile_and_flags_about_20_percent(dev, tmp_path):
    sets, _ = dev
    proba = cross_val_predict(sets, None, _groups(sets), l2=1.0, n_splits=5, seed=0)
    top1 = np.array([p.max() for p in proba])
    tau = fit_tau(top1)
    assert tau == float(np.quantile(top1, 0.2))
    frac = float(np.mean(flag(top1, tau)))
    assert abs(frac - 0.2) <= 1.0 / len(top1) + 1e-9
    assert flag(tau - 1e-9, tau) is True and flag(tau, tau) is False
    rf = ReviewFlag.fit(top1)
    loaded = ReviewFlag.load(rf.save(tmp_path / "review_flag.json"))
    assert loaded == rf and np.array_equal(loaded(top1), flag(top1, tau))


def test_json_round_trip_gives_identical_predictions(dev, tmp_path):
    sets, _ = dev
    m = ConditionalLogit(l2=0.5).fit(sets)
    m2 = ConditionalLogit.load(m.save(tmp_path / "clogit.json"))
    for cs in sets[:40]:
        np.testing.assert_array_equal(m.predict_proba(cs), m2.predict_proba(cs))
    assert m2.coef_ == m.coef_


# --------------------------------------------------------------------------------------------------------------------
# (10) image score
# --------------------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _phantoms():
    specs = [(4.6, 4.4, 3.0), (5.2, 4.8, 3.2), (5.8, 5.2, 3.4), (6.4, 5.8, 3.6)]
    return [make_phantom(shape=(48, 48, 48), width_1=w, width_2=0.9 * w, height=h, neck_width=nk, seed=i)
            for i, (w, h, nk) in enumerate(specs)]


def test_descriptor_image_scorer_end_to_end():
    ph = _phantoms()
    scorer = SacImageScorer()
    F = scorer.extract_many([p.image for p in ph], [p.labels for p in ph])
    assert F.shape == (4, 12) and np.isfinite(F).all()
    y = np.array([0, 1, 1, 2])
    s = scorer.fit(F, y).predict(F)
    assert s.shape == (4,) and np.isfinite(s).all()
    assert np.isfinite(scorer.score_case(ph[0].image, ph[0].labels))
    oof = SacImageScorer().fit_predict_oof(F, y, groups=["a", "b", "c", "d"], n_splits=2)
    assert np.isfinite(oof).all()
    logit = SacImageScorer(target="oversized").fit(F, y).predict(F)
    assert np.isfinite(logit).all()


def test_cnn_image_scorer_runs():
    pytest.importorskip("torch")
    ph = _phantoms()
    scorer = SacImageScorer(backend="cnn", crop_shape=(16, 16, 16), crop_mm=16.0, epochs=2, batch_size=2, device="cpu")
    F = scorer.extract_many([p.image for p in ph], [p.labels for p in ph])
    assert F.shape == (4, 2, 16, 16, 16)
    s = scorer.fit(F, [0, 1, 1, 2]).predict(F)
    assert s.shape == (4,) and np.isfinite(s).all()
