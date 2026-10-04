"""End-to-end development, locking and external planning on synthetic phantoms (no torch needed: the phantom
label maps stand in for the segmentation output)."""
import numpy as np
import pytest

from webplanner.data.splits import external_split
from webplanner.data.synthetic import make_phantom, operator_choice, synthetic_cases
from webplanner.device.catalogue import Catalogue
from webplanner.device.deployment import VirtualDeployer
from webplanner.evaluation.sizing import sizing_agreement
from webplanner.pipeline import WEBPlanner, analyse_cases, build_candidates, fit_image_scores, run_loco, train_ranker, write_report
from webplanner.ranking import FeatureBuilder, SacImageScorer


@pytest.fixture(scope="module")
def cohort():
    rng = np.random.default_rng(7)
    cat = Catalogue.default()
    cases, phantoms = [], {}
    for rec, spec in synthetic_cases(36, seed=11, weights=[0.5, 0.3, 0.2]):
        spec.shape = (80, 80, 80)
        ph = make_phantom(spec)
        t = ph.truth
        rec.implanted = operator_choice(t["mean_width"], t["height"], t["neck_width"], t["irregular"], t["entry_angle_deg"], cat, rng)
        phantoms[rec.case_id] = ph
        cases.append(rec)
    return cases, phantoms, cat


def test_development_lock_and_external_planning(cohort, tmp_path):
    cases, phantoms, cat = cohort
    dev, ext = external_split(cases, "Centre 3")
    deployer = VirtualDeployer(k=0.45)
    scorer = SacImageScorer("descriptor")
    arts, plog = analyse_cases(dev, lambda r: phantoms[r.case_id].labels, cat, deployer,
                               images=lambda r: phantoms[r.case_id].image, scorer=scorer)
    assert plog.n_out + len(plog.failures) == plog.n_in
    assert plog.n_out >= 0.8 * plog.n_in
    final_scorer = fit_image_scores(arts, cat, scorer, n_splits=3)
    assert final_scorer is not None and all(np.isfinite(a.image_score) for a in arts)

    builder = FeatureBuilder(cat)
    sets = build_candidates(arts, builder)
    tr = train_ranker(sets, grid=[0.3, 3.0, 30.0], n_splits=3, seed=0)
    assert 0.0 < tr.review.tau < 1.0
    assert tr.oof_metrics["top3"] >= tr.oof_metrics["top1"]

    planner = WEBPlanner(tr.model, tr.review, cat, deployer, builder, final_scorer)
    locked = planner.lock(tmp_path / "locked", extra={"development_centres": ["Centre 1", "Centre 2"]})
    loaded = WEBPlanner.from_directory(locked)

    tops, implanted = [], []
    for rec in ext:
        ph = phantoms[rec.case_id]
        rep = loaded.plan(rec, image=ph.image, labels=ph.labels)
        probs = [r.probability for r in rep.top]
        assert len(rep.top) == 3 and probs == sorted(probs, reverse=True) and sum(probs) <= 1.0 + 1e-9
        assert all(cat.contains(r.size) for r in rep.top)
        assert rep.review_flag == (probs[0] < rep.tau)
        tops.append([r.size for r in rep.top])
        implanted.append(rec.implanted)
    out = write_report(rep, tmp_path / "reports")
    assert (out / f"{rep.case_id}_plan.md").read_text(encoding="utf-8").startswith("# WEB planning report")

    agree = sizing_agreement(tops, implanted, cat)
    assert agree["n"] == len(ext)

    # tampering with the locked model is detected
    (locked / "review_flag.json").write_text('{"tau": 0.0}', encoding="utf-8")
    with pytest.raises(ValueError):
        WEBPlanner.from_directory(locked)


def test_leave_one_centre_out_predicts_every_case_once(cohort):
    cases, phantoms, cat = cohort
    arts, _ = analyse_cases(cases, lambda r: phantoms[r.case_id].labels, cat, VirtualDeployer())
    sets = build_candidates(arts, FeatureBuilder(cat, image_score=False))
    res, metrics = run_loco(sets, grid=[1.0, 10.0], n_inner=3)
    assert len(res.proba) == len(sets)
    assert set(res.l2) == {"Centre 1", "Centre 2", "Centre 3"}
    assert 0.0 <= metrics["top1"] <= metrics["top3"] <= 1.0
