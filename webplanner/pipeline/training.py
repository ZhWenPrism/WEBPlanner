"""Development of the planner on labelled cases: anatomy, deployment and candidate sets per case, the image score
with out-of-fold predictions, inner cross-validation of the ranking penalty, the review threshold and the locked
model. Leave-one-centre-out refits use the same functions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..device.catalogue import Catalogue
from ..device.deployment import VirtualDeployer, deploy_catalogue
from ..device.sizing_chart import ChartDecision, chart_from_morphometry
from ..ranking import (CVResult, ConditionalLogit, FeatureBuilder, ReviewFlag, SacImageScorer, leave_one_centre_out,
                       nested_cv, ranking_metrics, select_l2, width_step_target)
from ..ranking.features import CandidateSet
from ..types import AnatomyResult, CaseRecord, DeploymentResult, Volume
from ..utils.repro import get_logger

log = get_logger("webplanner.training")

LabelSource = Callable[[CaseRecord], Volume]
ImageSource = Callable[[CaseRecord], Optional[Volume]]


@dataclass
class CaseArtifacts:
    record: CaseRecord
    anatomy: AnatomyResult
    deployments: List[DeploymentResult]
    chart: ChartDecision
    descriptors: Optional[np.ndarray] = None
    image_score: float = 0.0
    candidates: Optional[CandidateSet] = None


@dataclass
class ProcessingLog:
    n_in: int = 0
    n_out: int = 0
    failures: Dict[str, str] = field(default_factory=dict)


def analyse_case(record: CaseRecord, labels: Volume, catalogue: Catalogue, deployer: VirtualDeployer,
                 thresholds=None) -> CaseArtifacts:
    """Geometry, planning factors, virtual deployment of every size and the chart size for one case."""
    from ..geometry import analyze

    anatomy = analyze(labels, ruptured=record.ruptured, tortuosity=record.access_tortuosity, thresholds=thresholds)
    deps = deploy_catalogue(labels, anatomy.frame, catalogue, deployer)
    return CaseArtifacts(record, anatomy, deps, chart_from_morphometry(anatomy.morphometry, catalogue))


def analyse_cases(records: Sequence[CaseRecord], labels: LabelSource, catalogue: Catalogue, deployer: VirtualDeployer,
                  thresholds=None, images: Optional[ImageSource] = None, scorer: Optional[SacImageScorer] = None
                  ) -> Tuple[List[CaseArtifacts], ProcessingLog]:
    """Run :func:`analyse_case` on every record. Failures are logged and reported, never dropped silently."""
    out, plog = [], ProcessingLog(n_in=len(records))
    for rec in records:
        try:
            lab = labels(rec)
            art = analyse_case(rec, lab, catalogue, deployer, thresholds)
            if scorer is not None and images is not None:
                img = images(rec)
                if img is not None:
                    art.descriptors = scorer.extract(img, lab)
            out.append(art)
        except Exception as exc:                       # keep going, but report every excluded case
            plog.failures[rec.case_id] = f"{type(exc).__name__}: {exc}"
            log.warning("case %s excluded: %s", rec.case_id, exc)
    plog.n_out = len(out)
    log.info("analysed %d of %d cases (%d excluded)", plog.n_out, plog.n_in, plog.n_in - plog.n_out)
    return out, plog


def fit_image_scores(arts: Sequence[CaseArtifacts], catalogue: Catalogue, scorer: SacImageScorer, n_splits: int = 5,
                     seed: int = 0) -> Optional[SacImageScorer]:
    """Out-of-fold sac image scores for the training cases (no case sees a score fitted on itself), then a final
    scorer fitted on all of them for new cases. Returns None when descriptors are unavailable."""
    if not arts or any(a.descriptors is None for a in arts):
        return None
    F = np.stack([a.descriptors for a in arts])
    y = np.array([width_step_target(a.chart.size, a.record.implanted, catalogue) for a in arts], float)
    groups = [a.record.patient_id for a in arts]
    oof = scorer.fit_predict_oof(F, y, groups, n_splits=min(n_splits, len(set(groups))), seed=seed)
    for a, s in zip(arts, oof):
        a.image_score = float(s)
    return scorer.fit(F, y)


def build_candidates(arts: Sequence[CaseArtifacts], builder: FeatureBuilder) -> List[CandidateSet]:
    sets = []
    for a in arts:
        a.candidates = builder.build(a.record, a.anatomy, a.deployments, a.chart.size, a.image_score)
        sets.append(a.candidates)
    return sets


@dataclass
class RankerTraining:
    model: ConditionalLogit
    review: ReviewFlag
    l2: float
    cv_table: List[Dict[str, float]]
    oof: CVResult
    oof_metrics: Dict[str, float]


def train_ranker(sets: Sequence[CandidateSet], grid: Sequence[float], n_splits: int = 5, seed: int = 0,
                 metric: str = "log_loss", review_quantile: float = 0.2) -> RankerTraining:
    """Penalty by patient-grouped inner CV; tau from nested out-of-fold top-1 probabilities; final refit on all."""
    sets = [s for s in sets if s.chosen_index is not None]
    if len(sets) < 2 * n_splits:
        raise ValueError(f"too few labelled cases ({len(sets)}) for {n_splits}-fold cross-validation")
    groups = [s.patient_id for s in sets]
    chosen = np.array([s.chosen_index for s in sets])
    k = min(n_splits, len(set(groups)))
    best, table = select_l2(sets, chosen, groups, grid, k, seed, metric)
    oof = nested_cv(sets, chosen, groups, grid, n_outer=k, n_inner=k, seed=seed, metric=metric)
    review = ReviewFlag.fit(oof.top1_probs(), review_quantile)
    model = ConditionalLogit(l2=best).fit(list(sets), chosen)
    m = oof.metrics(chosen)
    log.info("ranker: l2=%.3g, nested CV top-1 %.3f, top-3 %.3f, tau=%.3f", best, m["top1"], m["top3"], review.tau)
    return RankerTraining(model, review, best, table, oof, m)


def run_loco(sets: Sequence[CandidateSet], grid: Sequence[float], n_inner: int = 5, seed: int = 0,
             metric: str = "log_loss") -> Tuple[CVResult, Dict[str, float]]:
    """Leave-one-centre-out predictions for every labelled case (penalty chosen on the other centres)."""
    sets = [s for s in sets if s.chosen_index is not None]
    chosen = np.array([s.chosen_index for s in sets])
    res = leave_one_centre_out(sets, chosen, [s.centre for s in sets], [s.patient_id for s in sets], grid, n_inner, seed, metric)
    return res, ranking_metrics(res.proba, chosen)
