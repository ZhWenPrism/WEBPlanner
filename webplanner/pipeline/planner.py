"""End-to-end planning for one case and the locked-model directory.

    planner = WEBPlanner.from_directory("outputs/development/locked")
    report = planner.plan(record, image=volume)            # segmentation -> anatomy -> deployment -> ranking
    write_report(report, "report/")

A locked directory holds everything the external evaluation needs and nothing it may refit: the catalogue, the
deployment configuration (with the fitted elongation coefficient), the ranking coefficients with their
standardisation statistics, the review threshold, the image scorer and, optionally, segmentation checkpoints and the
preprocessor. SHA-256 checksums are written at lock time and verified at load time.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, fields
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union


from .. import __version__
from ..device.catalogue import Catalogue
from ..device.deployment import DeploymentConfig, VirtualDeployer, deploy_catalogue
from ..device.sizing_chart import chart_from_morphometry
from ..ranking import ConditionalLogit, FeatureBuilder, ReviewFlag, SacImageScorer
from ..types import CaseRecord, PlanReport, Recommendation, Volume
from ..utils.io import read_json, write_json

Segmenter = Callable[[Volume], Volume]
LOCK_FILES = ("catalogue.json", "deployment.json", "ranker.json", "review_flag.json", "features.json")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class WEBPlanner:
    """Chains segmentation, anatomy analysis, virtual deployment and size ranking for one case."""

    def __init__(self, ranker: ConditionalLogit, review: ReviewFlag, catalogue: Optional[Catalogue] = None,
                 deployer: Optional[VirtualDeployer] = None, builder: Optional[FeatureBuilder] = None,
                 image_scorer: Optional[SacImageScorer] = None, segmenter: Optional[Segmenter] = None,
                 thresholds=None, top_k: int = 3) -> None:
        self.catalogue = catalogue or Catalogue.default()
        self.ranker, self.review = ranker, review
        self.deployer = deployer or VirtualDeployer()
        self.builder = builder or FeatureBuilder(self.catalogue)
        self.image_scorer, self.segmenter = image_scorer, segmenter
        self.thresholds, self.top_k = thresholds, int(top_k)

    # ---------------------------------------------------------------- planning
    def segment(self, image: Volume) -> Volume:
        if self.segmenter is None:
            raise RuntimeError("no segmentation model attached; pass labels= or set a segmenter")
        return self.segmenter(image)

    def plan(self, record: CaseRecord, image: Optional[Volume] = None, labels: Optional[Volume] = None) -> PlanReport:
        from ..geometry import analyze

        if labels is None:
            if image is None:
                raise ValueError("an image or a label map is required")
            labels = self.segment(image)
        notes: List[str] = []
        anatomy = analyze(labels, ruptured=record.ruptured, tortuosity=record.access_tortuosity, thresholds=self.thresholds)
        deps = deploy_catalogue(labels, anatomy.frame, self.catalogue, self.deployer)
        chart = chart_from_morphometry(anatomy.morphometry, self.catalogue)
        notes += [f"sizing chart: {f}" for f in chart.flags]
        score = 0.0
        if self.image_scorer is not None and image is not None and self.image_scorer.model_ is not None:
            score = self.image_scorer.score_case(image, labels)
        cs = self.builder.build(record, anatomy, deps, chart.size, score)
        if cs.fallback:
            notes.append("no device width covers the largest neck: ranking restricted to the widest sizes")
        top = self.ranker.rank(cs, self.top_k)
        recs = [Recommendation(size, float(p), deps[self.catalogue.index(size)]) for size, p in top]
        flagged = bool(self.review(recs[0].probability))
        if flagged:
            notes.append(f"top-1 probability {recs[0].probability:.2f} below tau {self.review.tau:.2f}: expert review advised")
        if not anatomy.factors.simple_case:
            notes.append("outside the manufacturer's simple-case characteristics: " + ", ".join(anatomy.factors.simple_case_failures))
        return PlanReport(record.case_id, recs, chart.size, flagged, float(self.review.tau), anatomy, notes)

    # ---------------------------------------------------------------- locked model
    def lock(self, out_dir: Union[str, Path], extra: Optional[dict] = None) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_json(self.catalogue.to_dict(), out / "catalogue.json")
        write_json(asdict(self.deployer.config), out / "deployment.json")
        self.ranker.save(out / "ranker.json")
        self.review.save(out / "review_flag.json")
        write_json({"locations": list(self.builder.locations), "interactions": self.builder.interactions,
                    "image_score": self.builder.image_score, "names": self.builder.names, "top_k": self.top_k}, out / "features.json")
        files = list(LOCK_FILES)
        if self.image_scorer is not None and self.image_scorer.model_ is not None and self.image_scorer.backend == "descriptor":
            with open(out / "image_scorer.pkl", "wb") as fh:
                pickle.dump(self.image_scorer, fh)
            files.append("image_scorer.pkl")
        card = {"name": "WEBPlanner", "version": __version__, "files": {f: _sha256(out / f) for f in files}, **(extra or {})}
        (out / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
        return out

    @classmethod
    def from_directory(cls, path: Union[str, Path], segmenter: Optional[Segmenter] = None, verify: bool = True) -> "WEBPlanner":
        p = Path(path)
        card = read_json(p / "model_card.json")
        if verify:
            for f, digest in card["files"].items():
                if _sha256(p / f) != digest:
                    raise ValueError(f"checksum mismatch for {f}: the locked model was modified")
        cat = Catalogue.from_dict(read_json(p / "catalogue.json"))
        dcfg = read_json(p / "deployment.json")
        known = {f.name for f in fields(DeploymentConfig)}
        deployer = VirtualDeployer(DeploymentConfig(**{k: v for k, v in dcfg.items() if k in known}))
        feat = read_json(p / "features.json")
        builder = FeatureBuilder(cat, feat["locations"], feat["interactions"], feat["image_score"])
        if builder.names != feat["names"]:
            raise ValueError("feature definition differs from the locked model")
        scorer = None
        if (p / "image_scorer.pkl").exists():
            with open(p / "image_scorer.pkl", "rb") as fh:
                scorer = pickle.load(fh)
        return cls(ConditionalLogit.load(p / "ranker.json"), ReviewFlag.load(p / "review_flag.json"), cat, deployer, builder,
                   scorer, segmenter, top_k=int(feat.get("top_k", 3)))


def checkpoint_segmenter(checkpoints: Sequence[Union[str, Path]], patch_size: Sequence[int] = (128, 128, 128),
                         preprocessor=None, device: Optional[str] = None) -> Segmenter:
    """Segmenter from native-trainer checkpoints (five-fold ensemble with mirroring), with post-processing."""
    from ..segmentation.inference import predict_volume
    from ..segmentation.trainer import SegmentationTrainer

    models = [SegmentationTrainer.from_checkpoint(c).model.eval() for c in checkpoints]
    if device:
        models = [m.to(device) for m in models]

    def run(image: Volume) -> Volume:
        return predict_volume(models, image, preprocessor=preprocessor, patch_size=patch_size, postprocess=True)

    return run
