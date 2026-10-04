"""Command-line interface: ``webplanner <command> [options]``.

Every command takes ``--config`` (YAML, see ``configs/``) and ``--set key.sub=value`` overrides where relevant, and
writes the resolved configuration and an environment record next to its outputs.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .. import __version__
from ..config import Config, load_config
from ..utils.io import read_json, save_volume, write_json
from ..utils.repro import get_logger, seed_everything, write_environment

log = get_logger("webplanner")


# --------------------------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------------------------
def _cfg(a) -> Config:
    cfg = load_config(getattr(a, "config", None), getattr(a, "set", None) or [])
    seed_everything(cfg.seed)
    return cfg


def _outdir(path, cfg: Optional[Config] = None) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    write_environment(out)
    if cfg is not None:
        cfg.save(out / "config.yaml")
    return out


def _records(cfg: Config):
    from ..data.cohort import read_cohort

    recs = read_cohort(cfg.data.manifest)
    log.info("cohort: %d patients from %s", len(recs), cfg.data.manifest)
    return recs


def _manifest(cfg: Config, recs):
    from ..data.splits import SplitManifest, make_manifest

    p = Path(cfg.data.split_file)
    if p.exists():
        return SplitManifest.from_json(p)
    m = make_manifest(recs, cfg.data.external_centre, cfg.data.n_folds, cfg.data.seed)
    m.to_json(p)
    return m


def _catalogue(cfg: Config):
    from ..device.catalogue import Catalogue

    p = Path(cfg.deployment.catalogue)
    return Catalogue.from_yaml(p) if p.exists() else Catalogue.default()


def _deployer(cfg: Config, elongation: Optional[str] = None):
    from ..device.deployment import VirtualDeployer

    k = cfg.deployment.elongation_k
    if elongation:
        k = float(read_json(elongation)["k"])
    return VirtualDeployer(k=k, n_angles=cfg.deployment.n_angles, n_levels=cfg.deployment.n_levels)


def _thresholds(cfg: Config):
    from ..geometry.planning_factors import Thresholds

    return Thresholds(significant_angulation_deg=cfg.geometry.significant_angulation_deg, asymmetry_deg=cfg.geometry.asymmetry_deg,
                      irregular_solidity=cfg.geometry.irregular_solidity)


# --------------------------------------------------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------------------------------------------------
def cmd_synth(a) -> None:
    """Synthetic cohort (phantoms, labels, cohort.csv, VasoCT-like measurements) for tests and demonstrations."""
    from ..data.cohort import write_cohort
    from ..data.synthetic import make_phantom, operator_choice, synthetic_cases
    from ..device.catalogue import Catalogue

    out = _outdir(a.out)
    rng = np.random.default_rng(a.seed)
    cat = Catalogue.default()
    recs, vas = [], []
    for rec, spec in synthetic_cases(a.n, seed=a.seed, weights=[0.5, 0.3, 0.2]):
        spec.shape = tuple([a.size] * 3)
        ph = make_phantom(spec)
        t = ph.truth
        rec.implanted = operator_choice(t["mean_width"], t["height"], t["neck_width"], t["irregular"], t["entry_angle_deg"], cat, rng)
        rec.image_path = str(save_volume(ph.image, out / "images" / f"{rec.case_id}.npz"))
        rec.label_path = str(save_volume(ph.labels, out / "labels" / f"{rec.case_id}.npz"))
        recs.append(rec)
        c = max(0.0, 1 - t["mean_width"] / rec.implanted.width) + rng.normal(0, 0.02)
        vas.append({"patient_id": rec.patient_id, "nominal_width": rec.implanted.width, "nominal_height": rec.implanted.height,
                    "compression": c, "observed_height": rec.implanted.height + 0.45 * rec.implanted.width * c + rng.normal(0, 0.15)})
    write_cohort(recs, out / "cohort.csv")
    pd.DataFrame(vas).to_csv(out / "vasoct_measurements.csv", index=False)
    log.info("wrote %d synthetic cases to %s", len(recs), out)


def cmd_split(a) -> None:
    from ..data.splits import make_manifest

    cfg = _cfg(a)
    recs = _records(cfg)
    m = make_manifest(recs, cfg.data.external_centre, cfg.data.n_folds, cfg.data.seed)
    p = m.to_json(cfg.data.split_file)
    log.info("development %d, external %d (%s), %d folds, %d LOCO folds -> %s", len(m.development), len(m.external),
             m.external_centre, len(m.folds), len(m.loco), p)


def cmd_export_nnunet(a) -> None:
    from ..data.cohort import select
    from ..data.nnunet_export import export_nnunet_dataset
    from ..utils.io import load_volume

    cfg = _cfg(a)
    recs = _records(cfg)
    m = _manifest(cfg, recs)
    dev = [r for r in select(recs, m.development) if r.label_path]
    cases = [(r.case_id, load_volume(r.image_path), load_volume(r.label_path)) for r in dev]
    out = export_nnunet_dataset(cases, a.out, cfg.segmentation.nnunet_dataset_id, "WEBPlanner", folds=m.nnunet_splits())
    log.info("nnU-Net raw dataset with %d development cases -> %s", len(cases), out)
    print(f"next: nnUNetv2_plan_and_preprocess -d {cfg.segmentation.nnunet_dataset_id} --verify_dataset_integrity\n"
          f"      copy splits_final.json into nnUNet_preprocessed/Dataset{cfg.segmentation.nnunet_dataset_id:03d}_WEBPlanner/\n"
          f"      nnUNetv2_train {cfg.segmentation.nnunet_dataset_id} 3d_fullres FOLD -tr nnUNetTrainerWEBPlanner")


def cmd_train_seg(a) -> None:
    from ..data.cohort import select
    from ..data.preprocessing import Preprocessor
    from ..segmentation.trainer import SegmentationTrainer, TrainerConfig
    from ..utils.io import load_volume

    cfg = _cfg(a)
    if cfg.segmentation.backend != "native":
        print("segmentation.backend is 'nnunet': train with nnUNetv2_train (see `webplanner export-nnunet`), "
              "or rerun with --set segmentation.backend=native")
        return
    recs = _records(cfg)
    m = _manifest(cfg, recs)
    fold = m.folds[a.fold]
    out = _outdir(Path(cfg.output_dir) / "seg" / cfg.segmentation.network / fold.name, cfg)
    load = lambda ids: [(load_volume(r.image_path), load_volume(r.label_path)) for r in select(recs, ids) if r.label_path]
    train, val = load(fold.train), load(fold.val)
    log.info("%s: %d training and %d validation cases", fold.name, len(train), len(val))
    pre = Preprocessor(spacing=cfg.data.spacing_mm).fit([v for v, _ in train], [l.array for _, l in train], seed=cfg.seed)
    pre.save(out / "preprocessor.json")
    prep = lambda cases: [(pre.transform(v).array, pre.transform_labels(l).array) for v, l in cases]
    s = cfg.segmentation
    tcfg = TrainerConfig(network=s.network, deep_supervision=s.deep_supervision, epochs=s.epochs,
                         iterations_per_epoch=s.iterations_per_epoch, batch_size=s.batch_size, patch_size=tuple(s.patch_size),
                         lr=s.learning_rate, weight_decay=s.weight_decay, seed=cfg.seed, out_dir=str(out), resume=a.resume)
    SegmentationTrainer(tcfg).fit(prep(train), prep(val) if val else None)


def cmd_predict_seg(a) -> None:
    from ..data.cohort import select
    from ..data.preprocessing import Preprocessor
    from ..segmentation.inference import predict_volume
    from ..segmentation.trainer import SegmentationTrainer
    from ..utils.io import load_volume

    cfg = _cfg(a)
    recs = _records(cfg)
    m = _manifest(cfg, recs)
    ids = {"external": m.external, "development": m.development, "all": [r.case_id for r in recs]}[a.cases]
    ckpts = sorted(Path(a.checkpoints).rglob("checkpoint_best.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint_best.pt under {a.checkpoints}")
    pre = Preprocessor.load(ckpts[0].parent / "preprocessor.json")
    models = [SegmentationTrainer.from_checkpoint(c).model.eval() for c in ckpts]
    out = _outdir(a.out, cfg)
    for r in select(recs, ids):
        lab = predict_volume(models, load_volume(r.image_path), preprocessor=pre, patch_size=tuple(cfg.segmentation.patch_size),
                             postprocess=True)
        save_volume(lab, out / f"{r.case_id}.nii.gz" if a.nifti else out / f"{r.case_id}.npz")
    log.info("segmented %d cases with a %d-model ensemble -> %s", len(ids), len(models), out)


def cmd_analyze(a) -> None:
    from ..data.cohort import label_loader
    from ..geometry import analyze

    cfg = _cfg(a)
    recs = _records(cfg)
    load = label_loader(a.labels)
    out = _outdir(a.out, cfg)
    rows = []
    for r in recs:
        try:
            res = analyze(load(r), r.ruptured, r.access_tortuosity, _thresholds(cfg))
        except Exception as exc:
            log.warning("%s excluded: %s", r.case_id, exc)
            continue
        write_json({"frame": res.frame, "morphometry": res.morphometry.to_dict(), "factors": res.factors.to_dict()}, out / f"{r.case_id}.json")
        rows.append({"case_id": r.case_id, "centre": r.centre, **res.morphometry.to_dict(),
                     **{k: v for k, v in res.factors.to_dict().items() if not isinstance(v, list)}})
    pd.DataFrame(rows).to_csv(out / "anatomy.csv", index=False)
    log.info("anatomy for %d of %d cases -> %s", len(rows), len(recs), out)


def cmd_fit_elongation(a) -> None:
    from ..device.elongation import fit_elongation

    df = pd.read_csv(a.vasoct)
    n0 = len(df)
    df = df.dropna(subset=["nominal_width", "nominal_height", "compression", "observed_height"])
    log.info("VasoCT measurements: %d rows, %d complete", n0, len(df))
    fit = fit_elongation(df.nominal_width, df.nominal_height, df.compression, df.observed_height,
                         groups=df.patient_id if "patient_id" in df else None, seed=a.seed)
    write_json(fit, a.out)
    log.info("k = %.3f (95%% CI %.3f to %.3f), RMSE %.2f mm, n = %d -> %s", fit.k, fit.ci_low, fit.ci_high, fit.rmse, fit.n, a.out)


def cmd_deploy(a) -> None:
    from ..data.cohort import label_loader
    from ..device.deployment import deploy_catalogue
    from ..geometry import analyze

    cfg = _cfg(a)
    recs = _records(cfg)
    cat, dep = _catalogue(cfg), _deployer(cfg, a.elongation)
    load = label_loader(a.labels)
    out = _outdir(a.out, cfg)
    rows = []
    for r in recs:
        lab = load(r)
        res = analyze(lab, r.ruptured, r.access_tortuosity, _thresholds(cfg))
        for d in deploy_catalogue(lab, res.frame, cat, dep):
            rows.append({"case_id": r.case_id, "size": d.size.label, "width": d.size.width, "height": d.size.height,
                         "deployed_width": d.deployed_width, "deployed_height": d.deployed_height, "free_height": d.free_height,
                         "h_sac": d.h_sac, **d.features()})
    pd.DataFrame(rows).to_csv(out / "deployments.csv", index=False)
    log.info("deployed %d sizes in %d cases -> %s", len(cat), len(recs), out)


def _develop(cfg: Config, recs, ids, label_dir, elongation, out: Path):
    """Analyse development cases, fit image scores out of fold, train the ranker and return the planner pieces."""
    from ..data.cohort import image_loader, label_loader, select
    from ..pipeline import WEBPlanner, analyse_cases, build_candidates, fit_image_scores, train_ranker
    from ..ranking import FeatureBuilder, SacImageScorer

    cat, dep = _catalogue(cfg), _deployer(cfg, elongation)
    dev = [r for r in select(recs, ids) if r.treatment == "WEB" and r.implanted is not None]
    log.info("WEB cases with an implanted size: %d of %d development patients", len(dev), len(ids))
    scorer = SacImageScorer(cfg.ranking.image_score_backend, seed=cfg.seed)
    arts, plog = analyse_cases(dev, label_loader(label_dir), cat, dep, _thresholds(cfg), image_loader(), scorer)
    write_json(plog.failures, out / "excluded_cases.json")
    scorer = fit_image_scores(arts, cat, scorer, cfg.ranking.n_splits, cfg.seed)
    builder = FeatureBuilder(cat, image_score=scorer is not None)
    sets = build_candidates(arts, builder)
    tr = train_ranker(sets, cfg.ranking.l2_grid, cfg.ranking.n_splits, cfg.seed, cfg.ranking.selection_metric,
                      cfg.ranking.review_quantile)
    pd.DataFrame(tr.cv_table).to_csv(out / "l2_selection.csv", index=False)
    write_json(tr.oof_metrics, out / "development_cv_metrics.json")
    planner = WEBPlanner(tr.model, tr.review, cat, dep, builder, scorer, thresholds=_thresholds(cfg), top_k=cfg.ranking.top_k)
    return planner, sets, tr


def cmd_train_ranker(a) -> None:
    from ..pipeline import build_candidates, run_loco
    from ..pipeline.training import analyse_cases
    from ..data.cohort import label_loader
    from ..ranking import FeatureBuilder

    cfg = _cfg(a)
    recs = _records(cfg)
    m = _manifest(cfg, recs)
    out = _outdir(Path(cfg.output_dir) / ("loco" if a.loco else "ranker"), cfg)
    if a.loco:
        web = [r for r in recs if r.treatment == "WEB" and r.implanted is not None]
        arts, plog = analyse_cases(web, label_loader(a.labels), _catalogue(cfg), _deployer(cfg, a.elongation), _thresholds(cfg))
        sets = build_candidates(arts, FeatureBuilder(_catalogue(cfg), image_score=False))
        res, metrics = run_loco(sets, cfg.ranking.l2_grid, cfg.ranking.n_splits, cfg.seed, cfg.ranking.selection_metric)
        rows = [{"case_id": s.case_id, "centre": s.centre, "chosen": s.sizes[s.chosen_index].label,
                 "top1": s.sizes[int(np.argmax(p))].label, "p_top1": float(p.max()),
                 "top3": ";".join(s.sizes[i].label for i in np.argsort(-p)[:3])} for s, p in zip(sets, res.proba)]
        pd.DataFrame(rows).to_csv(out / "loco_predictions.csv", index=False)
        write_json({"metrics": metrics, "l2_by_centre": res.l2, "excluded": plog.failures}, out / "loco_summary.json")
        log.info("LOCO: top-1 %.3f, top-3 %.3f over %d cases -> %s", metrics["top1"], metrics["top3"], len(sets), out)
        return
    planner, _, tr = _develop(cfg, recs, m.development, a.labels, a.elongation, out)
    locked = planner.lock(out / "locked", extra={"external_centre": m.external_centre, "l2": tr.l2, "tau": tr.review.tau,
                                                  "development_cases": len(m.development)})
    log.info("locked model -> %s (l2=%.3g, tau=%.3f)", locked, tr.l2, tr.review.tau)


def _plan_cases(planner, recs, label_dir):
    from ..data.cohort import image_loader, label_loader

    load_l, load_i = label_loader(label_dir), image_loader()
    out = []
    for r in recs:
        try:
            out.append((r, planner.plan(r, image=load_i(r), labels=load_l(r))))
        except Exception as exc:
            log.warning("%s excluded from planning: %s", r.case_id, exc)
    log.info("planned %d of %d cases", len(out), len(recs))
    return out


def cmd_plan(a) -> None:
    from ..pipeline import WEBPlanner, checkpoint_segmenter, write_report
    from ..types import CaseRecord
    from ..utils.io import load_volume

    seg = checkpoint_segmenter(sorted(Path(a.checkpoints).rglob("checkpoint_best.pt"))) if a.checkpoints else None
    planner = WEBPlanner.from_directory(a.model, segmenter=seg)
    rec = CaseRecord(a.case_id, a.case_id, "unknown", location=a.location,
                     ruptured=None if a.ruptured is None else a.ruptured == "yes", access_tortuosity=a.tortuosity)
    image = load_volume(a.image) if a.image else None
    labels = load_volume(a.labels) if a.labels else None
    rep = planner.plan(rec, image=image, labels=labels)
    out = write_report(rep, a.out)
    print(json.dumps(rep.summary()["top"], indent=2))
    log.info("report -> %s", out)


def cmd_evaluate(a) -> None:
    from ..data.cohort import select
    from ..device.catalogue import Catalogue
    from ..evaluation.discrimination import auroc_delong
    from ..evaluation.sizing import compare_methods, sizing_agreement, sizing_cases, subgroup_table
    from ..pipeline import WEBPlanner

    cfg = _cfg(a)
    recs = _records(cfg)
    m = _manifest(cfg, recs)
    planner = WEBPlanner.from_directory(a.model)
    cat: Catalogue = planner.catalogue
    ext = [r for r in select(recs, m.external) if r.treatment == "WEB" and r.implanted is not None]
    planned = _plan_cases(planner, ext, a.labels)
    out = _outdir(a.out or Path(cfg.output_dir) / "external", cfg)
    tops = [[x.size for x in rep.top] for _, rep in planned]
    imp = [r.implanted for r, _ in planned]
    charts = [rep.chart_size for _, rep in planned]
    df = sizing_cases(tops, imp, cat, case_ids=[r.case_id for r, _ in planned])
    df["chart"] = [c.label for c in charts]
    df["chart_exact"] = [abs(c.width - i.width) < 1e-6 and abs(c.height - i.height) < 1e-6 for c, i in zip(charts, imp)]
    df["p_top1"] = [rep.top[0].probability for _, rep in planned]
    df["review_flag"] = [rep.review_flag for _, rep in planned]
    df["simple_case"] = [rep.anatomy.factors.simple_case for _, rep in planned]
    df["location"] = [r.location for r, _ in planned]
    df.to_csv(out / "external_cases.csv", index=False)
    agree = sizing_agreement(tops, imp, cat)
    cmp = compare_methods(df.exact_top1.tolist(), df.chart_exact.tolist())
    res = {"n": agree["n"], "agreement": {k: v for k, v in agree.items() if k != "n"}, "ai_vs_chart": cmp}
    y = df.exact_top1.astype(int).to_numpy()
    if 0 < y.sum() < len(y):
        res["auroc_top1_probability"] = auroc_delong(y, df.p_top1.to_numpy())
    res["subgroups"] = subgroup_table(df, "simple_case", ("exact_top1", "chart_exact")).to_dict(orient="records")
    write_json(res, out / "external_results.json")
    e, c = agree["exact_top1"], cmp["chart"]
    log.info("external centre: %d cases, exact top-1 %d/%d, sizing chart %d/%d -> %s", agree["n"], e.k, e.n, c.k, c.n, out)


def cmd_reader_study(a) -> None:
    from ..data.cohort import read_cohort
    from ..device.catalogue import Catalogue
    from ..evaluation.reader_study import ReaderStudy

    cat = Catalogue.default()
    reads = pd.read_csv(a.reads)
    if "correct" not in reads or "acceptable" not in reads:
        imp = {r.case_id: r.implanted for r in read_cohort(a.cases) if r.implanted is not None}
        n0 = len(reads)
        reads = reads[reads.case_id.astype(str).isin(imp)].copy()
        log.info("reads with an implanted reference: %d of %d", len(reads), n0)
        from ..types import DeviceSize

        plan = [DeviceSize(w, h) for w, h in zip(reads.chosen_width, reads.chosen_height)]
        ref = [imp[str(c)] for c in reads.case_id]
        reads["correct"] = [abs(p.width - i.width) < 1e-6 and abs(p.height - i.height) < 1e-6 for p, i in zip(plan, ref)]
        reads["acceptable"] = [cat.clinically_acceptable(p, i) for p, i in zip(plan, ref)]
    rs = ReaderStudy(reads)
    out = _outdir(a.out)
    rs.to_table(n_boot=a.n_boot, seed=a.seed).to_csv(out / "reader_study_results.csv", index=False)
    write_json({"accuracy_gee": rs.accuracy_gee(), "interaction": rs.interaction_gee(), "time": rs.time_mixed_model(),
                "corrections": rs.corrections()}, out / "reader_study_models.json")
    log.info("reader study (%d reads, %d readers) -> %s", len(reads), len(rs.readers), out)


def cmd_demo(a) -> None:
    """Synthetic end-to-end run: cohort, splits, optional segmentation smoke training, ranker, locked model,
    external evaluation and one planning report."""
    out = Path(a.out)
    if out.exists() and a.clean:
        shutil.rmtree(out)
    data = out / "data"
    cmd_synth(argparse.Namespace(out=str(data), n=a.n, seed=a.seed, size=a.size))
    sets = [f"data.manifest={data / 'cohort.csv'}", f"data.split_file={out / 'splits' / 'manifest.json'}",
            f"output_dir={out / 'run'}", "ranking.n_splits=3", "ranking.l2_grid=[0.3, 3.0, 30.0]"]
    base = dict(config="configs/experiment/smoke_test.yaml" if Path("configs/experiment/smoke_test.yaml").exists() else None, set=sets)
    cmd_split(argparse.Namespace(**base))
    cmd_fit_elongation(argparse.Namespace(vasoct=str(data / "vasoct_measurements.csv"), out=str(out / "run" / "elongation.json"), seed=a.seed))
    import importlib.util

    if a.segmentation and importlib.util.find_spec("torch") is not None:
        cmd_train_seg(argparse.Namespace(**base, fold=0, resume=False))
    elif a.segmentation:
        log.info("PyTorch not installed: segmentation smoke run skipped (reference label maps are used)")
    elong = str(out / "run" / "elongation.json")
    cmd_train_ranker(argparse.Namespace(**base, loco=False, labels=None, elongation=elong))
    cmd_train_ranker(argparse.Namespace(**base, loco=True, labels=None, elongation=elong))
    model = out / "run" / "ranker" / "locked"
    cmd_evaluate(argparse.Namespace(**base, model=str(model), labels=None, out=str(out / "run" / "external")))
    cfg = load_config(base["config"], sets)
    from ..data.cohort import read_cohort, select
    from ..data.splits import SplitManifest

    m = SplitManifest.from_json(cfg.data.split_file)
    rec = select(read_cohort(cfg.data.manifest), m.external)[0]
    cmd_plan(argparse.Namespace(model=str(model), checkpoints=None, case_id=rec.case_id, location=rec.location,
                                ruptured="yes" if rec.ruptured else "no", tortuosity=rec.access_tortuosity,
                                image=rec.image_path, labels=rec.label_path, out=str(out / "run" / "reports")))
    log.info("demo finished: %s", out)


# --------------------------------------------------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="webplanner", description="AI planning of WEB device size from 3D rotational angiography")
    p.add_argument("--version", action="version", version=f"webplanner {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, help_, config=True):
        s = sub.add_parser(name, help=help_, description=help_)
        if config:
            s.add_argument("--config", default="configs/experiment/development.yaml")
            s.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="configuration overrides")
        s.set_defaults(func=fn)
        return s

    s = add("synth", cmd_synth, "generate a synthetic cohort of phantoms", config=False)
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=60)
    s.add_argument("--size", type=int, default=80, help="phantom grid size (voxels per side)")
    s.add_argument("--seed", type=int, default=0)

    add("split", cmd_split, "external centre, patient-level folds and leave-one-centre-out folds")
    s = add("export-nnunet", cmd_export_nnunet, "write the development cohort in nnU-Net v2 raw format with our folds")
    s.add_argument("--out", required=True)
    s = add("train-seg", cmd_train_seg, "train the anatomy network on one fold (native backend)")
    s.add_argument("--fold", type=int, default=0)
    s.add_argument("--resume", action="store_true")
    s = add("predict-seg", cmd_predict_seg, "segment cases with the fold ensemble")
    s.add_argument("--checkpoints", required=True)
    s.add_argument("--cases", choices=["external", "development", "all"], default="external")
    s.add_argument("--out", required=True)
    s.add_argument("--nifti", action="store_true")
    s = add("analyze", cmd_analyze, "neck frame, morphometry and planning factors for every case")
    s.add_argument("--labels", default=None, help="label directory (default: label paths in the cohort file)")
    s.add_argument("--out", required=True)
    s = add("fit-elongation", cmd_fit_elongation, "fit the elongation coefficient k on post-deployment VasoCT", config=False)
    s.add_argument("--vasoct", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=0)
    s = add("deploy", cmd_deploy, "virtual deployment of every catalogue size")
    s.add_argument("--labels", default=None)
    s.add_argument("--elongation", default=None)
    s.add_argument("--out", required=True)
    s = add("train-ranker", cmd_train_ranker, "develop and lock the size-ranking model (or --loco)")
    s.add_argument("--labels", default=None)
    s.add_argument("--elongation", default=None)
    s.add_argument("--loco", action="store_true")
    s = add("plan", cmd_plan, "plan one case with a locked model", config=False)
    s.add_argument("--model", required=True)
    s.add_argument("--image", default=None)
    s.add_argument("--labels", default=None)
    s.add_argument("--checkpoints", default=None, help="segmentation checkpoints (used when --labels is absent)")
    s.add_argument("--case-id", default="case")
    s.add_argument("--location", default="other")
    s.add_argument("--ruptured", choices=["yes", "no"], default=None)
    s.add_argument("--tortuosity", choices=["minimal", "moderate", "severe"], default=None)
    s.add_argument("--out", default="report")
    s = add("evaluate", cmd_evaluate, "external validation of a locked model")
    s.add_argument("--model", required=True)
    s.add_argument("--labels", default=None)
    s.add_argument("--out", default=None)
    s = add("reader-study", cmd_reader_study, "analysis of the clinical evaluation reads", config=False)
    s.add_argument("--reads", required=True)
    s.add_argument("--cases", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n-boot", type=int, default=4000)
    s.add_argument("--seed", type=int, default=0)
    s = add("demo", cmd_demo, "synthetic end-to-end run", config=False)
    s.add_argument("--out", default="outputs/demo")
    s.add_argument("--n", type=int, default=36)
    s.add_argument("--size", type=int, default=80)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--segmentation", action="store_true", help="also run a short segmentation training (needs PyTorch)")
    s.add_argument("--clean", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    a.func(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
