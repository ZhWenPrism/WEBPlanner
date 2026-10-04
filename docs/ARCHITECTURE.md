# Architecture and module interfaces

WEBPlanner turns a pre-treatment 3D rotational angiogram (3D-RA) into a ranked list of Woven EndoBridge (WEB) sizes.
The pipeline has four components (Anatomy, Morphometry and planning factors, Virtual deployment, Size ranking) and
an evaluation layer. This page fixes the interfaces between packages.

```text
3D-RA volume ──► data.preprocessing ──► segmentation (nnU-Net / residual U-Net, 5-fold ensemble)
                                              │  labels: 1 sac · 2 parent · 3 branch · 4 neck landing zone
                                              ▼
                                   geometry.analyze()  ──► NeckFrame · Morphometry · PlanningFactors
                                              │
                                              ▼
                          device.deployment.deploy_catalogue()  ──► DeploymentResult × 29 sizes   (eq. 1)
                                              │
                                              ▼
                ranking.features ──► ranking.ConditionalLogit (feasible set C(x), eq. 2) ──► top-3 + review flag
                                              │
                                              ▼
                                   pipeline.planner.PlanReport ──► JSON / Markdown report
```

## Conventions

* Arrays are `(z, y, x)`; `spacing` is `(sz, sy, sx)` in mm. Geometry is computed in the array frame
  (mm = index × spacing).
* Label map: `0` background, `1` aneurysm sac, `2` parent artery, `3` branch arteries, `4` neck landing zone
  (`webplanner.types`). The landing zone is part of the sac (the base layer over the neck); "sac mask" means
  labels `{1, 4}`.
* Shared dataclasses live in `webplanner/types.py`: `Volume`, `DeviceSize`, `NeckFrame`, `VesselOrigin`,
  `Morphometry`, `PlanningFactors`, `AnatomyResult`, `DeploymentResult`, `CaseRecord`, `Recommendation`, `PlanReport`.
* The device catalogue and the manufacturer's sizing chart are in `webplanner/device/catalogue.py` and
  `sizing_chart.py` (29 SL sizes, widths 4 to 11 mm, heights 2 to 9 mm).
* Every random procedure takes a `seed`; split logic runs before any fitting.

## Packages

### `webplanner.data`
| Module | Public interface |
|---|---|
| `synthetic.py` | `PhantomSpec`, `make_phantom(spec) -> Phantom(image, labels, spec, truth)`, `random_spec(rng)`, `operator_choice(...)`, `synthetic_cases(n, centres, seed)` |
| `dicom.py` | `load_3dra_series(folder) -> Volume`: reads a Philips 3D-RA series, sorts slices, returns an isotropic volume |
| `preprocessing.py` | `resample_isotropic(vol, spacing)`, `crop_cube(vol, center_mm, size_mm=40)`, `normalise_intensity(vol, lo=0.5, hi=99.5)`, `Preprocessor` (fit on training folds only) |
| `splits.py` | `external_split(records, external_centre)`, `patient_kfold(records, k=5, seed)`, `leave_one_centre_out(records)`, `SplitManifest` (JSON), `assert_no_leakage(...)` |
| `transforms.py` | 3D augmentation (rotation, scaling, elastic, gamma, noise, blur, mirroring), NumPy implementations |
| `dataset.py` | `PatchDataset` (PyTorch): foreground-oversampled random patches, `collate` |
| `nnunet_export.py` | `export_nnunet_dataset(records, out_dir, dataset_id, folds)`: nnU-Net v2 raw format plus `splits_final.json` from our patient folds |

### `webplanner.segmentation`
| Module | Public interface |
|---|---|
| `networks/residual_unet.py` | `ResidualUNet3D(in_ch, n_classes, features, deep_supervision)` (nnU-Net-style residual encoder) |
| `networks/unet3d.py` | `UNet3D` (plain comparator) |
| `networks/swin.py` | `build_swin_unetr(...)` (MONAI comparator, optional dependency) |
| `networks/__init__.py` | `build_network(name, **kw)` with names `residual_unet`, `unet3d`, `swin_unetr` |
| `losses.py` | `DiceCELoss` (L = L_Dice + L_CE), `DeepSupervisionLoss` |
| `trainer.py` | `SegmentationTrainer(config).fit(train_ds, val_ds)`: SGD Nesterov, poly LR, AMP, checkpoints, validation Dice |
| `inference.py` | `SlidingWindowPredictor` (Gaussian weighting, mirroring TTA), `ensemble_predict(models, volume)` |
| `postprocess.py` | `postprocess(labels)`: largest sac component, landing zone constrained to the sac base |
| `nnunet_trainer.py` | `nnUNetTrainerWEBPlanner` (nnU-Net v2 trainer variant, imported only when nnU-Net is installed) |

### `webplanner.geometry`
| Module | Public interface |
|---|---|
| `neck.py` | `fit_neck_frame(labels: Volume) -> NeckFrame` (PCA of landing-zone voxels; normal = eigenvector of the smallest eigenvalue, oriented to the dome), `vessel_origins(labels, frame, radius_mm=1.5) -> list[VesselOrigin]` |
| `morphometry.py` | `measure(labels, frame, origins) -> Morphometry` |
| `centerline.py` | skeleton-based centrelines, terminal direction of the parent artery, take-off directions of daughter vessels |
| `planning_factors.py` | `assess(labels, frame, origins, morph, ruptured, tortuosity, thresholds) -> PlanningFactors`, `is_simple_case(...)` |
| `__init__.py` | `analyze(labels, ruptured=None, tortuosity=None, thresholds=None) -> AnatomyResult` |

### `webplanner.device`
| Module | Public interface |
|---|---|
| `deployment.py` | `VirtualDeployer(k=0.45, ...)`, `.deploy(labels, frame, size, return_mask=False) -> DeploymentResult`, `deploy_catalogue(labels, frame, catalogue, deployer) -> list[DeploymentResult]` |
| `elongation.py` | `fit_elongation(nominal_w, nominal_h, compression, observed_h) -> ElongationFit(k, ci)` (least squares through eq. 1, bootstrap CI) |
| `fluoro.py` | `render_stages(labels, frame, deployment, view_dir=None) -> dict[str, ndarray]` (expected appearance at the manufacturer's deployment stages) |

### `webplanner.ranking`
| Module | Public interface |
|---|---|
| `features.py` | `FeatureBuilder(catalogue, locations).build(record, anatomy, deployments, chart, image_score) -> CandidateSet(X, sizes, feasible, names)` |
| `conditional_logit.py` | `ConditionalLogit(l2, standardize=True).fit(sets, chosen)`, `.predict_proba(set)`, `.rank(set, k=3)` |
| `model_selection.py` | `select_l2(sets, chosen, groups, grid, n_splits, seed)`, `cross_val_predict(...)`, `leave_one_centre_out(...)` |
| `review_flag.py` | `fit_tau(top1_probs, quantile=0.2)`, `flag(p_top1, tau)` |
| `image_score.py` | `SacImageScorer`: descriptor-based ridge score of the sac (default) or a small 3D CNN on the sac crop (PyTorch, optional); out-of-fold scores for training cases |

### `webplanner.evaluation`
| Module | Public interface |
|---|---|
| `segmentation_metrics.py` | `dice`, `hd95`, `per_class_metrics` |
| `agreement.py` | `icc_2_1`, `bland_altman`, `cohen_kappa`, `fleiss_kappa` |
| `proportions.py` | `wilson_ci`, `mcnemar_exact`, `paired_bootstrap_diff` |
| `discrimination.py` | `auroc_delong(y, score) -> (auc, lo, hi)`, `delong_test(y, s1, s2)` |
| `sizing.py` | `sizing_agreement(pred, implanted, catalogue)`: exact, top-3, within one size, acceptable plan, narrower/wider |
| `reader_study.py` | `ReaderStudy(reads).accuracy_gee()`, `.time_mixed_model()`, `.fleiss_by_session()`, `.paired_summary()` |

### `webplanner.pipeline` and `webplanner.cli`
`pipeline.planner.WEBPlanner` chains the components for one case and returns a `PlanReport`; `pipeline.report`
writes JSON and Markdown. `webplanner.cli` exposes the steps as sub-commands (`webplanner --help`).
