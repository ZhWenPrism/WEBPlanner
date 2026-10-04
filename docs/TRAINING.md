# Training and evaluation workflow

All steps are sub-commands of `webplanner` (`webplanner <command> --help`). Paths and hyperparameters come from the
YAML files in `configs/`; any value can be overridden with `--set key.sub=value`.

## 0. Smoke test on synthetic phantoms

```bash
webplanner demo --out outputs/demo            # phantoms, segmentation smoke run, geometry, deployment, ranking, report
```

## 1. Cohort and splits

```bash
webplanner split --config configs/experiment/development.yaml
```

Centre 3 is removed first. The development cohort is divided into five patient-level folds before any statistic is
estimated. The leave-one-centre-out folds are written to the same manifest.

## 2. Anatomy network

nnU-Net v2 backend (default):

```bash
webplanner export-nnunet --config configs/experiment/development.yaml   # raw dataset + splits_final.json from our folds
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
# copy outputs/.../splits_final.json to nnUNet_preprocessed/Dataset501_WEBPlanner/
for f in 0 1 2 3 4; do nnUNetv2_train 501 3d_fullres $f -tr nnUNetTrainerWEBPlanner; done
```

Built-in PyTorch trainer (no nnU-Net installation):

```bash
webplanner train-seg --config configs/experiment/development.yaml --set segmentation.backend=native --fold 0
webplanner train-seg ... --set segmentation.network=unet3d      # comparator
webplanner train-seg ... --set segmentation.network=swin_unetr  # comparator (MONAI)
```

Inference with the five-fold ensemble and mirroring test-time augmentation:

```bash
webplanner predict-seg --config configs/experiment/development.yaml --checkpoints outputs/development/seg --cases external
```

## 3. Anatomy, morphometry and planning factors

```bash
webplanner analyze --labels outputs/development/pred --out outputs/development/anatomy
```

## 4. Virtual deployment

```bash
webplanner fit-elongation --vasoct data/vasoct_measurements.csv --out outputs/development/elongation.json
webplanner deploy --anatomy outputs/development/anatomy --labels outputs/development/pred --elongation outputs/development/elongation.json
```

## 5. Size ranking

```bash
webplanner train-ranker --config configs/experiment/development.yaml      # inner CV for the L2 penalty, tau, locked model
webplanner train-ranker --config configs/experiment/loco.yaml --loco      # three refits, one centre held out each time
```

## 6. Planning and evaluation

```bash
webplanner plan --image path/to/3dra.nii.gz --model outputs/development/ranker --out report/   # one case
webplanner evaluate --config configs/experiment/development.yaml       # external centre: primary and secondary outcomes
webplanner reader-study --reads data/reads.csv --cases data/cohort.csv --out outputs/clinical_evaluation
```

The locked model (segmentation checkpoints, elongation coefficient, ranking coefficients, standardisation statistics
and tau) is written once, before the external cases are processed.
