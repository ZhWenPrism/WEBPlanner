# Reproducibility

* **Seeds.** `webplanner.utils.repro.seed_everything(seed)` seeds Python, NumPy and PyTorch and enables deterministic
  cuDNN. Every random procedure (splits, bootstrap, augmentation, cross-validation) takes an explicit seed from the
  configuration.
* **Environment record.** Each command writes `environment.json` (Python, library versions, CUDA, GPUs, time) next
  to its outputs, plus the resolved configuration (`config.yaml`).
* **Split before fit.** `webplanner split` writes the manifest first; preprocessing statistics, the elongation
  coefficient, the ranking coefficients and the review threshold are estimated from training folds only.
  `assert_no_leakage` fails if a patient appears in two partitions.
* **Locked model.** The external evaluation reads a frozen directory (`model_card.json`, segmentation checkpoints,
  `elongation.json`, `ranker.json`, `review_flag.json`) whose SHA-256 checksums are written at lock time.
* **No silent filtering.** Every exclusion step logs the number of cases before and after.
* **Tests.** `pytest` runs the unit and integration tests on synthetic phantoms (geometry against analytic truth,
  deployment equations, gradient checks of the ranking likelihood, statistics against reference implementations).
* **Software used for the article.** Python 3.12.4, statsmodels 0.14.2, scikit-learn 1.4.2.
