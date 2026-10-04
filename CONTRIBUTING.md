# Contributing

Contributions should improve correctness and reproducibility without exposing clinical data.

## Suitable contributions

- Bug fixes and tests for geometry, virtual deployment, ranking and the statistics modules.
- Catalogue files for other WEB generations (with the source of every value).
- Data loaders for other vendors' 3D-RA exports.
- Documentation and examples on synthetic data.

## Research safeguards

- Do not commit patient images, identifiers, split manifests, checkpoints or derived records.
- Keep the external centre and the clinical-evaluation cohort isolated from training at patient level.
- Report segmentation, sizing and clinical-evaluation results separately, with uncertainty and the protocol used.

## Pull requests

Use a focused branch, describe the purpose, the affected module and the validation performed, and run
`pytest` before opening the request. Keep generated artifacts out of version control.
