# Data interface

No clinical data are distributed with this repository. This page describes the files the pipeline expects so that a
site can run it on its own archive.

## Cohort manifest (`data/cohort.csv`)

One row per patient (one treated index aneurysm per patient). Study IDs only; no names, hospital numbers or dates.

| Column | Type | Description |
|---|---|---|
| `case_id` | str | study ID of the aneurysm (e.g. `C1-0001`) |
| `patient_id` | str | study ID of the patient; the unit of every split |
| `centre` | str | `Centre 1`, `Centre 2`, `Centre 3` |
| `treatment` | str | `WEB`, `FD` or `other`; only WEB cases enter the sizing analyses, all cases train the anatomy network |
| `location` | str | `MCA`, `AComA`, `basilar tip`, `ICA terminus`, `other` |
| `ruptured` | bool | rupture status at treatment |
| `access_tortuosity` | str | `minimal`, `moderate`, `severe` (from the procedure record) |
| `implanted_width`, `implanted_height` | float | nominal size of the implanted WEB (mm), from procedure records and device labels |
| `image` | path | pre-treatment 3D-RA (`.nii.gz` or `.npz`) |
| `label` | path | expert four-class segmentation, if available (training and reference) |
| `vasoct_width`, `vasoct_height`, `vasoct_compression` | float | optional: deployed dimensions measured on post-deployment VasoCT |

## Images and labels

* 3D-RA reconstructions are read with `webplanner.data.dicom.load_3dra_series` (single-frame stacks or enhanced
  multi-frame objects) and stored as NIfTI. Arrays are handled as `(z, y, x)` with spacing in mm.
* Volumes are resampled to an isotropic grid (default 0.30 mm) and cropped to a 40 mm cube around the aneurysm.
* Label values: `0` background, `1` aneurysm sac, `2` parent artery (inflow artery), `3` branch arteries,
  `4` neck landing zone (the ostial surface between sac and arteries on which the proximal end of an intrasaccular
  device sits, labelled as the base layer of the sac).

## Reader-study table (`reads.csv`)

One row per read: `case_id`, `reader_id`, `experience` (`junior`/`senior`), `session` (`unaided`/`assisted`),
`chosen_width`, `chosen_height`, `minutes`, `tlx`, `confidence`, and for assisted reads the structured ratings
(`seg_score`, `neck_score`, `measurement_change`, `factors_score`, `deployment_plausible`, `top1_decision`,
`usefulness`, `trust`, `would_use`, `harm_if_top1`). `correct` and `acceptable` are derived against the implanted
size by `webplanner reader-study`.

## Split manifest (`data/splits/manifest.json`)

Written by `webplanner split`; lists case IDs of the development and external cohorts, the five patient-level folds
and the leave-one-centre-out folds. It is excluded from version control.

## What must never be committed

Images, labels, manifests with real IDs, model checkpoints trained on clinical data, and any derived per-patient
record. `.gitignore` blocks the common formats; check `git status` before every commit.
