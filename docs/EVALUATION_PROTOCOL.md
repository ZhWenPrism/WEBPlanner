# Evaluation protocol

## Partition

* Development: Centres 1 and 2. External validation: Centre 3, designated before model development; none of its
  data are used for training, tuning or threshold selection.
* Within development, five patient-level folds; all preprocessing statistics are fitted within folds.
* Leave-one-centre-out (LOCO): the models are retrained three times, each time with one centre held out.
* Clinical evaluation: all WEB-treated patients with archived imaging at one centre; planner outputs come from the
  model trained without that centre.

## Reference standards

* Sizing: width and height of the implanted WEB, from procedure records and device labels.
* Anatomy (external centre): consensus of two neurointerventionalists blinded to the AI output and the implanted
  device (four-class segmentation, neck plane, width, height, neck, entry and daughter-vessel angles).
* Deployment: deployed width and height, lateral compression and proximal-marker position on post-deployment VasoCT,
  measured blind to the virtual deployment; protrusion on post-deployment angiography.

## Comparator

The manufacturer's sizing chart applied as in routine practice (`webplanner.device.sizing_chart`), to expert
measurements and to the automated measurements. Segmentation comparators: 3D U-Net and Swin-UNETR trained on the
same folds.

## Outcomes

| Level | Outcome |
|---|---|
| Primary | exact agreement (width and height) of the top-1 recommendation with the implanted size, external centre |
| Secondary | top-3 agreement; within one size; LOCO agreement; simple vs complex aneurysms (prespecified); discrimination of correct vs incorrect recommendations by the top-1 probability |
| Technical | Dice and HD95 per class; neck-plane centre distance and orientation error; branch-incorporation sensitivity/specificity; agreement of planning factors; virtual vs observed deployment; morphometric agreement |
| Clinical evaluation | exact agreement and clinically acceptable plans (implanted size, or within one step and not narrower); planning time; NASA-TLX; confidence; inter-reader agreement; structured ratings of the planner output |

## Statistics (`webplanner.evaluation`)

Wilson 95% CIs; exact McNemar test for paired proportions; Wilcoxon signed-rank test for paired continuous or ordinal
outcomes; ICC(2,1) (two-way random effects, absolute agreement) and Bland-Altman limits; Cohen's and Fleiss' kappa;
AUROC with DeLong CIs. Reader study: GEE logistic model (exchangeable correlation within case, reader as fixed
effect), linear mixed model for log time (random intercept for case), Fleiss' kappa per session with a bootstrap
test, bootstrap CIs of paired differences resampling cases. Tests are two-sided at alpha = 0.05; secondary analyses
are not adjusted for multiplicity and are labelled exploratory where not prespecified.
