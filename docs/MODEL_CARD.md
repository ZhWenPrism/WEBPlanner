# Model card

| Field | Description |
|---|---|
| Name | WEBPlanner 0.1.0 |
| Task | Rank the sizes (width x height) of the Woven EndoBridge SL device for a bifurcation aneurysm from the pre-treatment 3D-RA |
| Intended use | Research on pre-procedural planning support for neurointerventionalists. Outputs are proposals for expert review, not instructions |
| Out of scope | Treatment selection (WEB vs other devices), device shape choice, other WEB generations without a matching catalogue file, non-saccular aneurysms, non-3D-RA imaging |
| Inputs | Isotropic 3D-RA crop (40 mm), rupture status, access tortuosity, location |
| Outputs | Four-class segmentation, morphometry, planning factors, virtual deployment of every catalogue size, top-3 sizes with probabilities, review flag |
| Components | nnU-Net 3D segmentation (Dice + cross-entropy); PCA neck plane; geometric deployment model (eq. 1); conditional logit over the feasible sizes (eq. 2) |
| Training data | Three-centre retrospective cohort (two development centres, one external centre); not distributed |
| Evaluation | See `docs/EVALUATION_PROTOCOL.md`. Results will be reported in the accompanying article |
| Known limitations | Reference standard is the operator's implanted device although several sizes may be acceptable; one angiography vendor; geometric deployment without braid mechanics; selection table of the participating centres; small external cohort |
| Human oversight | The review flag marks low-confidence cases; every recommendation must be checked against the anatomy by a qualified operator |
| Regulatory status | Research software. Not a medical device; not cleared for clinical use |
