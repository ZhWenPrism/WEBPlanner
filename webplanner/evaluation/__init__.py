"""Evaluation layer: segmentation metrics, proportions, agreement, discrimination, sizing and the reader study."""
from .agreement import (BlandAltmanResult, FleissResult, ICCResult, KappaResult, bland_altman, cohen_kappa,
                        fleiss_kappa, fleiss_kappa_bootstrap, icc_2_1, ratings_to_table)
from .discrimination import auroc_delong, auroc_difference_ci, auroc_se, delong_cov, delong_test
from .proportions import (BootstrapResult, Proportion, fmt_ci, fmt_p, fmt_pct, fmt_prop, mcnemar_exact,
                          newcombe_paired_diff_ci, paired_bootstrap_diff, proportion, proportion_of, wilson_ci)
from .reader_study import ReaderStudy
from .segmentation_metrics import (average_surface_distance, dice, dice_per_class, hausdorff, hd95, per_class_metrics,
                                   surface_distances, surface_metrics)
from .sizing import agreement_table, compare_methods, sizing_agreement, sizing_cases, subgroup_table

__all__ = [
    "BlandAltmanResult", "BootstrapResult", "FleissResult", "ICCResult", "KappaResult", "Proportion", "ReaderStudy",
    "agreement_table", "auroc_delong", "auroc_difference_ci", "auroc_se", "average_surface_distance", "bland_altman",
    "cohen_kappa", "compare_methods", "delong_cov", "delong_test", "dice", "dice_per_class", "fleiss_kappa",
    "fleiss_kappa_bootstrap", "fmt_ci", "fmt_p", "fmt_pct", "fmt_prop", "hausdorff", "hd95", "icc_2_1",
    "mcnemar_exact", "newcombe_paired_diff_ci", "paired_bootstrap_diff", "per_class_metrics", "proportion",
    "proportion_of", "ratings_to_table", "sizing_agreement", "sizing_cases", "subgroup_table", "surface_distances",
    "surface_metrics", "wilson_ci",
]
