import numpy as np
import pytest

from webplanner.evaluation import bland_altman, icc_2_1


def test_icc_does_not_treat_infinite_ratings_as_omittable_missing_data():
    with pytest.raises(ValueError, match="infinite"):
        icc_2_1([[1, 1], [2, np.inf], [3, 3]], nan_policy="omit")


def test_bland_altman_rejects_nonfinite_measurements():
    with pytest.raises(ValueError, match="finite"):
        bland_altman([1, 2, np.inf], [1, 2, 3])


def test_confidence_levels_are_validated_even_for_perfect_agreement():
    with pytest.raises(ValueError, match="alpha"):
        icc_2_1([[1, 1], [2, 2], [3, 3]], alpha=0)
    with pytest.raises(ValueError, match="alpha"):
        bland_altman([1, 2, 3], [1, 2, 3], alpha=0)
    with pytest.raises(ValueError, match="loa"):
        bland_altman([1, 2, 3], [1, 2, 3], loa=1)
