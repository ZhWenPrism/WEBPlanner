import numpy as np
import pytest

from webplanner.ranking import fit_tau, flag


@pytest.mark.parametrize("values", [[0.8, np.nan], [-0.1, 0.8], [0.8, 1.1], [[0.1, 0.9]]])
def test_fitting_requires_a_vector_of_probabilities(values):
    with pytest.raises(ValueError, match="probabilit"):
        fit_tau(values)


@pytest.mark.parametrize("probability, tau", [(np.nan, 0.2), (1.1, 0.2), (0.3, np.nan), (0.3, 1.1)])
def test_invalid_inputs_cannot_suppress_or_force_a_review(probability, tau):
    with pytest.raises(ValueError, match="probabilit|tau"):
        flag(probability, tau)


def test_review_flags_retain_strict_comparison_and_array_shape():
    values = np.array([[0.0, 0.3, 1.0]])
    np.testing.assert_array_equal(flag(values, 0.3), [[True, False, False]])
    assert flag(0.3, 0.3) is False
