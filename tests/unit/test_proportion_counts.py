import numpy as np
import pytest

from webplanner.evaluation import proportion, wilson_ci


@pytest.mark.parametrize("k, n", [(0.5, 2), (1, 2.5), (np.nan, 2), (1, np.inf)])
def test_wilson_intervals_require_finite_integer_counts(k, n):
    with pytest.raises(ValueError, match="integer counts"):
        wilson_ci(k, n)


def test_proportion_does_not_report_truncated_counts():
    with pytest.raises(ValueError, match="integer count"):
        proportion(1.5, 3)


def test_vectorized_intervals_keep_zero_trials_undefined():
    low, high = wilson_ci([0, 1], [0, 2])
    assert np.isnan(low[0]) and np.isnan(high[0])
    assert 0 < low[1] < 0.5 < high[1] < 1
