import numpy as np
import pytest

from webplanner.evaluation.proportions import paired_bootstrap_diff


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_bootstrap_rejects_nonfinite_paired_observations(value):
    with pytest.raises(ValueError, match="paired observations must be finite"):
        paired_bootstrap_diff([0.0, value], [1.0, 3.0], n_boot=5)


@pytest.mark.parametrize(("options", "match"), [({"n_boot": True}, "positive integer"),
                                                ({"n_boot": 2.5}, "positive integer"),
                                                ({"alpha": 0}, "alpha"),
                                                ({"alpha": 1}, "alpha")])
def test_invalid_bootstrap_settings_fail_before_evaluating_the_statistic(options, match):
    def stat(_):
        pytest.fail("statistic was called before validating settings")

    with pytest.raises(ValueError, match=match):
        paired_bootstrap_diff([0.0], [1.0], stat=stat, **options)
