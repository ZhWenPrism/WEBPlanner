import numpy as np
import pytest

from webplanner.ranking.model_selection import ranking_metrics


@pytest.mark.parametrize("proba,chosen", [
    ([np.array([0.2, 0.8])], [1, 0]), ([np.array([0.2, 0.8])], [0.5]),
    ([np.array([0.2, 0.8])], [-1]), ([np.array([0.2, 0.8])], [2]),
    ([np.array([np.nan, 0.8])], [1]), ([np.array([0.2, 0.4])], [1]),
    ([np.array([-0.2, 1.2])], [1]), ([np.array([[0.2, 0.8]])], [0]),
    ([], []),
])
def test_invalid_ranking_input_is_rejected(proba, chosen):
    with pytest.raises(ValueError):
        ranking_metrics(proba, chosen)


def test_list_probabilities_and_zero_chosen_probability():
    result = ranking_metrics([[0., 1.], [0.2, 0.8]], [0, 1])
    assert result["n"] == 2 and result["top1"] == 0.5 and result["top3"] == 1.
    assert result["log_loss"] == pytest.approx((-np.log(1e-12) - np.log(0.8)) / 2)


def test_float32_probability_normalization_allows_roundoff():
    proba = np.array([0.2, 0.8], dtype=np.float32)
    result = ranking_metrics([proba], [1])
    assert result["n"] == 1 and result["top1"] == 1.
    assert result["log_loss"] == pytest.approx(-np.log(float(proba[1])))
