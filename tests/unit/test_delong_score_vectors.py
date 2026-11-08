import numpy as np
import pytest

from webplanner.evaluation import delong_cov


def test_one_score_argument_cannot_expand_to_two_models():
    labels = [0, 1, 0, 1]
    with pytest.raises(ValueError, match="one-dimensional"):
        delong_cov(labels, np.array([[0.1, 0.9, 0.3, 0.7], [0.2, 0.8, 0.4, 0.6]]))


def test_two_score_vectors_return_a_two_model_covariance():
    labels = [0, 1, 0, 1]
    aucs, covariance = delong_cov(labels, [0.1, 0.9, 0.3, 0.7], [0.2, 0.8, 0.4, 0.6])
    np.testing.assert_allclose(aucs, [1.0, 1.0])
    assert covariance.shape == (2, 2)
