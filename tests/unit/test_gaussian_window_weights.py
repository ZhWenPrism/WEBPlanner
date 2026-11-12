import numpy as np
import pytest


def test_narrow_gaussian_on_even_grid_has_finite_positive_weights():
    pytest.importorskip("torch")
    from webplanner.segmentation.inference import gaussian_importance_map

    with np.errstate(divide="raise", invalid="raise"):
        weights = gaussian_importance_map((8, 10, 12), sigma_scale=0.001)
    assert weights.shape == (8, 10, 12)
    assert np.isfinite(weights).all() and np.all(weights > 0)
    assert weights[3, 4, 5] == weights.max() == 10.0
    np.testing.assert_array_equal(weights, weights[::-1, ::-1, ::-1])
