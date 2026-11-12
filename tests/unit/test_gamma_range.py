import numpy as np
import pytest

from webplanner.data.transforms import Gamma


@pytest.mark.parametrize("bounds", [(0.5, 0.8), (0.5, 0.5), (1.2, 1.5)])
def test_gamma_samples_stay_within_one_sided_ranges(bounds):
    image = np.array([[[0.0, 0.25, 1.0]]], dtype=np.float32)
    label = np.array([[[0, 1, 0]]], dtype=np.uint8)
    rng = np.random.default_rng(0)
    transform = Gamma(bounds, retain_stats=False, p=1)
    for _ in range(20):
        out, unchanged = transform(image, label, rng)
        assert 0.25 ** bounds[1] - 1e-7 <= out[0, 0, 1] <= 0.25 ** bounds[0] + 1e-7
        np.testing.assert_array_equal(unchanged, label)
