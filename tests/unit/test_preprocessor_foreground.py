import numpy as np
import pytest

from webplanner.data.preprocessing import Preprocessor
from webplanner.types import Volume


def test_fit_does_not_pair_a_lower_dimensional_mask_with_a_volume():
    vol = Volume(np.arange(8).reshape(2, 2, 2), (1, 1, 1))
    with pytest.raises(ValueError, match="foreground mask shape .* does not match volume shape"):
        Preprocessor().fit([vol], [np.ones((2, 2))])


@pytest.mark.parametrize("n_masks", [0, 2])
def test_fit_requires_one_foreground_mask_per_training_volume(n_masks):
    vol = Volume(np.arange(8).reshape(2, 2, 2), (1, 1, 1))
    masks = (np.ones(vol.shape) for _ in range(n_masks))
    with pytest.raises(ValueError, match="zip.*argument"):
        Preprocessor().fit(iter([vol]), masks)


def test_fit_reports_missing_foreground_before_percentile_estimation():
    vol = Volume(np.ones((2, 2, 2)), (1, 1, 1))
    with pytest.raises(ValueError, match="no foreground voxels"):
        Preprocessor().fit([vol])
