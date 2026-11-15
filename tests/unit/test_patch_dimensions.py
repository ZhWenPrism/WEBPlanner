import numpy as np
import pytest

from webplanner.data.dataset import PatchDataset, crop_with_padding


@pytest.mark.parametrize("size", [0, True, 2.7, (3, 2, 0)])
def test_patch_size_has_three_positive_integer_dimensions(size):
    volume = np.zeros((3, 3, 3))
    with pytest.raises(ValueError, match="positive integer dimensions"):
        PatchDataset([(volume, volume)], patch_size=size)


def test_crop_rejects_incomplete_coordinates_instead_of_leaving_an_axis_unsliced():
    volume = np.zeros((3, 3, 3))
    with pytest.raises(ValueError, match="one entry per array dimension"):
        crop_with_padding(volume, (0, 0), (2, 2))


def test_integral_crop_can_pad_a_negative_start():
    volume = np.arange(9).reshape(3, 3)
    result = crop_with_padding(volume, (np.int64(-1), 1), (2, 2), -1)
    np.testing.assert_array_equal(result, [[-1, -1], [1, 2]])
