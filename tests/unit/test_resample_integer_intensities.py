import numpy as np
from scipy import ndimage as ndi
from webplanner.data.preprocessing import resample_isotropic
from webplanner.types import Volume


def test_linear_resampling_keeps_fractional_intensities_from_integer_images():
    array = np.tile(np.array([0, 1], dtype=np.int16), (2, 2, 1))
    volume = Volume(array, (1., 1., 1.))
    result = resample_isotropic(volume, .5, order=1)
    expected = ndi.zoom(array.astype(np.float32), 2, order=1, mode="nearest", prefilter=False)
    assert result.array.dtype.kind == "f"
    np.testing.assert_allclose(result.array, expected)
    assert np.any((result.array > 0) & (result.array < 1))


def test_nearest_neighbour_label_resampling_retains_integer_codes():
    array = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
    result = resample_isotropic(Volume(array, (1., 1., 1.)), .5, order=0)
    assert result.array.dtype == array.dtype
    assert set(np.unique(result.array)) == set(np.unique(array))


def test_existing_floating_images_keep_the_same_interpolation():
    array = np.arange(8, dtype=np.float64).reshape(2, 2, 2) / 3
    result = resample_isotropic(Volume(array, (1., 1., 1.)), .5, order=1)
    expected = ndi.zoom(array, 2, order=1, mode="nearest", prefilter=False)
    assert result.array.dtype == array.dtype
    np.testing.assert_array_equal(result.array, expected)
