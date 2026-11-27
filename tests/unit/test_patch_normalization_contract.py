import numpy as np
import pytest
from webplanner.data.dataset import PatchDataset


@pytest.mark.parametrize("normalize,message", [
    (lambda x: x[1:], "preserve the image grid"),
    (lambda x: x[0], "preserve the image grid"),
    (lambda x: np.full_like(x, np.nan), "finite"),
])
def test_normalization_must_preserve_shape_and_finite_voxels(normalize, message):
    image = np.ones((3, 3, 3), np.float32)
    with pytest.raises(ValueError, match=message):
        PatchDataset([(image, np.zeros(image.shape, np.uint8))], patch_size=3, normalize=normalize)


@pytest.mark.parametrize("shape", [(3, 3), (0, 3, 3)])
def test_patch_cases_require_nonempty_3d_arrays(shape):
    with pytest.raises(ValueError, match="non-empty 3D"):
        PatchDataset([(np.ones(shape), np.zeros(shape, np.uint8))], patch_size=3)


def test_valid_normalization_preserves_patch_labels_and_output_types():
    image = np.ones((3, 3, 3), np.float32)
    label = np.zeros(image.shape, np.uint8)
    dataset = PatchDataset([(image, label)], patch_size=3, normalize=lambda x: x * 2 + 1)
    patch, target = dataset.sample(0)
    np.testing.assert_array_equal(patch, np.full(image.shape, 3, np.float32))
    np.testing.assert_array_equal(target, label)
    assert patch.dtype == np.float32 and target.dtype == np.int64
