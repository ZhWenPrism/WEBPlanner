import numpy as np
import pytest

from webplanner.segmentation.postprocess import postprocess


@pytest.mark.parametrize("value", [-1, 5, 256, 1.5, np.nan])
def test_cleanup_rejects_invalid_labels_before_casting(value):
    with pytest.raises(ValueError, match="label map must contain only values 0 through 4"):
        postprocess(np.array([[[value]]]))


@pytest.mark.parametrize("spacing", [(1, 1), (1, 1, 0), (1, 1, np.inf)])
def test_cleanup_rejects_invalid_voxel_spacing(spacing):
    with pytest.raises(ValueError, match="three positive finite voxel sizes"):
        postprocess(np.zeros((3, 3, 3)), spacing=spacing)
