import numpy as np
import pytest

from webplanner.evaluation import dice, dice_per_class, per_class_metrics, surface_metrics


@pytest.mark.parametrize("metric", [dice, dice_per_class, per_class_metrics, surface_metrics])
def test_nonfinite_voxels_are_not_interpreted_as_mask_values(metric):
    pred = np.zeros((3, 3, 3), dtype=np.float32)
    pred[0, 0, 0] = np.nan
    ref = np.zeros_like(pred, dtype=np.uint8)
    ref[0, 0, 0] = 1
    with pytest.raises(ValueError, match="finite voxel"):
        metric(pred, ref)
