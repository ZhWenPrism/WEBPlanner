import numpy as np
import pytest

from webplanner.evaluation.segmentation_metrics import hd95, per_class_metrics, surface_distances
from webplanner.types import Volume


@pytest.fixture
def volumes():
    p = np.zeros((3, 3, 3), np.uint8)
    r = p.copy()
    p[0, 1, 1] = r[1, 1, 1] = 1
    return Volume(p, (2.0, 1.0, 1.0)), Volume(r, (1.0, 1.0, 1.0))


@pytest.mark.parametrize("metric", [surface_distances, hd95, per_class_metrics])
def test_surface_distances_do_not_silently_take_one_of_two_different_spacings(metric, volumes):
    with pytest.raises(ValueError, match="voxel spacing differ"):
        metric(*volumes)


def test_explicit_spacing_overrides_volume_metadata(volumes):
    assert hd95(*volumes, spacing=(1.5, 1.0, 1.0)) == pytest.approx(1.5)
