import numpy as np
import pytest

from webplanner.evaluation.segmentation_metrics import average_surface_distance, hd95, per_class_metrics


def masks():
    a = np.zeros((3, 3, 3), bool)
    b = a.copy()
    a[0, 1, 1] = b[1, 1, 1] = True
    return a, b


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("metric", [hd95, average_surface_distance, per_class_metrics])
def test_surface_spacing_must_be_finite(metric, value):
    with pytest.raises(ValueError, match="spacing"):
        metric(*masks(), spacing=(value, 1., 1.))


def test_surface_distance_respects_valid_anisotropic_spacing():
    assert hd95(*masks(), spacing=(2., 1., 1.)) == pytest.approx(2.)
