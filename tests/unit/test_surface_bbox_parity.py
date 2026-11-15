import numpy as np
from scipy import ndimage as ndi

from webplanner.evaluation.segmentation_metrics import surface, surface_distances


def test_cropped_distances_match_full_grid_for_anisotropic_border_masks():
    pred = np.zeros((10, 12, 14), bool)
    ref = pred.copy()
    pred[:3, 2:6, 8:13] = True
    ref[1:4, 3:9, 10:] = True
    spacing = (2.0, 1.5, 0.7)
    p, r = surface(pred), surface(ref)
    expected = ndi.distance_transform_edt(~r, sampling=spacing)[p], ndi.distance_transform_edt(~p, sampling=spacing)[r]
    for actual, full in zip(surface_distances(pred, ref, spacing), expected):
        np.testing.assert_allclose(actual, full)
