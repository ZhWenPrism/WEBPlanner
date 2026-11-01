import numpy as np
import pytest

from webplanner.data.preprocessing import crop_cube
from webplanner.types import Volume


def test_crop_origin_tracks_offset_in_scanner_coordinates():
    volume = Volume(np.zeros((10, 10, 10)), (2.0, 1.0, 0.5), (30.0, 20.0, 10.0),
                    (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    cropped, offset = crop_cube(volume, (8.0, 4.0, 2.0), size_mm=8.0)
    assert offset.tolist() == [2, 0, -4]
    assert cropped.origin == pytest.approx((34.0, 18.0, 10.0))
    assert cropped.direction == volume.direction
    assert cropped.spacing == volume.spacing

