import pytest
from webplanner.device.catalogue import Catalogue


@pytest.mark.parametrize("height,expected", [(3.4999, 3.0), (3.5, 3.0), (3.5001, 4.0)])
def test_nearest_height_uses_distance_then_lower_height(height, expected):
    catalogue = Catalogue.default()
    assert catalogue.nearest_height(6.0, height) == expected


def test_equal_distance_ties_do_not_depend_on_catalogue_storage_order():
    catalogue = Catalogue.default()
    catalogue.heights[6.0] = [4.0, 3.0, 2.0]
    assert catalogue.nearest_height(6.0, 3.5) == 3.0
