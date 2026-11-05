import numpy as np
import pytest

from webplanner.types import Volume


@pytest.mark.parametrize("spacing, metadata, message", [
    ((1., 1.), {}, "spacing"),
    ((1., 0., 1.), {}, "spacing"),
    ((1., -1., 1.), {}, "spacing"),
    ((1., float("nan"), 1.), {}, "spacing"),
    ((1., 1., 1.), {"origin": [0., float("inf"), 0.]}, "origin"),
    ((1., 1., 1.), {"origin": [0., 0.]}, "origin"),
    ((1., 1., 1.), {"direction": [1.] * 8}, "direction"),
    ((1., 1., 1.), {"direction": [float("nan")] * 9}, "direction"),
])
def test_invalid_physical_metadata_is_rejected(spacing, metadata, message):
    with pytest.raises(ValueError, match=message):
        Volume(np.zeros((2, 2, 2)), spacing, **metadata)


def test_valid_metadata_is_independent_of_mutable_inputs():
    spacing, origin = [2., 1., 0.5], [3., 2., 1.]
    direction = np.eye(3).ravel().tolist()
    volume = Volume(np.zeros((2, 2, 2)), spacing, origin, direction)
    spacing[:], origin[:], direction[:] = [0.]*3, [0.]*3, [0.]*9
    assert volume.spacing == (2., 1., 0.5)
    assert volume.origin == (3., 2., 1.)
    assert volume.direction == tuple(np.eye(3).ravel())
