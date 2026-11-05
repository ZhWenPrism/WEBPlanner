from itertools import permutations
from unittest.mock import patch

import numpy as np
import pytest

from webplanner.data.synthetic import make_phantom, random_spec
from webplanner.geometry import fit_neck_frame, measure
from webplanner.types import Volume


@pytest.fixture(scope="module")
def circular_neck_volume():
    return make_phantom(random_spec(np.random.default_rng(13))).labels


@pytest.mark.parametrize("angle", [0.1, 0.5, 1.0])
def test_equivalent_eigenbases_preserve_widths(circular_neck_volume, angle):
    volume = circular_neck_volume
    reference_frame = fit_neck_frame(volume)
    reference = measure(volume, reference_frame)
    eigh = np.linalg.eigh

    def rotated_eigh(matrix):
        values, vectors = eigh(matrix)
        if np.isclose(values[-1], values[-2], rtol=1e-12, atol=1e-12):
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            vectors[:, -2:] = vectors[:, -2:] @ rotation
        return values, vectors

    with patch("webplanner.geometry.neck.np.linalg.eigh", side_effect=rotated_eigh):
        frame = fit_neck_frame(volume)
    assert abs(frame.e1 @ reference_frame.e1) == pytest.approx(1., abs=1e-12)
    result = measure(volume, frame)
    assert result.mean_width == pytest.approx(reference.mean_width, abs=1e-9)
    assert result.max_width == pytest.approx(reference.max_width, abs=1e-9)


@pytest.mark.parametrize("axes", list(permutations(range(3))))
def test_circular_neck_measurements_survive_axis_permutations(circular_neck_volume, axes):
    volume = circular_neck_volume
    reference = measure(volume, fit_neck_frame(volume))
    moved = Volume(np.ascontiguousarray(volume.array.transpose(axes)),
                   tuple(volume.spacing[i] for i in axes))
    observed = measure(moved, fit_neck_frame(moved))
    for name in ("mean_width", "max_width", "height", "neck_width", "volume_mm3", "solidity", "sac_axis_extent"):
        assert getattr(observed, name) == pytest.approx(getattr(reference, name), abs=1e-6), name
