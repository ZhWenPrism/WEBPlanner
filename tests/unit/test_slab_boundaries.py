import numpy as np

from webplanner.geometry.morphometry import slab_extents


def test_machine_roundoff_does_not_move_points_across_a_slab_boundary():
    h = np.repeat([0.5, 1.0], 8)
    uv = np.column_stack((np.repeat([0., 10.], 8), np.zeros(16)))
    directions = np.array([[1., 0.]])
    expected = slab_extents(h, uv, directions, slab_mm=1.)
    perturbed = h.copy()
    perturbed[8:] = np.nextafter(1., 0.)
    np.testing.assert_array_equal(slab_extents(perturbed, uv, directions, slab_mm=1.), expected)
    np.testing.assert_array_equal(expected, [0.])


def test_points_resolvably_below_a_boundary_keep_their_slab():
    h = np.repeat([0.5, 1. - 1e-6], 8)
    uv = np.column_stack((np.repeat([0., 10.], 8), np.zeros(16)))
    np.testing.assert_array_equal(slab_extents(h, uv, np.array([[1., 0.]]), slab_mm=1.), [10.])
