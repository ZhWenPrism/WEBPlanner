import builtins

import numpy as np
import pytest

from webplanner.types import Volume
from webplanner.utils.io import load_volume, save_volume


def _volume():
    return Volume(np.arange(24, dtype=np.int16).reshape(2, 3, 4), (2.0, 1.0, 0.5),
                  (30.0, 20.0, 10.0), (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))


def _assert_round_trip(original, restored):
    assert np.array_equal(restored.array, original.array)
    assert restored.spacing == pytest.approx(original.spacing)
    assert restored.origin == pytest.approx(original.origin)
    assert restored.direction == pytest.approx(original.direction)


def test_npz_round_trip_keeps_direction(tmp_path):
    original = _volume()
    _assert_round_trip(original, load_volume(save_volume(original, tmp_path / "volume.npz")))


def test_legacy_npz_without_direction_remains_readable(tmp_path):
    original = _volume()
    path = tmp_path / "legacy.npz"
    np.savez(path, array=original.array, spacing=original.spacing)
    assert load_volume(path).direction == Volume(original.array, original.spacing).direction


def test_simpleitk_round_trip_keeps_direction(tmp_path):
    pytest.importorskip("SimpleITK")
    original = _volume()
    _assert_round_trip(original, load_volume(save_volume(original, tmp_path / "volume.nii.gz")))


def test_nibabel_fallback_round_trip_and_affine(tmp_path, monkeypatch):
    nib = pytest.importorskip("nibabel")
    real_import = builtins.__import__
    def without_sitk(name, *args, **kwargs):
        if name == "SimpleITK":
            raise ImportError("test fallback")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_sitk)
    original = _volume()
    path = save_volume(original, tmp_path / "volume.nii.gz")
    affine = np.array([[0.0, 1.0, 0.0, -10.0],
                       [-0.5, 0.0, 0.0, -20.0],
                       [0.0, 0.0, 2.0, 30.0],
                       [0.0, 0.0, 0.0, 1.0]])
    assert np.allclose(nib.load(path).affine, affine)
    _assert_round_trip(original, load_volume(path))

