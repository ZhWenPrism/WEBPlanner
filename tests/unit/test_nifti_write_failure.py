import builtins
from pathlib import Path
import stat
import numpy as np
import pytest
from webplanner.types import Volume
from webplanner.utils.io import load_volume, save_volume


def volume():
    return Volume(np.arange(24, dtype=np.int16).reshape(2, 3, 4), (1., 2., 3.),
                  origin=(4., 5., 6.))


@pytest.mark.parametrize("ending", [".nii", ".nii.gz"])
def test_partial_simpleitk_output_preserves_the_existing_volume(tmp_path, monkeypatch, ending):
    sitk = pytest.importorskip("SimpleITK")
    path = save_volume(volume(), tmp_path / ("image" + ending))
    original = path.read_bytes()
    def fail(image, destination, **kwargs):
        Path(destination).write_bytes(b"partial")
        raise RuntimeError("simulated NIfTI write failure")
    monkeypatch.setattr(sitk, "WriteImage", fail)
    with pytest.raises(RuntimeError, match="simulated"):
        save_volume(volume(), path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_partial_nibabel_fallback_preserves_the_existing_volume(tmp_path, monkeypatch):
    nib = pytest.importorskip("nibabel")
    original_import = builtins.__import__
    def without_sitk(name, *args, **kwargs):
        if name == "SimpleITK":
            raise ImportError("test fallback")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_sitk)
    path = save_volume(volume(), tmp_path / "image.nii.gz")
    original = path.read_bytes()
    def fail(image, destination):
        Path(destination).write_bytes(b"partial")
        raise OSError("simulated fallback failure")
    monkeypatch.setattr(nib, "save", fail)
    with pytest.raises(OSError, match="simulated"):
        save_volume(volume(), path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_successful_replacement_keeps_array_geometry_and_existing_permissions(tmp_path):
    pytest.importorskip("SimpleITK")
    path = save_volume(volume(), tmp_path / "image.nii.gz")
    path.chmod(0o640)
    mode = stat.S_IMODE(path.stat().st_mode)
    save_volume(volume(), path)
    restored = load_volume(path)
    np.testing.assert_array_equal(restored.array, volume().array)
    assert restored.spacing == volume().spacing and restored.origin == volume().origin
    assert stat.S_IMODE(path.stat().st_mode) == mode
