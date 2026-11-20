from pathlib import Path
import numpy as np
import pytest
from webplanner.types import Volume
from webplanner.utils.io import load_volume, save_volume


def test_partial_volume_archives_do_not_replace_a_completed_file(tmp_path, monkeypatch):
    path = tmp_path / "volume.npz"
    path.write_bytes(b"previous complete volume")
    def fail(destination, **arrays):
        Path(destination).write_bytes(b"partial volume")
        raise OSError("simulated archive failure")
    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError, match="simulated"):
        save_volume(Volume(np.zeros((2, 3, 4)), (1., 2., 3.)), path)
    assert path.read_bytes() == b"previous complete volume"
    assert list(tmp_path.iterdir()) == [path]


def test_successful_archive_keeps_array_and_geometry(tmp_path):
    volume = Volume(np.arange(24).reshape(2, 3, 4), (1., 2., 3.), origin=(4., 5., 6.))
    restored = load_volume(save_volume(volume, tmp_path / "volume.npz"))
    np.testing.assert_array_equal(restored.array, volume.array)
    assert restored.spacing == volume.spacing and restored.origin == volume.origin
