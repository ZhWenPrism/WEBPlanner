from dataclasses import asdict
from pathlib import Path
import numpy as np
import pytest
from webplanner.data.preprocessing import Preprocessor
from webplanner.types import Volume


def fitted():
    volume = Volume(np.arange(8, dtype=np.float32).reshape(2, 2, 2), (1., 1., 1.))
    return Preprocessor().fit([volume], [np.ones(volume.shape)])


def test_partial_preprocessing_state_does_not_replace_a_complete_file(tmp_path, monkeypatch):
    state = fitted()
    path = state.save(tmp_path / "state.json")
    original = path.read_bytes()
    write_text = Path.write_text
    def fail(destination, text, *args, **kwargs):
        write_text(destination, text[:12], *args, **kwargs)
        raise OSError("simulated state write failure")
    monkeypatch.setattr(Path, "write_text", fail)
    with pytest.raises(OSError, match="simulated"):
        state.save(path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_saved_state_round_trips_through_the_shared_encoder(tmp_path):
    state = fitted()
    assert asdict(Preprocessor.load(state.save(tmp_path / "state.json"))) == asdict(state)
