from pathlib import Path
import pytest
from webplanner.config import Config, load_config
from webplanner.utils.io import read_json, write_json


@pytest.mark.parametrize("kind", ["json", "config"])
def test_partial_writes_preserve_existing_metadata(tmp_path, monkeypatch, kind):
    path = tmp_path / ("saved.json" if kind == "json" else "saved.yaml")
    path.write_bytes(b"previous completed metadata")
    def fail(self, text, **kwargs):
        self.write_bytes(b"partial")
        raise OSError("simulated write failure")
    monkeypatch.setattr(Path, "write_text", fail)
    with pytest.raises(OSError, match="simulated"):
        write_json({"epoch": 1}, path) if kind == "json" else Config().save(path)
    assert path.read_bytes() == b"previous completed metadata"
    assert list(tmp_path.iterdir()) == [path]


def test_successful_metadata_round_trips(tmp_path):
    path = write_json({"epoch": 3}, tmp_path / "saved.json")
    assert read_json(path) == {"epoch": 3}
    config = Config(seed=19)
    assert load_config(config.save(tmp_path / "config.yaml")).seed == 19
