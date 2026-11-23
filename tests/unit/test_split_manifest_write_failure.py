from pathlib import Path

import pytest

from webplanner.data.splits import Fold, SplitManifest


def manifest():
    return SplitManifest(["case-1", "case-2"], ["case-3"], "external",
                         [Fold("fold_0", ["case-1"], ["case-2"]), Fold("fold_1", ["case-2"], ["case-1"])],
                         seed=7)


def test_partial_split_write_preserves_the_previous_manifest(tmp_path, monkeypatch):
    target = tmp_path / "splits.json"
    target.write_bytes(b"previous manifest")
    original = Path.write_text

    def fail(path, text, *args, **kwargs):
        original(path, "partial manifest", encoding="utf-8")
        raise OSError("manifest write failed")

    monkeypatch.setattr(Path, "write_text", fail)
    with pytest.raises(OSError, match="manifest write failed"):
        manifest().to_json(target)
    assert target.read_bytes() == b"previous manifest"
    assert [p.name for p in tmp_path.iterdir()] == ["splits.json"]


def test_successful_split_manifest_round_trips(tmp_path):
    value = manifest()
    path = value.to_json(tmp_path / "nested/splits.json")
    assert SplitManifest.from_json(path) == value
