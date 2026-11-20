from types import SimpleNamespace
import pytest
from webplanner.data import dicom


@pytest.mark.parametrize("counts", [(1, 1, 1), (2, 3), (4,)])
def test_series_summary_counts_frames_across_all_files(tmp_path, monkeypatch, counts):
    records = {}
    for i, count in enumerate(counts):
        path = tmp_path / f"{i}.dcm"
        path.touch()
        records[path.name] = SimpleNamespace(SeriesInstanceUID="1.2.3", Rows=2, Columns=2, NumberOfFrames=count)
    monkeypatch.setattr(dicom, "_read", lambda path, **kwargs: records[path.name])
    series = dicom.list_series(tmp_path)
    assert len(series) == 1
    assert series[0].n_files == len(counts)
    assert series[0].n_frames == sum(counts)
