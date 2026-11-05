import pytest

from webplanner.data.dicom import find_3dra_series, list_series, load_3dra_series


def test_non_image_files_do_not_form_a_series(tmp_path):
    (tmp_path / "notes.txt").write_text("synthetic archive notes", encoding="utf-8")
    assert list_series(tmp_path) == []
    assert find_3dra_series(tmp_path, min_slices=1) is None


def test_unknown_series_reports_missing_image_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="requested DICOM image series"):
        load_3dra_series(tmp_path, series_uid="1.2.3.4")
