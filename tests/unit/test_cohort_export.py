from pathlib import Path

import pytest

from webplanner.data.cohort import read_cohort, write_cohort
from webplanner.types import CaseRecord


def test_export_round_trip_with_images_outside_manifest_directory(tmp_path):
    path = tmp_path / "metadata" / "cohort.csv"
    image = tmp_path / "images" / "study.npz"
    label = path.parent / "labels" / "study.npz"
    record = CaseRecord("c1", "p1", "Centre 1", image_path=str(image), label_path=str(label),
                        extra={"vasoct_width": 4.5})
    write_cohort([record], path)
    [restored] = read_cohort(path)
    assert Path(restored.image_path) == image
    assert Path(restored.label_path) == label
    assert restored.extra == record.extra
    assert "labels" in path.read_text(encoding="utf-8")


def test_export_rejects_metadata_that_would_override_patient_identity(tmp_path):
    record = CaseRecord("c1", "p1", "Centre 1", extra={"patient_id": "p2"})
    with pytest.raises(ValueError, match="patient_id"):
        write_cohort([record], tmp_path / "cohort.csv")


def test_empty_cohort_round_trips(tmp_path):
    path = tmp_path / "cohort.csv"
    assert read_cohort(write_cohort([], path)) == []
