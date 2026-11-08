import pytest

from webplanner.data.cohort import read_cohort


@pytest.mark.parametrize("row, column", [
    (",patient-1,Centre 1", "case_id"),
    ("case-1, ,Centre 1", "patient_id"),
    ("case-1,patient-1,", "centre"),
])
def test_missing_identifiers_are_rejected(tmp_path, row, column):
    path = tmp_path / "cohort.csv"
    path.write_text("case_id,patient_id,centre\n" + row + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=f"missing {column}"):
        read_cohort(path)


def test_identifier_tokens_keep_leading_zeroes_and_literal_na(tmp_path):
    path = tmp_path / "cohort.csv"
    path.write_text("case_id,patient_id,centre\n001,NA,Centre 1\n", encoding="utf-8")
    [record] = read_cohort(path)
    assert (record.case_id, record.patient_id, record.centre) == ("001", "NA", "Centre 1")
