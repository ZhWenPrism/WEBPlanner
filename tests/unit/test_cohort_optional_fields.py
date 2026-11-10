import pytest

from webplanner.data.cohort import read_cohort


def write_manifest(path, suffix):
    path.write_text("case_id,patient_id,centre,ruptured,implanted_width,implanted_height\n"
                    + suffix, encoding="utf-8")


def test_numeric_rupture_status_survives_an_empty_field(tmp_path):
    path = tmp_path / "cohort.csv"
    write_manifest(path, "c1,p1,Centre 1,1,5,3\nc2,p2,Centre 1,,,\nc3,p3,Centre 1,0,,\n")
    records = read_cohort(path)
    assert [record.ruptured for record in records] == [True, None, False]
    assert records[0].implanted.label == "5x3"


@pytest.mark.parametrize("fields, message", [
    ("maybe,5,3", "ruptured"),
    ("true,5,", "both implanted"),
    ("true,,3", "both implanted"),
    ("true,inf,3", "positive finite"),
    ("true,5,0", "positive finite"),
])
def test_malformed_optional_values_are_rejected(tmp_path, fields, message):
    path = tmp_path / "cohort.csv"
    write_manifest(path, "c1,p1,Centre 1," + fields + "\n")
    with pytest.raises(ValueError, match=message):
        read_cohort(path)
