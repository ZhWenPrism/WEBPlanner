import pytest
from webplanner.data.cohort import read_cohort


@pytest.mark.parametrize("column", ["case_id", "patient_id", "centre", "image"])
def test_duplicate_headers_do_not_get_silently_mangled(tmp_path, column):
    path = tmp_path / "cohort.csv"
    path.write_text("case_id,patient_id,centre,image," + column + "\ncase-01,study-01,site,scan.npz,other\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicated manifest columns"):
        read_cohort(path)


def test_bom_and_literal_identifiers_remain_supported(tmp_path):
    path = tmp_path / "cohort.csv"
    path.write_text("case_id,patient_id,centre\n001,002,NA\n", encoding="utf-8-sig")
    result = read_cohort(path)
    assert (result[0].case_id, result[0].patient_id, result[0].centre) == ("001", "002", "NA")
