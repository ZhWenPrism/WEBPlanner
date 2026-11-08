import numpy as np
import pytest

from webplanner.data.cohort import label_loader
from webplanner.types import CaseRecord, Volume
from webplanner.utils.io import save_volume


def test_case_id_cannot_read_a_label_from_a_sibling_directory(tmp_path):
    folder = tmp_path / "labels"
    folder.mkdir()
    save_volume(Volume(np.zeros((2, 2, 2), dtype=np.uint8), (1, 1, 1)), tmp_path / "sibling.npz")
    record = CaseRecord("../sibling", "p1", "Centre 1")
    with pytest.raises(ValueError, match="case_id"):
        label_loader(folder)(record)


def test_label_directory_reads_a_matching_case_file(tmp_path):
    data = np.ones((2, 2, 2), dtype=np.uint8)
    save_volume(Volume(data, (1, 1, 1)), tmp_path / "c1.npz")
    record = CaseRecord("c1", "p1", "Centre 1")
    np.testing.assert_array_equal(label_loader(tmp_path)(record).array, data)
