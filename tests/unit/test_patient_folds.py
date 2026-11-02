import numpy as np
import pytest

from webplanner.data.splits import patient_kfold
from webplanner.types import CaseRecord


def records():
    return [CaseRecord(case_id="a", patient_id="p1", centre="a"),
            CaseRecord(case_id="b", patient_id="p1", centre="a"),
            CaseRecord(case_id="c", patient_id="p2", centre="a")]


@pytest.mark.parametrize("k", [0, 1, -1, 3, 2.0, True])
def test_invalid_fold_count_is_rejected(k):
    with pytest.raises(ValueError, match="k"):
        patient_kfold(records(), k=k)


def test_valid_folds_keep_patient_cases_together():
    folds = patient_kfold(records(), k=np.int64(2))
    assert len(folds) == 2
    assert sorted(c for f in folds for c in f.val) == ["a", "b", "c"]
    for f in folds:
        assert bool("a" in f.val) == bool("b" in f.val)
        assert f.train and f.val and not (set(f.train) & set(f.val))
