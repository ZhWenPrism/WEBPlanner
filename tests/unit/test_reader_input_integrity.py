import numpy as np
import pandas as pd
import pytest

from webplanner.evaluation.reader_study import ReaderStudy


def reads(**overrides):
    return pd.DataFrame({"case_id": ["case-1", "case-1"], "reader_id": ["reader-1", "reader-1"],
                         "experience": ["senior", "senior"], "session": ["unaided", "assisted"],
                         "correct": [0, 1], **overrides})


@pytest.mark.parametrize("name", ["case_id", "reader_id"])
def test_blank_identifiers_do_not_form_a_study_group(name):
    with pytest.raises(ValueError, match=f"missing {name}"):
        ReaderStudy(reads(**{name: ["", "  "]}))


def test_infinite_reading_time_is_rejected_but_missing_time_is_allowed():
    with pytest.raises(ValueError, match="minutes.*finite numeric"):
        ReaderStudy(reads(minutes=[1.0, np.inf]))
    assert ReaderStudy(reads(minutes=[1.0, np.nan])).reads["minutes"].isna().sum() == 1
