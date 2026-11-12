import numpy as np
import pytest

from webplanner.ranking.conditional_logit import resolve_chosen
from webplanner.ranking.features import CandidateSet
from webplanner.types import DeviceSize


@pytest.mark.parametrize("index", [0.9, "1", True])
def test_resolve_chosen_does_not_coerce_noninteger_indices(index):
    case = CandidateSet("case-1", [DeviceSize(5, 3), DeviceSize(6, 4)], [[0], [1]], [True, True], ["feature"])
    with pytest.raises(ValueError, match="case case-1: chosen index must be an integer"):
        resolve_chosen([case], [index])


def test_resolve_chosen_accepts_numpy_integer_indices():
    case = CandidateSet("case-1", [DeviceSize(5, 3), DeviceSize(6, 4)], [[0], [1]], [True, True], ["feature"])
    np.testing.assert_array_equal(resolve_chosen([case], [np.int64(1)]), [1])
