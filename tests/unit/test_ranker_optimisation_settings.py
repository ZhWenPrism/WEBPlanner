import numpy as np
import pytest

from webplanner.ranking import ConditionalLogit


@pytest.mark.parametrize("name,value", [
    ("l2", -1.), ("l2", np.nan), ("l2", np.inf),
    ("max_iter", 0), ("max_iter", -1), ("max_iter", 1.5), ("max_iter", True),
    ("tol", 0.), ("tol", -1.), ("tol", np.nan), ("tol", np.inf),
])
def test_invalid_ranker_settings_are_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        ConditionalLogit(**{name: value})


def test_unpenalised_model_and_numpy_integer_budget_remain_valid():
    model = ConditionalLogit(l2=0., max_iter=np.int64(20), tol=1e-6)
    assert model.l2 == 0. and model.max_iter == 20 and model.tol == 1e-6
