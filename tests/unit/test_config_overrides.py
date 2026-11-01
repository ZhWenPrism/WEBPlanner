import pytest

from webplanner.config import parse_overrides


@pytest.mark.parametrize("override", ["seed", "=1", "data..n_folds=3", ".seed=1", "data.=2"])
def test_malformed_override_is_rejected(override):
    with pytest.raises(ValueError):
        parse_overrides([override])


def test_nested_override_below_a_scalar_has_a_clear_error():
    with pytest.raises(ValueError):
        parse_overrides(["data=1", "data.n_folds=3"])


def test_valid_nested_override_keeps_yaml_value_types():
    assert parse_overrides(["data.n_folds=3", "segmentation.mirror_tta=false"]) == {
        "data": {"n_folds": 3}, "segmentation": {"mirror_tta": False}}

