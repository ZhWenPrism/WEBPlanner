import json
import pytest
from webplanner.data.preprocessing import Preprocessor


@pytest.mark.parametrize("values", [
    {"spacing": 0}, {"crop_mm": float("nan")},
    {"lower_percentile": 80, "upper_percentile": 20},
    {"lower_percentile": -1}, {"upper_percentile": 101},
    {"mean": 0}, {"clip_low": 0, "clip_high": 1, "mean": 0, "std": 0},
    {"clip_low": 2, "clip_high": 1, "mean": 0, "std": 1},
])
def test_invalid_preprocessing_parameters_fail_at_construction_and_load(tmp_path, values):
    with pytest.raises(ValueError):
        Preprocessor(**values)
    path = tmp_path / "state.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(ValueError):
        Preprocessor.load(path)


def test_unfitted_and_constant_fitted_states_remain_valid():
    assert Preprocessor().mean is None
    state = Preprocessor(clip_low=2, clip_high=2, mean=2, std=1e-8)
    assert state.clip_low == state.clip_high == 2
