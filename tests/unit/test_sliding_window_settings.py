import numpy as np
import pytest

torch = pytest.importorskip("torch")
from webplanner.segmentation.inference import SlidingWindowPredictor


@pytest.mark.parametrize("overrides, message", [
    ({"patch_size": (2, 2)}, "patch_size"),
    ({"patch_size": (2, 2, 0)}, "patch_size"),
    ({"patch_size": (2, 2, 2.5)}, "patch_size"),
    ({"batch_size": 0}, "batch_size"),
    ({"batch_size": 1.5}, "batch_size"),
    ({"mirror_axes": (0, 0)}, "mirror_axes"),
    ({"mirror_axes": (-1,)}, "mirror_axes"),
])
def test_invalid_settings_fail_before_model_runs(overrides, message):
    options = {"patch_size": (2, 2, 2)}
    options.update(overrides)
    with pytest.raises(ValueError, match=message):
        SlidingWindowPredictor(torch.nn.Identity(), **options)


def test_array_settings_work_in_sliding_window_inference():
    predictor = SlidingWindowPredictor(torch.nn.Identity(), patch_size=np.array([2, 2, 2]),
                                       batch_size=np.int64(2), mirror_axes=np.array([0, 2]))
    probabilities = predictor.predict_probabilities(np.ones((3, 3, 3), dtype=np.float32))
    np.testing.assert_allclose(probabilities, np.ones((1, 3, 3, 3)))
