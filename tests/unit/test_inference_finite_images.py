import numpy as np
import pytest
import torch
from webplanner.segmentation.inference import SlidingWindowPredictor


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
    def forward(self, x):
        self.calls += 1
        return torch.cat((x[:, :1], -x[:, :1]), dim=1)


@pytest.mark.parametrize("image", [np.full((2, 2, 2), np.nan),
                                   np.full((1, 2, 2, 2), np.inf),
                                   np.full((2, 2, 2), -np.inf), np.empty((0, 2, 2))])
def test_invalid_images_fail_before_the_model_is_called(image):
    model = Model()
    predictor = SlidingWindowPredictor(model, patch_size=(2, 2, 2),
                                       mirror_axes=(), use_gaussian=False)
    with pytest.raises(ValueError, match="finite voxels"):
        predictor.predict_probabilities(image)
    assert model.calls == 0


def test_valid_images_keep_normalized_probabilities_and_grid_shape():
    image = np.linspace(-1, 1, 8, dtype=np.float32).reshape(2, 2, 2)
    predictor = SlidingWindowPredictor(Model(), patch_size=(2, 2, 2),
                                       mirror_axes=(), use_gaussian=False)
    probabilities = predictor.predict_probabilities(image)
    assert probabilities.shape == (2, *image.shape)
    np.testing.assert_allclose(probabilities.sum(axis=0), 1, atol=1e-6)
