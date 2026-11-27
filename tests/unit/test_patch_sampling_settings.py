import numpy as np
import pytest
from webplanner.data.dataset import PatchDataset


def cases():
    image = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    return [(image, np.zeros(image.shape, np.uint8))]


@pytest.mark.parametrize("name,value", [
    ("foreground_oversample", np.nan), ("foreground_oversample", -.1),
    ("foreground_oversample", 1.1), ("samples_per_epoch", 0),
    ("samples_per_epoch", -1), ("samples_per_epoch", 1.5), ("samples_per_epoch", True),
    ("max_locations", 0), ("max_locations", 1.5), ("seed", -1), ("seed", 1.5), ("seed", True),
])
def test_invalid_sampling_settings_are_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        PatchDataset(cases(), patch_size=2, **{name: value})


def test_valid_sampling_settings_preserve_deterministic_samples():
    options = dict(patch_size=2, samples_per_epoch=4, max_locations=2, seed=17,
                   foreground_oversample=0)
    left, right = PatchDataset(cases(), **options), PatchDataset(cases(), **options)
    assert len(left) == 4 and len(PatchDataset(cases(), patch_size=2)) == 1
    for index in range(4):
        for a, b in zip(left.sample(index), right.sample(index)):
            np.testing.assert_array_equal(a, b)
