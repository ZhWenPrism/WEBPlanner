import numpy as np

from webplanner.data.dataset import PatchDataset


def test_in_place_augmentation_does_not_change_the_case():
    image = np.ones((4, 4, 4), dtype=np.float32)
    label = np.zeros(image.shape, dtype=np.uint8)

    def transform(image_patch, label_patch, rng):
        image_patch += 2
        label_patch += 1
        return image_patch, label_patch

    dataset = PatchDataset([(image, label)], patch_size=4, transform=transform)
    for index in (0, 1, 0):
        image_patch, label_patch = dataset.sample(index)
        np.testing.assert_array_equal(image_patch, 3)
        np.testing.assert_array_equal(label_patch, 1)
    np.testing.assert_array_equal(image, 1)
    np.testing.assert_array_equal(label, 0)
