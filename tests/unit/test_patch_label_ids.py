import numpy as np
import pytest

from webplanner.data.dataset import PatchDataset


@pytest.mark.parametrize("bad", [-1, 1.5, np.nan, np.inf, np.uint64(2**63)])
def test_patch_dataset_rejects_labels_that_cannot_be_cast_exactly(bad):
    labels = np.full((2, 2, 2), bad)
    with pytest.raises(ValueError, match="integer class IDs"):
        PatchDataset([(np.zeros(labels.shape), labels)], patch_size=2)


def test_custom_label_ids_above_uint8_are_preserved():
    labels = np.full((2, 2, 2), 256, dtype=np.uint16)
    dataset = PatchDataset([(np.zeros(labels.shape), labels)], patch_size=2, foreground_classes=(256,))
    np.testing.assert_array_equal(dataset.sample(0)[1], labels)
