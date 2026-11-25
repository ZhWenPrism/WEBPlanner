import numpy as np
import pytest
from webplanner.data.nnunet_export import _check_folds, export_nnunet_dataset
from webplanner.types import Volume


@pytest.mark.parametrize("fold", [
    {"train": ["a", "a"], "val": ["b"]},
    {"train": ["a"], "val": ["b", "b"]},
])
def test_repeated_fold_members_are_rejected_before_writing(tmp_path, fold):
    image = Volume(np.zeros((2, 2, 2), dtype=np.uint8), (1., 1., 1.))
    with pytest.raises(ValueError, match="duplicate case ids"):
        export_nnunet_dataset([("a", image, image), ("b", image, image)], tmp_path,
                              folds=[fold], file_ending=".npz")
    assert not list(tmp_path.iterdir())


def test_valid_fold_members_retain_their_order():
    folds = [{"train": ["c", "a"], "val": ["b"]}]
    assert _check_folds(folds, ["a", "b", "c"]) == folds
