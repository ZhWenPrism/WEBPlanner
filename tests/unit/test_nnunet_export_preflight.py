import numpy as np
import pytest

from webplanner.data.nnunet_export import export_nnunet_dataset
from webplanner.types import Volume


def _volume(**metadata):
    return Volume(np.zeros((2, 2, 2), dtype=np.uint8), (1., 1., 1.), **metadata)


@pytest.mark.parametrize("metadata, field", [
    ({"origin": (1., 0., 0.)}, "origin"),
    ({"direction": (0., -1., 0., 1., 0., 0., 0., 0., 1.)}, "direction"),
])
def test_misaligned_labels_are_rejected_before_writing(tmp_path, metadata, field):
    image = _volume()
    cases = [("valid", image, image), ("misaligned", image, _volume(**metadata))]
    with pytest.raises(ValueError, match=field):
        export_nnunet_dataset(cases, tmp_path, file_ending=".npz")
    assert not list(tmp_path.iterdir())


def test_invalid_folds_leave_output_untouched(tmp_path):
    image = _volume()
    with pytest.raises(ValueError, match="both train and val"):
        export_nnunet_dataset([("case", image, image)], tmp_path,
                              folds=[{"train": ["case"], "val": ["case"]}], file_ending=".npz")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("case_id, options, message", [
    ("../case", {"file_ending": ".npz"}, "case_id"),
    ("a\\case", {"file_ending": ".npz"}, "case_id"),
    ("case", {"file_ending": ".npz", "dataset_name": "site/../../escape"}, "dataset_name"),
    ("case", {"file_ending": ".bad/../../out.npz"}, "file_ending"),
])
def test_export_rejects_path_components(tmp_path, case_id, options, message):
    image = _volume()
    with pytest.raises(ValueError, match=message):
        export_nnunet_dataset([(case_id, image, image)], tmp_path, **options)
    assert not list(tmp_path.iterdir())
