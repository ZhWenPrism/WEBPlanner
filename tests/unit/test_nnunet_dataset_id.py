import numpy as np
import pytest
from webplanner.data.nnunet_export import export_nnunet_dataset
from webplanner.types import Volume


@pytest.mark.parametrize("dataset_id", [True, 501.5, "501", 0, 1000])
def test_invalid_dataset_ids_do_not_create_output(tmp_path, dataset_id):
    image = Volume(np.zeros((2, 2, 2), dtype=np.uint8), (1., 1., 1.))
    with pytest.raises(ValueError, match="dataset ids"):
        export_nnunet_dataset([("case", image, image)], tmp_path,
                              dataset_id=dataset_id, file_ending=".npz")
    assert not list(tmp_path.iterdir())


def test_numpy_integer_ids_keep_the_existing_folder_format(tmp_path):
    image = Volume(np.zeros((2, 2, 2), dtype=np.uint8), (1., 1., 1.))
    root = export_nnunet_dataset([("case", image, image)], tmp_path,
                                dataset_id=np.int64(501), file_ending=".npz")
    assert root.name == "Dataset501_WEBPlanner"
    assert (root / "imagesTr/case_0000.npz").is_file()
