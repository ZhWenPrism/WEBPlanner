from dataclasses import asdict
import numpy as np
import pytest
from webplanner.data.preprocessing import Preprocessor
from webplanner.types import Volume


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_foreground_cannot_create_invalid_fitted_statistics(value):
    array = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    array.flat[0] = value
    with pytest.raises(ValueError, match="finite voxels"):
        Preprocessor().fit([Volume(array, (1., 1., 1.))], [np.ones(array.shape)])


def test_failed_refit_preserves_the_previous_fitted_state():
    volume = Volume(np.arange(8, dtype=np.float32).reshape(2, 2, 2), (1., 1., 1.))
    preprocessor = Preprocessor().fit([volume], [np.ones(volume.shape)])
    before = asdict(preprocessor)
    bad = Volume(np.full(volume.shape, np.nan), volume.spacing)
    with pytest.raises(ValueError, match="finite voxels"):
        preprocessor.fit([bad], [np.ones(volume.shape)])
    assert asdict(preprocessor) == before
