import numpy as np
import pytest

from webplanner.ranking import SacImageScorer


@pytest.mark.parametrize("target", ["width_step", "oversized"])
@pytest.mark.parametrize("values", [[0., np.nan], [0., np.inf], [0., -np.inf], [[0.], [1.]], []])
def test_invalid_targets_cannot_be_converted_to_valid_binary_labels(target, values):
    with pytest.raises(ValueError, match="targets"):
        SacImageScorer(target=target)._y(values)


@pytest.mark.parametrize("n_labels,n_groups", [(3, 3), (4, 3), (3, 4)])
def test_oof_inputs_must_align_before_fitting(monkeypatch, n_labels, n_groups):
    monkeypatch.setattr(SacImageScorer, "fit", lambda self, *a: self)
    monkeypatch.setattr(SacImageScorer, "predict", lambda self, values: np.zeros(len(values)))
    with pytest.raises(ValueError, match="aligned rows"):
        SacImageScorer().fit_predict_oof(np.zeros((4, 2)), np.zeros(n_labels),
                                         np.arange(n_groups), n_splits=2)


def test_signed_width_targets_keep_their_declared_meaning():
    raw = np.array([-2., 0., 1.])
    np.testing.assert_array_equal(SacImageScorer(target="width_step")._y(raw), raw)
    np.testing.assert_array_equal(SacImageScorer(target="oversized")._y(raw), [0., 0., 1.])
