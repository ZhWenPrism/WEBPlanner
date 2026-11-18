import pytest
import torch

from webplanner.segmentation.losses import DeepSupervisionLoss, DiceCELoss


@pytest.mark.parametrize("weights", [[], [0., 0.], [-1., 2.], [float("nan"), 1.], [float("inf")]])
def test_invalid_weight_lists_are_rejected(weights):
    with pytest.raises(ValueError, match="weights"):
        DeepSupervisionLoss(DiceCELoss(), weights)


def test_extra_weights_and_empty_outputs_cannot_be_silently_ignored():
    logits = torch.zeros(1, 3, 2, 2, 2)
    labels = torch.ones(1, 2, 2, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="outputs"):
        DeepSupervisionLoss(DiceCELoss(), [.4, .6])([logits], labels)
    with pytest.raises(ValueError, match="at least one output"):
        DeepSupervisionLoss(DiceCELoss())([], labels)
