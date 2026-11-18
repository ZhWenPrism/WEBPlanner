import pytest
import torch

from webplanner.segmentation.losses import DiceCELoss, DiceLoss


@pytest.mark.parametrize("loss", [DiceLoss(), DiceCELoss()])
@pytest.mark.parametrize("value", [0.5, -1., 3., float("nan")])
def test_invalid_labels_fail_before_loss_computation(loss, value):
    logits = torch.zeros(1, 3, 2, 2, 2)
    target = torch.zeros(1, 2, 2, 2)
    target[0, 0, 0, 0] = value
    with pytest.raises(ValueError, match="integer class indices"):
        loss(logits, target)


def test_integral_float_and_singleton_channel_targets_preserve_the_loss():
    logits = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 2, 2) / 10
    target = torch.ones(1, 2, 2, 2, dtype=torch.long)
    loss = DiceCELoss()
    torch.testing.assert_close(loss(logits, target), loss(logits, target[:, None].float()))
