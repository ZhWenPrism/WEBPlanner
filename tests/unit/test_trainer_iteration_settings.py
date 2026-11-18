import pytest

from webplanner.segmentation.trainer import TrainerConfig


@pytest.mark.parametrize("name", ["epochs", "iterations_per_epoch", "val_iterations", "batch_size", "val_every", "save_every"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_training_iteration_counts_fail_at_configuration(name, value):
    with pytest.raises(ValueError, match=name):
        TrainerConfig.from_dict({name: value})


def test_valid_checkpoint_configuration_round_trips():
    config = TrainerConfig(epochs=2, iterations_per_epoch=1, val_iterations=1, save_every=2)
    assert TrainerConfig.from_dict(config.to_dict()) == config
