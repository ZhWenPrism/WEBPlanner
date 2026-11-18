import torch
from torch import nn

from webplanner.segmentation import trainer


def test_selected_checkpoint_contains_its_epoch_and_validation_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "_environment", lambda: {})
    config = trainer.TrainerConfig(epochs=1, iterations_per_epoch=1, val_iterations=1,
                                   batch_size=1, patch_size=(4, 4, 4), augment=False,
                                   amp=False, verbose=False, device="cpu", out_dir=str(tmp_path))
    model = trainer.SegmentationTrainer(config, nn.Conv3d(1, 5, 1))
    inputs = [(torch.zeros(4, 4, 4), torch.ones(4, 4, 4, dtype=torch.long))]
    model.fit(inputs, inputs)
    checkpoint = torch.load(tmp_path / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    assert len(checkpoint["history"]) == checkpoint["epoch"] == 1
    assert checkpoint["history"][-1]["epoch"] == checkpoint["best_epoch"]
    assert checkpoint["history"][-1]["val_dice_ema"] == checkpoint["best_ema"]
