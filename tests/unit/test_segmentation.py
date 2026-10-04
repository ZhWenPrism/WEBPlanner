"""Segmentation networks, losses, trainer, sliding-window inference and post-processing.

Tests that need PyTorch call ``pytest.importorskip("torch")`` first and are skipped where it is not installed.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from webplanner.data.synthetic import make_phantom
from webplanner.segmentation.postprocess import postprocess
from webplanner.types import BRANCH, NECK, PARENT, SAC, Volume

ROOT = Path(__file__).resolve().parents[2]
TINY = dict(features=(8, 16, 32))


@lru_cache(maxsize=None)
def _phantom(seed: int = 0, **kw):
    return make_phantom(shape=(48, 48, 48), seed=seed, **kw)


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-8)


def _csv_rows(path: Path):
    return list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))


# --------------------------------------------------------------------------------------------------------------------
# post-processing and packaging (no torch)
# --------------------------------------------------------------------------------------------------------------------
def test_postprocess_keeps_clean_phantom_unchanged():
    ph = _phantom()
    np.testing.assert_array_equal(postprocess(ph.labels.array, ph.labels.spacing), ph.labels.array)


def test_postprocess_removes_disconnected_sac_blob_and_vessel_island():
    ph = _phantom()
    lab = ph.labels.array.copy()
    assert (lab[:8, :8, :8] == 0).all() and (lab[40:, 40:, :4] == 0).all()
    lab[1:6, 1:6, 1:6] = SAC                      # spurious sac component
    lab[2:4, 2:4, 6] = NECK                       # touching it, so labels 1 and 4 form one component
    lab[42:44, 42:44, 1:3] = BRANCH               # 8 voxels = 0.216 mm^3 vessel island
    out = postprocess(lab, ph.labels.spacing, min_vessel_mm3=1.0)
    np.testing.assert_array_equal(out, ph.labels.array)
    kept = postprocess(lab, ph.labels.spacing, min_vessel_mm3=0.0)
    assert (kept[42:44, 42:44, 1:3] == BRANCH).all()


def test_postprocess_relabels_landing_zone_away_from_arteries():
    ph = _phantom()
    lab = ph.labels.array.copy()
    sac = np.argwhere(lab == SAC)
    top = sac[:, 0].max()                         # the aneurysm axis is array z, the dome is the top
    dome = (lab == SAC) & (np.arange(48)[:, None, None] >= top - 1)
    lab[dome] = NECK
    out = postprocess(lab, ph.labels.spacing)
    assert (out[dome] == SAC).all()
    np.testing.assert_array_equal(out == NECK, ph.labels.array == NECK)
    assert (out == PARENT).sum() == (lab == PARENT).sum()


def test_postprocess_accepts_volume():
    ph = _phantom()
    out = postprocess(ph.labels)
    assert isinstance(out, Volume) and out.spacing == ph.labels.spacing and out.array.dtype == np.uint8


def test_segmentation_package_imports_without_torch():
    code = ("import sys; import webplanner.segmentation as s, webplanner.segmentation.networks; "
            "assert 'torch' not in sys.modules, 'torch imported'; assert callable(s.postprocess)")
    res = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_nnunet_trainer_module_without_nnunet(tmp_path):
    from webplanner.segmentation import nnunet_trainer as m

    if not m.NNUNET_AVAILABLE:
        with pytest.raises(ImportError, match="nnunetv2"):
            m.nnUNetTrainerWEBPlanner()
    stub = m.install_trainer(tmp_path)
    assert "nnUNetTrainerWEBPlanner" in stub.read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------------------------------------
# networks
# --------------------------------------------------------------------------------------------------------------------
def test_residual_unet_deep_supervision_shapes():
    torch = pytest.importorskip("torch")
    from webplanner.segmentation.networks import ResidualUNet3D

    net = ResidualUNet3D(1, 5, features=(8, 16, 32), deep_supervision=True)
    x = torch.randn(2, 1, 32, 32, 32)
    outs = net.train()(x)
    assert isinstance(outs, list) and [tuple(o.shape) for o in outs] == [(2, 5, 32, 32, 32), (2, 5, 16, 16, 16)]
    with torch.no_grad():
        y = net.eval()(x)
        assert tuple(y.shape) == (2, 5, 32, 32, 32)
        assert tuple(net(torch.randn(1, 1, 30, 28, 34)).shape) == (1, 5, 30, 28, 34)
    deep = ResidualUNet3D(1, 3, features=(4, 8, 8, 8), blocks_per_stage=1, strides=[1, 2, (1, 2, 2), 2])
    shapes = [tuple(o.shape[2:]) for o in deep.train()(torch.randn(1, 1, 16, 32, 32))]
    assert shapes == [(16, 32, 32), (8, 16, 16), (8, 8, 8)]


def test_build_network_and_unet3d():
    torch = pytest.importorskip("torch")
    from webplanner.segmentation.networks import ResidualUNet3D, UNet3D, build_network

    assert isinstance(build_network("residual_unet", **TINY), ResidualUNet3D)
    net = build_network("unet3d", in_ch=1, n_classes=5, deep_supervision=False, **TINY)
    assert isinstance(net, UNet3D)
    assert tuple(net.train()(torch.randn(1, 1, 32, 32, 32)).shape) == (1, 5, 32, 32, 32)
    with pytest.raises(ValueError):
        build_network("vnet")


def test_swin_unetr_builder():
    torch = pytest.importorskip("torch")
    pytest.importorskip("monai")
    from webplanner.segmentation.networks import build_network

    net = build_network("swin_unetr", in_ch=1, n_classes=5, img_size=64, feature_size=12, depths=(1, 1, 1, 1))
    with torch.no_grad():
        assert tuple(net.eval()(torch.randn(1, 1, 64, 64, 64)).shape) == (1, 5, 64, 64, 64)
    for bad in ((96, 64, 80), 32):
        with pytest.raises(ValueError):
            build_network("swin_unetr", img_size=bad)


# --------------------------------------------------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------------------------------------------------
def test_dice_losses_perfect_and_random_prediction():
    torch = pytest.importorskip("torch")
    from webplanner.segmentation.losses import DiceCELoss, DiceLoss

    g = torch.Generator().manual_seed(0)
    target = torch.randint(0, 5, (2, 12, 12, 12), generator=g)
    perfect = torch.nn.functional.one_hot(target, 5).permute(0, 4, 1, 2, 3).float() * 40.0
    for loss in (DiceLoss(), DiceLoss(include_background=True), DiceLoss(batch_dice=True), DiceCELoss()):
        assert float(loss(perfect, target)) < 1e-4
    assert float(DiceLoss()(perfect, target[:, None])) < 1e-4          # singleton channel accepted
    noise = torch.randn(2, 5, 12, 12, 12, generator=g, requires_grad=True)
    value = DiceCELoss(weight_dice=1.0, weight_ce=0.5)(noise, target)
    assert float(value) > 0.5
    value.backward()
    assert noise.grad is not None and torch.isfinite(noise.grad).all()
    empty = torch.zeros(1, 12, 12, 12, dtype=torch.long)                 # absent classes score Dice 1
    assert float(DiceLoss()(torch.nn.functional.one_hot(empty, 5).permute(0, 4, 1, 2, 3).float() * 40, empty)) < 1e-4


def test_deep_supervision_loss_weights_and_target_downsampling():
    torch = pytest.importorskip("torch")
    from webplanner.segmentation.losses import DeepSupervisionLoss, DiceCELoss, downsample_target

    assert np.allclose(DeepSupervisionLoss.nnunet_weights(3), [2 / 3, 1 / 3, 0.0])
    assert DeepSupervisionLoss.nnunet_weights(1) == [1.0]
    target = torch.zeros(2, 16, 16, 16, dtype=torch.long)
    target[:, 4:12, 4:12, 4:12] = 1
    target[:, 6:10, 6:10, 6:10] = 3
    small = downsample_target(target, (8, 8, 8))
    assert tuple(small.shape) == (2, 8, 8, 8) and set(small.unique().tolist()) <= {0, 1, 3}

    def logits(t):
        return torch.nn.functional.one_hot(t, 5).permute(0, 4, 1, 2, 3).float() * 40.0

    ds = DeepSupervisionLoss(DiceCELoss())
    outs = [logits(target), logits(small), torch.randn(2, 5, 4, 4, 4)]   # the last scale has weight 0
    assert float(ds(outs, target)) < 1e-3
    assert float(ds(logits(target), target)) < 1e-3
    assert float(DeepSupervisionLoss(DiceCELoss(), weights=[0.5, 0.5])(outs[:2], target)) < 1e-3


# --------------------------------------------------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------------------------------------------------
def _tiny_net(seed: int = 0):
    import torch
    from webplanner.segmentation.networks import ResidualUNet3D

    torch.manual_seed(seed)
    return ResidualUNet3D(1, 5, features=(4, 8, 16), blocks_per_stage=1).eval()


def test_gaussian_importance_map():
    pytest.importorskip("torch")
    from webplanner.segmentation.inference import gaussian_importance_map, window_starts

    g = gaussian_importance_map((16, 16, 16))
    assert g.shape == (16, 16, 16) and np.isclose(g.max(), 10.0) and (g > 0).all()
    np.testing.assert_allclose(g, g[::-1, ::-1, ::-1], rtol=1e-6)
    assert window_starts(40, 32, 0.5) == [0, 8] and window_starts(20, 32, 0.5) == [0]
    assert window_starts(96, 32, 0.5) == [0, 16, 32, 48, 64]


def test_sliding_window_probabilities_and_padding():
    torch = pytest.importorskip("torch")
    from webplanner.segmentation.inference import SlidingWindowPredictor

    rng = np.random.default_rng(0)
    pred = SlidingWindowPredictor(_tiny_net(), patch_size=(16, 16, 16), mirror_axes=(0, 2), batch_size=3)
    image = rng.normal(size=(24, 20, 28)).astype(np.float32)
    labels, probs = pred.predict(image, return_probabilities=True)
    assert probs.shape == (5, 24, 20, 28) and probs.dtype == np.float32
    np.testing.assert_allclose(probs.sum(0), 1.0, atol=1e-5)
    assert labels.shape == (24, 20, 28) and labels.dtype == np.uint8
    np.testing.assert_array_equal(labels, probs.argmax(0))
    small = SlidingWindowPredictor(_tiny_net(), patch_size=(16, 16, 16), mirror_axes=None)
    p_small = small.predict_probabilities(rng.normal(size=(10, 12, 9)).astype(np.float32))
    assert p_small.shape == (5, 10, 12, 9)
    np.testing.assert_allclose(p_small.sum(0), 1.0, atol=1e-5)


def test_ensemble_and_predict_volume():
    torch = pytest.importorskip("torch")
    from webplanner.segmentation.inference import SlidingWindowPredictor, ensemble_predict, predict_volume

    ph = _phantom()
    vol = Volume(_zscore(ph.image.array)[8:40, 8:40, 8:40].copy(), ph.image.spacing)
    nets = [_tiny_net(0), _tiny_net(1)]
    kw = dict(patch_size=(16, 16, 16), mirror_axes=(1,))
    ens = ensemble_predict(nets, vol, **kw)
    single = [SlidingWindowPredictor(n, **kw).predict_probabilities(vol.array) for n in nets]
    np.testing.assert_allclose(ens, (single[0] + single[1]) / 2, atol=1e-5)
    seg, probs = predict_volume(nets, vol, return_probabilities=True, postprocess=True, **kw)
    assert isinstance(seg, Volume) and seg.array.dtype == np.uint8 and seg.shape == vol.shape
    assert seg.spacing == vol.spacing and probs.shape == (5,) + vol.shape
    pre = predict_volume(nets[0], Volume(ph.image.array[8:40, 8:40, 8:40].copy(), ph.image.spacing),
                         preprocessor=lambda v: _zscore(v.array), **kw)
    assert pre.shape == vol.shape


# --------------------------------------------------------------------------------------------------------------------
# trainer
# --------------------------------------------------------------------------------------------------------------------
def test_trainer_fit_checkpoints_resume_and_predict(tmp_path):
    torch = pytest.importorskip("torch")
    from webplanner.segmentation.trainer import SegmentationTrainer, TrainerConfig

    phantoms = [_phantom(0), _phantom(1, width_1=5.0, width_2=4.6, height=4.5, neck_width=3.4, azimuth_deg=40.0)]
    cases = [(_zscore(p.image.array), p.labels.array) for p in phantoms]
    cfg = TrainerConfig(network_kwargs=dict(features=(8, 16, 32), blocks_per_stage=1), epochs=3, iterations_per_epoch=2,
                        val_iterations=1, batch_size=2, patch_size=(32, 32, 32), device="cpu",
                        out_dir=str(tmp_path / "run"), verbose=False, seed=1)
    trainer = SegmentationTrainer(cfg)
    summary = trainer.fit(cases, cases)
    run = tmp_path / "run"
    for name in ("checkpoint_last.pt", "checkpoint_best.pt", "training_log.csv", "config.json", "environment.json"):
        assert (run / name).exists(), name
    assert summary["epochs_run"] == 3 and summary["best_epoch"] >= 0
    losses = [h["train_loss"] for h in summary["history"]]
    assert len(losses) == 3 and all(np.isfinite(losses))
    rows = _csv_rows(run / "training_log.csv")
    assert len(rows) == 3 and "dice_aneurysm_sac" in rows[0] and rows[-1]["val_dice_ema"] != ""
    lrs = [float(r["lr"]) for r in rows]
    assert np.isclose(lrs[0], 1e-2) and lrs[0] > lrs[1] > lrs[2] > 0
    ckpt = torch.load(run / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 3 and ckpt["config"]["patch_size"] == (32, 32, 32) and "packages" in ckpt["environment"]

    resumed = SegmentationTrainer(TrainerConfig(**{**cfg.to_dict(), "epochs": 4, "resume": True}))
    summary2 = resumed.fit(cases, cases)
    assert summary2["epochs_run"] == 4 and len(summary2["history"]) == 4
    assert len(_csv_rows(run / "training_log.csv")) == 4

    best = SegmentationTrainer.from_checkpoint(run / "checkpoint_best.pt", device="cpu")
    vol = Volume(cases[0][0], phantoms[0].image.spacing)
    seg = best.predict(vol, mirror_axes=None)
    assert isinstance(seg, Volume) and seg.shape == (48, 48, 48) and seg.array.dtype == np.uint8
