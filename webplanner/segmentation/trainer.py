"""Patch-based training loop for the anatomy segmentation networks (nnU-Net-like recipe).

SGD with Nesterov momentum 0.99, initial learning rate 1e-2 with polynomial decay (exponent 0.9, stepped per
epoch), weight decay 3e-5, Dice + cross-entropy with deep supervision, gradient-norm clipping at 12, mixed precision
on CUDA. Every ``val_every`` epochs the model is evaluated on validation patches (per-class Dice from TP/FP/FN
accumulated over all patches, as nnU-Net's pseudo-Dice); an exponential moving average of the mean foreground Dice
selects ``checkpoint_best.pt``. ``checkpoint_last.pt`` allows resuming. Checkpoints carry the configuration and the
software environment; every epoch is appended to ``training_log.csv``.
"""
from __future__ import annotations

import csv
import dataclasses
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from ..data.dataset import PatchDataset, collate, infinite_loader
from ..data.transforms import default_pipeline, enlarged_size
from ..types import LABELS, Volume
from ..utils.io import write_json
from ..utils.repro import environment, get_logger, seed_everything
from .inference import SlidingWindowPredictor, predict_volume
from .losses import DeepSupervisionLoss, DiceCELoss
from .networks import build_network

_ENV: Optional[Dict[str, object]] = None


def _environment() -> Dict[str, object]:
    global _ENV
    if _ENV is None:          # importing the optional packages is slow; record once per process
        _ENV = environment()
    return _ENV


@dataclass
class TrainerConfig:
    """Training configuration (stored in every checkpoint)."""

    network: str = "residual_unet"
    network_kwargs: Dict[str, Any] = field(default_factory=dict)
    in_ch: int = 1
    n_classes: int = 5
    deep_supervision: bool = True
    epochs: int = 1000
    iterations_per_epoch: int = 250
    val_iterations: int = 50
    batch_size: int = 2
    patch_size: Tuple[int, int, int] = (128, 128, 128)
    lr: float = 1e-2
    momentum: float = 0.99
    nesterov: bool = True
    weight_decay: float = 3e-5
    poly_exponent: float = 0.9
    grad_clip: float = 12.0
    amp: bool = True
    weight_dice: float = 1.0
    weight_ce: float = 1.0
    batch_dice: bool = False
    val_every: int = 1
    ema_alpha: float = 0.9
    foreground_oversample: float = 0.33
    augment: bool = True
    num_workers: int = 0
    seed: int = 0
    deterministic: bool = True
    device: str = "auto"
    out_dir: str = "runs/segmentation"
    resume: bool = False
    save_every: int = 1
    verbose: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainerConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        kw = {k: v for k, v in d.items() if k in names}
        if "patch_size" in kw:
            kw["patch_size"] = tuple(int(p) for p in kw["patch_size"])
        return cls(**kw)


def _torch_load(path: Union[str, Path], device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:                                   # torch < 1.13 has no weights_only
        return torch.load(path, map_location=device)


def _grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):                  # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


class SegmentationTrainer:
    """Train a segmentation network on patches and save checkpoints under ``config.out_dir``.

    ``model`` overrides the network built from ``config.network``. :meth:`fit` accepts torch datasets returning
    ``(image (1, z, y, x), label (z, y, x))`` or sequences of ``(image, label)`` pairs, which are wrapped in a
    :class:`~webplanner.data.dataset.PatchDataset` (with augmentation for training when ``config.augment``).
    """

    CSV_NAME = "training_log.csv"

    def __init__(self, config: Optional[TrainerConfig] = None, model: Optional[nn.Module] = None) -> None:
        self.config = cfg = config or TrainerConfig()
        seed_everything(cfg.seed, deterministic=cfg.deterministic)
        if cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.device)
        self.use_amp = bool(cfg.amp and self.device.type == "cuda")
        self.out_dir = Path(cfg.out_dir)
        if model is None:
            kw = dict(cfg.network_kwargs)
            if cfg.network == "swin_unetr":
                kw.setdefault("img_size", tuple(cfg.patch_size))
            model = build_network(cfg.network, in_ch=cfg.in_ch, n_classes=cfg.n_classes,
                                  deep_supervision=cfg.deep_supervision, **kw)
        self.model = model.to(self.device)
        base = DiceCELoss(cfg.weight_dice, cfg.weight_ce, batch_dice=cfg.batch_dice)
        self.loss = DeepSupervisionLoss(base).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                                         nesterov=cfg.nesterov, weight_decay=cfg.weight_decay)
        self.scaler = _grad_scaler(self.use_amp)
        self.epoch = 0                      # next epoch to run
        self.ema: Optional[float] = None
        self.best_ema = -math.inf
        self.best_epoch = -1
        self.history: List[Dict[str, float]] = []
        self.logger = get_logger("webplanner.segmentation")

    # ---------------------------------------------------------------------------------------------------------------
    # data
    # ---------------------------------------------------------------------------------------------------------------
    def _as_dataset(self, data, train: bool) -> Dataset:
        if isinstance(data, Dataset):
            return data
        cfg = self.config
        transform = sample_size = None
        if train and cfg.augment:
            transform = default_pipeline(cfg.patch_size)
            sample_size = enlarged_size(cfg.patch_size)
        return PatchDataset(data, cfg.patch_size, transform=transform, sample_size=sample_size,
                            foreground_oversample=cfg.foreground_oversample, seed=cfg.seed + (0 if train else 1))

    # ---------------------------------------------------------------------------------------------------------------
    # optimisation
    # ---------------------------------------------------------------------------------------------------------------
    def poly_lr(self, epoch: int) -> float:
        cfg = self.config
        return cfg.lr * (1.0 - epoch / max(cfg.epochs, 1)) ** cfg.poly_exponent

    def train_step(self, image: torch.Tensor, label: torch.Tensor) -> float:
        image = image.to(self.device, non_blocking=True)
        label = label.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            loss = self.loss(self.model(image), label)
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
        return float(loss.detach())

    @torch.no_grad()
    def validate(self, dataset: Dataset) -> Tuple[float, np.ndarray]:
        """Mean loss and per-class Dice (classes 1..C-1) over ``val_iterations`` batches of ``dataset``."""
        cfg = self.config
        n = cfg.val_iterations * cfg.batch_size
        if not isinstance(dataset, PatchDataset):
            n = min(n, len(dataset))
        loader = DataLoader(Subset(dataset, range(n)), batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, collate_fn=collate)
        self.model.eval()
        k = cfg.n_classes - 1
        tp, fp, fn = np.zeros(k), np.zeros(k), np.zeros(k)
        losses = []
        for image, label in loader:
            image, label = image.to(self.device), label.to(self.device)
            if label.ndim == image.ndim:
                label = label[:, 0]
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                out = self.model(image)
            if isinstance(out, (list, tuple)):
                out = out[0]
            losses.append(float(self.loss(out.float(), label)))
            pred = out.argmax(1)
            for c in range(1, cfg.n_classes):
                p, g = pred == c, label == c
                tp[c - 1] += float((p & g).sum())
                fp[c - 1] += float((p & ~g).sum())
                fn[c - 1] += float((~p & g).sum())
        denom = 2 * tp + fp + fn
        dice = np.divide(2 * tp, denom, out=np.full(k, np.nan), where=denom > 0)
        return float(np.mean(losses)) if losses else float("nan"), dice

    # ---------------------------------------------------------------------------------------------------------------
    # checkpoints and logs
    # ---------------------------------------------------------------------------------------------------------------
    def save_checkpoint(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "scaler": self.scaler.state_dict(), "epoch": self.epoch, "ema": self.ema, "best_ema": self.best_ema,
                 "best_epoch": self.best_epoch, "history": self.history, "config": self.config.to_dict(),
                 "environment": _environment()}
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)
        return path

    def load_checkpoint(self, path: Union[str, Path], weights_only: bool = False) -> Dict[str, Any]:
        """Restore the model (and, unless ``weights_only``, optimiser, scaler and training state)."""
        ckpt = _torch_load(path, self.device)
        self.model.load_state_dict(ckpt["model"])
        if not weights_only:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if ckpt.get("scaler"):
                self.scaler.load_state_dict(ckpt["scaler"])
            self.epoch = int(ckpt["epoch"])
            self.ema, self.best_ema = ckpt.get("ema"), float(ckpt.get("best_ema", -math.inf))
            self.best_epoch = int(ckpt.get("best_epoch", -1))
            self.history = list(ckpt.get("history", []))
        return ckpt

    @classmethod
    def from_checkpoint(cls, path: Union[str, Path], device: Optional[str] = None) -> "SegmentationTrainer":
        """Rebuild the network from the configuration stored in a checkpoint and load its weights."""
        ckpt = _torch_load(path, torch.device("cpu"))
        cfg = TrainerConfig.from_dict(ckpt["config"])
        if device is not None:
            cfg.device = device
        trainer = cls(cfg)
        trainer.model.load_state_dict(ckpt["model"])
        trainer.model.eval()
        return trainer

    def _class_names(self) -> List[str]:
        n = self.config.n_classes
        return [LABELS[c] if n == len(LABELS) else f"class_{c}" for c in range(1, n)]

    def _log_row(self, row: Dict[str, float]) -> None:
        path = self.out_dir / self.CSV_NAME
        fields = ["epoch", "lr", "train_loss", "val_loss", "val_dice_mean", "val_dice_ema"] + \
                 [f"dice_{n}" for n in self._class_names()] + ["epoch_seconds"]
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in fields})

    # ---------------------------------------------------------------------------------------------------------------
    # main loop
    # ---------------------------------------------------------------------------------------------------------------
    def fit(self, train_dataset, val_dataset=None) -> Dict[str, Any]:
        """Train for ``config.epochs`` epochs (continuing from ``checkpoint_last.pt`` when ``config.resume``)."""
        cfg = self.config
        train_ds = self._as_dataset(train_dataset, train=True)
        val_ds = self._as_dataset(val_dataset, train=False) if val_dataset is not None else None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        last, best = self.out_dir / "checkpoint_last.pt", self.out_dir / "checkpoint_best.pt"
        if cfg.resume and last.exists():
            self.load_checkpoint(last)
            if cfg.verbose:
                self.logger.info("resumed from %s at epoch %d", last, self.epoch)
        write_json(cfg.to_dict(), self.out_dir / "config.json")
        write_json(_environment(), self.out_dir / "environment.json")

        loader = infinite_loader(train_ds, cfg.batch_size, num_workers=cfg.num_workers, seed=cfg.seed,
                                 start=self.epoch * cfg.iterations_per_epoch * cfg.batch_size,
                                 pin_memory=self.device.type == "cuda")
        names = self._class_names()
        for epoch in range(self.epoch, cfg.epochs):
            t0 = time.time()
            lr = self.poly_lr(epoch)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self.model.train()
            losses = [self.train_step(*next(loader)) for _ in range(cfg.iterations_per_epoch)]
            self.epoch = epoch + 1
            row: Dict[str, float] = {"epoch": epoch, "lr": lr, "train_loss": float(np.mean(losses))}
            if val_ds is not None and ((epoch + 1) % cfg.val_every == 0 or epoch == cfg.epochs - 1):
                val_loss, dice = self.validate(val_ds)
                mean = float(np.nanmean(dice)) if np.isfinite(dice).any() else float("nan")
                if np.isfinite(mean):
                    self.ema = mean if self.ema is None else cfg.ema_alpha * self.ema + (1 - cfg.ema_alpha) * mean
                row.update({"val_loss": val_loss, "val_dice_mean": mean,
                            "val_dice_ema": self.ema if self.ema is not None else float("nan")})
                row.update({f"dice_{n}": float(d) for n, d in zip(names, dice)})
                if self.ema is not None and self.ema > self.best_ema:
                    self.best_ema, self.best_epoch = self.ema, epoch
                    self.save_checkpoint(best)
            row["epoch_seconds"] = time.time() - t0
            self.history.append(row)
            if self.epoch % cfg.save_every == 0 or self.epoch == cfg.epochs:
                self.save_checkpoint(last)
            self._log_row(row)
            if cfg.verbose:
                self.logger.info("epoch %d/%d lr %.5f train %.4f val dice %s ema %s", epoch + 1, cfg.epochs, lr,
                                 row["train_loss"], _fmt(row.get("val_dice_mean")), _fmt(row.get("val_dice_ema")))
        return {"epochs_run": self.epoch, "best_ema": self.best_ema, "best_epoch": self.best_epoch,
                "history": self.history, "out_dir": str(self.out_dir)}

    # ---------------------------------------------------------------------------------------------------------------
    # inference
    # ---------------------------------------------------------------------------------------------------------------
    def predict(self, volume: Union[Volume, np.ndarray], **kw) -> Union[Volume, np.ndarray]:
        """Sliding-window prediction with ``config.patch_size``; keywords go to :class:`SlidingWindowPredictor`."""
        kw.setdefault("patch_size", self.config.patch_size)
        if isinstance(volume, Volume):
            return predict_volume(self.model, volume, **kw)
        return SlidingWindowPredictor(self.model, **kw).predict(volume)


def _fmt(x: Optional[float]) -> str:
    return "-" if x is None or not np.isfinite(x) else f"{x:.4f}"
