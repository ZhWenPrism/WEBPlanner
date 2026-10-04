"""Segmentation losses: soft Dice, Dice + cross-entropy and nnU-Net deep supervision.

Inputs are logits ``(B, C, z, y, x)`` and integer targets ``(B, z, y, x)`` (a singleton channel axis is accepted).
The Dice terms are computed in float32 even under mixed precision.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import nn


def _target_index(target: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    if target.ndim == logits.ndim:
        if target.shape[1] != 1:
            raise ValueError("target must hold class indices, not one-hot channels")
        target = target[:, 0]
    return target.long()


def downsample_target(target: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Nearest-neighbour resize of an index target ``(B, z, y, x)`` to ``size``."""
    if tuple(target.shape[-3:]) == tuple(size):
        return target
    return F.interpolate(target[:, None].float(), size=tuple(size), mode="nearest")[:, 0].long()


class DiceLoss(nn.Module):
    """Soft multi-class Dice loss, ``1 - mean_c (2 I_c + s_nr) / (P_c + G_c + s_dr)``.

    Args:
        include_background: include class 0 in the mean.
        batch_dice: sum over the batch before computing Dice (nnU-Net ``batch_dice``) instead of per sample.
        smooth_nr, smooth_dr: smoothing terms of numerator and denominator; a class absent from target and
            prediction scores 1.
        apply_softmax: apply softmax to the input (set False for probabilities).
    """

    def __init__(self, include_background: bool = False, batch_dice: bool = False, smooth_nr: float = 1e-5,
                 smooth_dr: float = 1e-5, apply_softmax: bool = True) -> None:
        super().__init__()
        self.include_background, self.batch_dice = include_background, batch_dice
        self.smooth_nr, self.smooth_dr, self.apply_softmax = smooth_nr, smooth_dr, apply_softmax

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = _target_index(target, logits)
        probs = torch.softmax(logits.float(), dim=1) if self.apply_softmax else logits.float()
        onehot = torch.zeros_like(probs).scatter_(1, target[:, None], 1.0)
        if not self.include_background:
            probs, onehot = probs[:, 1:], onehot[:, 1:]
        dims = (0,) + tuple(range(2, probs.ndim)) if self.batch_dice else tuple(range(2, probs.ndim))
        inter = (probs * onehot).sum(dims)
        denom = probs.sum(dims) + onehot.sum(dims)
        dice = (2.0 * inter + self.smooth_nr) / (denom + self.smooth_dr).clamp_min(1e-8)
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    """``L = w_dice * L_Dice + w_ce * L_CE`` (nnU-Net default with both weights 1)."""

    def __init__(self, weight_dice: float = 1.0, weight_ce: float = 1.0, include_background: bool = False,
                 batch_dice: bool = False, smooth_nr: float = 1e-5, smooth_dr: float = 1e-5,
                 class_weights: Optional[Sequence[float]] = None) -> None:
        super().__init__()
        self.weight_dice, self.weight_ce = weight_dice, weight_ce
        self.dice = DiceLoss(include_background, batch_dice, smooth_nr, smooth_dr)
        self.register_buffer("class_weights", None if class_weights is None
                             else torch.as_tensor(class_weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = _target_index(target, logits)
        loss = logits.new_zeros((), dtype=torch.float32)
        if self.weight_dice:
            loss = loss + self.weight_dice * self.dice(logits, target)
        if self.weight_ce:
            loss = loss + self.weight_ce * F.cross_entropy(logits.float(), target, weight=self.class_weights)
        return loss


class DeepSupervisionLoss(nn.Module):
    """Weighted sum of a loss over deep-supervision outputs (full resolution first).

    Default weights are nnU-Net's: ``1 / 2**i``, the coarsest scale set to 0, normalised to sum to 1. Targets are
    resized to each output with nearest-neighbour interpolation. A plain tensor input is passed straight to ``loss``.
    """

    def __init__(self, loss: nn.Module, weights: Optional[Sequence[float]] = None) -> None:
        super().__init__()
        self.loss = loss
        self.weights = None if weights is None else [float(w) for w in weights]

    @staticmethod
    def nnunet_weights(n: int) -> List[float]:
        w = [1.0 / 2 ** i for i in range(n)]
        if n > 1:
            w[-1] = 0.0
        s = sum(w)
        return [x / s for x in w]

    def forward(self, outputs: Union[torch.Tensor, Sequence[torch.Tensor]], target: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return self.loss(outputs, target)
        weights = self.weights if self.weights is not None else self.nnunet_weights(len(outputs))
        if len(weights) < len(outputs):
            raise ValueError(f"{len(outputs)} outputs but {len(weights)} weights")
        target = _target_index(target, outputs[0])
        total = outputs[0].new_zeros((), dtype=torch.float32)
        for out, w in zip(outputs, weights):
            if w:
                total = total + w * self.loss(out, downsample_target(target, out.shape[2:]))
        return total
