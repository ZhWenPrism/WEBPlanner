"""Sliding-window inference with Gaussian weighting, mirroring test-time augmentation and fold ensembling.

For every window the softmax output is averaged over models (folds) and mirror combinations, multiplied by a
Gaussian importance map (sigma = ``sigma_scale`` x patch size, 1/8 by default) and accumulated; the accumulated
probabilities are divided by the accumulated weights, so every voxel holds a weighted mean of probability vectors.
Windows overlap by ``overlap`` (0.5) of the patch; volumes smaller than the patch are padded and cropped back.
"""
from __future__ import annotations

from itertools import combinations, product
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

from ..types import Volume
from .postprocess import postprocess as _postprocess

ModelOrModels = Union[nn.Module, Sequence[nn.Module]]


def gaussian_importance_map(patch_size: Sequence[int], sigma_scale: float = 1.0 / 8, value_scale: float = 10.0) -> np.ndarray:
    """Gaussian centred on the patch with sigma ``sigma_scale * patch_size`` per axis (maximum ``value_scale``)."""
    axes = [((np.arange(n) - (n - 1) / 2.0) / max(n * sigma_scale, 1e-6)) ** 2 for n in patch_size]
    g = np.exp(-0.5 * (axes[0][:, None, None] + axes[1][None, :, None] + axes[2][None, None, :]))
    g = (g / g.max() * value_scale).astype(np.float32)
    return np.maximum(g, g[g > 0].min())


def window_starts(size: int, patch: int, overlap: float) -> List[int]:
    """Window start positions covering ``size`` with step ``patch * (1 - overlap)`` (nnU-Net rule)."""
    if size <= patch:
        return [0]
    step = patch * (1.0 - overlap)
    n = int(np.ceil((size - patch) / max(step, 1.0))) + 1
    return [int(round(x)) for x in np.linspace(0, size - patch, n)]


class SlidingWindowPredictor:
    """Patch-wise prediction of a whole volume.

    Args:
        model: a network or a sequence of networks (folds); outputs are averaged in probability space.
        patch_size: window size (should be divisible by the network's total stride).
        overlap: fractional overlap between neighbouring windows.
        sigma_scale: Gaussian sigma as a fraction of the patch; ``use_gaussian=False`` gives uniform weights.
        mirror_axes: spatial axes for mirroring TTA (all 2**k flip combinations); ``None`` or ``()`` disables it.
        batch_size: windows per forward pass.
        device: defaults to the device of the first model.
        amp: autocast on CUDA.
        pad_value: padding for small volumes; ``None`` uses the image minimum.
        accumulate_on_device: keep the accumulators on ``device`` (faster on GPU, more memory).
    """

    def __init__(self, model: ModelOrModels, patch_size: Sequence[int] = (128, 128, 128), overlap: float = 0.5,
                 sigma_scale: float = 1.0 / 8, use_gaussian: bool = True, mirror_axes: Optional[Sequence[int]] = (0, 1, 2),
                 batch_size: int = 1, device: Optional[Union[str, torch.device]] = None, amp: bool = True,
                 pad_value: Optional[float] = None, accumulate_on_device: bool = True) -> None:
        self.models: List[nn.Module] = list(model) if isinstance(model, (list, tuple)) else [model]
        if not self.models:
            raise ValueError("no model given")
        if not 0.0 <= overlap < 1.0:
            raise ValueError("overlap must be in [0, 1)")
        if device is None:
            try:
                device = next(self.models[0].parameters()).device
            except StopIteration:
                device = "cpu"
        self.device = torch.device(device)
        for m in self.models:
            m.to(self.device).eval()
        self.patch_size = tuple(int(p) for p in patch_size)
        self.overlap, self.batch_size, self.pad_value = float(overlap), int(batch_size), pad_value
        self.amp = amp and self.device.type == "cuda"
        self.accumulate_on_device = accumulate_on_device
        g = gaussian_importance_map(self.patch_size, sigma_scale) if use_gaussian else np.ones(self.patch_size, np.float32)
        self.importance = torch.from_numpy(g)
        axes = tuple(mirror_axes or ())
        self.mirror_sets: List[Tuple[int, ...]] = [c for r in range(len(axes) + 1) for c in combinations(axes, r)]

    @torch.no_grad()
    def _predict_batch(self, x: torch.Tensor) -> torch.Tensor:
        total, n = None, 0
        for model in self.models:
            for axes in self.mirror_sets:
                dims = [a + 2 for a in axes]
                xi = torch.flip(x, dims) if dims else x
                with torch.autocast(device_type=self.device.type, enabled=self.amp):
                    out = model(xi)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                p = torch.softmax(out.float(), dim=1)
                if dims:
                    p = torch.flip(p, dims)
                total = p if total is None else total + p
                n += 1
        return total / n

    @torch.no_grad()
    def predict_probabilities(self, image: np.ndarray) -> np.ndarray:
        """Softmax probabilities ``(C, z, y, x)`` (float32) for an image ``(z, y, x)`` or ``(c, z, y, x)``."""
        img = np.asarray(image, dtype=np.float32)
        if img.ndim == 3:
            img = img[None]
        if img.ndim != 4:
            raise ValueError(f"expected (z, y, x) or (c, z, y, x), got {img.shape}")
        shape = np.asarray(img.shape[1:])
        target = np.maximum(shape, self.patch_size)
        before = (target - shape) // 2
        if np.any(target > shape):
            pad = float(img.min()) if self.pad_value is None else float(self.pad_value)
            img = np.pad(img, [(0, 0)] + [(int(b), int(t - s - b)) for b, t, s in zip(before, target, shape)],
                         mode="constant", constant_values=pad)
        starts = list(product(*[window_starts(int(s), p, self.overlap) for s, p in zip(target, self.patch_size)]))
        windows = [tuple(slice(s, s + p) for s, p in zip(st, self.patch_size)) for st in starts]

        acc_device = self.device if self.accumulate_on_device else torch.device("cpu")
        weight = self.importance.to(acc_device)
        acc: Optional[torch.Tensor] = None
        norm = torch.zeros(tuple(int(t) for t in target), dtype=torch.float32, device=acc_device)
        for i in range(0, len(windows), self.batch_size):
            batch = windows[i:i + self.batch_size]
            x = torch.from_numpy(np.stack([img[(slice(None),) + w] for w in batch])).to(self.device)
            probs = self._predict_batch(x).to(acc_device)
            if acc is None:
                acc = torch.zeros((probs.shape[1],) + tuple(int(t) for t in target), dtype=torch.float32, device=acc_device)
            for w, p in zip(batch, probs):
                acc[(slice(None),) + w] += p * weight
                norm[w] += weight
        acc /= norm
        crop = tuple(slice(int(b), int(b + s)) for b, s in zip(before, shape))
        return acc[(slice(None),) + crop].cpu().numpy()

    def predict(self, image: np.ndarray, return_probabilities: bool = False
                ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Label map ``uint8 (z, y, x)`` by argmax over the probabilities."""
        probs = self.predict_probabilities(image)
        labels = probs.argmax(0).astype(np.uint8)
        return (labels, probs) if return_probabilities else labels


def ensemble_predict(models: ModelOrModels, volume: Union[Volume, np.ndarray], patch_size: Sequence[int] = (128, 128, 128),
                     **kw) -> np.ndarray:
    """Softmax probabilities ``(C, z, y, x)`` averaged over ``models`` (e.g. the five cross-validation folds)."""
    arr = volume.array if isinstance(volume, Volume) else volume
    return SlidingWindowPredictor(models, patch_size, **kw).predict_probabilities(arr)


def predict_volume(model: ModelOrModels, volume: Volume, preprocessor=None, patch_size: Sequence[int] = (128, 128, 128),
                   return_probabilities: bool = False, postprocess: bool = False, **kw
                   ) -> Union[Volume, Tuple[Volume, np.ndarray]]:
    """Segment a :class:`Volume` with one model or an ensemble.

    ``preprocessor`` is applied first: an object with ``.transform(volume)`` or a callable, returning a
    :class:`Volume` (whose grid then defines the output) or an array on the input grid. ``postprocess=True`` runs
    :func:`webplanner.segmentation.postprocess.postprocess`. Returns a ``uint8`` label :class:`Volume`, plus the
    probabilities ``(C, z, y, x)`` when ``return_probabilities``.
    """
    ref = volume
    if preprocessor is not None:
        out = preprocessor.transform(volume) if hasattr(preprocessor, "transform") else preprocessor(volume)
        if isinstance(out, Volume):
            ref = out
        else:
            out = np.asarray(out)
            if out.shape != volume.shape:
                raise ValueError("the preprocessor changed the grid; return a Volume so that the spacing is known")
            ref = Volume(out, volume.spacing, volume.origin, volume.direction)
    labels, probs = SlidingWindowPredictor(model, patch_size, **kw).predict(ref.array, return_probabilities=True)
    if postprocess:
        labels = _postprocess(labels, ref.spacing)
    result = Volume(labels.astype(np.uint8), ref.spacing, ref.origin, ref.direction)
    return (result, probs) if return_probabilities else result
