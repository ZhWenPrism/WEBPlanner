"""Patch sampling for segmentation training.

:class:`PatchDataset` draws random patches with foreground oversampling (nnU-Net style): with probability
``foreground_oversample`` the patch is centred on a random voxel of a random foreground class present in the case,
otherwise its position is uniform. Volumes smaller than the patch are padded. The random state of sample ``i`` is
derived from ``(seed, i)`` only, so a sample does not depend on worker scheduling, and the sampling logic
(:meth:`PatchDataset.sample`) runs without PyTorch.
"""
from __future__ import annotations

from itertools import count
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..types import FOREGROUND_CLASSES, Volume
from ..utils.io import load_volume

try:
    import torch
    from torch.utils.data import Dataset as _BaseDataset
except ImportError:          # the sampling logic stays usable without torch
    torch = None
    _BaseDataset = object

ArrayLike = Union[np.ndarray, Volume, str, Path]


def _as_array(x: ArrayLike) -> np.ndarray:
    if isinstance(x, Volume):
        return x.array
    if isinstance(x, (str, Path)):
        return load_volume(x).array
    return np.asarray(x)


def _triple(size: Union[int, Sequence[int]]) -> Tuple[int, int, int]:
    t = (int(size),) * 3 if np.isscalar(size) else tuple(int(s) for s in size)
    if len(t) != 3:
        raise ValueError(f"expected a 3D size, got {size}")
    return t


def crop_with_padding(arr: np.ndarray, start: Sequence[int], size: Sequence[int], pad_value: float = 0) -> np.ndarray:
    """Crop ``arr[start:start+size]``; the parts outside the array are filled with ``pad_value``.

    Returns a view when the crop lies inside the array (callers must not modify it in place).
    """
    if all(s >= 0 and s + n <= dim for s, n, dim in zip(start, size, arr.shape)):
        return arr[tuple(slice(int(s), int(s) + int(n)) for s, n in zip(start, size))]
    out = np.full(tuple(size), pad_value, dtype=arr.dtype)
    src, dst = [], []
    for s, n, dim in zip(start, size, arr.shape):
        lo, hi = max(s, 0), min(s + n, dim)
        if hi <= lo:
            return out
        src.append(slice(lo, hi))
        dst.append(slice(lo - s, hi - s))
    out[tuple(dst)] = arr[tuple(src)]
    return out


def center_crop_or_pad(image: np.ndarray, label: Optional[np.ndarray], size: Sequence[int], pad_image: float = 0.0
                       ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Centre crop or pad an image/label pair to ``size``."""
    shape = np.asarray(image.shape)
    if tuple(shape) == tuple(size):
        return image, label
    start = (shape - np.asarray(size)) // 2
    image = crop_with_padding(image, start, size, pad_image)
    if label is not None:
        label = crop_with_padding(label, start, size, 0)
    return image, label


class PatchDataset(_BaseDataset):
    """Random 3D patches from a list of cases.

    Args:
        cases: sequence of ``(image, label)``; each item an ndarray ``(z, y, x)``, a :class:`Volume` or a path
            (``.npz`` / ``.nii.gz``, read with :func:`webplanner.utils.io.load_volume`).
        patch_size: output patch size.
        transform: joint augmentation ``t(image, label, rng)`` (see :mod:`webplanner.data.transforms`).
        foreground_oversample: probability that a patch is centred on a foreground voxel.
        foreground_classes: classes eligible for oversampling (those present in each case are used).
        samples_per_epoch: dataset length; defaults to the number of cases.
        sample_size: region cut before augmentation (e.g. :func:`~webplanner.data.transforms.enlarged_size`); the
            result is centre-cropped to ``patch_size`` afterwards. Defaults to ``patch_size``.
        normalize: optional function applied once to every image (e.g. z-scoring fitted on training folds).
        pad_value: image padding value; ``None`` uses the case minimum.
        max_locations: foreground voxel positions stored per class and case.
    """

    def __init__(self, cases: Sequence[Tuple[ArrayLike, ArrayLike]], patch_size: Union[int, Sequence[int]] = 128,
                 transform: Optional[Callable] = None, foreground_oversample: float = 0.33,
                 foreground_classes: Sequence[int] = FOREGROUND_CLASSES, samples_per_epoch: Optional[int] = None,
                 sample_size: Optional[Union[int, Sequence[int]]] = None,
                 normalize: Optional[Callable[[np.ndarray], np.ndarray]] = None, pad_value: Optional[float] = None,
                 max_locations: int = 10000, seed: int = 0) -> None:
        if len(cases) == 0:
            raise ValueError("PatchDataset needs at least one case")
        self.patch_size = _triple(patch_size)
        self.sample_size = _triple(sample_size) if sample_size is not None else self.patch_size
        self.transform = transform
        self.foreground_oversample = float(foreground_oversample)
        self.foreground_classes = tuple(int(c) for c in foreground_classes)
        self.seed = int(seed)
        self.images: List[np.ndarray] = []
        self.labels: List[np.ndarray] = []
        self.locations: List[Dict[int, np.ndarray]] = []
        self.pad_values: List[float] = []
        rng = np.random.default_rng(self.seed)
        for i, (img, lab) in enumerate(cases):
            img = _as_array(img).astype(np.float32, copy=False)
            lab = _as_array(lab)
            if img.shape != lab.shape:
                raise ValueError(f"case {i}: image {img.shape} and label {lab.shape} differ")
            if normalize is not None:
                img = np.asarray(normalize(img), dtype=np.float32)
            lab = lab.astype(np.uint8 if lab.max(initial=0) <= 255 else np.int64, copy=False)
            locs = {}
            for c in self.foreground_classes:
                idx = np.argwhere(lab == c)
                if len(idx) > max_locations:
                    idx = idx[rng.choice(len(idx), max_locations, replace=False)]
                if len(idx):
                    locs[c] = idx
            self.images.append(img)
            self.labels.append(lab)
            self.locations.append(locs)
            self.pad_values.append(float(img.min()) if pad_value is None else float(pad_value))
        self.samples_per_epoch = int(samples_per_epoch) if samples_per_epoch else len(self.images)

    @property
    def n_cases(self) -> int:
        return len(self.images)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _start(self, case: int, foreground: bool, rng: np.random.Generator) -> np.ndarray:
        shape = np.asarray(self.labels[case].shape)
        size = np.asarray(self.sample_size)
        locs = self.locations[case]
        if foreground and locs:
            cls = list(locs)[int(rng.integers(len(locs)))]
            voxel = locs[cls][int(rng.integers(len(locs[cls])))]
            start = voxel - size // 2
            return np.where(shape >= size, np.clip(start, 0, np.maximum(shape - size, 0)), -((size - shape) // 2))
        return np.array([rng.integers(0, s - n + 1) if s >= n else -((n - s) // 2) for s, n in zip(shape, size)])

    def sample(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """NumPy sample ``index``: image ``float32 (z, y, x)`` and label ``int64 (z, y, x)`` of ``patch_size``."""
        if index < 0:
            raise IndexError("negative sample index")
        rng = np.random.default_rng([self.seed, int(index)])
        case = int(index) % self.n_cases
        foreground = rng.random() < self.foreground_oversample
        start = self._start(case, foreground, rng)
        image = crop_with_padding(self.images[case], start, self.sample_size, self.pad_values[case])
        label = crop_with_padding(self.labels[case], start, self.sample_size, 0)
        if self.transform is not None:
            image, label = self.transform(image, label, rng)
        image, label = center_crop_or_pad(image, label, self.patch_size, self.pad_values[case])
        return np.array(image, dtype=np.float32), np.array(label, dtype=np.int64)       # always copies

    def __getitem__(self, index: int):
        if torch is None:
            raise ImportError("PatchDataset.__getitem__ returns torch tensors; install torch or use .sample()")
        image, label = self.sample(int(index))
        return torch.from_numpy(image[None]), torch.from_numpy(label)


def collate(batch: Sequence[Tuple["torch.Tensor", "torch.Tensor"]]) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Stack ``(image (1, z, y, x), label (z, y, x))`` pairs into ``(B, 1, z, y, x)`` and ``(B, z, y, x)``."""
    images, labels = zip(*batch)
    return torch.stack(images), torch.stack(labels)


class InfiniteSampler:
    """Endless index stream: a fresh permutation of the ``n`` cases in every round.

    With ``unique=True`` the yielded index is ``round * n + case``, so every draw has its own random state in
    :class:`PatchDataset` while ``index % n`` still visits each case once per round. ``start`` skips the first draws
    (used when resuming training).
    """

    def __init__(self, n: int, shuffle: bool = True, seed: int = 0, start: int = 0, unique: bool = True) -> None:
        if n <= 0:
            raise ValueError("n must be positive")
        self.n, self.shuffle, self.seed, self.start, self.unique = int(n), shuffle, int(seed), int(start), unique

    def _order(self, rnd: int) -> np.ndarray:
        if not self.shuffle:
            return np.arange(self.n)
        return np.random.default_rng([self.seed, rnd]).permutation(self.n)

    def __iter__(self) -> Iterator[int]:
        rnd, pos = divmod(self.start, self.n)
        order = self._order(rnd)
        for _ in count():
            case = int(order[pos])
            yield rnd * self.n + case if self.unique else case
            pos += 1
            if pos == self.n:
                rnd, pos = rnd + 1, 0
                order = self._order(rnd)


def infinite_loader(dataset, batch_size: int, num_workers: int = 0, seed: int = 0, start: int = 0,
                    pin_memory: bool = False) -> Iterator[Tuple["torch.Tensor", "torch.Tensor"]]:
    """Iteration-based loader: an endless iterator of batches (draw ``iterations_per_epoch`` per epoch)."""
    from torch.utils.data import DataLoader

    is_patch = isinstance(dataset, PatchDataset)
    sampler = InfiniteSampler(dataset.n_cases if is_patch else len(dataset), seed=seed, start=start, unique=is_patch)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, drop_last=True,
                        collate_fn=collate, pin_memory=pin_memory, persistent_workers=num_workers > 0)
    return iter(loader)
