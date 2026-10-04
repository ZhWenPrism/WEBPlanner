"""Image-based sac score: one scalar per case summarising sac appearance that the morphometry may miss.

The score is trained to predict how the operator moved away from the chart, either the signed width step of the
implanted size relative to the chart size (``target="width_step"``) or whether the operator oversized
(``target="oversized"``, score = log-odds). In the ranking model it enters only as ``dw_x_image_score``.

Backends
--------
* ``"descriptor"`` (default, NumPy/SciPy/scikit-learn): radiomics-like descriptors of the sac (intensity
  percentiles, surface-to-volume ratio, sphericity, solidity, PCA elongation and flatness, boundary gradient) fed to a
  standardised ridge (``width_step``) or logistic (``oversized``) model.
* ``"cnn"`` (optional, PyTorch imported lazily): a small 3D CNN on a fixed physical-size, sac-centred crop with two
  channels (intensity normalised by the sac median, sac mask) and a minimal training loop.

When the score is used as a ranking feature on the same cases it was trained on, use :meth:`fit_predict_oof` with
patient groups so that every case receives a score from a model that did not see it.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull

from ..types import NECK, SAC, DeviceSize, Volume

DESCRIPTOR_NAMES = ["log_volume_mm3", "surface_to_volume", "sphericity", "solidity", "elongation", "flatness",
                    "major_axis_mm", "int_p10_rel", "int_p90_rel", "int_cv", "sac_bg_contrast", "boundary_gradient"]

ArrayOrVolume = Union[Volume, np.ndarray]


def _as_volume(x: ArrayOrVolume, spacing: Optional[Sequence[float]] = None) -> Volume:
    if isinstance(x, Volume):
        return x
    return Volume(np.asarray(x), tuple(spacing) if spacing is not None else (1.0, 1.0, 1.0))


def sac_mask(labels: ArrayOrVolume) -> np.ndarray:
    lab = labels.array if isinstance(labels, Volume) else np.asarray(labels)
    return np.isin(lab, (SAC, NECK))


def width_step_target(chart: DeviceSize, implanted: DeviceSize, catalogue: Any) -> int:
    """Signed width step from the chart size to the implanted size."""
    return int(catalogue.steps_between(chart, implanted)[0])


def _surface_area(mask: np.ndarray, spacing: Sequence[float]) -> float:
    m = np.pad(mask, 1).astype(np.int8)
    sz, sy, sx = spacing
    face = (sy * sx, sz * sx, sz * sy)
    return float(sum(np.abs(np.diff(m, axis=a)).sum() * face[a] for a in range(3)))


def _hull_volume(boundary_idx: np.ndarray, spacing: Sequence[float]) -> float:
    corners = np.array([[i, j, k] for i in (-0.5, 0.5) for j in (-0.5, 0.5) for k in (-0.5, 0.5)])
    pts = np.unique((boundary_idx[:, None, :] + corners[None]).reshape(-1, 3), axis=0) * np.asarray(spacing)
    try:
        return float(ConvexHull(pts).volume)
    except Exception:
        return float("nan")


def sac_descriptors(image: ArrayOrVolume, labels: ArrayOrVolume, spacing: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """Intensity and shape descriptors of the sac (labels 1 and 4). Intensities are divided by the sac median so that
    the descriptors do not depend on the scanner's intensity scale."""
    img, lab = _as_volume(image, spacing), _as_volume(labels, spacing)
    sp = lab.spacing
    m = sac_mask(lab)
    n = int(m.sum())
    if n < 8:
        raise ValueError(f"sac mask has {n} voxels; descriptors need at least 8")
    lo = np.maximum(np.argwhere(m).min(axis=0) - 4, 0)
    hi = np.argwhere(m).max(axis=0) + 5
    box = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))       # sac bounding box with a 4-voxel margin
    m, lab_arr, a = m[box], lab.array[box], img.array[box].astype(float)
    vox = float(np.prod(sp))
    volume = n * vox
    area = _surface_area(m, sp)
    boundary = m & ~ndi.binary_erosion(m)
    bidx = np.argwhere(boundary)
    hull = _hull_volume(bidx, sp)
    solidity = float(min(1.0, volume / hull)) if np.isfinite(hull) and hull > 0 else 1.0

    pts = np.argwhere(m) * np.asarray(sp)
    ev = np.sort(np.linalg.eigvalsh(np.cov(pts.T)))[::-1]
    ev = np.maximum(ev, 1e-12)

    inside = a[m]
    med = float(np.median(inside))
    med = med if abs(med) > 1e-12 else 1.0
    shell = ndi.binary_dilation(m, iterations=2) & (lab_arr == 0)
    bg = float(np.median(a[shell])) if shell.any() else 0.0
    grad = np.sqrt(sum(g ** 2 for g in np.gradient(a / med, *sp)))
    return {
        "log_volume_mm3": float(np.log(volume)),
        "surface_to_volume": area / volume,
        "sphericity": float(np.pi ** (1 / 3) * (6 * volume) ** (2 / 3) / area),
        "solidity": solidity,
        "elongation": float(np.sqrt(ev[1] / ev[0])),
        "flatness": float(np.sqrt(ev[2] / ev[0])),
        "major_axis_mm": float(4 * np.sqrt(ev[0])),
        "int_p10_rel": float(np.percentile(inside, 10) / med),
        "int_p90_rel": float(np.percentile(inside, 90) / med),
        "int_cv": float(inside.std() / max(abs(inside.mean()), 1e-12)),
        "sac_bg_contrast": (med - bg) / med,
        "boundary_gradient": float(grad[boundary].mean()),
    }


def sac_crop(image: ArrayOrVolume, labels: ArrayOrVolume, shape: Tuple[int, int, int] = (32, 32, 32),
             extent_mm: float = 20.0, spacing: Optional[Sequence[float]] = None) -> np.ndarray:
    """Two-channel crop ``(2, *shape)`` centred on the sac centroid with a fixed physical extent (so that the CNN sees
    absolute size): channel 0 intensity / sac median (clipped to [0, 3]), channel 1 sac mask."""
    img, lab = _as_volume(image, spacing), _as_volume(labels, spacing)
    m = sac_mask(lab)
    if not m.any():
        raise ValueError("empty sac mask")
    sp = np.asarray(lab.spacing)
    centre_mm = np.argwhere(m).mean(axis=0) * sp
    axes = [centre_mm[i] + np.linspace(-extent_mm / 2, extent_mm / 2, shape[i]) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=0) / sp[:, None, None, None]
    a = img.array.astype(float)
    med = float(np.median(a[m])) or 1.0
    chan0 = ndi.map_coordinates(a / med, grid, order=1, mode="constant", cval=0.0)
    chan1 = ndi.map_coordinates(m.astype(float), grid, order=1, mode="constant", cval=0.0)
    return np.stack([np.clip(chan0, 0.0, 3.0), chan1]).astype(np.float32)


# --------------------------------------------------------------------------------------------------------------------
# CNN backend (PyTorch imported lazily)
# --------------------------------------------------------------------------------------------------------------------
def _build_cnn(in_ch: int = 2, width: int = 8):
    import torch.nn as nn

    def block(ci: int, co: int, stride: int):
        return [nn.Conv3d(ci, co, 3, stride=stride, padding=1), nn.InstanceNorm3d(co, affine=True), nn.LeakyReLU(0.01)]

    w = width
    return nn.Sequential(*block(in_ch, w, 1), *block(w, 2 * w, 2), *block(2 * w, 4 * w, 2), *block(4 * w, 4 * w, 2),
                         nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(4 * w, 1))


class SacImageScorer:
    """Scalar sac image score from descriptors (default) or a small 3D CNN.

    Usage: ``F = scorer.extract_many(images, labels)``; ``scorer.fit(F, y)``; ``scorer.predict(F)``. ``extract``
    returns a descriptor vector (``"descriptor"``) or a two-channel crop (``"cnn"``), so large volumes need not be
    held in memory.
    """

    def __init__(self, backend: str = "descriptor", target: str = "width_step", alpha: float = 1.0,
                 crop_shape: Tuple[int, int, int] = (32, 32, 32), crop_mm: float = 20.0, epochs: int = 30,
                 batch_size: int = 8, lr: float = 1e-3, width: int = 8, device: Optional[str] = None, seed: int = 0) -> None:
        if backend not in ("descriptor", "cnn"):
            raise ValueError("backend must be 'descriptor' or 'cnn'")
        if target not in ("width_step", "oversized"):
            raise ValueError("target must be 'width_step' or 'oversized'")
        self.backend, self.target, self.alpha = backend, target, float(alpha)
        self.crop_shape, self.crop_mm = tuple(crop_shape), float(crop_mm)
        self.epochs, self.batch_size, self.lr, self.width = int(epochs), int(batch_size), float(lr), int(width)
        self.device, self.seed = device, int(seed)
        self.model_: Any = None

    # ---------------------------------------------------------------- inputs
    def extract(self, image: ArrayOrVolume, labels: ArrayOrVolume, spacing: Optional[Sequence[float]] = None) -> np.ndarray:
        if self.backend == "descriptor":
            d = sac_descriptors(image, labels, spacing)
            return np.array([d[k] for k in DESCRIPTOR_NAMES], float)
        return sac_crop(image, labels, self.crop_shape, self.crop_mm, spacing)

    def extract_many(self, images: Sequence[ArrayOrVolume], labels: Sequence[ArrayOrVolume]) -> np.ndarray:
        if len(images) != len(labels):
            raise ValueError("images and labels must have the same length")
        return np.stack([self.extract(i, l) for i, l in zip(images, labels)])

    def _y(self, y: Sequence[float]) -> np.ndarray:
        y = np.asarray(y, float)
        return (y > 0).astype(float) if self.target == "oversized" else y

    # ---------------------------------------------------------------- fit / predict
    def fit(self, F: np.ndarray, y: Sequence[float]) -> "SacImageScorer":
        """``y``: signed width steps of the implanted size relative to the chart (``oversized`` uses ``y > 0``)."""
        F, yy = np.asarray(F, float), self._y(y)
        if len(F) != len(yy):
            raise ValueError("F and y must have the same length")
        if self.backend == "descriptor":
            from sklearn.linear_model import LogisticRegression, Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            if self.target == "oversized":
                if len(np.unique(yy)) < 2:
                    raise ValueError("target 'oversized' needs both classes in the training data")
                est = LogisticRegression(C=1.0 / self.alpha, max_iter=1000)
            else:
                est = Ridge(alpha=self.alpha)
            self.model_ = make_pipeline(StandardScaler(), est).fit(F, yy)
        else:
            self.model_ = self._fit_cnn(F, yy)
        return self

    def predict(self, F: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("scorer is not fitted")
        F = np.asarray(F, float)
        if self.backend == "descriptor":
            if self.target == "oversized":
                return np.asarray(self.model_.decision_function(F), float)
            return np.asarray(self.model_.predict(F), float)
        return self._predict_cnn(F)

    def score_case(self, image: ArrayOrVolume, labels: ArrayOrVolume, spacing: Optional[Sequence[float]] = None) -> float:
        return float(self.predict(self.extract(image, labels, spacing)[None])[0])

    def fit_predict_oof(self, F: np.ndarray, y: Sequence[float], groups: Sequence[Any], n_splits: int = 5,
                        seed: Optional[int] = None) -> np.ndarray:
        """Out-of-fold scores from patient-grouped folds (no case is scored by a model trained on it)."""
        from .model_selection import group_kfold

        F, y = np.asarray(F, float), np.asarray(y, float)
        out = np.full(len(F), np.nan)
        for tr, te in group_kfold(groups, n_splits, self.seed if seed is None else seed):
            clone = SacImageScorer(self.backend, self.target, self.alpha, self.crop_shape, self.crop_mm, self.epochs,
                                   self.batch_size, self.lr, self.width, self.device, self.seed)
            out[te] = clone.fit(F[tr], y[tr]).predict(F[te])
        return out

    # ---------------------------------------------------------------- CNN internals
    def _torch_device(self):
        import torch

        return torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _fit_cnn(self, F: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        import torch

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        dev = self._torch_device()
        net = _build_cnn(F.shape[1], self.width).to(dev)
        if self.target == "width_step":
            mu, sd = float(y.mean()), float(y.std()) or 1.0
            loss_fn = torch.nn.MSELoss()
        else:
            mu, sd = 0.0, 1.0
            loss_fn = torch.nn.BCEWithLogitsLoss()
        t = (y - mu) / sd
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-4)
        n, history = len(F), []
        for _ in range(self.epochs):
            net.train()
            perm, total = rng.permutation(n), 0.0
            for s in range(0, n, self.batch_size):
                b = perm[s:s + self.batch_size]
                xb = F[b].copy()
                for ax in (2, 3, 4):                                   # random mirroring per batch
                    if rng.random() < 0.5:
                        xb = np.flip(xb, axis=ax)
                xb = torch.from_numpy(np.ascontiguousarray(xb, np.float32)).to(dev)
                tb = torch.from_numpy(t[b].astype(np.float32)).to(dev)
                opt.zero_grad()
                loss = loss_fn(net(xb)[:, 0], tb)
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(b)
            history.append(total / n)
        return {"net": net, "mu": mu, "sd": sd, "history": history}

    def _predict_cnn(self, F: np.ndarray) -> np.ndarray:
        import torch

        net, dev = self.model_["net"], self._torch_device()
        net.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(F), self.batch_size):
                xb = torch.from_numpy(np.ascontiguousarray(F[s:s + self.batch_size], np.float32)).to(dev)
                out.append(net(xb)[:, 0].cpu().numpy())
        return np.concatenate(out).astype(float) * self.model_["sd"] + self.model_["mu"]
