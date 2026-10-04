"""3D data augmentation in NumPy/SciPy, applied jointly to an image and its label map.

Every transform is a callable ``t(image, label, rng) -> (image, label)``:

* ``image`` is ``(z, y, x)`` or channel-first ``(c, z, y, x)``; ``label`` is ``(z, y, x)`` or ``None``;
* ``rng`` is a :class:`numpy.random.Generator`, so a pipeline is reproducible from one seed;
* spatial transforms resample the label with nearest-neighbour interpolation, so label values never leave the label
  set; intensity transforms leave the label untouched.

:func:`default_pipeline` mirrors the nnU-Net defaults (rotation and scaling with p = 0.2, Gaussian noise 0.1,
Gaussian blur 0.2, brightness 0.15, contrast 0.15, inverted gamma 0.1, gamma 0.3, mirroring on all axes).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage as ndi

Range = Tuple[float, float]
Pair = Tuple[np.ndarray, Optional[np.ndarray]]


def _channels(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Return a channel-first view and whether a channel axis was added."""
    if image.ndim == 3:
        return image[None], True
    if image.ndim == 4:
        return image, False
    raise ValueError(f"expected a 3D or channel-first 4D image, got shape {image.shape}")


def _uniform(rng: np.random.Generator, r: Union[float, Range]) -> float:
    if np.isscalar(r):
        return float(r)
    lo, hi = r
    return float(lo) if lo == hi else float(rng.uniform(lo, hi))


def _rotation_matrix(angles_rad: Sequence[float]) -> np.ndarray:
    """Rotation about the z, y and x axes (in that order) in the ``(z, y, x)`` index frame."""
    az, ay, ax = angles_rad
    cz, sz, cy, sy, cx, sx = np.cos(az), np.sin(az), np.cos(ay), np.sin(ay), np.cos(ax), np.sin(ax)
    rz = np.array([[1, 0, 0], [0, cz, -sz], [0, sz, cz]])      # rotation in the (y, x) plane
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])      # rotation in the (z, x) plane
    rx = np.array([[cx, -sx, 0], [sx, cx, 0], [0, 0, 1]])      # rotation in the (z, y) plane
    return rz @ ry @ rx


class Transform:
    """Base class: applies :meth:`apply` with probability ``p``."""

    def __init__(self, p: float = 1.0) -> None:
        self.p = float(p)

    def __call__(self, image: np.ndarray, label: Optional[np.ndarray], rng: np.random.Generator) -> Pair:
        if self.p < 1.0 and rng.random() >= self.p:
            return image, label
        return self.apply(image, label, rng)

    def apply(self, image: np.ndarray, label: Optional[np.ndarray], rng: np.random.Generator) -> Pair:
        raise NotImplementedError

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({args})"


class Compose:
    """Apply transforms in order."""

    def __init__(self, transforms: Sequence) -> None:
        self.transforms: List = list(transforms)

    def __call__(self, image: np.ndarray, label: Optional[np.ndarray], rng: np.random.Generator) -> Pair:
        for t in self.transforms:
            image, label = t(image, label, rng)
        return image, label

    def __repr__(self) -> str:
        return "Compose([\n  " + ",\n  ".join(repr(t) for t in self.transforms) + "\n])"


# --------------------------------------------------------------------------------------------------------------------
# spatial transforms
# --------------------------------------------------------------------------------------------------------------------
class SpatialTransform(Transform):
    """Rotation, isotropic scaling and elastic deformation in a single resampling step.

    Each component is drawn independently with its own probability. The output grid has ``patch_size`` (centred on
    the input centre) when given, otherwise the input shape, so a larger input region can be cropped to the patch
    after rotation without empty corners.

    Args:
        rotation_deg: per-axis angle ranges ``((lo, hi),) * 3`` for rotations about z, y and x, or one symmetric
            bound applied to all axes.
        scale_range: zoom factor range; a factor above 1 enlarges the content.
        elastic: enable elastic deformation (displacement field = Gaussian-smoothed uniform noise).
        elastic_alpha: displacement magnitude range in voxels; ``elastic_sigma``: smoothing range in voxels.
        order_image: spline order for the image (labels always use order 0).
        mode: boundary mode passed to :func:`scipy.ndimage.map_coordinates`.
    """

    def __init__(self, rotation_deg: Union[float, Sequence[Range], None] = 30.0, scale_range: Optional[Range] = (0.7, 1.4),
                 elastic: bool = False, elastic_alpha: Range = (0.0, 9.0), elastic_sigma: Range = (9.0, 13.0),
                 p_rotation: float = 0.2, p_scale: float = 0.2, p_elastic: float = 0.2,
                 patch_size: Optional[Sequence[int]] = None, order_image: int = 3, mode: str = "nearest",
                 p: float = 1.0) -> None:
        super().__init__(p)
        if rotation_deg is not None and np.isscalar(rotation_deg):
            rotation_deg = [(-float(rotation_deg), float(rotation_deg))] * 3
        self.rotation_deg = None if rotation_deg is None else [tuple(map(float, r)) for r in rotation_deg]
        if self.rotation_deg is not None and len(self.rotation_deg) != 3:
            raise ValueError("rotation_deg needs one range per axis (z, y, x)")
        self.scale_range = scale_range
        self.elastic = elastic
        self.elastic_alpha, self.elastic_sigma = elastic_alpha, elastic_sigma
        self.p_rotation, self.p_scale, self.p_elastic = p_rotation, p_scale, p_elastic
        self.patch_size = None if patch_size is None else tuple(int(s) for s in patch_size)
        self.order_image, self.mode = order_image, mode

    def apply(self, image: np.ndarray, label: Optional[np.ndarray], rng: np.random.Generator) -> Pair:
        img, squeeze = _channels(image)
        in_shape = np.asarray(img.shape[1:])
        out_shape = tuple(self.patch_size) if self.patch_size is not None else tuple(in_shape)

        matrix = np.eye(3)
        if self.rotation_deg is not None and rng.random() < self.p_rotation:
            matrix = _rotation_matrix([np.deg2rad(_uniform(rng, r)) for r in self.rotation_deg]) @ matrix
        if self.scale_range is not None and rng.random() < self.p_scale:
            matrix = matrix / _uniform(rng, self.scale_range)
        do_elastic = self.elastic and rng.random() < self.p_elastic
        identity = np.allclose(matrix, np.eye(3)) and not do_elastic
        if identity and out_shape == tuple(in_shape):
            return image, label
        if identity and all(o <= i for o, i in zip(out_shape, in_shape)):      # plain centre crop
            lo = (in_shape - np.asarray(out_shape)) // 2
            sl = tuple(slice(int(a), int(a) + o) for a, o in zip(lo, out_shape))
            out = np.ascontiguousarray(img[(slice(None),) + sl])
            return (out[0] if squeeze else out), (None if label is None else np.ascontiguousarray(label[sl]))

        grid = np.stack(np.meshgrid(*[np.arange(n, dtype=np.float32) for n in out_shape], indexing="ij"))
        centre_out = (np.asarray(out_shape) - 1) / 2.0
        centre_in = (in_shape - 1) / 2.0
        offsets = grid.reshape(3, -1) - centre_out[:, None]
        coords = (matrix @ offsets + centre_in[:, None]).reshape((3,) + out_shape).astype(np.float32)
        if do_elastic:
            alpha, sigma = _uniform(rng, self.elastic_alpha), _uniform(rng, self.elastic_sigma)
            for d in range(3):
                coords[d] += ndi.gaussian_filter(rng.uniform(-1, 1, out_shape), sigma, mode="constant") * alpha

        # resample only the bounding box of the sampled coordinates (plus the spline support)
        lo = np.clip(np.floor(coords.reshape(3, -1).min(1)) - 4, 0, in_shape - 1).astype(int)
        hi = np.clip(np.ceil(coords.reshape(3, -1).max(1)) + 5, lo + 1, in_shape).astype(int)
        box = tuple(slice(a, b) for a, b in zip(lo, hi))
        coords -= lo.reshape(3, 1, 1, 1).astype(np.float32)
        out = np.stack([ndi.map_coordinates(np.ascontiguousarray(c[box], dtype=np.float32), coords,
                                            order=self.order_image, mode=self.mode) for c in img])
        out_label = None
        if label is not None:
            out_label = ndi.map_coordinates(np.ascontiguousarray(label[box]), coords, order=0, mode=self.mode,
                                            output=label.dtype)
        return (out[0] if squeeze else out), out_label


class RandomRotation(SpatialTransform):
    """Random rotation with per-axis angle ranges (degrees)."""

    def __init__(self, rotation_deg: Union[float, Sequence[Range]] = 30.0, p: float = 0.2, **kw) -> None:
        super().__init__(rotation_deg=rotation_deg, scale_range=None, elastic=False, p_rotation=1.0, p=p, **kw)


class RandomScaling(SpatialTransform):
    """Random isotropic zoom."""

    def __init__(self, scale_range: Range = (0.7, 1.4), p: float = 0.2, **kw) -> None:
        super().__init__(rotation_deg=None, scale_range=scale_range, elastic=False, p_scale=1.0, p=p, **kw)


class ElasticDeformation(SpatialTransform):
    """Random elastic deformation (alpha and sigma in voxels)."""

    def __init__(self, alpha: Range = (0.0, 9.0), sigma: Range = (9.0, 13.0), p: float = 0.2, **kw) -> None:
        super().__init__(rotation_deg=None, scale_range=None, elastic=True, elastic_alpha=alpha, elastic_sigma=sigma,
                         p_elastic=1.0, p=p, **kw)


class Mirror(Transform):
    """Flip image and label along each axis in ``axes`` independently with probability ``p_axis``."""

    def __init__(self, axes: Sequence[int] = (0, 1, 2), p_axis: float = 0.5, p: float = 1.0) -> None:
        super().__init__(p)
        self.axes, self.p_axis = tuple(axes), p_axis

    def apply(self, image: np.ndarray, label: Optional[np.ndarray], rng: np.random.Generator) -> Pair:
        img, squeeze = _channels(image)
        flip = [a for a in self.axes if rng.random() < self.p_axis]
        if not flip:
            return image, label
        img = np.ascontiguousarray(np.flip(img, axis=[a + 1 for a in flip]))
        if label is not None:
            label = np.ascontiguousarray(np.flip(label, axis=flip))
        return (img[0] if squeeze else img), label


# --------------------------------------------------------------------------------------------------------------------
# intensity transforms (image only)
# --------------------------------------------------------------------------------------------------------------------
class GaussianNoise(Transform):
    """Additive Gaussian noise; the variance is drawn from ``variance``."""

    def __init__(self, variance: Range = (0.0, 0.1), p: float = 0.1) -> None:
        super().__init__(p)
        self.variance = variance

    def apply(self, image, label, rng):
        sd = np.sqrt(_uniform(rng, self.variance))
        return (image + rng.normal(0.0, sd, image.shape)).astype(image.dtype, copy=False), label


class GaussianBlur(Transform):
    """Gaussian smoothing with a sigma (voxels) drawn per axis from ``sigma``."""

    def __init__(self, sigma: Range = (0.5, 1.0), per_axis: bool = True, p: float = 0.2) -> None:
        super().__init__(p)
        self.sigma, self.per_axis = sigma, per_axis

    def apply(self, image, label, rng):
        img, squeeze = _channels(image)
        s = [_uniform(rng, self.sigma) for _ in range(3)] if self.per_axis else [_uniform(rng, self.sigma)] * 3
        out = np.stack([ndi.gaussian_filter(c, s) for c in img]).astype(image.dtype, copy=False)
        return (out[0] if squeeze else out), label


class BrightnessContrast(Transform):
    """Multiplicative brightness and contrast change around the mean (range preserved for contrast)."""

    def __init__(self, brightness: Range = (0.75, 1.25), contrast: Range = (0.75, 1.25), p_brightness: float = 0.15,
                 p_contrast: float = 0.15, preserve_range: bool = True, p: float = 1.0) -> None:
        super().__init__(p)
        self.brightness, self.contrast = brightness, contrast
        self.p_brightness, self.p_contrast, self.preserve_range = p_brightness, p_contrast, preserve_range

    def apply(self, image, label, rng):
        out = image.astype(np.float32, copy=True)
        if rng.random() < self.p_brightness:
            out *= _uniform(rng, self.brightness)
        if rng.random() < self.p_contrast:
            lo, hi, mean = out.min(), out.max(), out.mean()
            out = (out - mean) * _uniform(rng, self.contrast) + mean
            if self.preserve_range:
                out = np.clip(out, lo, hi)
        return out.astype(image.dtype, copy=False), label


class Gamma(Transform):
    """Gamma transform on the image rescaled to [0, 1]; optionally on the inverted image; mean and SD retained."""

    def __init__(self, gamma_range: Range = (0.7, 1.5), invert: bool = False, retain_stats: bool = True,
                 p: float = 0.3) -> None:
        super().__init__(p)
        self.gamma_range, self.invert, self.retain_stats = gamma_range, invert, retain_stats

    def apply(self, image, label, rng):
        x = image.astype(np.float32, copy=True)
        if self.invert:
            x = -x
        mean, sd = x.mean(), x.std()
        g_lo, g_hi = self.gamma_range
        if g_lo < 1.0 and rng.random() < 0.5:
            g = rng.uniform(g_lo, 1.0)              # half of the draws below 1, as in nnU-Net
        else:
            g = rng.uniform(max(g_lo, 1.0), g_hi)
        lo, hi = x.min(), x.max()
        span = max(hi - lo, 1e-7)
        x = np.power((x - lo) / span, g) * span + lo
        if self.retain_stats:
            x = (x - x.mean()) / max(x.std(), 1e-7) * sd + mean
        if self.invert:
            x = -x
        return x.astype(image.dtype, copy=False), label


# --------------------------------------------------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------------------------------------------------
def default_pipeline(patch_size: Optional[Sequence[int]] = None, rotation_deg: Union[float, Sequence[Range]] = 30.0,
                     scale_range: Range = (0.7, 1.4), elastic: bool = False, mirror_axes: Optional[Sequence[int]] = (0, 1, 2),
                     order_image: int = 3) -> Compose:
    """nnU-Net-like augmentation pipeline.

    With ``patch_size`` the spatial step crops the output to the patch, so the input may be a larger region (see
    :func:`enlarged_size`).
    """
    ts: List = [SpatialTransform(rotation_deg=rotation_deg, scale_range=scale_range, elastic=elastic, p_rotation=0.2,
                                 p_scale=0.2, p_elastic=0.2, patch_size=patch_size, order_image=order_image),
                GaussianNoise(p=0.1), GaussianBlur(p=0.2), BrightnessContrast(p_brightness=0.15, p_contrast=0.15),
                Gamma(invert=True, p=0.1), Gamma(invert=False, p=0.3)]
    if mirror_axes:
        ts.append(Mirror(axes=mirror_axes))
    return Compose(ts)


def enlarged_size(patch_size: Sequence[int], rotation_deg: Union[float, Sequence[Range]] = 30.0,
                  scale_range: Range = (0.85, 1.25)) -> Tuple[int, ...]:
    """Input region that covers ``patch_size`` after the largest rotation and a zoom-out to ``min(scale_range)``.

    The default zoom bound (0.85) is nnU-Net's compromise between coverage and memory: stronger zoom-outs replicate
    the border of the region (``mode="nearest"``) instead of reading further context.
    """
    if np.isscalar(rotation_deg):
        rotation_deg = [(-float(rotation_deg), float(rotation_deg))] * 3
    half = (np.asarray(patch_size, float) - 1) / 2.0
    corners = np.array(np.meshgrid(*[[-h, h] for h in half], indexing="ij")).reshape(3, -1)
    angles = [np.deg2rad(max(abs(lo), abs(hi))) for lo, hi in rotation_deg]
    extent = np.zeros(3)
    for signs in np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij")).reshape(3, -1).T:
        r = _rotation_matrix(signs * np.asarray(angles))
        extent = np.maximum(extent, np.abs(r @ corners).max(axis=1))
    extent /= min(scale_range) if scale_range else 1.0
    return tuple(int(np.ceil(2 * e + 1)) for e in extent)
