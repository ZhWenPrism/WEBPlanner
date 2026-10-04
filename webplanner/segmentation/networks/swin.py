"""SwinUNETR comparator (MONAI, optional dependency).

The network has no deep supervision outputs; the trainer and the sliding-window predictor accept a single tensor.
Input patch sizes must be divisible by 32 and at least 64.
"""
from __future__ import annotations

import inspect
import warnings
from typing import Sequence, Tuple, Union


def build_swin_unetr(in_ch: int = 1, n_classes: int = 5, img_size: Union[int, Sequence[int]] = (96, 96, 96),
                     feature_size: int = 48, depths: Sequence[int] = (2, 2, 2, 2), num_heads: Sequence[int] = (3, 6, 12, 24),
                     drop_rate: float = 0.0, attn_drop_rate: float = 0.0, dropout_path_rate: float = 0.0,
                     use_checkpoint: bool = False, use_v2: bool = False, deep_supervision: bool = False, **kwargs):
    """Return ``monai.networks.nets.SwinUNETR``.

    ``img_size`` is passed only to MONAI versions that still accept it (it was deprecated in 1.3 and later removed);
    it is checked (divisible by 32, at least 64) in every case. ``deep_supervision`` is accepted for a uniform
    :func:`~webplanner.segmentation.networks.build_network` interface and ignored.
    """
    try:
        from monai.networks.nets import SwinUNETR
    except ImportError as e:            # pragma: no cover - depends on the environment
        raise ImportError("swin_unetr needs MONAI (pip install 'monai>=1.3'); residual_unet and unet3d do not") from e

    size: Tuple[int, ...] = (int(img_size),) * 3 if isinstance(img_size, int) else tuple(int(s) for s in img_size)
    if len(size) != 3 or any(s % 32 or s < 64 for s in size):
        raise ValueError(f"SwinUNETR needs a 3D patch size divisible by 32 and at least 64 (the bottleneck is "
                         f"size / 32 and its instance norm needs more than one voxel), got {img_size}")
    params = inspect.signature(SwinUNETR.__init__).parameters
    kw = dict(in_channels=in_ch, out_channels=n_classes, depths=tuple(depths), num_heads=tuple(num_heads),
              feature_size=feature_size, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
              dropout_path_rate=dropout_path_rate, use_checkpoint=use_checkpoint, spatial_dims=3, **kwargs)
    if "img_size" in params:
        kw["img_size"] = size
    if "use_v2" in params:
        kw["use_v2"] = use_v2
    elif use_v2:
        raise ValueError("this MONAI version has no SwinUNETR-V2 (use_v2)")
    with warnings.catch_warnings():         # MONAI 1.3/1.4 still require img_size but flag it as deprecated
        warnings.filterwarnings("ignore", message=".*img_size.*", category=FutureWarning)
        return SwinUNETR(**kw)
