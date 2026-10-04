"""Plain 3D U-Net comparator (nnU-Net ``PlainConvUNet`` layout).

Each encoder stage has ``convs_per_stage`` conv-norm-act layers, the first one strided for downsampling; the
decoder, deep supervision behaviour and weight initialisation are shared with :class:`ResidualUNet3D`, so the two
networks differ only in the residual encoder.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from .residual_unet import ConvNormAct, Stride, UNetDecoder, as_triple, default_strides, expand_per_stage, init_weights


class UNet3D(nn.Module):
    """Plain convolutional U-Net with the same signature as :class:`ResidualUNet3D`."""

    def __init__(self, in_ch: int = 1, n_classes: int = 5, features: Sequence[int] = (32, 64, 128, 256, 320, 320),
                 deep_supervision: bool = True, strides: Optional[Sequence[Stride]] = None,
                 convs_per_stage: Union[int, Sequence[int]] = 2, decoder_convs: Union[int, Sequence[int]] = 2,
                 kernel_size: int = 3, neg_slope: float = 1e-2) -> None:
        super().__init__()
        n = len(features)
        if n < 2:
            raise ValueError("need at least two stages")
        self.strides = [as_triple(s) for s in strides] if strides is not None else default_strides(n)
        if len(self.strides) != n:
            raise ValueError("strides must have one entry per stage")
        convs = [max(c, 1) for c in expand_per_stage(convs_per_stage, n)]
        self.features, self.n_classes, self.deep_supervision = list(features), n_classes, deep_supervision
        self.stages = nn.ModuleList()
        prev = in_ch
        for f, s, nc in zip(features, self.strides, convs):
            layers: List[nn.Module] = [ConvNormAct(prev, f, kernel_size, s, neg_slope)]
            layers += [ConvNormAct(f, f, kernel_size, 1, neg_slope) for _ in range(nc - 1)]
            self.stages.append(nn.Sequential(*layers))
            prev = f
        self.decoder = UNetDecoder(features, self.strides, n_classes, decoder_convs, kernel_size, neg_slope)
        self.apply(lambda m: init_weights(m, neg_slope))

    @property
    def total_stride(self) -> Tuple[int, int, int]:
        out = [1, 1, 1]
        for s in self.strides:
            out = [a * b for a, b in zip(out, s)]
        return tuple(out)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        ds = self.deep_supervision and self.training
        outputs = self.decoder(skips, all_heads=ds)
        return outputs if ds else outputs[0]
