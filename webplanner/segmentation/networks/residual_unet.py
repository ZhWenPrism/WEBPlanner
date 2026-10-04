"""nnU-Net-style residual encoder U-Net (ResEnc) for 3D segmentation.

Encoder: a stem convolution followed by stages of residual blocks; the first block of every stage after the first
downsamples with a strided convolution (the skip path uses average pooling and a 1x1x1 convolution, as in
nnU-Net's ``BasicBlockD``). Decoder: transposed-convolution upsampling, concatenation with the encoder skip,
``decoder_convs`` conv-norm-act layers per stage and a 1x1x1 segmentation head per stage. Instance normalisation and
leaky ReLU (negative slope 0.01) throughout. With ``deep_supervision=True`` and ``model.training`` the forward pass
returns a list of logits from full resolution to the coarsest decoder stage; otherwise a single tensor.

Defaults follow the nnU-Net ResEnc-M 3d_fullres configuration (6 stages, 32 to 320 features, 1-3-4-6-6-6 blocks).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

Stride = Union[int, Sequence[int]]


def as_triple(v: Stride) -> Tuple[int, int, int]:
    return (int(v),) * 3 if isinstance(v, int) else tuple(int(x) for x in v)


def default_strides(n_stages: int) -> List[Tuple[int, int, int]]:
    return [(1, 1, 1)] + [(2, 2, 2)] * (n_stages - 1)


def expand_per_stage(value: Union[int, Sequence[int]], n: int) -> List[int]:
    """Per-stage setting from an int or a sequence (truncated, or extended with its last entry)."""
    vals = [int(value)] * n if isinstance(value, int) else [int(v) for v in value]
    return (vals + vals[-1:] * max(0, n - len(vals)))[:n]


def init_weights(module: nn.Module, neg_slope: float = 1e-2) -> None:
    """He initialisation used by nnU-Net (``InitWeights_He``)."""
    if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(module.weight, a=neg_slope)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Pad or crop ``x`` to the spatial size of ``ref`` (inputs not divisible by the total stride)."""
    diff = [r - s for r, s in zip(ref.shape[2:], x.shape[2:])]
    if not any(diff):
        return x
    pad = []
    for d in reversed(diff):
        pad += [max(d, 0) // 2, max(d, 0) - max(d, 0) // 2]
    x = F.pad(x, pad)
    return x[(slice(None), slice(None)) + tuple(slice(0, r) for r in ref.shape[2:])]


class ConvNormAct(nn.Sequential):
    """Conv3d, InstanceNorm3d (affine), LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: Stride = 1, neg_slope: float = 1e-2,
                 act: bool = True) -> None:
        layers: List[nn.Module] = [nn.Conv3d(in_ch, out_ch, kernel_size, as_triple(stride), padding=kernel_size // 2),
                                   nn.InstanceNorm3d(out_ch, eps=1e-5, affine=True)]
        if act:
            layers.append(nn.LeakyReLU(neg_slope, inplace=True))
        super().__init__(*layers)


class ResidualBlock(nn.Module):
    """Two 3x3x3 conv-norm layers with an identity or (avg-pool + 1x1x1 conv + norm) skip."""

    def __init__(self, in_ch: int, out_ch: int, stride: Stride = 1, kernel_size: int = 3, neg_slope: float = 1e-2) -> None:
        super().__init__()
        stride = as_triple(stride)
        self.conv1 = ConvNormAct(in_ch, out_ch, kernel_size, stride, neg_slope)
        self.conv2 = ConvNormAct(out_ch, out_ch, kernel_size, 1, neg_slope, act=False)
        self.act = nn.LeakyReLU(neg_slope, inplace=True)
        skip: List[nn.Module] = []
        if any(s != 1 for s in stride):
            skip.append(nn.AvgPool3d(stride, stride, ceil_mode=True))
        if in_ch != out_ch or skip:
            skip += [nn.Conv3d(in_ch, out_ch, 1, bias=False), nn.InstanceNorm3d(out_ch, eps=1e-5, affine=True)]
        self.skip = nn.Sequential(*skip) if skip else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return self.act(out + match_size(self.skip(x), out))


class UNetDecoder(nn.Module):
    """Transposed-conv upsampling, skip concatenation, conv stack and a segmentation head per stage."""

    def __init__(self, features: Sequence[int], strides: Sequence[Tuple[int, int, int]], n_classes: int,
                 convs_per_stage: Union[int, Sequence[int]] = 1, kernel_size: int = 3, neg_slope: float = 1e-2) -> None:
        super().__init__()
        n_up = len(features) - 1
        convs_per_stage = [max(c, 1) for c in expand_per_stage(convs_per_stage, n_up)]
        self.up = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.heads = nn.ModuleList()
        for i in range(n_up):                       # i = 0 is the deepest decoder stage
            deep, skip = features[-1 - i], features[-2 - i]
            stride = strides[-1 - i]
            self.up.append(nn.ConvTranspose3d(deep, skip, stride, stride))
            layers = [ConvNormAct(2 * skip, skip, kernel_size, 1, neg_slope)]
            layers += [ConvNormAct(skip, skip, kernel_size, 1, neg_slope) for _ in range(convs_per_stage[i] - 1)]
            self.blocks.append(nn.Sequential(*layers))
            self.heads.append(nn.Conv3d(skip, n_classes, 1))

    def forward(self, skips: List[torch.Tensor], all_heads: bool) -> List[torch.Tensor]:
        """Returns the logits from full resolution to coarse (only the full-resolution one if not ``all_heads``)."""
        x = skips[-1]
        outputs = []
        for i, (up, block, head) in enumerate(zip(self.up, self.blocks, self.heads)):
            ref = skips[-2 - i]
            x = block(torch.cat([match_size(up(x), ref), ref], dim=1))
            if all_heads or i == len(self.up) - 1:
                outputs.append(head(x))
        return outputs[::-1]


class ResidualUNet3D(nn.Module):
    """Residual encoder U-Net.

    Args:
        in_ch: input channels.
        n_classes: output classes (5 for the WEBPlanner label map).
        features: channels per stage; the number of stages is ``len(features)``.
        deep_supervision: return logits of every decoder stage while training.
        strides: per-stage stride (int or 3-tuple); default 1 for the first stage, 2 afterwards.
        blocks_per_stage: residual blocks per encoder stage.
        decoder_convs: conv-norm-act layers per decoder stage.
    """

    def __init__(self, in_ch: int = 1, n_classes: int = 5, features: Sequence[int] = (32, 64, 128, 256, 320, 320),
                 deep_supervision: bool = True, strides: Optional[Sequence[Stride]] = None,
                 blocks_per_stage: Union[int, Sequence[int]] = (1, 3, 4, 6, 6, 6),
                 decoder_convs: Union[int, Sequence[int]] = 1, kernel_size: int = 3, neg_slope: float = 1e-2) -> None:
        super().__init__()
        n = len(features)
        if n < 2:
            raise ValueError("need at least two stages")
        self.strides = [as_triple(s) for s in strides] if strides is not None else default_strides(n)
        blocks_per_stage = expand_per_stage(blocks_per_stage, n)
        if len(self.strides) != n:
            raise ValueError("strides must have one entry per stage")
        self.features, self.n_classes, self.deep_supervision = list(features), n_classes, deep_supervision
        self.stem = ConvNormAct(in_ch, features[0], kernel_size, 1, neg_slope)
        self.stages = nn.ModuleList()
        prev = features[0]
        for f, s, nb in zip(features, self.strides, blocks_per_stage):
            blocks = [ResidualBlock(prev, f, s, kernel_size, neg_slope)]
            blocks += [ResidualBlock(f, f, 1, kernel_size, neg_slope) for _ in range(max(nb, 1) - 1)]
            self.stages.append(nn.Sequential(*blocks))
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
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        ds = self.deep_supervision and self.training
        outputs = self.decoder(skips, all_heads=ds)
        return outputs if ds else outputs[0]
