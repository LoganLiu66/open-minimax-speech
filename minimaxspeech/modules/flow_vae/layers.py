#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Reference:
- https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/layers.py
- https://github.com/facebookresearch/dacvae/blob/main/dacvae/nn/layers.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


# Scripting this brings model speed up 1.4x
# @torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


def apply_parametrization_norm(module: nn.Module, norm: str = "none"):
    assert norm in ["none", "weight_norm"]
    if norm == "weight_norm":
        return weight_norm(module)
    else:
        return module

class NormConv1d(nn.Conv1d):
    """1D Causal Convolution with padding"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        norm: str = "weight_norm",  # Normalization method, value: "none", "weight_norm", "spectral_norm"
        causal: bool = False,
        pad_mode: str = "none",  # Padding mode, value: "none", "auto"
        **kwargs
    ):
        if pad_mode == "none":
            pad = (kernel_size - stride) * dilation // 2
        else:
            pad = 0

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            dilation=dilation,
            **kwargs
        )

        apply_parametrization_norm(self, norm)

        self.causal = causal
        self.pad_mode = pad_mode

    def pad(self, x: torch.Tensor):
        if self.pad_mode == "none":
            return x

        length = x.shape[-1]
        kernel_size = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0]
        dilation = self.dilation if isinstance(self.dilation, int) else self.dilation[0]
        stride = self.stride if isinstance(self.stride, int) else self.stride[0]

        effective_kernel_size = (kernel_size - 1) * dilation + 1
        padding_total = (effective_kernel_size - stride)
        n_frames = (length - effective_kernel_size + padding_total) / stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
        extra_padding = ideal_length - length

        if self.causal:
            pad_x = F.pad(x, (padding_total, extra_padding))
        else:
            padding_right = extra_padding // 2
            padding_left = padding_total - padding_right
            pad_x = F.pad(x, (padding_left, padding_right + extra_padding))

        return pad_x

    def forward(self, x: torch.Tensor):
        x = self.pad(x)
        return super().forward(x)