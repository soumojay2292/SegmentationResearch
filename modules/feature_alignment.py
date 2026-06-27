"""
Feature Alignment module (synopsis §1.4 — stabilised MAFFNet).

Aligns the activation statistics of two feature streams (e.g. SAM encoder
features and frequency-branch features) before fusion, by per-sample,
per-channel instance normalisation followed by a 1x1 channel projection and a
ReLU non-linearity.

    InstanceNorm2d  ->  1x1 Conv  ->  ReLU

Standalone and reusable: depends only on torch. Not integrated anywhere yet.
Does not import or modify the current MAFFNet implementation.
"""

import torch
import torch.nn as nn


class FeatureAlignment(nn.Module):
    """Instance-normalise + project + activate, for arbitrary channel widths.

    Args:
        in_channels:  number of input channels C_in (arbitrary, > 0).
        out_channels: number of output channels C_out. Defaults to in_channels.
        affine:       if True, InstanceNorm2d learns per-channel scale/shift.
                      Default False (pure normalisation for alignment).
        eps:          numerical epsilon for InstanceNorm2d.

    Shape:
        input:  (B, C_in,  H, W)
        output: (B, C_out, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        affine: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm = nn.InstanceNorm2d(in_channels, eps=eps, affine=affine)
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.proj(self.norm(x)))
