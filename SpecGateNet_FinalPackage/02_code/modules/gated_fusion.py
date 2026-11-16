"""
Lightweight Gated Fusion module (synopsis §1.4 — stabilised MAFFNet).

Fuses a spatial feature stream and a frequency feature stream with a
spatially-varying, channel-wise sigmoid gate at linear spatial complexity:

    g   = sigmoid( Conv1x1( [F_spatial || F_freq] ) )
    out = g * F_spatial + (1 - g) * F_freq

No attention, no MultiHeadAttention, no Transformer. The gate is a single 1x1
convolution over the concatenated streams. Both inputs and the output share the
same channel width C and spatial size (H, W).

Standalone and reusable: depends only on torch. Not integrated anywhere yet.
Does not import or modify the current MAFFNet implementation or the trainer.
"""

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """Sigmoid-gated convex fusion of two feature streams of equal width.

    Args:
        channels: common channel count C of both input streams and the output.

    Shape:
        f_spatial: (B, C, H, W)
        f_freq:    (B, C, H, W)
        output:    (B, C, H, W)

    (If the two streams have different widths, align them to a common C first,
    e.g. with the FeatureAlignment module.)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        # 1x1 conv: 2C -> C, produces a per-channel, per-pixel gate logit.
        self.gate = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=True)

    def forward(self, f_spatial: torch.Tensor, f_freq: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([f_spatial, f_freq], dim=1)))
        return g * f_spatial + (1.0 - g) * f_freq
