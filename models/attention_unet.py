"""Attention U-Net (Oktay et al., 2018) for binary segmentation.

Input:  (B, 3, H, W)  — RGB image
Output: (B, 1, H, W)  — raw logits (no sigmoid applied)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two 3×3 conv layers with BN + ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class AttentionGate(nn.Module):
    """Additive soft attention gate from Oktay et al. (2018).

    Produces a spatial attention map alpha in [0,1] and returns x * alpha,
    where x is the encoder skip and g is the decoder gating signal.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(F_g,   F_int, 1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.Wx = nn.Sequential(
            nn.Conv2d(F_l,   F_int, 1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        g : gating signal from decoder  (B, F_g, H, W)
        x : skip connection from encoder (B, F_l, H, W)
        returns attention-weighted x    (B, F_l, H, W)
        """
        g1 = self.Wg(g)
        x1 = self.Wx(x)
        # Guard against rounding differences from ConvTranspose2d
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(g1, size=x1.shape[-2:], mode="bilinear", align_corners=True)
        alpha = self.psi(self.relu(g1 + x1))   # (B, 1, H, W)
        return x * alpha


class AttentionUNet(nn.Module):
    """Attention U-Net: 4-level encoder, 1024-ch bottleneck, attention gates on every skip.

    Encoder channels : [64, 128, 256, 512]
    Bottleneck       : 1024
    Decoder channels : [512, 256, 128, 64]
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────
        self.enc1 = _conv_block(in_ch,  64)
        self.enc2 = _conv_block(64,    128)
        self.enc3 = _conv_block(128,   256)
        self.enc4 = _conv_block(256,   512)
        self.pool = nn.MaxPool2d(2)

        # ── Bottleneck ───────────────────────────────────────────
        self.bottleneck = _conv_block(512, 1024)

        # ── Attention gates ──────────────────────────────────────
        # F_g = decoder channels after upsample, F_l = encoder skip channels
        self.att4 = AttentionGate(F_g=512,  F_l=512,  F_int=256)
        self.att3 = AttentionGate(F_g=256,  F_l=256,  F_int=128)
        self.att2 = AttentionGate(F_g=128,  F_l=128,  F_int=64)
        self.att1 = AttentionGate(F_g=64,   F_l=64,   F_int=32)

        # ── Decoder ──────────────────────────────────────────────
        self.up4  = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = _conv_block(512 + 512, 512)    # cat(up, att_skip)

        self.up3  = nn.ConvTranspose2d(512,  256, kernel_size=2, stride=2)
        self.dec3 = _conv_block(256 + 256, 256)

        self.up2  = nn.ConvTranspose2d(256,  128, kernel_size=2, stride=2)
        self.dec2 = _conv_block(128 + 128, 128)

        self.up1  = nn.ConvTranspose2d(128,  64,  kernel_size=2, stride=2)
        self.dec1 = _conv_block(64  + 64,  64)

        self.final = nn.Conv2d(64, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Encoder ──────────────────────────────────────────────
        e1 = self.enc1(x)                   # (B,  64, H,    W)
        e2 = self.enc2(self.pool(e1))        # (B, 128, H/2,  W/2)
        e3 = self.enc3(self.pool(e2))        # (B, 256, H/4,  W/4)
        e4 = self.enc4(self.pool(e3))        # (B, 512, H/8,  W/8)

        # ── Bottleneck ───────────────────────────────────────────
        b  = self.bottleneck(self.pool(e4))  # (B,1024, H/16, W/16)

        # ── Decoder with attention gates ─────────────────────────
        d4 = self.up4(b)                                           # (B, 512, H/8, W/8)
        d4 = self.dec4(torch.cat([d4, self.att4(d4, e4)], dim=1)) # (B, 512, H/8, W/8)

        d3 = self.up3(d4)                                          # (B, 256, H/4, W/4)
        d3 = self.dec3(torch.cat([d3, self.att3(d3, e3)], dim=1)) # (B, 256, H/4, W/4)

        d2 = self.up2(d3)                                          # (B, 128, H/2, W/2)
        d2 = self.dec2(torch.cat([d2, self.att2(d2, e2)], dim=1)) # (B, 128, H/2, W/2)

        d1 = self.up1(d2)                                          # (B,  64, H,   W)
        d1 = self.dec1(torch.cat([d1, self.att1(d1, e1)], dim=1)) # (B,  64, H,   W)

        return self.final(d1)                                       # (B,   1, H,   W)
