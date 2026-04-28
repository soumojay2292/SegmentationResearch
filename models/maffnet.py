"""
MAFFNet: Multi-scale Attention Feature Fusion Network for skin lesion segmentation.
Paper-accurate implementation.

Architecture:
  Backbone  : SAM2 Hiera-Large (frozen), extracts 4-level features
  MMPA      : Multi-scale Morphological Pyramid Attention (dilated conv r=3,5,7)
  FGFF      : Frequency-Guided Feature Fusion (FFT magnitude + spatial fusion)
  Decoder   : Full U-Net style (no channel reduction)
  RDB       : Reconstruction Decoder Block (RGB reconstruction from deep features)

Outputs: p1, p2, p3, p4  (segmentation logits, each upsampled to 384×384)
         rec              (reconstructed RGB image, 384×384)
"""
import os
import csv as _csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1, groups=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, padding=d * (k // 2) if p == 1 else p,
                      dilation=d, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


# ---------------------------------------------------------------------------
# MMPA – Multi-scale Morphological Pyramid Attention
# ---------------------------------------------------------------------------
class MMPA(nn.Module):
    """
    Multi-scale dilated convolution (rates 3, 5, 7) + channel attention.
    No channel reduction – keeps full channel width throughout.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.branch3 = ConvBNReLU(channels, channels, k=3, d=3)
        self.branch5 = ConvBNReLU(channels, channels, k=3, d=5)
        self.branch7 = ConvBNReLU(channels, channels, k=3, d=7)
        self.branch1 = ConvBNReLU(channels, channels, k=1, p=0)  # context

        # Fuse 4 branches → same channel width
        self.fuse = ConvBNReLU(channels * 4, channels, k=1, p=0)

        # Channel attention (SE-style)
        self.ca_avg = nn.AdaptiveAvgPool2d(1)
        self.ca_max = nn.AdaptiveMaxPool2d(1)
        self.ca_fc  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        b1 = self.branch1(x)

        out = self.fuse(torch.cat([b3, b5, b7, b1], dim=1))

        # Channel attention
        ca = self.ca_fc(self.ca_avg(out) + self.ca_max(out))
        ca = ca.view(out.size(0), -1, 1, 1)
        return out * ca + x  # residual


# ---------------------------------------------------------------------------
# FGFF – Frequency-Guided Feature Fusion
# ---------------------------------------------------------------------------
class FGFF(nn.Module):
    """
    FFT → 2D magnitude spectrum → normalise → conv → scale spatial features.
    No channel reduction.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.freq_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.spatial_conv = ConvBNReLU(channels, channels)
        self.fuse = ConvBNReLU(channels * 2, channels, k=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frequency branch
        # --- Force FFT in float32 to avoid cuFFT fp16 limitation ---
        x_float = x.float()
        x_fft = torch.fft.rfft2(x_float, norm='ortho')
        mag     = torch.abs(x_fft)                    # (B, C, H, W//2+1)
        mag_full = F.interpolate(mag, size=x.shape[2:], mode='bilinear',
                                 align_corners=False)
        # Normalise per-channel to [0,1]
        mag_min  = mag_full.flatten(2).min(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
        mag_max  = mag_full.flatten(2).max(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
        mag_norm = (mag_full - mag_min) / (mag_max - mag_min + 1e-8)

        freq_feat = self.freq_conv(mag_norm.to(x.dtype))
        spatial_feat = self.spatial_conv(x)
        fused        = self.fuse(torch.cat([spatial_feat, freq_feat], dim=1))
        return fused + x  # residual


# ---------------------------------------------------------------------------
# Decoder block (U-Net style, no channel reduction)
# ---------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch + skip_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# RDB – Reconstruction Decoder Block
# ---------------------------------------------------------------------------
class RDB(nn.Module):
    """
    Reconstruct RGB image from deepest encoder feature + skip connections.
    Progressive upsampling with skip additions at each level.
    """
    def __init__(self, channels: List[int]):
        """channels: [c4, c3, c2, c1] (deep → shallow)"""
        super().__init__()
        c4, c3, c2, c1 = channels
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(c4, c3),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(c3 * 2, c2),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(c2 * 2, c1),
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(c1 * 2, 64),
        )
        self.out_conv = nn.Sequential(
            ConvBNReLU(64, 32),
            nn.Conv2d(32, 3, 1),
            nn.Tanh(),
        )

    def forward(self, f4, f3, f2, f1) -> torch.Tensor:
        x = self.up4(f4)
        if x.shape[2:] != f3.shape[2:]:
            x = F.interpolate(x, size=f3.shape[2:], mode='bilinear', align_corners=False)
        x = self.up3(torch.cat([x, f3], dim=1))
        if x.shape[2:] != f2.shape[2:]:
            x = F.interpolate(x, size=f2.shape[2:], mode='bilinear', align_corners=False)
        x = self.up2(torch.cat([x, f2], dim=1))
        if x.shape[2:] != f1.shape[2:]:
            x = F.interpolate(x, size=f1.shape[2:], mode='bilinear', align_corners=False)
        x = self.up1(torch.cat([x, f1], dim=1))
        return self.out_conv(x)


# ---------------------------------------------------------------------------
# SAM2 Hiera backbone wrapper  (used when checkpoint= path is given)
# ---------------------------------------------------------------------------
class SAM2Encoder(nn.Module):
    """
    Wraps a SAM2 ImageEncoder module (Hiera trunk + FPN neck) to produce
    4 multi-scale feature maps compatible with MAFFNet.

    SAM2 Hiera-Large with scalp=1 yields 3 FPN outputs:
        backbone_fpn[0] : highest res  (~stride 4)
        backbone_fpn[1] : mid res      (~stride 8)
        backbone_fpn[2] : low res      (~stride 16)   ← vision_features
    We derive a 4th scale by halving backbone_fpn[2].
    All outputs are 256-channel (FPN d_model).
    """
    channels: list = [256, 256, 256, 256]

    def __init__(self, image_encoder: nn.Module):
        super().__init__()
        self.encoder      = image_encoder
        self.out_channels = self.channels
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            out = self.encoder(x)

        if isinstance(out, dict):
            features = out["backbone_fpn"]   # list ordered high→low res
        else:
            # Fallback: single tensor → synthesise 4 scales via upsampling
            feat = out
            f1 = F.interpolate(feat, scale_factor=8.0, mode='bilinear', align_corners=False)
            f2 = F.interpolate(feat, scale_factor=4.0, mode='bilinear', align_corners=False)
            f3 = F.interpolate(feat, scale_factor=2.0, mode='bilinear', align_corners=False)
            return f1, f2, f3, feat

        n = len(features)
        if n >= 4:
            return features[0], features[1], features[2], features[3]
        elif n == 3:
            f4 = F.interpolate(features[2], scale_factor=0.5,
                               mode='bilinear', align_corners=False)
            return features[0], features[1], features[2], f4
        else:
            # Fewer than expected — upsample from the coarsest available
            feat = features[-1]
            f1   = F.interpolate(feat, scale_factor=4.0, mode='bilinear', align_corners=False)
            f2   = F.interpolate(feat, scale_factor=2.0, mode='bilinear', align_corners=False)
            f3   = feat
            f4   = F.interpolate(feat, scale_factor=0.5, mode='bilinear', align_corners=False)
            return f1, f2, f3, f4


# ---------------------------------------------------------------------------
# SAM ViT-H backbone wrapper  (used when encoder= object is passed directly)
# ---------------------------------------------------------------------------
class SAMEncoder(nn.Module):
    """Wraps SAM ViT-H image_encoder to produce 4 multi-scale feature maps."""
    channels: list = [256, 256, 256, 256]

    def __init__(self, sam_model):
        super().__init__()
        self.encoder = sam_model.image_encoder
        self.out_channels = self.channels
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            feat = self.encoder(x)          # [B, 256, H/16, W/16]
        # Build 4 scales: f1 largest (stride 2) → f4 smallest (stride 16)
        f4 = feat
        f3 = F.interpolate(feat, scale_factor=2.0, mode='bilinear', align_corners=False)
        f2 = F.interpolate(feat, scale_factor=4.0, mode='bilinear', align_corners=False)
        f1 = F.interpolate(feat, scale_factor=8.0, mode='bilinear', align_corners=False)
        return f1, f2, f3, f4

# ---------------------------------------------------------------------------
# MAFFNet – full model
# ---------------------------------------------------------------------------
class MAFFNet(nn.Module):
    """
    Multi-scale Attention Feature Fusion Network.

    Input : (B, 3, 384, 384)
    Output: p1, p2, p3, p4  – (B, 1, 384, 384) each  [logits]
            rec              – (B, 3, 384, 384)        [Tanh, ≈[-1,1]]
    """

    def __init__(self, encoder=None, checkpoint: str = None):
        super().__init__()

        # ---------- Backbone ----------
        if encoder is not None:
            # Caller already constructed a SAM model object — wrap its image_encoder
            self.backbone = SAMEncoder(encoder)
        elif checkpoint is not None:
            # Load SAM2 Hiera-Large via the project's sam2_loader (no segment_anything needed)
            from models.sam2_loader import load_sam2_encoder
            image_encoder = load_sam2_encoder(
                config_path="src/sam2/sam2/configs/sam2/sam2_hiera_l.yaml",
                checkpoint=checkpoint,
                device="cpu",
            )
            self.backbone = SAM2Encoder(image_encoder)
        else:
            raise ValueError("Provide either 'encoder' (a SAM model) or 'checkpoint' (path to .pth).")
        print("Encoder type:", type(self.backbone.encoder))
        c1, c2, c3, c4 = self.backbone.channels   # all 256 for SAM ViT-H

        # ---------- MMPA per level ----------
        self.mmpa1 = MMPA(c1)
        self.mmpa2 = MMPA(c2)
        self.mmpa3 = MMPA(c3)
        self.mmpa4 = MMPA(c4)

        # ---------- FGFF per level ----------
        self.fgff1 = FGFF(c1)
        self.fgff2 = FGFF(c2)
        self.fgff3 = FGFF(c3)
        self.fgff4 = FGFF(c4)

        # ---------- Decoder (U-Net, no channel reduction) ----------
        self.dec4 = DecoderBlock(c4,     c3, c3)    # up: c4 → c3
        self.dec3 = DecoderBlock(c3,     c2, c2)    # up: c3 → c2
        self.dec2 = DecoderBlock(c2,     c1, c1)    # up: c2 → c1
        self.dec1 = DecoderBlock(c1,     c1, c1)    # up: c1 → c1 (no skip at lowest)

        # ---------- Segmentation heads (one per decoder level) ----------
        self.head4 = nn.Conv2d(c3, 1, 1)
        self.head3 = nn.Conv2d(c2, 1, 1)
        self.head2 = nn.Conv2d(c1, 1, 1)
        self.head1 = nn.Conv2d(c1, 1, 1)

        # ---------- RDB ----------
        self.rdb = RDB([c4, c3, c2, c1])

        # Input size for upsampling predictions
        self._input_size = (384, 384)

    def forward(self, x: torch.Tensor):
        H, W = x.shape[2], x.shape[3]
        self._input_size = (H, W)

        # --- Backbone ---
        f1, f2, f3, f4 = self.backbone(x)   # shallow → deep

        # --- MMPA ---
        f1 = self.mmpa1(f1)
        f2 = self.mmpa2(f2)
        f3 = self.mmpa3(f3)
        f4 = self.mmpa4(f4)

        # --- FGFF ---
        f1 = self.fgff1(f1)
        f2 = self.fgff2(f2)
        f3 = self.fgff3(f3)
        f4 = self.fgff4(f4)

        # --- Decoder ---
        d4 = self.dec4(f4, f3)     # stride ~16
        d3 = self.dec3(d4, f2)     # stride ~8
        d2 = self.dec2(d3, f1)     # stride ~4
        # Final upsample: no skip connection at level 1 (use zero skip)
        up_d2 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        zero_skip = torch.zeros_like(up_d2)
        d1 = self.dec1.conv(torch.cat([up_d2, zero_skip], dim=1))

        # --- Segmentation heads → upsample to input size ---
        def _up(t):
            return F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False)

        p4 = _up(self.head4(d4))
        p3 = _up(self.head3(d3))
        p2 = _up(self.head2(d2))
        p1 = _up(self.head1(d1))

        # --- RDB: reconstruct RGB ---
        rec = self.rdb(f4, f3, f2, f1)
        rec = F.interpolate(rec, size=(H, W), mode='bilinear', align_corners=False)

        return p1, p2, p3, p4, rec