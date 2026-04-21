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
from sam2.build_sam import build_sam2 

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
        x_fft   = torch.fft.rfft2(x, norm='ortho')
        mag     = torch.abs(x_fft)                    # (B, C, H, W//2+1)
        mag_full = F.interpolate(mag, size=x.shape[2:], mode='bilinear',
                                 align_corners=False)
        # Normalise per-channel to [0,1]
        mag_min  = mag_full.flatten(2).min(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
        mag_max  = mag_full.flatten(2).max(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
        mag_norm = (mag_full - mag_min) / (mag_max - mag_min + 1e-8)

        freq_feat    = self.freq_conv(mag_norm)
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
# SAM2 Hiera-Large backbone wrapper
# ---------------------------------------------------------------------------
class SAM2HieraBackbone(nn.Module):
    """
    Wraps SAM2's image encoder (Hiera-Large).
    Returns 4-level feature maps at strides ~[4, 8, 16, 32].
    Channel counts for Hiera-Large: 144, 288, 576, 1152
    """
    HIERA_LARGE_CHANNELS = (144, 288, 576, 1152)

    def __init__(self, checkpoint: str = None):
        super().__init__()
        self._encoder = None
        self._channels = self.HIERA_LARGE_CHANNELS

        self._build_sam2(checkpoint)

    def _build_sam2(self, checkpoint):
        import os
        from hydra import initialize, compose
        from hydra.core.global_hydra import GlobalHydra

        # 🔥 Reset Hydra (important for repeated runs)
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        # 🔥 Point Hydra to SAM2 config folder
        config_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../src/sam2/sam2/configs")
        )

        with initialize(config_path=config_dir, version_base=None):
            sam2_model = build_sam2(
                config_file="sam2_hiera_l",
                ckpt_path=checkpoint
            )

        self._encoder = sam2_model.image_encoder

        for p in self._encoder.parameters():
            p.requires_grad = False

        self._encoder.eval()

        print("✅ SAM2 loaded correctly")

    @property
    def channels(self):
        return self._channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Returns (f1, f2, f3, f4) feature maps from shallow to deep.
        Each fi: (B, Ci, Hi, Wi)
        """
        with torch.no_grad():
            features = self._encoder(x)  # dict or list depending on SAM2 version
        # SAM2 image encoder returns backbone features as list/dict
        if isinstance(features, dict):
            # Typical SAM2 output: features['backbone_fpn'] = list of tensors
            feats = features.get('backbone_fpn', list(features.values()))
        elif isinstance(features, (list, tuple)):
            feats = features
        else:
            raise ValueError(f"Unexpected backbone output type: {type(features)}")

        # Ensure we have 4 levels; pad or trim as needed
        feats = list(feats)
        assert len(feats) == 4, f"Expected 4 feature levels, got {len(feats)}"
        return tuple(feats)   # (f1, f2, f3, f4) shallow→deep


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

    def __init__(self, checkpoint: str = None):
        super().__init__()

        # ---------- Backbone ----------
        self.backbone = SAM2HieraBackbone(checkpoint)
        c1, c2, c3, c4 = self.backbone.channels   # 144, 288, 576, 1152

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