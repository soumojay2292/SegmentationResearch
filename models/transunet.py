"""
TransUNet — official R50 + ViT-B/16 hybrid (Route B, via timm).

Replaces the legacy in-name-only TransUNet (archived in transunet_legacy.py).
Encoder: ResNet-50 stem + ViT-B/16 (12 layers, 768 dim, 12 heads, MLP 3072),
ImageNet-21k→1k pretrained — timm `vit_base_r50_s16_384.orig_in21k_ft_in1k`.
Decoder: CUP-style (cascaded upsampling) /16 → /1.

Public API preserved: TransUNet(in_ch=3, out_ch=1) → forward(x) returns
(B, out_ch, H, W) raw logits, so the existing trainer / loss / report /
benchmark all work unchanged. ImageNet normalisation is applied INSIDE forward
(input assumed in [0,1] from ToTensor), so no dataloader/trainer edit is needed.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

_TIMM_MODEL = "vit_base_r50_s16_384.orig_in21k_ft_in1k"
_PATCH_STRIDE = 16  # R50+ViT-B/16 total stride


class _CUPBlock(nn.Module):
    """Cascaded-Upsampler block: ×2 upsample + two conv-BN-ReLU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(self.up(x))


class TransUNet(nn.Module):
    """Official-style R50+ViT-B/16 hybrid + CUP decoder for binary segmentation."""

    def __init__(self, in_ch: int = 3, out_ch: int = 1,
                 pretrained: bool = True, img_size: int = 384):
        super().__init__()
        self.img_size = img_size

        # ── Hybrid encoder (ResNet-50 stem + ViT-B/16), ImageNet-21k→1k ──
        try:
            self.vit = timm.create_model(
                _TIMM_MODEL, pretrained=pretrained,
                num_classes=0, global_pool="", img_size=img_size,
            )
            self.pretrained_loaded = bool(pretrained)
        except Exception as e:                       # offline / download failure
            print(f"[transunet] pretrained load failed ({type(e).__name__}: {e}); "
                  f"falling back to random init")
            self.vit = timm.create_model(
                _TIMM_MODEL, pretrained=False,
                num_classes=0, global_pool="", img_size=img_size,
            )
            self.pretrained_loaded = False

        self.embed_dim  = self.vit.embed_dim                       # 768
        self.num_prefix = getattr(self.vit, "num_prefix_tokens", 1)  # 1 (cls)

        # ── ImageNet normalisation (input assumed [0,1]) ──
        self.register_buffer("norm_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("norm_std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # ── CUP decoder: /16 → /1 (four ×2 stages) ──
        self.decoder = nn.ModuleList([
            _CUPBlock(self.embed_dim, 256),
            _CUPBlock(256, 128),
            _CUPBlock(128, 64),
            _CUPBlock(64, 32),
        ])
        self.seg_head = nn.Conv2d(32, out_ch, 1)

    def _tokens_to_grid(self, feats: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # feats: (B, prefix+N, C) -> (B, C, h, w)
        feats = feats[:, self.num_prefix:, :]
        B, N, C = feats.shape
        h, w = H // _PATCH_STRIDE, W // _PATCH_STRIDE
        if h * w != N:
            h = w = int(round(math.sqrt(N)))
        return feats.transpose(1, 2).reshape(B, C, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        x = (x - self.norm_mean) / self.norm_std         # ImageNet norm (TransUNet-only)

        feats = self.vit.forward_features(x)             # (B, prefix+N, C)
        f = self._tokens_to_grid(feats, H, W)            # (B, C, H/16, W/16)

        for blk in self.decoder:
            f = blk(f)                                   # ×16 upsample total
        if f.shape[-2:] != (H, W):
            f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
        return self.seg_head(f)                          # (B, out_ch, H, W) logits
