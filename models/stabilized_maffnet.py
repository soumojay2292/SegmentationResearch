"""
StabilizedMAFFNet — the synopsis architecture (§1.4), assembled from approved
reusable modules. A NEW model; it does not modify or replace models/maffnet.py.

Pipeline (per encoder scale):
    SAM2 encoder feature  ->  { spatial: FeatureAlignment
                                frequency: FrequencyBranch -> FeatureAlignment }
                          ->  GatedFusion  ->  fused feature
    fused features  ->  projected (1x1) skip connections  ->  reused decoder
                    ->  segmentation heads (+ reused RDB reconstruction)

Reuses (import, unchanged): SAM2Encoder / SAMEncoder, DecoderBlock, RDB,
ConvBNReLU from models.maffnet; FeatureAlignment, FrequencyBranch, GatedFusion
from modules.

Output interface is identical to MAFFNet: returns (p1, p2, p3, p4, rec), so the
existing MAFFNetTrainer and its boundary supervision work without changes.
Constructor mirrors MAFFNet(encoder=..., checkpoint=...) for train.py drop-in.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.maffnet import SAM2Encoder, SAMEncoder, DecoderBlock, RDB, ConvBNReLU
from modules.feature_alignment import FeatureAlignment
from modules.frequency_branch import FrequencyBranch
from modules.gated_fusion import GatedFusion


class StabilizedMAFFNet(nn.Module):
    """SAM2 encoder + dual (spatial / frequency) streams + gated fusion + decoder.

    Input : (B, 3, 384, 384)
    Output: p1, p2, p3, p4  — (B, 1, 384, 384) each [logits]
            rec             — (B, 3, 384, 384)      [Tanh, ~[-1,1]]
    """

    def __init__(self, encoder=None, checkpoint: str = None, cutoff: float = 0.1):
        super().__init__()

        # ---------- Backbone (reused, frozen) ----------
        if encoder is not None:
            self.backbone = SAMEncoder(encoder)
        elif checkpoint is not None:
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

        c1, c2, c3, c4 = self.backbone.channels          # all 256 for SAM2 FPN
        chans = [c1, c2, c3, c4]

        # ---------- Per-scale dual streams ----------
        self.freq    = nn.ModuleList([FrequencyBranch(cutoff)  for _ in chans])   # frequency stream
        self.align_s = nn.ModuleList([FeatureAlignment(c, c)   for c in chans])   # spatial alignment
        self.align_f = nn.ModuleList([FeatureAlignment(c, c)   for c in chans])   # frequency alignment
        self.fuse    = nn.ModuleList([GatedFusion(c)           for c in chans])   # gated fusion

        # ---------- Projected skip connections (1x1) for the 3 skip levels ----------
        self.skip_proj = nn.ModuleList([
            nn.Conv2d(c1, c1, kernel_size=1),
            nn.Conv2d(c2, c2, kernel_size=1),
            nn.Conv2d(c3, c3, kernel_size=1),
        ])

        # ---------- Decoder (reused DecoderBlock; mirrors MAFFNet layout) ----------
        self.dec4 = DecoderBlock(c4, c3, c3)    # up: c4 -> c3, skip = proj(fused3)
        self.dec3 = DecoderBlock(c3, c2, c2)    # up: c3 -> c2, skip = proj(fused2)
        self.dec2 = DecoderBlock(c2, c1, c1)    # up: c2 -> c1, skip = proj(fused1)
        self.dec1_up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1_conv = nn.Sequential(ConvBNReLU(c1, c1), ConvBNReLU(c1, c1))

        # ---------- Segmentation heads ----------
        self.head4 = nn.Conv2d(c3, 1, 1)
        self.head3 = nn.Conv2d(c2, 1, 1)
        self.head2 = nn.Conv2d(c1, 1, 1)
        self.head1 = nn.Conv2d(c1, 1, 1)

        # ---------- Auxiliary reconstruction (reused RDB) ----------
        self.rdb = RDB([c4, c3, c2, c1])

        # ---------- ImageNet/SAM input normalisation ----------
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor):
        H, W = x.shape[-2], x.shape[-1]
        x_norm = (x - self.img_mean) / self.img_std

        # --- Backbone (frozen): shallow -> deep ---
        f1, f2, f3, f4 = self.backbone(x_norm)
        feats = [f1, f2, f3, f4]

        # --- Dual-stream alignment + gated fusion, per scale ---
        fused = []
        for i, f in enumerate(feats):
            fs = self.align_s[i](f)                       # spatial stream
            ff = self.align_f[i](self.freq[i](f))         # frequency stream (high-pass -> align)
            fused.append(self.fuse[i](fs, ff))            # gated fusion
        g1, g2, g3, g4 = fused

        # --- Projected skip connections ---
        s1 = self.skip_proj[0](g1)
        s2 = self.skip_proj[1](g2)
        s3 = self.skip_proj[2](g3)

        # --- Decoder ---
        d4 = self.dec4(g4, s3)
        d3 = self.dec3(d4, s2)
        d2 = self.dec2(d3, s1)
        d1 = self.dec1_conv(self.dec1_up(d2))             # finest skip-less refine

        def _up(t):
            return F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)

        p4 = _up(self.head4(d4))
        p3 = _up(self.head3(d3))
        p2 = _up(self.head2(d2))
        p1 = _up(self.head1(d1))

        # --- Auxiliary reconstruction ---
        rec = self.rdb(g4, g3, g2, g1)
        rec = F.interpolate(rec, size=(H, W), mode="bilinear", align_corners=False)

        return p1, p2, p3, p4, rec
