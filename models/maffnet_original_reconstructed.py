"""
maffnet_original_reconstructed.py
=================================
Paper-faithful MAFF-Net reproduction, built strictly from the frozen blueprint
in ``reports/implementation/MAFFNET_FINAL_DESIGN_SPECIFICATION.md`` (the single
source of truth). All hidden widths are fixed integers taken from that document;
no symbolic dimension remains.

This file is SELF-CONTAINED and does not import from ``models/maffnet.py``,
``models/stabilized_maffnet.py``, or ``models/maffnet_original.py``.

--------------------------------------------------------------------------------
MILESTONE 1 SCOPE (this file currently implements ONLY):
    1. Frozen SAM Encoder
    2. MMPA
    3. Projection Layers (spatial)

NOT YET IMPLEMENTED (later milestones): Frequency Encoder, FGFF, EDB, RDB,
Deep Supervision, Segmentation Heads, Loss, Decoder.

forward() returns only:
    {
        "encoder_features":  [X1, X2, X3, X4],
        "mmpa_features":     [S1, S2, S3, S4],
        "projected_features":[Ps1, Ps2, Ps3, Ps4],
    }
--------------------------------------------------------------------------------

Frozen constants (Design Freeze §3, §8):
    Input                : 3 x 384 x 384
    Encoder stages       : [144, 288, 576, 1152] @ strides [4, 8, 16, 32]
                           -> resolutions [96, 48, 24, 12]
    MMPA reduced dim     : 32   (projection C -> 32)
    MMPA perception      : full 3x3 convs @ 32, dilations {3, 5, 7}
    WetGeneration        : on NATIVE channel C  (GAP -> C -> C/2 -> 3 -> softmax)
    Gamma                : R^32, init 1e-7
    Spatial projection   : Conv1x1(C_i -> 80) for every stage
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Frozen widths (Design Freeze §3).
ENC_CHANNELS: Tuple[int, int, int, int] = (144, 288, 576, 1152)
ENC_STRIDES: Tuple[int, int, int, int] = (4, 8, 16, 32)
MMPA_PROJ_DIM: int = 32
MMPA_DILATIONS: Tuple[int, int, int] = (3, 5, 7)
SPATIAL_PROJ_WIDTH: int = 80

# Frozen frequency-stream widths (Design Freeze §1.4, §4).
FREQ_EMBED: int = 128
FREQ_BLOCKS_PER_STAGE: int = 2
FREQ_HEADS: int = 4
FREQ_FFN_RATIO: int = 2
FREQ_DOWNSAMPLE: float = 0.25          # interp factor 384 -> 96
FREQ_PROJ_WIDTH: int = 80

# Frozen FGFF widths (Design Freeze §1.5).
FUSION_WIDTH: int = 80
FGFF_DILATIONS: Tuple[int, int, int, int] = (1, 3, 5, 7)

# Frozen decoder width (Design Freeze §1.6, §5).
DECODER_WIDTH: int = 64


# ============================================================================
#  1. Frozen SAM Encoder  (Design Freeze §1.1)
# ============================================================================

class _DummyHieraTrunk(nn.Module):
    """Shape-compatible placeholder used ONLY when the real SAM2 Hiera-Large
    weights are unavailable. Emits four maps at strides 4/8/16/32 with the frozen
    channel widths so the network is constructible and shape-testable without the
    checkpoint. Not part of the architecture."""

    def __init__(self, channels: Tuple[int, int, int, int] = ENC_CHANNELS):
        super().__init__()
        self.stem = nn.Conv2d(3, channels[0], kernel_size=7, stride=4, padding=3)
        self.down = nn.ModuleList([
            nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1),
            nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1),
            nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1),
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x1 = self.stem(x)        # H/4
        x2 = self.down[0](x1)    # H/8
        x3 = self.down[1](x2)    # H/16
        x4 = self.down[2](x3)    # H/32
        return [x1, x2, x3, x4]


class FrozenSAMEncoder(nn.Module):
    """Frozen SAM2 Hiera-Large image encoder (Design Freeze §1.1).

    Parameters are frozen (``requires_grad = False``). Loads the real SAM2 trunk
    when available, else falls back to a shape-compatible placeholder. Output:
    ``[X1, X2, X3, X4]`` at strides 4/8/16/32, channels (144, 288, 576, 1152).
    """

    def __init__(self, checkpoint: Optional[str] = None,
                 config_file: str = "sam2_hiera_l.yaml"):
        super().__init__()
        self.out_channels: Tuple[int, int, int, int] = ENC_CHANNELS
        self._pretrained = False
        try:
            from sam2.build_sam import build_sam2
            sam2 = build_sam2(config_file=config_file, ckpt_path=checkpoint,
                              device="cpu")
            self.trunk = sam2.image_encoder.trunk
            self._pretrained = True
        except Exception:
            self.trunk = _DummyHieraTrunk(self.out_channels)
            self._pretrained = False

        for p in self.parameters():
            p.requires_grad_(False)

    @property
    def is_pretrained(self) -> bool:
        return self._pretrained

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.trunk(x)
        if isinstance(feats, dict):
            feats = list(feats.values())
        return list(feats)[:4]


# ============================================================================
#  2. MMPA - Mixed Multi-scale Perception Adaptor  (Design Freeze §1.2)
# ============================================================================

class PerceptionNet(nn.Module):
    """One perceptionNet: two FULL 3x3 convolutions at the 32-d reduced space,
    a standard one (d=1) then a dilated one (d=r), with a GELU between
    (Design Freeze §1.2 / Eq. 3). Dilation r gives effective RF (1 + 2r)."""

    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=dilation,
                               dilation=dilation)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.act(self.conv1(z)))


class WeightGenerator(nn.Module):
    """WetGeneration operating on the NATIVE encoder channel C (Design Freeze
    §1.2): GAP -> Linear(C -> C/2) -> GELU -> Linear(C/2 -> 3) -> softmax."""

    def __init__(self, in_channels: int, n_experts: int = 3):
        super().__init__()
        hidden = max(1, in_channels // 2)
        self.fc1 = nn.Linear(in_channels, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, n_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = F.adaptive_avg_pool2d(x, 1).flatten(1)   # (B, C) from native feature
        a = self.fc2(self.act(self.fc1(g)))          # (B, n_experts)
        return torch.softmax(a, dim=-1)


class MMPA(nn.Module):
    """Mixed Multi-scale Perception Adaptor (Design Freeze §1.2, Eqs. 2-5).

        Z          = GELU(Conv1x1(LN(X)))            # project C -> 32
        E_j(Z)     = perceptionNet_j(Z),  j=1..3     # full 3x3, dil 3/5/7
        alpha      = WetGeneration(X)                # softmax over 3 experts (native C)
        E_fused    = sum_j alpha_j * E_j(Z)
        P_enh      = Z + (E_fused * Gamma)           # Gamma in R^32, init 1e-7
        X_adapted  = GELU(Conv1x1(P_enh))            # reconstruct 32 -> C
    """

    def __init__(self, in_channels: int, proj_dim: int = MMPA_PROJ_DIM,
                 dilations: Tuple[int, ...] = MMPA_DILATIONS):
        super().__init__()
        self.in_channels = in_channels
        self.proj_dim = proj_dim

        self.ln = nn.GroupNorm(1, in_channels)                 # LN over channels
        self.proj = nn.Conv2d(in_channels, proj_dim, kernel_size=1)
        self.proj_act = nn.GELU()

        self.experts = nn.ModuleList([
            PerceptionNet(proj_dim, d) for d in dilations
        ])
        self.weight_gen = WeightGenerator(in_channels, n_experts=len(dilations))

        self.gamma = nn.Parameter(torch.full((proj_dim,), 1e-7))

        self.recon = nn.Conv2d(proj_dim, in_channels, kernel_size=1)
        self.recon_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj_act(self.proj(self.ln(x)))               # (B, 32, H, W)
        alpha = self.weight_gen(x)                             # (B, 3) native C

        e_fused = torch.zeros_like(z)
        for j, expert in enumerate(self.experts):
            e_fused = e_fused + alpha[:, j].view(-1, 1, 1, 1) * expert(z)

        p_enh = z + e_fused * self.gamma.view(1, -1, 1, 1)
        return self.recon_act(self.recon(p_enh))               # (B, C, H, W)


# ============================================================================
#  3. Projection Layers  (Design Freeze §1.3)
# ============================================================================

class SpatialProjection(nn.Module):
    """Conv1x1(C_i -> 80) per encoder stage (Design Freeze §1.3)."""

    def __init__(self, channels: Tuple[int, ...] = ENC_CHANNELS,
                 width: int = SPATIAL_PROJ_WIDTH):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Conv2d(c, width, kernel_size=1) for c in channels
        ])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.proj[i](f) for i, f in enumerate(feats)]


# ============================================================================
#  Milestone 2A — FFT preprocessing  (Design Freeze §1.4)
# ============================================================================

class FFTPreprocess(nn.Module):
    """RGB -> grayscale -> x255 -> FFT2 -> fftshift -> magnitude -> min-max ->
    pseudo-RGB -> downsample to 96x96 (Design Freeze §1.4 / Eqs. 6-7).

        B x 3 x 384 x 384
          -> grayscale (mean over RGB) x 255            B x 384 x 384
          -> FFT2 -> fftshift -> |.|                    B x 384 x 384
          -> per-sample min-max in [0,1]
          -> replicate to pseudo-RGB                    B x 3 x 384 x 384
          -> interpolate x0.25                          B x 3 x 96 x 96
    """

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        x_gray = x.mean(dim=1) * 255.0                  # (B, H, W)

        f = torch.fft.fft2(x_gray)                      # complex (B, H, W)
        f = torch.fft.fftshift(f, dim=(-2, -1))         # centre spectrum
        mag = torch.abs(f)                              # |F_shifted|

        b = mag.shape[0]
        flat = mag.view(b, -1)
        mn = flat.min(dim=1).values.view(b, 1, 1)
        mx = flat.max(dim=1).values.view(b, 1, 1)
        m = (mag - mn) / (mx - mn)                      # paper-faithful (no eps)

        pseudo_rgb = m.unsqueeze(1).repeat(1, 3, 1, 1)  # (B, 3, 384, 384)
        return F.interpolate(pseudo_rgb, scale_factor=FREQ_DOWNSAMPLE,
                             mode="bilinear", align_corners=False)  # (B,3,96,96)


# ============================================================================
#  Milestone 2A — Frequency Encoder  (Design Freeze §1.4, §4)
# ============================================================================

class FreqTransformerBlock(nn.Module):
    """Lightweight frequency Transformer block (Design Freeze §4 / Eq. 8).

        P         = Conv1x1(x)
        Q,K,V     = Split(DWConv3x3(Conv1x1(P)))
        Attn      = Softmax((Q K^T / ||.||) * T) V        # channel attention (MDTA-style)
        Z         = P + Conv1x1(Attn)
        out       = GroupNorm(Z) + FFN(GroupNorm(Z))

    E=128, heads=4 (d_k=32), FFN ratio 2, depthwise Q/K/V, learnable temperature
    T in R^{heads x 1 x 1}, GroupNorm(1). Channel (head_dim x head_dim) attention
    is used so the operation is independent of the H*W token count.
    """

    def __init__(self, dim: int = FREQ_EMBED, num_heads: int = FREQ_HEADS,
                 ffn_ratio: int = FREQ_FFN_RATIO):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.proj_in = nn.Conv2d(dim, dim, kernel_size=1)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1,
                                groups=dim * 3)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1)

        self.norm1 = nn.GroupNorm(1, dim)
        self.norm2 = nn.GroupNorm(1, dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * ffn_ratio, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * ffn_ratio, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        p = self.proj_in(x)

        q, k, v = self.qkv_dw(self.qkv(p)).chunk(3, dim=1)

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(b, self.num_heads, self.head_dim, h * w)

        q, k, v = heads(q), heads(k), heads(v)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature   # (B,heads,d,d)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(b, c, h, w)
        out = self.proj_out(out)

        z = p + out
        z = self.norm1(z)
        return z + self.ffn(self.norm2(z))


class FrequencyEncoder(nn.Module):
    """Hierarchical frequency-domain encoder (Design Freeze §1.4).

    FFT pseudo-RGB at 96x96 -> stem Conv3x3(3->128) -> 4 stages of 2 Transformer
    blocks each, with stride-2 downsampling between stages. Outputs four maps at
    128 channels and resolutions 96/48/24/12.
    """

    def __init__(self, embed: int = FREQ_EMBED,
                 blocks_per_stage: int = FREQ_BLOCKS_PER_STAGE,
                 n_stages: int = 4):
        super().__init__()
        self.fft = FFTPreprocess()
        self.stem = nn.Conv2d(3, embed, kernel_size=3, padding=1)
        self.stages = nn.ModuleList([
            nn.Sequential(*[FreqTransformerBlock(embed)
                            for _ in range(blocks_per_stage)])
            for _ in range(n_stages)
        ])
        # Stride-2 downsamples between the n_stages (n_stages - 1 of them).
        self.downs = nn.ModuleList([
            nn.Conv2d(embed, embed, kernel_size=3, stride=2, padding=1)
            for _ in range(n_stages - 1)
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        f = self.stem(self.fft(x))            # (B, 128, 96, 96)
        feats: List[torch.Tensor] = []
        h = self.stages[0](f)
        feats.append(h)                       # F1 @96
        for down, stage in zip(self.downs, self.stages[1:]):
            h = stage(down(h))
            feats.append(h)                   # F2 @48, F3 @24, F4 @12
        return feats


# ============================================================================
#  Milestone 2A — Frequency Projection Layers  (Design Freeze §1.3)
# ============================================================================

class FrequencyProjection(nn.Module):
    """Conv1x1(128 -> 80) per stage (Design Freeze §1.3)."""

    def __init__(self, embed: int = FREQ_EMBED, width: int = FREQ_PROJ_WIDTH,
                 n_stages: int = 4):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Conv2d(embed, width, kernel_size=1) for _ in range(n_stages)
        ])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.proj[i](f) for i, f in enumerate(feats)]


# ============================================================================
#  Milestone 2B — FGFF (Frequency-Guided Feature Fusion)  (Design Freeze §1.5)
# ============================================================================

class FGFF(nn.Module):
    """Frequency-Guided Feature Fusion (Design Freeze §1.5 / Eq. 9), width 80.

    Spatial branch (multi-scale dilated convolutions on Ps):
        b_i              = DilConv3x3(Conv1x1(Ps), dilation=d_i),  d_i in {1,3,5,7}
        f_spatial_fused  = Conv3x3(Cat[b0,b1,b2,b3])
    Frequency branch (attention from Pf):
        f_freq_attn      = sigma(Conv3x3([AvgPool_c(Pf); MaxPool_c(Pf)]))
    Fusion (beta scaling + residual addition, Milestone 2B requirement):
        G                = f_spatial_fused + beta * (f_freq_attn * f_spatial_fused)

    `beta in R^80` is a learnable per-channel weight on the frequency-guided
    increment. (Eq. 9 states the pure gate `beta*(attn*spatial)`; this milestone
    adds it residually so the spatial pathway is preserved.) Output width 80.
    """

    def __init__(self, width: int = FUSION_WIDTH,
                 dilations: Tuple[int, ...] = FGFF_DILATIONS):
        super().__init__()
        # Spatial branch: four dilation branches (1x1 then dilated 3x3).
        self.branch_1x1 = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=1) for _ in dilations
        ])
        self.branch_dil = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=3, padding=d, dilation=d)
            for d in dilations
        ])
        # Channel fusion of the concatenated branches.
        self.spatial_fuse = nn.Conv2d(width * len(dilations), width,
                                      kernel_size=3, padding=1)
        # Frequency attention from channel-wise avg/max pooling -> 1xHxW.
        self.freq_attn = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        # Learnable per-channel beta on the frequency-guided increment.
        self.beta = nn.Parameter(torch.ones(width))

    def forward(self, ps: torch.Tensor, pf: torch.Tensor) -> torch.Tensor:
        if pf.shape[-2:] != ps.shape[-2:]:
            pf = F.interpolate(pf, size=ps.shape[-2:], mode="bilinear",
                               align_corners=False)

        # Spatial branch: multi-scale dilated convolutions + concat + fuse.
        branches = [dil(c1(ps))
                    for c1, dil in zip(self.branch_1x1, self.branch_dil)]
        spatial_fused = self.spatial_fuse(torch.cat(branches, dim=1))

        # Frequency branch: channel-pooled attention map.
        avg = pf.mean(dim=1, keepdim=True)
        mx = pf.max(dim=1, keepdim=True).values
        freq_attn = torch.sigmoid(self.freq_attn(torch.cat([avg, mx], dim=1)))

        # Beta scaling + residual fusion.
        gated = self.beta.view(1, -1, 1, 1) * (freq_attn * spatial_fused)
        return spatial_fused + gated


# ============================================================================
#  Milestone 3 — EDB Decoder + Skip Connections  (Design Freeze §1.6, §5)
# ============================================================================

class DoubleConv(nn.Module):
    """Two consecutive Conv3x3 -> BatchNorm -> ReLU blocks (Design Freeze §5)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EDB(nn.Module):
    """Enhanced Decoder Block (Design Freeze §1.6, §5 / Eqs. 10-11).

        x_up   = bilinear-upsample(x_high) to skip resolution
        x_cat  = Concat(x_up, x_skip)
        x      = DoubleConv(x_cat) -> out width 64
        attn   = sigma(Conv7x7([avg_c(x); max_c(x)]))        # spatial, 1xHxW
        D      = x + gamma * (attn * x)                       # residual, gamma init 0
    """

    def __init__(self, high_channels: int, skip_channels: int,
                 out_channels: int = DECODER_WIDTH):
        super().__init__()
        self.double_conv = DoubleConv(high_channels + skip_channels, out_channels)
        self.attn_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x_high: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x_up = F.interpolate(x_high, size=x_skip.shape[-2:], mode="bilinear",
                             align_corners=False)
        x = self.double_conv(torch.cat([x_up, x_skip], dim=1))
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True).values
        attn = torch.sigmoid(self.attn_conv(torch.cat([avg, mx], dim=1)))
        return x + self.gamma * (attn * x)


# ============================================================================
#  Model (Milestones 1 + 2A + 2B + 3)
# ============================================================================

class MAFFNetOriginalReconstructed(nn.Module):
    """MAFF-Net reconstruction — Milestones 1 + 2A.

    Spatial stream (M1): frozen SAM encoder + MMPA + spatial projection.
    Frequency stream (M2A): FFT preprocessing + frequency encoder + frequency
    projection.

    forward returns a dict with ``encoder_features``, ``mmpa_features``,
    ``projected_features`` (M1, unchanged), ``frequency_features`` and
    ``frequency_projected`` (M2A, unchanged), ``fgff_features`` (M2B, unchanged),
    plus ``decoder_features`` (M3). No segmentation / deep supervision / loss yet.
    """

    def __init__(self, sam_checkpoint: Optional[str] = None):
        super().__init__()
        # --- Milestone 1 (verified, unchanged) ---
        self.encoder = FrozenSAMEncoder(checkpoint=sam_checkpoint)
        self.mmpa = nn.ModuleList([MMPA(c) for c in ENC_CHANNELS])
        self.spatial_proj = SpatialProjection(ENC_CHANNELS, SPATIAL_PROJ_WIDTH)
        # --- Milestone 2A (verified, unchanged) ---
        self.freq_encoder = FrequencyEncoder()
        self.freq_proj = FrequencyProjection()
        # --- Milestone 2B (verified, unchanged) ---
        self.fgff = nn.ModuleList([FGFF() for _ in ENC_CHANNELS])
        # --- Milestone 3: EDB decoder (deepest -> finest) ---
        # g4(80,12²) ⊕ g3(80,24²) -> 64 ; -> ⊕ g2(80,48²) -> 64 ; -> ⊕ g1(80,96²) -> 64
        self.edb3 = EDB(FUSION_WIDTH, FUSION_WIDTH, DECODER_WIDTH)
        self.edb2 = EDB(DECODER_WIDTH, FUSION_WIDTH, DECODER_WIDTH)
        self.edb1 = EDB(DECODER_WIDTH, FUSION_WIDTH, DECODER_WIDTH)

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        # Spatial stream (Milestone 1).
        enc = self.encoder(x)
        mm = [self.mmpa[i](enc[i]) for i in range(len(enc))]
        proj = self.spatial_proj(mm)
        # Frequency stream (Milestone 2A).
        freq = self.freq_encoder(x)
        freq_proj = self.freq_proj(freq)
        # Frequency-guided fusion (Milestone 2B).
        fgff = [self.fgff[i](proj[i], freq_proj[i]) for i in range(len(proj))]

        # EDB decoder (Milestone 3): g[0..3] = G1..G4 (finest..coarsest).
        d4 = fgff[3]                      # bottleneck feature (80, 12²)
        d3 = self.edb3(fgff[3], fgff[2])  # 64, 24²
        d2 = self.edb2(d3, fgff[1])       # 64, 48²
        d1 = self.edb1(d2, fgff[0])       # 64, 96²
        decoder = [d1, d2, d3, d4]        # D1..D4 (finest..coarsest)

        return {
            "encoder_features": enc,
            "mmpa_features": mm,
            "projected_features": proj,
            "frequency_features": freq,
            "frequency_projected": freq_proj,
            "fgff_features": fgff,
            "decoder_features": decoder,
        }


# ============================================================================
#  Milestone-1 verification (compile/forward/shape/param check)
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    model = MAFFNetOriginalReconstructed()
    print(f"[encoder] pretrained SAM trunk: {model.encoder.is_pretrained}")

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_mmpa = sum(p.numel() for p in model.mmpa.parameters() if p.requires_grad)
    n_sproj = sum(p.numel() for p in model.spatial_proj.parameters()
                  if p.requires_grad)
    n_freq = sum(p.numel() for p in model.freq_encoder.parameters()
                 if p.requires_grad)
    n_fproj = sum(p.numel() for p in model.freq_proj.parameters()
                  if p.requires_grad)
    n_fgff = sum(p.numel() for p in model.fgff.parameters() if p.requires_grad)
    n_edb = sum(p.numel() for p in
                list(model.edb3.parameters()) + list(model.edb2.parameters())
                + list(model.edb1.parameters()) if p.requires_grad)
    print(f"Total params:     {n_total/1e6:.4f} M")
    print(f"Trainable params: {n_train/1e6:.4f} M")
    print(f"  - MMPA x4:           {n_mmpa/1e6:.4f} M")
    print(f"  - Spatial proj x4:   {n_sproj/1e6:.4f} M")
    print(f"  - Frequency encoder: {n_freq/1e6:.4f} M")
    print(f"  - Frequency proj x4: {n_fproj/1e6:.4f} M")
    print(f"  - FGFF x4:           {n_fgff/1e6:.4f} M")
    print(f"  - EDB decoder x3:    {n_edb/1e6:.4f} M")

    x = torch.randn(2, 3, 384, 384)
    out = model(x)

    expected_res = [96, 48, 24, 12]
    print("\nencoder_features:")
    for i, f in enumerate(out["encoder_features"]):
        print(f"  X{i+1}: {tuple(f.shape)}  (expect B,{ENC_CHANNELS[i]},"
              f"{expected_res[i]},{expected_res[i]})")
    print("mmpa_features:")
    for i, f in enumerate(out["mmpa_features"]):
        print(f"  S{i+1}: {tuple(f.shape)}  (expect B,{ENC_CHANNELS[i]},"
              f"{expected_res[i]},{expected_res[i]})")
    print("projected_features:")
    for i, f in enumerate(out["projected_features"]):
        print(f"  Ps{i+1}: {tuple(f.shape)}  (expect B,80,"
              f"{expected_res[i]},{expected_res[i]})")
    print("frequency_features:")
    for i, f in enumerate(out["frequency_features"]):
        print(f"  F{i+1}: {tuple(f.shape)}  (expect B,128,"
              f"{expected_res[i]},{expected_res[i]})")
    print("frequency_projected:")
    for i, f in enumerate(out["frequency_projected"]):
        print(f"  Pf{i+1}: {tuple(f.shape)}  (expect B,80,"
              f"{expected_res[i]},{expected_res[i]})")
    print("fgff_features:")
    for i, f in enumerate(out["fgff_features"]):
        print(f"  G{i+1}: {tuple(f.shape)}  (expect B,80,"
              f"{expected_res[i]},{expected_res[i]})")
    # D1..D4 = finest..coarsest; D4 is the 80-ch bottleneck, D1..D3 are 64-ch.
    dec_expect = [(64, 96), (64, 48), (64, 24), (80, 12)]
    print("decoder_features:")
    for i, f in enumerate(out["decoder_features"]):
        c, r = dec_expect[i]
        print(f"  D{i+1}: {tuple(f.shape)}  (expect B,{c},{r},{r})")

    # Assertions against the Design Freeze.
    ok = True
    for i in range(4):
        r = expected_res[i]
        if tuple(out["encoder_features"][i].shape) != (2, ENC_CHANNELS[i], r, r):
            ok = False
        if tuple(out["mmpa_features"][i].shape) != (2, ENC_CHANNELS[i], r, r):
            ok = False
        if tuple(out["projected_features"][i].shape) != (2, 80, r, r):
            ok = False
        if tuple(out["frequency_features"][i].shape) != (2, 128, r, r):
            ok = False
        if tuple(out["frequency_projected"][i].shape) != (2, 80, r, r):
            ok = False
        if tuple(out["fgff_features"][i].shape) != (2, 80, r, r):
            ok = False
        dc, dr = dec_expect[i]
        if tuple(out["decoder_features"][i].shape) != (2, dc, dr, dr):
            ok = False
    print(f"\nShape compliance with Design Freeze: {'PASS' if ok else 'FAIL'}")
