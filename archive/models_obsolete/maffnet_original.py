"""
maffnet_original.py
====================
Faithful reproduction of the ORIGINAL MAFF-Net architecture as published in:

    Y. Zhou, H. Gong, R. Qiu, S. Wei, Z. Li, W. Zhang,
    "MAFF-Net: SAM-powered Mixed Multi-scale Perception Adaptation and
     frequency-guided feature fusion for robust skin lesion segmentation",
    Biomedical Signal Processing and Control 113 (2026) 109222.
    https://doi.org/10.1016/j.bspc.2025.109222

This module reconstructs the network exactly as described in Section 3 of the
paper (Eqs. 1-14, Figs. 1-6). It is intentionally SELF-CONTAINED and does NOT
import or reuse any block from ``models/maffnet.py`` or
``models/stabilized_maffnet.py``; those are separate research models. Where the
existing repository implementations diverge from the paper (e.g. encoder channel
widths, MMPA dilation rates, projection width), the PAPER definition is used
here.

Paper-faithful design choices implemented:
  * Frozen SAM (Hiera, sam2_hiera_large) image encoder, 4 hierarchical stages
        X1: H/4 x W/4 x 144
        X2: H/8 x W/8 x 288
        X3: H/16 x W/16 x 576
        X4: H/32 x W/32 x 1152                                      (Eq. 1)
  * Mixed Multi-scale Perception Adaptor (MMPA): projection C->32, three
    parallel perceptionNets with dilations {3, 5, 7}, a WetGeneration weight
    generator (softmax over 3 experts), fixed scaling vector Gamma init 1e-7,
    residual reconstruction back to C.                          (Eqs. 2-5)
  * Lightweight frequency encoder: RGB->grayscale, 2D FFT, fft-shift, magnitude
    spectrum, min-max normalisation, pseudo-RGB replication, 0.25x interpolation,
    and a 4-stage lightweight Transformer with depthwise Q/K/V and a learnable
    temperature.                                                (Eqs. 6-8)
  * Frequency-Guided Feature Fusion (FGFF): multi-branch dilated convolutions
    {1, 3, 5, 7}, frequency attention from avg/max pooling, learnable channel
    weight beta.                                                (Eq. 9)
  * Enhanced Decoder Block (EDB): bilinear upsample, pad-to-match, concat,
    double-conv, plus a channel-attention residual with learnable gamma, under
    multi-level deep supervision (p1..p4).                      (Eqs. 10-11)
  * Reconstruction Decoder Block (RDB): grouped transposed convolutions and
    hierarchical skip concatenations, train-only auxiliary reconstruction. (Eq. 12)
  * Loss: weighted multi-scale Dice deep supervision with weights {1, 1, 1, 0.01}
    plus reconstruction MSE.                                    (Eq. 13)

Assumptions (paper under-specifies some hyperparameters; see
ORIGINAL_MAFFNET_IMPLEMENTATION_REPORT.md for the full list):
  * WetGeneration linear widths operate on the 32-d projected feature
    (32 -> 16 -> 3), consistent with the C->C/2->3 description applied to Z.
  * The paper's explicit Permute in Eq. 3 is an implementation artefact to place
    the 32 reduced channels on the conv channel axis; depthwise convolutions with
    groups=32 over (B, 32, H, W) yield the equivalent behaviour and are used here.
  * The EDB skip "channel attention" produces a 1xHxW spatial map (avg/max taken
    over the channel axis), matching the stated output shape [0,1]^{1xHxW}.
  * Frequency-Transformer embedding width / heads / depth are not given; faithful
    but reasonable values are chosen so frequency features pair channel-for-channel
    with the spatial stages {144, 288, 576, 1152}.

No engineering additions from our later research are present here: there is no
feature-alignment module, no lightweight gated fusion, no boundary-aware
supervision, no bf16-specific code, and no extra numerical stabilisation beyond
what the paper specifies.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Paper Eq. (1): hierarchical SAM/Hiera stage channel widths.
PAPER_ENC_CHANNELS: Tuple[int, int, int, int] = (144, 288, 576, 1152)
# MMPA reduced (projection) dimension. Paper Section 3.1: "reduces ... from C to 32".
MMPA_PROJ_DIM: int = 32
# MMPA perceptionNet dilations. Paper Eq. (3) and Table 6 (optimal = 3 nets).
MMPA_DILATIONS: Tuple[int, int, int] = (3, 5, 7)
# FGFF multi-branch dilation rates. Paper Eq. (9): d_i in {1, 3, 5, 7}.
FGFF_DILATIONS: Tuple[int, int, int, int] = (1, 3, 5, 7)


# ============================================================================
#  Frozen SAM (Hiera) image encoder
# ============================================================================

class _DummyHieraTrunk(nn.Module):
    """Shape-compatible placeholder used ONLY when the real SAM2/Hiera weights
    are unavailable. It emits four feature maps at 1/4, 1/8, 1/16, 1/32 with the
    paper channel widths so that the rest of the network can be constructed and
    shape-checked without the pretrained checkpoint. It is NOT part of the
    architecture and carries no learnable adaptation."""

    def __init__(self, channels: Tuple[int, int, int, int] = PAPER_ENC_CHANNELS):
        super().__init__()
        self.stem = nn.Conv2d(3, channels[0], kernel_size=7, stride=4, padding=3)
        self.down = nn.ModuleList([
            nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1),
            nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1),
            nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1),
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x1 = self.stem(x)               # H/4
        x2 = self.down[0](x1)           # H/8
        x3 = self.down[1](x2)           # H/16
        x4 = self.down[2](x3)           # H/32
        return [x1, x2, x3, x4]


class FrozenSAMEncoder(nn.Module):
    """Frozen SAM image encoder (Hiera, sam2_hiera_large) per Eq. (1).

    The encoder parameters are frozen (``requires_grad = False``), reproducing
    the paper's choice to avoid full fine-tuning. The wrapper attempts to load
    the real SAM2 Hiera-large trunk and, only if it is unavailable, falls back to
    a shape-compatible placeholder so the model remains importable and testable.

    Output: list ``[X1, X2, X3, X4]`` at strides 4/8/16/32 with channels
    ``(144, 288, 576, 1152)``.
    """

    def __init__(self, checkpoint: Optional[str] = None,
                 config_file: str = "sam2_hiera_l.yaml"):
        super().__init__()
        self.out_channels: Tuple[int, int, int, int] = PAPER_ENC_CHANNELS
        self._real = False
        self.trunk: nn.Module
        try:
            from sam2.build_sam import build_sam2
            sam2 = build_sam2(config_file=config_file, ckpt_path=checkpoint,
                              device="cpu")
            # Hiera trunk exposes the 4 hierarchical stage features.
            self.trunk = sam2.image_encoder.trunk
            self._real = True
        except Exception:
            self.trunk = _DummyHieraTrunk(self.out_channels)
            self._real = False

        for p in self.parameters():
            p.requires_grad_(False)

    @property
    def is_pretrained(self) -> bool:
        return self._real

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.trunk(x)
        if isinstance(feats, dict):
            feats = list(feats.values())
        feats = list(feats)[:4]
        return feats


# ============================================================================
#  MMPA - Mixed Multi-scale Perception Adaptor  (Section 3.1, Eqs. 2-5)
# ============================================================================

class PerceptionNet(nn.Module):
    """A single perceptionNet (Fig. 3, Eq. 3).

    Two depthwise (groups = 32) 3x3 convolutions: a standard one (d = 1) followed
    by a dilated one (d = r), with a GELU in between. A dilation r yields an
    effective receptive field of (1 + 2r): 7x7, 11x11, 15x15 for r in {3, 5, 7}.
    """

    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.dwconv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1,
                                 groups=dim, bias=True)
        self.act = nn.GELU()
        self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=dilation,
                                 dilation=dilation, groups=dim, bias=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.dwconv2(self.act(self.dwconv1(z)))


class WeightGenerator(nn.Module):
    """WetGeneration weight generator (Eq. 4).

    Global average pooling, then two linear layers (proj_dim -> proj_dim/2 ->
    n_experts) with a GELU and a final softmax, producing normalised per-expert
    fusion weights alpha.
    """

    def __init__(self, dim: int, n_experts: int = 3):
        super().__init__()
        hidden = max(1, dim // 2)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, n_experts)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        zp = F.adaptive_avg_pool2d(z, 1).flatten(1)   # (B, dim)
        a = self.fc2(self.act(self.fc1(zp)))          # (B, n_experts)
        return torch.softmax(a, dim=-1)               # alpha


class MMPA(nn.Module):
    """Mixed Multi-scale Perception Adaptor (Fig. 2, Eqs. 2-5).

    Pipeline for an input stage feature X in R^{B x C x H x W}:
        Z          = GELU(W_p . LN(X) + b_p)          # project C -> 32  (Eq. 2)
        E_j(Z)     = perceptionNet_j(Z), j = 1..3     # dilations 3,5,7  (Eq. 3)
        alpha      = WetGeneration(Z)                 # softmax weights   (Eq. 4)
        E_fused    = sum_j alpha_j * E_j(Z)                              (Eq. 5)
        P_enh      = Z + (E_fused * Gamma)            # Gamma init 1e-7   (Eq. 5)
        X_adapted  = GELU(W_rec . P_enh + b_rec)      # reconstruct -> C  (Eq. 5)
    A residual structure preserves the frozen SAM feature distribution while
    performing gradual domain adaptation.
    """

    def __init__(self, in_channels: int, proj_dim: int = MMPA_PROJ_DIM,
                 dilations: Tuple[int, ...] = MMPA_DILATIONS):
        super().__init__()
        self.in_channels = in_channels
        self.proj_dim = proj_dim

        # Eq. 2: layer-norm over channels then 1x1 projection C -> 32 + GELU.
        self.ln = nn.GroupNorm(1, in_channels)         # LN over channel axis
        self.proj = nn.Conv2d(in_channels, proj_dim, kernel_size=1, bias=True)
        self.proj_act = nn.GELU()

        # Eq. 3: three perceptionNets with distinct dilations.
        self.experts = nn.ModuleList([
            PerceptionNet(proj_dim, d) for d in dilations
        ])

        # Eq. 4: adaptive fusion weight generator.
        self.weight_gen = WeightGenerator(proj_dim, n_experts=len(dilations))

        # Eq. 5: fixed scaling vector Gamma in R^{32}, initialised to 1e-7.
        self.gamma = nn.Parameter(torch.full((proj_dim,), 1e-7))

        # Eq. 5: reconstruct channel dimension 32 -> C + GELU.
        self.recon = nn.Conv2d(proj_dim, in_channels, kernel_size=1, bias=True)
        self.recon_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj_act(self.proj(self.ln(x)))       # (B, 32, H, W)

        expert_outs = [e(z) for e in self.experts]     # 3 x (B, 32, H, W)
        alpha = self.weight_gen(z)                      # (B, 3)

        e_fused = torch.zeros_like(z)
        for j, eo in enumerate(expert_outs):
            e_fused = e_fused + alpha[:, j].view(-1, 1, 1, 1) * eo

        scaled = e_fused * self.gamma.view(1, -1, 1, 1)
        p_enhanced = z + scaled                         # residual (Eq. 5)
        x_adapted = self.recon_act(self.recon(p_enhanced))
        return x_adapted


# ============================================================================
#  Frequency encoder  (Section 3.2, Eqs. 6-8)
# ============================================================================

class FFTMagnitude(nn.Module):
    """RGB -> grayscale -> 2D FFT -> fft-shift -> magnitude -> min-max norm ->
    pseudo-RGB (Eq. 6).

    Only the magnitude spectrum is retained (phase discarded). The normalised
    single-channel magnitude is replicated to three channels to match the input
    format expected by standard visual networks.
    """

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        # x_gray = mean over the 3 RGB channels, mapped to the 8-bit range.
        x_gray = x.mean(dim=1) * 255.0                  # (B, H, W)

        f = torch.fft.fft2(x_gray)                      # complex (B, H, W)
        f = torch.fft.fftshift(f, dim=(-2, -1))         # centre the spectrum
        mag = torch.abs(f)                              # |F_shifted|

        # Per-sample min-max normalisation to [0, 1]  (Eq. 6).
        b = mag.shape[0]
        flat = mag.view(b, -1)
        mn = flat.min(dim=1).values.view(b, 1, 1)
        mx = flat.max(dim=1).values.view(b, 1, 1)
        m = (mag - mn) / (mx - mn)                      # paper-faithful (no eps)

        return m.unsqueeze(1).repeat(1, 3, 1, 1)        # pseudo-RGB (B, 3, H, W)


class FreqTransformerBlock(nn.Module):
    """Lightweight Transformer block for the frequency stream (Eq. 8).

        P            = Conv1x1(f_in)
        Q, K, V      = Split(DWConv(Conv1x1(P)))
        Attention    = Softmax((Q K^T / sqrt(d_k)) * T) V
        Z            = P + Attention
        f_out        = LayerNorm(Z) + FFN(LayerNorm(Z))

    A learnable temperature T in R^{heads x 1 x 1} scales the attention logits to
    amplify high-frequency responses. Depthwise convolution before Q/K/V keeps
    the parameter count low.
    """

    def __init__(self, dim: int, num_heads: int = 4, ffn_ratio: int = 2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.proj_in = nn.Conv2d(dim, dim, kernel_size=1)
        self.qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1)
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

        qkv = self.qkv_dw(self.qkv_conv(p))
        q, k, v = qkv.chunk(3, dim=1)

        def reshape_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(b, self.num_heads, self.head_dim, h * w)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature   # (B, heads, d, d)
        attn = attn.softmax(dim=-1)
        out = attn @ v                                        # (B, heads, d, HW)
        out = out.reshape(b, c, h, w)
        out = self.proj_out(out)

        z = p + out                                           # residual
        z = self.norm1(z)
        z = z + self.ffn(self.norm2(z))
        return z


class FrequencyEncoder(nn.Module):
    """Hierarchical frequency-domain encoder (Section 3.2.2, Eqs. 7-8).

    The pseudo-RGB magnitude map is interpolated by 0.25x (Eq. 7) and then passed
    through four stages of lightweight Transformer blocks, each halving the
    spatial resolution and producing channels that pair stage-for-stage with the
    spatial encoder ``(144, 288, 576, 1152)`` at strides 4/8/16/32.
    """

    def __init__(self, channels: Tuple[int, int, int, int] = PAPER_ENC_CHANNELS):
        super().__init__()
        self.to_freq = FFTMagnitude()
        # Stem: pseudo-RGB (3ch) at 0.25x -> first stage width at H/4.
        self.stem = nn.Conv2d(3, channels[0], kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([FreqTransformerBlock(channels[0])])
        self.downs = nn.ModuleList()
        for i in range(1, len(channels)):
            self.downs.append(
                nn.Conv2d(channels[i - 1], channels[i], kernel_size=3,
                          stride=2, padding=1)
            )
            self.blocks.append(FreqTransformerBlock(channels[i]))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        f = self.to_freq(x)                                   # (B, 3, H, W)
        f_in = F.interpolate(f, scale_factor=0.25, mode="bilinear",
                             align_corners=False)             # Eq. 7 -> H/4
        feats: List[torch.Tensor] = []
        h = self.stem(f_in)
        h = self.blocks[0](h)
        feats.append(h)                                       # H/4
        for down, block in zip(self.downs, self.blocks[1:]):
            h = block(down(h))
            feats.append(h)                                   # H/8, H/16, H/32
        return feats


# ============================================================================
#  FGFF - Frequency-Guided Feature Fusion  (Section 3.2.3, Eq. 9, Fig. 4)
# ============================================================================

class FGFF(nn.Module):
    """Frequency-Guided Feature Fusion (Eq. 9).

        b_i              = DilatedConv(Conv1x1(f_spatial), dilation = d_i)
        f_spatial_fused  = Conv3x3(Cat([b0; b1; b2; b3]))
        f_freq_attn      = sigma(Conv3x3([AvgPool(f_freq); MaxPool(f_freq)]))
        f_fused          = beta * (f_freq_attn * f_spatial_fused)

    Dilations d_i in {1, 3, 5, 7}; sigma is the sigmoid; beta in R^C is a
    learnable per-channel weight on the frequency contribution.
    """

    def __init__(self, channels: int,
                 dilations: Tuple[int, ...] = FGFF_DILATIONS):
        super().__init__()
        self.branch_1x1 = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=1) for _ in dilations
        ])
        self.branch_dil = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=3, padding=d, dilation=d)
            for d in dilations
        ])
        self.spatial_fuse = nn.Conv2d(channels * len(dilations), channels,
                                      kernel_size=3, padding=1)
        # Frequency attention from channel-wise avg/max pooling -> 1xHxW.
        self.freq_attn = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.beta = nn.Parameter(torch.ones(channels))

    def forward(self, f_spatial: torch.Tensor,
                f_freq: torch.Tensor) -> torch.Tensor:
        if f_freq.shape[-2:] != f_spatial.shape[-2:]:
            f_freq = F.interpolate(f_freq, size=f_spatial.shape[-2:],
                                   mode="bilinear", align_corners=False)

        branches = []
        for c1, cd in zip(self.branch_1x1, self.branch_dil):
            branches.append(cd(c1(f_spatial)))
        spatial_fused = self.spatial_fuse(torch.cat(branches, dim=1))

        avg = f_freq.mean(dim=1, keepdim=True)
        mx = f_freq.max(dim=1, keepdim=True).values
        freq_attn = torch.sigmoid(self.freq_attn(torch.cat([avg, mx], dim=1)))

        fused = self.beta.view(1, -1, 1, 1) * (freq_attn * spatial_fused)
        return fused


# ============================================================================
#  EDB - Enhanced Decoder Block  (Section 3.3.1, Eqs. 10-11, Fig. 5)
# ============================================================================

class DoubleConv(nn.Module):
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
    """Enhanced Decoder Block (Eqs. 10-11).

    Upsamples a high-level feature, pads it to match the low-level (skip)
    feature, concatenates, and applies a double convolution (Eq. 10). A
    spatial-attention residual with a learnable scale gamma then sharpens
    boundary sensitivity (Eq. 11):
        attn   = sigma(Conv([avg; max]))           # over channel axis, 1xHxW
        x_out  = x + gamma * (attn * x)
    """

    def __init__(self, high_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = DoubleConv(high_channels + skip_channels, out_channels)
        self.attn_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x_high: torch.Tensor, x_low: torch.Tensor) -> torch.Tensor:
        x_up = F.interpolate(x_high, size=x_low.shape[-2:], mode="bilinear",
                             align_corners=False)
        x = self.double_conv(torch.cat([x_up, x_low], dim=1))

        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True).values
        attn = torch.sigmoid(self.attn_conv(torch.cat([avg, mx], dim=1)))
        return x + self.gamma * (attn * x)


class SegHead(nn.Module):
    """1x1 convolution producing single-channel logits, upsampled to target."""

    def __init__(self, in_channels: int, num_classes: int = 1):
        super().__init__()
        self.head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor,
                size: Tuple[int, int]) -> torch.Tensor:
        out = self.head(x)
        if out.shape[-2:] != size:
            out = F.interpolate(out, size=size, mode="bilinear",
                                align_corners=False)
        return out


# ============================================================================
#  RDB - Reconstruction Decoder Block  (Section 3.3.2, Eq. 12, Fig. 6)
# ============================================================================

class RDB(nn.Module):
    """Reconstruction Decoder Block (Eq. 12).

    Receives the four encoder-stage feature maps s1..s4 (deepest first) and
    progressively upsamples s1 via grouped transposed convolutions, concatenating
    each stage with the next shallower feature and fusing with a conv + ReLU.
    The final feature is restored to 3 channels and interpolated to the input
    size. Used only for the auxiliary reconstruction MSE during training.
    """

    def __init__(self, channels: Tuple[int, int, int, int] = PAPER_ENC_CHANNELS):
        super().__init__()
        # channels are given shallow->deep (144,288,576,1152); reconstruction
        # walks deep->shallow: s1=1152 (H/32) up to s4=144 (H/4).
        deep_to_shallow = list(reversed(channels))     # [1152, 576, 288, 144]
        self.up_convs = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()
        cur = deep_to_shallow[0]
        for nxt in deep_to_shallow[1:]:
            self.up_convs.append(
                nn.ConvTranspose2d(cur, cur, kernel_size=4, stride=2, padding=1,
                                   groups=_gcd(cur, cur))
            )
            self.fuse_convs.append(
                nn.Sequential(
                    nn.Conv2d(cur + nxt, nxt, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )
            cur = nxt
        self.to_rgb = nn.ConvTranspose2d(cur, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, feats: List[torch.Tensor],
                size: Tuple[int, int]) -> torch.Tensor:
        # feats given shallow->deep; reverse to deep->shallow.
        s = list(reversed(feats))                      # [s1(deep) .. s4(shallow)]
        x = s[0]
        for up, fuse, skip in zip(self.up_convs, self.fuse_convs, s[1:]):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                                  align_corners=False)
            x = fuse(torch.cat([x, skip], dim=1))
        x = self.to_rgb(x)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


# ============================================================================
#  MAFF-Net (full model)
# ============================================================================

class MAFFNetOriginal(nn.Module):
    """Original MAFF-Net (paper-faithful).

    Forward (training): returns ``(p1, p2, p3, p4, rec)`` where ``p1`` is the
    finest-scale prediction and ``rec`` is the auxiliary reconstruction.
    Forward (eval): returns ``p1`` only (RDB disabled).
    """

    def __init__(self, num_classes: int = 1,
                 sam_checkpoint: Optional[str] = None):
        super().__init__()
        self.num_classes = num_classes
        ch = PAPER_ENC_CHANNELS

        # Spatial stream: frozen SAM encoder + per-stage MMPA.
        self.encoder = FrozenSAMEncoder(checkpoint=sam_checkpoint)
        self.mmpa = nn.ModuleList([MMPA(c) for c in ch])

        # Frequency stream.
        self.freq_encoder = FrequencyEncoder(ch)

        # Per-stage frequency-guided fusion.
        self.fgff = nn.ModuleList([FGFF(c) for c in ch])

        # Main decoder: EDB stages from deepest fused feature up to finest.
        # g4(1152,H/32) -> +g3 -> 576 ; -> +g2 -> 288 ; -> +g1 -> 144.
        self.edb3 = EDB(high_channels=ch[3], skip_channels=ch[2], out_channels=ch[2])
        self.edb2 = EDB(high_channels=ch[2], skip_channels=ch[1], out_channels=ch[1])
        self.edb1 = EDB(high_channels=ch[1], skip_channels=ch[0], out_channels=ch[0])

        # Deep-supervision heads (p4 from bottleneck g4, p3..p1 from EDB stages).
        self.head4 = SegHead(ch[3], num_classes)
        self.head3 = SegHead(ch[2], num_classes)
        self.head2 = SegHead(ch[1], num_classes)
        self.head1 = SegHead(ch[0], num_classes)

        # Auxiliary reconstruction decoder (train-only).
        self.rdb = RDB(ch)

    def forward(self, x: torch.Tensor):
        size = (x.shape[-2], x.shape[-1])

        # Spatial stream: encoder (frozen) -> MMPA per stage.
        enc = self.encoder(x)                          # [X1..X4]
        spatial = [self.mmpa[i](enc[i]) for i in range(4)]

        # Frequency stream.
        freq = self.freq_encoder(x)                    # [F1..F4]

        # Frequency-guided fusion per stage.
        g = [self.fgff[i](spatial[i], freq[i]) for i in range(4)]

        # Main decoder with deep supervision.
        p4 = self.head4(g[3], size)
        d3 = self.edb3(g[3], g[2])
        p3 = self.head3(d3, size)
        d2 = self.edb2(d3, g[1])
        p2 = self.head2(d2, size)
        d1 = self.edb1(d2, g[0])
        p1 = self.head1(d1, size)

        if not self.training:
            return p1

        rec = self.rdb(g, size)
        return p1, p2, p3, p4, rec


# ============================================================================
#  Loss  (Section 3.4, Eq. 13)
# ============================================================================

class MAFFNetOriginalLoss(nn.Module):
    """Original MAFF-Net loss (Eq. 13).

        L = sum_i lambda_i * Dice(Upsample(p_i), y) + lambda_rec * MSE(rec, x)

    with deep-supervision weights ``lambda = {1, 1, 1, 0.01}`` over the four
    prediction heads and a reconstruction MSE weighted by ``lambda_rec``. The
    paper uses Dice for segmentation supervision; no BCE, boundary, or
    edge-weighted terms are present (those belong to later research models).
    """

    def __init__(self, ds_weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.01),
                 lambda_rec: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.ds_weights = ds_weights
        self.lambda_rec = lambda_rec
        self.eps = eps

    def dice_loss(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logit)
        num = 2.0 * (prob * target).sum(dim=(1, 2, 3)) + self.eps
        den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + self.eps
        return (1.0 - num / den).mean()

    def forward(self, outputs, target: torch.Tensor,
                image: torch.Tensor) -> torch.Tensor:
        p1, p2, p3, p4, rec = outputs
        preds = [p1, p2, p3, p4]
        seg = sum(w * self.dice_loss(p, target)
                  for w, p in zip(self.ds_weights, preds))
        rec_loss = F.mse_loss(rec, image)
        return seg + self.lambda_rec * rec_loss


# ============================================================================
#  Sanity check (shape only; uses the placeholder encoder if SAM2 is absent)
# ============================================================================

if __name__ == "__main__":
    model = MAFFNetOriginal()
    print(f"[MAFFNetOriginal] pretrained SAM encoder: "
          f"{model.encoder.is_pretrained}")

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {n_total/1e6:.2f} M")
    print(f"Trainable params: {n_train/1e6:.2f} M")

    x = torch.randn(2, 3, 384, 384)

    model.train()
    p1, p2, p3, p4, rec = model(x)
    print("Train outputs:")
    for name, t in zip(["p1", "p2", "p3", "p4", "rec"], [p1, p2, p3, p4, rec]):
        print(f"  {name}: {tuple(t.shape)}")

    y = (torch.rand(2, 1, 384, 384) > 0.5).float()
    loss = MAFFNetOriginalLoss()((p1, p2, p3, p4, rec), y, x)
    print(f"Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"Eval output: {tuple(out.shape)}")
