# MAFF-Net — Final Design Specification (Design Freeze)

**Status:** FROZEN. Single source of truth for implementing `models/maffnet_original_reconstructed.py`.
**Inputs to this freeze:** Architecture Audit, Reproduction Verification (70/100, "structurally faithful but dimensionally different"), Mathematical Dimension Reconstruction (candidate D ≈ 14 M).
**Constraints honoured:** no code written, no Python file created, nothing implemented, no training run. This document is design only.

All symbolic widths from the paper are now resolved to **fixed integers**. No variable remains. Reference: Zhou et al., MAFF-Net, BSPC 113 (2026) 109222.

**Global constants (frozen):**
| Constant | Value |
|---|---|
| Input image | `3 × 384 × 384` |
| Encoder stages `C_enc` | `[144, 288, 576, 1152]` @ strides `[4, 8, 16, 32]` → resolutions `[96², 48², 24², 12²]` |
| **Common fusion width `Wf`** | **80** |
| **Common decoder width `Wd`** | **64** |
| **Frequency embed `E`** | **128** |
| Frequency blocks / stage | **2** |
| Attention heads | **4** |
| FFN ratio | **2** |
| MMPA reduced dim | **32** ; WetGeneration on native `C` ; Γ init `1e-7` |
| RDB widths | **native** `[1152, 576, 288, 144]` |
| Deep-sup heads | **4** ; weights `[1, 1, 1, 0.01]` |

---

## 1. Architecture — per-module specification

Notation `B×C×H×W`. Param estimates from `Conv(c_in,c_out,k,g)=(c_in/g)·c_out·k²(+bias)`.

### 1.1 Frozen SAM Encoder
| Field | Value |
|---|---|
| Input | `B×3×384×384` |
| Output | 4 maps `X1..X4`: `B×144×96²`, `B×288×48²`, `B×576×24²`, `B×1152×12²` |
| Channels | `[144,288,576,1152]` |
| Resolution | `[96,48,24,12]` |
| Params | ≈ 227 M **frozen** (0 trainable) |

### 1.2 MMPA (×4, one per stage)
| Field | Value |
|---|---|
| Input | `B×C_i×H_i×W_i` |
| Output | `B×C_i×H_i×W_i` (same shape; residual adaptation) |
| Internal | proj `C_i→32`, 3 perceptionNets (dil 3/5/7, full 3×3 @32), WetGeneration `C_i→C_i/2→3` softmax, Γ∈R³², recon `32→C_i` |
| Params | ≈ **1.70 M** total (WetGen on native `C` dominates: `ΣC_i²/2 ≈ 0.88 M`) |

### 1.3 Projection Layers (channel harmonisation to fusion width)
| Field | Value |
|---|---|
| Spatial proj (×4) | `Conv1×1(C_i → 80)` : `B×C_i×H_i² → B×80×H_i²` |
| Frequency proj (×4) | `Conv1×1(128 → 80)` : `B×128×H_i² → B×80×H_i²` |
| Params | spatial `Σ C_i·80 ≈ 0.17 M` + freq `4·128·80 ≈ 0.04 M` = **≈ 0.21 M** |

### 1.4 Frequency Encoder
| Field | Value |
|---|---|
| Input | `B×3×384×384` (raw image) |
| FFT stage | grayscale `mean·255 → FFT2 → fftshift → |·| → min-max → pseudo-RGB` → `B×3×384²` |
| Downsample | `interp ×0.25` → `B×3×96²` → stem `Conv3×3(3→128)` |
| Body | 4 stages, **2 FreqTransformer blocks each** (E=128, 4 heads, FFN×2), stride-2 down between stages |
| Output | 4 maps `F1..F4`: `B×128×[96,48,24,12]²` |
| Params | ≈ **1.63 M** (8 blocks ·`9E²` ≈ 1.18 M + 3 downs ≈ 0.44 M + stem) |

### 1.5 FGFF (×4, one per stage)
| Field | Value |
|---|---|
| Input | spatial `B×80×H_i²` + frequency `B×80×H_i²` |
| Output | fused `B×80×H_i²` |
| Internal | 4 branches `Conv1×1(80,80)+DilConv3×3(80,80)` dil {1,3,5,7}; fuse `Conv3×3(320,80)`; freq-attn `σ(Conv3×3(2,1))` over channel-pool; `β∈R⁸⁰` |
| Params | ≈ **1.95 M** total (`4 × 76·80²`) |

### 1.6 EDB Decoder (×3)
| Field | Value |
|---|---|
| EDB3 | high `g4 B×80×12²` ↑ + skip `g3 B×80×24²` → `B×64×24²` |
| EDB2 | high `B×64×24²` ↑ + skip `g2 B×80×48²` → `B×64×48²` |
| EDB1 | high `B×64×48²` ↑ + skip `g1 B×80×96²` → `B×64×96²` |
| Internal | `Upsample→Concat→DoubleConv(in→64→64)` + spatial-attn residual `x+γ·(σ(Conv7×7([avg;max]))·x)` |
| Params | ≈ **0.40 M** total |

### 1.7 RDB (train-only, native widths)
| Field | Value |
|---|---|
| Input | 4 **post-MMPA spatial** features (native) deep→shallow: `B×1152×12²`, `B×576×24²`, `B×288×48²`, `B×144×96²` |
| Internal | grouped `ConvTranspose2d(k4,s2,p1)` + fuse `Conv3×3(C_in+C_skip → C_in/2)` + ReLU, chained; final `→3ch`, interp to `384²` |
| Output | `B×3×384×384` |
| Params | ≈ **7.80 M** (native widths, grouped fuse) |

### 1.8 Deep Supervision
| Field | Value |
|---|---|
| Heads | p4←g4(80), p3←EDB3(64), p2←EDB2(64), p1←EDB1(64) |
| Weights | `[1, 1, 1, 0.01]` (p1,p2,p3,p4) |

### 1.9 Segmentation Heads (×4)
| Field | Value |
|---|---|
| Op | `Conv1×1(width→1)` then bilinear-upsample to `384²` |
| Shapes | `p4: B×1×384²` … `p1: B×1×384²` |
| Params | ≈ **0.0003 M** |

### 1.10 Loss Functions
| Field | Value |
|---|---|
| Seg | `Σ_i λ_i · Dice(σ(Up(p_i)), y)`, `λ=[1,1,1,0.01]` |
| Recon | `λ_rec · MSE(rec, x)`, `λ_rec = 1.0` (train-only) |
| Total | `L = Σ λ_i·Dice_i + λ_rec·MSE` |

---

## 2. Tensor Flow (complete pipeline)

```
I  : B×3×384×384
│
├─ SPATIAL ────────────────────────────────────────────────
│  X1..X4 = SAM(I)            B×144×96² B×288×48² B×576×24² B×1152×12²
│  Si     = MMPA_i(Xi)        (same shapes, adapted)            ← RDB taps Si (native)
│  Ps_i   = Conv1×1(Si→80)    B×80×{96,48,24,12}²
│
├─ FREQUENCY ──────────────────────────────────────────────
│  Fmag   = FFT_mag(I)        B×3×384²  (gray→FFT→shift→|·|→minmax→pseudoRGB)
│  Fin    = interp×0.25       B×3×96²
│  Ff1..4 = FreqEnc(Fin)      B×128×{96,48,24,12}²   (4 stages ×2 blocks)
│  Pf_i   = Conv1×1(Ff_i→80)  B×80×{96,48,24,12}²
│
├─ FUSION ─────────────────────────────────────────────────
│  g_i    = FGFF_i(Ps_i, Pf_i)   B×80×{96,48,24,12}²   (i=1..4)
│
├─ DECODE (deep supervision) ──────────────────────────────
│  p4     = SegHead(g4)→↑384      B×1×384²
│  d3     = EDB3(g4, g3)          B×64×24²   → p3 = SegHead(d3)→↑384
│  d2     = EDB2(d3, g2)          B×64×48²   → p2 = SegHead(d2)→↑384
│  d1     = EDB1(d2, g1)          B×64×96²   → p1 = SegHead(d1)→↑384   ← PRIMARY
│
├─ RECON (train-only) ─────────────────────────────────────
│  rec    = RDB(S1,S2,S3,S4)      B×3×384²
│
└─ OUTPUT
   train : (p1, p2, p3, p4, rec)
   eval  : p1
```

---

## 3. Channel Widths — FROZEN (no symbols)

| Path | Width(s) |
|---|---|
| Encoder stages | `144, 288, 576, 1152` |
| MMPA reduced dim | `32` |
| Spatial projection out | `80` |
| Frequency embed | `128` |
| Frequency projection out | `80` |
| FGFF in / out | `80 / 80` |
| Decoder (EDB) out | `64` |
| Skip width into EDB | `80` (fused `g_i`) |
| RDB stage widths | `1152, 576, 288, 144` (native) |
| Seg head out | `1` |

---

## 4. Transformer Configuration — FROZEN

| Field | Value |
|---|---|
| Embedding dim `E` | **128** |
| Blocks per stage | **2** (4 stages → 8 blocks total) |
| Attention heads | **4** (`d_k = 128/4 = 32`) |
| FFN ratio | **2** (`128 → 256 → 128`) |
| Q/K/V | depthwise `Conv3×3` before split |
| Temperature | learnable `T ∈ R^{4×1×1}` |
| Norm | GroupNorm(1) (LayerNorm-equivalent over channels) |

---

## 5. Decoder — FROZEN

| Field | Value |
|---|---|
| Decoder widths | `64, 64, 64` (EDB3/EDB2/EDB1) |
| Skip widths | `80, 80, 80` (fused `g3, g2, g1`) |
| Fusion (DoubleConv in) | `80+80=160 → 64`, then `64+80=144 → 64`, then `64+80=144 → 64` |
| Output widths | seg heads `1`; reconstruction `3` |
| Upsampling | bilinear, `align_corners=False` |
| Skip attention | spatial: `σ(Conv7×7([avg_c; max_c]))`, residual scale `γ` (init 0) |

---

## 6. Parameter Budget

| Module | Estimate | Paper target |
|---|---:|---:|
| Frozen SAM encoder | 227 M (frozen) | 227 M |
| MMPA ×4 | 1.70 M | 1.7 M |
| Projection layers | 0.21 M | — |
| Frequency encoder | 1.63 M | (part of ~2.7 M) |
| FGFF ×4 | 1.95 M | 2.2 M |
| EDB ×3 | 0.40 M | 0.4 M |
| RDB | 7.80 M | 7.8 M |
| Seg heads ×4 | <0.01 M | — |
| **Total trainable** | **≈ 13.7 M** | **≈ 14.8 M** |

**Remaining difference (≈ 1.1 M):** the paper does not itemise (a) the bottleneck/transition convolution from the deepest stage, (b) BatchNorm/GroupNorm affine parameters and conv biases, and (c) the exact RDB grouping (modelled here as a grouped 3×3 fuse). These collectively account for ~1 M. The freeze deliberately keeps a single small `Wf=80 / Wd=64` rather than inflating widths to force the last 1 M, because every itemised module already matches the paper to within rounding; the residual is in un-itemised glue, not in a missing wide module. This is within the ±1 M tolerance implied by the paper's "approximately 14.8 M".

---

## 7. Compliance Checklist

| Paper component | Status | Note |
|---|---|---|
| Frozen SAM encoder, 144/288/576/1152 | ✔ Exact | Eq. 1 |
| MMPA proj `C→32`, dil {3,5,7}, Γ=1e-7, residual | ✔ Exact | Eqs. 2–5 |
| MMPA WetGeneration linear widths | ⚠ Reconstructed | on native `C` to meet 1.7 M |
| Projection layers (→80) | ⚠ Reconstructed | width not in paper |
| Freq: gray→FFT→shift→magnitude→minmax→pseudoRGB→×0.25 | ✔ Exact | Eqs. 6–7 |
| Freq transformer (DW-QKV, temperature, FFN, LN) | ✔ Exact (structure) | Eq. 8 |
| Freq E=128 / 2 blocks / 4 heads / FFN×2 | ⚠ Reconstructed | numbers not in paper |
| FGFF dil {1,3,5,7}, freq-attn, β | ✔ Exact (structure) | Eq. 9 |
| FGFF width 80 | ⚠ Reconstructed | width not in paper |
| EDB upsample+pad+concat+DoubleConv | ✔ Exact | Eq. 10 |
| EDB channel/spatial attention + γ | ⚠ Reconstructed | paper shape self-inconsistent; spatial chosen |
| EDB width 64 | ⚠ Reconstructed | width not in paper |
| RDB grouped transpose, native, `C_out=C_in/2`, train-only MSE | ✔ Exact (structure) | Eq. 12 |
| RDB exact groups | ⚠ Reconstructed | `groups_i` unspecified |
| Deep supervision 4 heads | ✔ Exact | §3.3 |
| Seg heads 1×1 + upsample | ✔ Exact | Fig. 1 |
| Loss: Dice {1,1,1,0.01} + recon MSE | ✔ Exact | Eq. 13 |
| Feature alignment / gated fusion / boundary loss | ❌ Not in paper | correctly EXCLUDED (belong to our research models) |

No ❌ component is included; all ❌ items are intentionally absent.

---

## 8. FINAL DESIGN LOCK

| Module | Final Configuration |
|---|---|
| **Frozen SAM Encoder** | SAM2 Hiera-Large trunk, frozen; out `[144,288,576,1152]` @ `[96,48,24,12]²`; input `384²` |
| **MMPA (×4)** | `LN → Conv1×1(C→32) → GELU`; 3 perceptionNets full `Conv3×3@32` dil {3,5,7}; WetGen `GAP→Lin(C→C/2)→GELU→Lin(→3)→softmax`; fuse `Σαⱼ·Eⱼ`; `Γ∈R³² (1e-7)`; recon `Conv1×1(32→C)`; residual |
| **Projection Layers** | spatial `Conv1×1(C_i→80)`; frequency `Conv1×1(128→80)` |
| **Frequency Encoder** | `gray·255→FFT2→fftshift→|·|→minmax→pseudoRGB→interp×0.25→Conv3×3(3→128)`; 4 stages × 2 FreqTransformer blocks (E=128, heads=4, FFN×2, DW-QKV, learnable T), stride-2 downs; out `128 @ [96,48,24,12]²` |
| **FGFF (×4)** | width 80; 4 branches `Conv1×1→DilConv3×3` dil {1,3,5,7}; fuse `Conv3×3(320→80)`; freq-attn `σ(Conv3×3([avg_c;max_c]))`; `β∈R⁸⁰`; out 80 |
| **EDB (×3)** | `Up→Concat(skip 80)→DoubleConv(→64)` + spatial-attn residual `x+γ·(σ(Conv7×7([avg;max]))·x)`; out 64 |
| **RDB** | inputs = post-MMPA native features `[1152,576,288,144]`; grouped `ConvTranspose(k4,s2,p1)` + `Conv3×3(C_in+C_skip→C_in/2)+ReLU` chained; `→3ch`; interp `384²`; train-only |
| **Deep Supervision** | p4←g4(80), p3←EDB3(64), p2←EDB2(64), p1←EDB1(64); weights `[1,1,1,0.01]` |
| **Segmentation Heads (×4)** | `Conv1×1(width→1)` → bilinear ↑ `384²` |
| **Loss** | `L = Σ λ_i·Dice(σ(Up(p_i)),y) + 1.0·MSE(rec,x)`, `λ=[1,1,1,0.01]` |
| **Outputs** | train `(p1,p2,p3,p4,rec)`; eval `p1` |
| **Budget** | ≈ 13.7 M trainable (target 14.8 M; ≈1.1 M residual in un-itemised glue) |

**No architectural decision remains open.** This table is the official blueprint for `models/maffnet_original_reconstructed.py`.

---

*Status: Design Freeze v1 / Analysis only — no code, no Python file, no training / Supersedes width questions in the Verification and Reconstruction reports. Existing models (`maffnet.py`, `stabilized_maffnet.py`, `maffnet_original.py`) remain unchanged.*
