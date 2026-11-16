# RDB (Reconstruction Decoder Block) — Architectural Audit & Design Specification

**Milestone 4A — audit only. No code written, no file edited, nothing implemented.**
**Sources:** original MAFF-Net paper (Zhou et al., BSPC 113 (2026) 109222) + `MAFFNET_FINAL_DESIGN_SPECIFICATION.md`.

---

## 1. Every RDB mention in the paper

| Location | Line(s) | Content |
|---|---|---|
| Abstract | 32–33 | "Reconstructed Decoder Block (RDB) constrains consistency between reconstructed features and input images via **auxiliary loss**, strengthening feature stability." |
| Intro, contribution 4 | 186–191 | RDB "constrains the consistency between reconstructed features and the input image through **auxiliary loss**, similar to a **self-supervised** learning approach … solves feature degradation in traditional decoders." |
| §3 overview | 383–384 | Boundary-optimization decoding = EDB (Fig. 5) + "a **reconstruction decoding module, as shown in Fig. 6**." |
| Parameter budget | 391 | "**RDB (about 7.8 million parameters)**." |
| §3.3 intro | 667–669 | "an **auxiliary reconstruction decoder as shown in Fig. 6**." |
| §3.3.2 (full text) | 731–785 | Architecture description (see §2). |
| Eq. 12 | 760–770 | Formal definition (grouped transposed conv + concat + conv/ReLU + interpolate). |
| §3.4 loss | 791–821 | Reconstruction loss = MSE(f_rec, x); "auxiliary Reconstruction Decoder (**grouped transposed convolutions and hierarchical skip connections**)"; weight λ₃; Eq. 13. |
| Fig. 6 caption | 844 | "The structure of the Reconstruction Decoder Block (RDB)." |
| Discussion/ablation | 1244, 1268 | EDB + reconstruction decoder ablated together; "self-supervised regularization enhances generalization." |

No appendix, no code release — Fig. 6 (a schematic) and Eq. 12 are the only structural sources.

---

## 2. Input / output / widths / pathway / upsampling / conv ordering

### Eq. 12 (verbatim structure)
```
s_i_up     = ConvTranspose2d(s_i, kernel=4, stride=2, padding=1, groups_i)
s_i_concat = Concat(s_i_up, s_{i+1}, dim=1)
C_in       = channels(s_i_up)
C_out      = in_channel / 2
w_i        = ReLU( Conv2d(s_i_concat, in_channel=C_in, out_channel=C_out) )
I          = Interpolate(w_4, scale_factor=2)
```

### §3.3.2 text
"The decoder receives four feature maps of different scales — **s1, s2, s3, s4 — generated during the encoding stage** as inputs. s1 undergoes deconvolution for upsampling and is then concatenated with s2 along the channel dimension … fusion via a convolutional block followed by ReLU … repeated iteratively … after upsampling s1 and fusing layer by layer with s2, s3, s4, the resultant feature map is **restored to the original 3 channels through deconvolution**. Subsequently … adjusted to match the original input through **interpolation** … the reconstructed image is output."

| Field | Value | Source |
|---|---|---|
| Input tensors | `s1..s4`, the four encoding-stage features (deepest → shallowest) | §3.3.2 text |
| Channel widths (native) | `s1=1152@12², s2=576@24², s3=288@48², s4=144@96²` | Eq. 1 + freeze |
| Upsampling | grouped `ConvTranspose2d(k=4, s=2, p=1)` → exact ×2 per step; final `Interpolate(×2)` to input size | Eq. 12 |
| Reconstruction pathway | progressive: deconv → concat next shallower → Conv+ReLU (halve channels), repeated; final deconv → 3 ch → interpolate | §3.3.2 + Eq. 12 |
| Conv ordering | `ConvTranspose → Concat → Conv2d → ReLU` per stage | Eq. 12 |
| Channel reduction | each fuse `C_out = C_in / 2` | Eq. 12 |
| Output tensor | `f_rec = I ∈ B×3×384×384` | §3.3.2 |

---

## 3. What does RDB consume? — RESOLVED

> The paper states the inputs are "**s1, s2, s3, s4 — generated during the encoding stage**."

| Candidate | Verdict |
|---|---|
| **Encoder features** | ✅ **YES** — "generated during the encoding stage" |
| Decoder features (EDB outputs) | ❌ No |
| FGFF features | ❌ No (FGFF is the encoder-side fusion, but the paper's wording is "encoding stage", not "fused/FGFF") |

**Decision (consistent with Design Freeze §1.7):** RDB consumes the four **encoding-stage spatial features** — i.e. the **post-MMPA spatial features** at native widths `[1152, 576, 288, 144]`. These are the closest faithful match to "generated during the encoding stage." (Feeding raw frozen-SAM features instead of MMPA-adapted ones is an alternative reading; see §6 Inferred.)

---

## 4. Reconstruction-loss connection

| Field | Value | Source |
|---|---|---|
| Loss type | Mean Squared Error between reconstructed image `f_rec` and input `x` | §3.4, L779–780 |
| Formula | `L_rec = (1/B·C·H·W) Σ (f_rec − x)²` (Eq. 13, line 1) | Eq. 13 |
| Total objective | `L = Σ_i λ_i·Dice(Up(p_i), y) + λ₃·MSE(f_rec, x)` | Eq. 13 |
| Weight | `λ₃` (reconstruction weight; value not numerically given — Design Freeze sets `λ_rec = 1.0`) | §3.4 / freeze §1.10 |
| Phase | **train-only**; RDB removed at inference "to reduce computational overhead" | §3.3.2 L778–785 |
| Purpose | self-supervised-style consistency regularizer; combats decoder feature degradation | Intro 4 / §3.3.2 |

RDB connects to the loss **only** through `MSE(f_rec, x)`; it produces no segmentation output and contributes no gradient at inference.

---

## 5. Expected parameter count

Native widths, deep→shallow `[1152, 576, 288, 144]`. Per-stage cost depends on two unknowns: the transposed-conv `groups_i` and the fuse-conv kernel size.

| Variant | Transpose | Fuse conv | Estimated total |
|---|---|---|---|
| Depthwise transpose + **k3** fuse `(C_in+C_skip)→C_in/2` | grouped (≈ depthwise) | 3×3 | **≈ 11.8 M** |
| Depthwise transpose + **k1** fuse | grouped | 1×1 | ≈ 1.35 M |
| Paper-reported | — | — | **≈ 7.8 M** |

Per-stage k3 fuse dominates: `(1728→576)=8.96M`, `(864→288)=2.24M`, `(432→144)=0.56M`. The paper's **7.8 M sits between** the k1 and k3 brackets → most consistent with **k3 fuse + grouped transposed convs**, with the deepest fuse partially grouped/reduced. Exact 7.8 M is not reachable without the unspecified `groups_i`. **Recommended target: native widths, grouped depthwise transpose, 3×3 fuse → ≈ 8–12 M; tune `groups_i`/kernel to land near 7.8 M.**

This is the heaviest trainable module by design (no width reduction), consistent with the budget (7.8 M of 14.8 M total).

---

## 6. Ambiguities — Known / Inferred / Unknown

### Known (explicit in paper)
- Inputs are four **encoding-stage** features `s1..s4`.
- `ConvTranspose2d(kernel=4, stride=2, padding=1)` for upsampling (exact ×2).
- Per-stage: deconv → concat next shallower → `Conv2d → ReLU`.
- Channel reduction `C_out = C_in / 2`.
- Final restore to **3 channels** via deconvolution, then **interpolate ×2** to input size.
- Output `f_rec ∈ B×3×H×W`; **train-only**; loss = `MSE(f_rec, x)` weighted `λ₃`.
- Budget ≈ 7.8 M.

### Inferred (reconstructed to satisfy the paper)
- `s1` = **deepest** stage (1152@12²), `s4` = **shallowest** (144@96²) — required so that deconv(s1)→concat(s2) aligns resolutions.
- Encoding-stage features = **post-MMPA spatial** features (native widths) — the faithful reading of "encoding stage" (Design Freeze §1.7).
- Fuse `Conv2d` operates on the **concatenated** tensor `(C_in + C_skip)` → `C_in/2` (Eq. 12's `in_channel=C_in` labelling is internally inconsistent; concat input is the only shape that type-checks).
- Fuse kernel = **3×3** (closest to "convolutional block"; matches the 7.8 M bracket better than 1×1).

### Unknown (not in paper, no supplementary)
- Exact `groups_i` of each transposed convolution.
- Exact fuse-conv kernel size (3×3 inferred).
- Whether BatchNorm/GroupNorm is applied inside RDB (text says only "Conv + ReLU" → assume **no norm**).
- Numerical value of `λ₃`.
- Whether `s1` is fed raw or MMPA-adapted (Inferred: MMPA-adapted).

---

## 7. Final implementation specification

### 7.1 Architecture diagram
```
encoding-stage features (post-MMPA, native widths, deep→shallow)
   s1: B×1152×12×12   s2: B×576×24×24   s3: B×288×48×48   s4: B×144×96×96
        │
        ▼  ConvTranspose2d(1152→1152, k4,s2,p1, groups)        → B×1152×24×24
        ├─ Concat(s2)                                          → B×1728×24×24
        ▼  Conv2d(1728→576, k3) → ReLU            [w1]         → B×576×24×24
        ▼  ConvTranspose2d(576→576, k4,s2,p1, groups)          → B×576×48×48
        ├─ Concat(s3)                                          → B×864×48×48
        ▼  Conv2d(864→288, k3) → ReLU             [w2]         → B×288×48×48
        ▼  ConvTranspose2d(288→288, k4,s2,p1, groups)          → B×288×96×96
        ├─ Concat(s4)                                          → B×432×96×96
        ▼  Conv2d(432→144, k3) → ReLU             [w3]         → B×144×96×96
        ▼  ConvTranspose2d(144→3, k4,s2,p1)       [→ 3 ch]     → B×3×192×192
        ▼  Interpolate(scale_factor=2)  (or to 384²)           → B×3×384×384
                                                               = f_rec
```

### 7.2 Tensor flow (shapes, batch B)
| Step | Op | Output |
|---|---|---|
| in | s1,s2,s3,s4 | `1152@12², 576@24², 288@48², 144@96²` |
| 1 | deconv s1 + concat s2 + Conv3×3→ReLU | `w1: 576@24²` |
| 2 | deconv w1 + concat s3 + Conv3×3→ReLU | `w2: 288@48²` |
| 3 | deconv w2 + concat s4 + Conv3×3→ReLU | `w3: 144@96²` |
| 4 | deconv w3 → 3ch | `3@192²` |
| 5 | interpolate → input size | `f_rec: 3@384²` |

### 7.3 Parameter estimate
| Stage | Transpose | Fuse (k3) | Subtotal |
|---|---:|---:|---:|
| s1 | grouped | 1728→576 ≈ 8.96 M | ~8.96 M |
| s2 | grouped | 864→288 ≈ 2.24 M | ~2.24 M |
| s3 | grouped | 432→144 ≈ 0.56 M | ~0.56 M |
| final | 144→3 ≈ 0.007 M | — | ~0.01 M |
| **Total (k3)** | | | **≈ 11.8 M** |
| Paper target | | | **≈ 7.8 M** (tune groups / reduce deepest fuse) |

### 7.4 Implementation checklist (Milestone 4B)
- [ ] Implement **inside** `models/maffnet_original_reconstructed.py` only; do not touch verified modules.
- [ ] Inputs = the four **post-MMPA spatial** features (native widths), passed deep→shallow.
- [ ] 3 reconstruction stages: `ConvTranspose2d(C→C, k4,s2,p1, groups=C)` → `Concat(next)` → `Conv2d((C+C_skip)→C//2, k3)` → `ReLU`.
- [ ] Final `ConvTranspose2d(144→3, k4,s2,p1)` then `F.interpolate` to `384²`.
- [ ] No normalization inside RDB (Conv + ReLU only).
- [ ] **Train-only**: compute `f_rec` only in `model.training`; skip at eval.
- [ ] Output shape `B×3×384×384`; expose as `reconstruction` (and later wire into `MSE(f_rec, x)` at Milestone 6).
- [ ] Verify: compile, dummy forward, shapes, RDB param count (target ≈ 7.8 M; document any residual), total params.
- [ ] Keep all prior forward outputs unchanged.

---

**Audit complete. No code implemented. Not proceeding to Milestone 4B.**

*Status: Milestone 4A (audit) / analysis only / no code or file changed / existing models unchanged.*
