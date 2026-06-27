# MAFFNet — Design & Contribution Analysis (Q1 Paper Planning)

**Original paper:** Zhou, Gong, Qiu, Wei, Li, Zhang. *"MAFF-Net: SAM-powered Mixed Multi-scale Perception Adaptation and frequency-guided feature fusion for robust skin lesion segmentation."* Biomedical Signal Processing and Control 113 (2026) 109222. Nanchang University.
**Our impl:** `models/maffnet.py` + `training/trainer.py` (MAFFNetLoss).
**Mode:** analysis only. No code modified. No architecture redesigned.

> **Blunt headline (read first).** The current implementation is **not a set of improvements over** MAFF-Net — it is a **simplified / partial re-implementation that downgrades the paper's three core contributions** (dynamic MMPA adaptor, frequency-transformer FGFF, EDB attention + reconstruction self-supervision) and adds **one genuine new element (boundary/edge supervision)**. For a Q1 paper that cites the original MAFF-Net as the base, claiming the simplified MMPA/FGFF as "novel" will be caught instantly. The realistic contribution surface is **narrow** and is spelled out honestly below.

---

## 1. Overall Design Philosophy

The original MAFF-Net philosophy: **freeze a strong SAM2 (Hiera-L) encoder; adapt it cheaply** with (a) a lightweight *dynamic* multi-scale **adaptor** (MMPA, ~1.7M params, bottleneck-32, softmax-weighted experts), (b) a *parallel frequency stream* with a small **transformer** + **cross-modal** fusion (FGFF), and (c) an attention-enhanced decoder (EDB) with **reconstruction self-supervision** (RDB + MSE). Only ~6.5% of params are trainable.

Our implementation keeps the **silhouette** (frozen SAM2 → per-scale modules → multi-head decoder) but replaces each adaptive sub-module with a **static, heavier, simpler** counterpart, and **drops the reconstruction loss**. Net: our model is closer to "SAM2 + ASPP-ish blocks + SE + plain U-Net decoder + boundary loss" than to the published MAFF-Net.

**Design stance for the paper (recommended):** *do not* market the simplifications as contributions. Instead, (i) restore what is half-built and free (reconstruction loss, EDB attention), and (ii) formalize the one true addition — **boundary-aware deep supervision** — plus any small, ablatable additions that survive scrutiny.

---

## 2. Module-by-Module Comparison

| Module | Original MAFF-Net | Current Version (ours) | Difference | Reason (technical) | Expected Benefit | Novelty |
|--------|-------------------|------------------------|------------|--------------------|------------------|---------|
| **Encoder** | Frozen **SAM2 Hiera-L**, raw trunk multi-scale `[144,288,576,1152]` (H/4..H/32) | Frozen **SAM2 Hiera-L**, **FPN-neck** outputs, uniform **256-ch** | **Modified (minor)** — same backbone, different feature tap | FPN gives channel-uniform maps (simpler wiring) | Simpler code; possible loss of native scale-specific capacity | **Low** |
| **Multi-scale extraction / MMPA** | *Mixed Multi-scale Perception Adaptor*: project C→32, 3 depthwise dilated experts (d=3,5,7), **WetGeneration softmax dynamic weights**, residual w/ Γ=1e-7. ~1.7M params | *MMPA renamed "Morphological Pyramid Attention"*: 3 full-width dilated convs (d=3,5,7) + 1×1 ctx + concat→1×1 + **SE channel attn** + residual | **Replaced** — static SE instead of dynamic softmax experts; no 32-bottleneck; full-width (heavier) | Reuses standard ASPP+SE blocks | Lower param-efficiency; **loses the paper's #1 contribution** (dynamic cross-scale weighting) | **Low** (regression) |
| **Frequency branch (encoder)** | Parallel **dual-stream** freq encoder on grayscale-image FFT + **lightweight Transformer** (DWConv-QKV, learnable temperature emphasising high-freq), 4-stage | FFT taken **per feature-map level**; only a **1×1 conv** on magnitude. No freq transformer, no separate stream | **Missing / heavily simplified** | Avoids a second encoder stream | Much cheaper; **loses long-range freq modelling + high-freq emphasis** | **Low** (regression) |
| **FFT processing** | grayscale→FFT→**fftshift**→magnitude→min-max→pseudo-RGB | feature-map `rfft2`→magnitude→interp→min-max (no shift, no grayscale) | **Modified** — different domain (feature vs image), no centering | Operates on deep features directly | Cheaper; semantics differ from paper | **Low** |
| **Attention modules** | Dynamic expert softmax (MMPA) + **cross-modal freq attn** (β) + **EDB channel attn** + freq temperature | **SE channel attn only** (in MMPA) | **Missing / replaced** | Single standard SE block | Fewer adaptive pathways | **Low** |
| **Feature fusion (FGFF)** | `f_fused = β⊙(f_freq_attn ⊗ f_spatial_fused)`; multi-dilation spatial (d=1,3,5,7); cross-modal | `concat([spatial_conv, freq_conv]) → 1×1 → +residual` | **Replaced** — concat instead of cross-modal gating; no β; no multi-dilation | Simpler fusion | **Loses cross-modal complementarity** (the FGFF selling point) | **Low** (regression) |
| **Decoder (EDB)** | Enhanced Decoder Block: upsample+pad+concat+DoubleConv **+ channel-attention skip** (γ-scaled) | Plain `DecoderBlock` (upsample+concat+DoubleConv); **zero-tensor skip at level 1** | **Modified** — EDB attention removed; L1 skip is zeros | Generic U-Net block | Loses boundary-focused skip attention; L1 zero-skip is a defect | **Low** (regression) |
| **Skip connections** | 3 attention-enhanced skips + 4-head deep supervision | Plain concat skips; **dec1 zero-skip**; 4-head deep supervision kept | **Modified** | — | Deep supervision retained (good); zero-skip wastes the finest level | **Low** |
| **Loss function** | `Σ λ_i·Dice(up(dec_i),y) + λ3·MSE(f_rec,x)`; λ={1,1,1,0.01}; **Dice + reconstruction MSE** | `Dice + BCEWithLogits(pos_weight=2) + 0.01·Sobel BoundaryLoss` (after warmup); **no reconstruction MSE** | **Replaced + Added** — **+Sobel boundary loss (new)**, +BCE; **−reconstruction MSE** | Edge emphasis + class-imbalance handling | **Boundary loss = the one genuine addition**; but RDB self-supervision (paper contribution) is dropped | **Medium** (boundary) / Low (BCE) |
| **Training strategy** | batch **12**, **100** epochs, AdamW 1e-3, wd 0.01, linear decay (1.0→0.001) | batch **2**, **20–40** epochs, AdamW 1e-3, wd 0.01, linear decay; **+early stopping** | **Modified** — scaled down + early stop | GPU/memory limits | Faster/cheaper; early-stop is a small infra add | **Low** (engineering) |
| **Data preprocessing** | Rich aug (colour jitter, posterize, sharpen, Gaussian blur/noise, rotation, hflip, affine) | Subset (hflip, vflip, rotate, affine, colour jitter); **no ImageNet norm**; anisotropic 384 resize | **Modified** — fewer augs; missing norm | Simpler pipeline | Less regularisation; norm gap may hurt frozen SAM2 | **Low** (partly regression) |
| **Inference pipeline** | RDB disabled at inference; seg-only; 0.08s/img @384 | RDB computed but unused; seg-only | **Identical (intent)** | — | — | **None** |

---

## 3. Answers to the 10 Questions

**1. Real research contributions of our paper (honest):**
- Only one stands today: **boundary-aware deep supervision** (Sobel/edge L1 added to the MAFF-Net multi-head decoder). Everything else is identical (backbone, inference) or a simplification/regression. As-is, this is **not enough for Q1**.

**2. Which modifications are *only* engineering:**
- FPN-neck feature tap; SE instead of dynamic adaptor (a swap, not an advance); concat instead of cross-modal FGFF; batch/epoch scaling; early stopping; reporting/benchmark tooling; AMP/fp16. None are paper-worthy "research."

**3. Which can *legitimately* be claimed as research contributions:**
- **Boundary/edge supervision** (with curriculum warmup) — yes, *if* ablation shows gain.
- Potentially **restoring + extending reconstruction self-supervision** if you make it non-trivial (e.g., perceptual/edge-weighted recon, not plain MSE).
- A *defensible* claim only if you ADD something new on top — current deltas alone are too thin/negative.

**4. Changes significant enough for the Methodology section:**
- Boundary-supervision loss (formula + warmup schedule). The decoder/loss configuration actually used. The encoder feature-tap choice (state honestly as a design choice, not a contribution).

**5. Changes deserving their own subsection:**
- **Boundary-Aware Deep Supervision** (one subsection). Nothing else currently rises to a standalone subsection without first being restored/upgraded.

**6. Modifications requiring new diagrams:**
- Boundary-loss pipeline (Sobel on sigmoid prob + GT, multi-scale aggregation). A revised overall architecture figure that **matches the code** (so the paper figure ≠ the original MAFF-Net figure, which the code does not implement).

**7. Modifications requiring mathematical formulation:**
- Boundary loss: `L_bnd = (1/4)Σ_i L1(Sobel(σ(p_i)), Sobel(y))`, total `L = L_seg + λ_bnd·L_bnd`, λ_bnd=0.01, active epoch≥10.
- If reconstruction restored: the MSE/edge-weighted recon term.
- The exact total-loss equation (currently Dice+BCE+boundary) must be written, since it **differs from the paper's** Dice+MSE.

**8. Modifications requiring ablation experiments:**
- Boundary loss on/off; λ_bnd sweep {0,0.001,0.01,0.05,0.1}; warmup vs always-on.
- BCE term on/off (vs Dice-only).
- FPN-neck vs raw-trunk features.
- (If restored) reconstruction loss on/off; EDB attention on/off; dynamic MMPA vs SE-MMPA.

**9. Modifications NOT worth mentioning in the paper:**
- Early stopping, AMP/fp16, reporting/benchmark utilities, the dec1 zero-skip defect (fix silently), `metrics.json` plumbing, FPN wiring details. These are housekeeping.

**10. As a Q1 reviewer, the primary novelty I would identify:**
- **Currently: none sufficient.** I would note the submission *removes* MAFF-Net's headline modules (dynamic MMPA, freq-transformer FGFF, reconstruction self-supervision) and adds a small boundary loss — i.e., a **weaker variant of the cited base** plus one common edge-loss trick. I would request either faithful restoration of the original modules **plus** a genuinely new component, or a re-scoped claim. The **only** salvageable primary-novelty candidate is **boundary-aware multi-scale supervision on a SAM2-adapted segmenter**, and only if ablations + SOTA comparison support it.

---

## 4. Final List of Research Contributions (realistic, honest)

**Tier 1 — defensible now (with experiments):**
1. **Boundary-aware deep supervision** for SAM2-adapted skin-lesion segmentation (Sobel-edge L1, curriculum warmup, multi-scale). Needs ablation + edge-metric (HD95/Boundary-IoU) evidence.

**Tier 2 — defensible only after small, in-scope additions (no full redesign):**
2. **Restored + upgraded reconstruction self-supervision** (RDB is already coded but unused — wire its MSE back, optionally edge-weighted). Turns a current *regression* into a contribution.
3. **Class-imbalance-aware compound loss** (Dice+BCE+boundary) with an ablated weighting — minor but real if shown to beat Dice-only/Dice+MSE.

**Not contributions (state as design choices or fix silently):** SE-MMPA, concat-FGFF, FPN tap, training scale-down, tooling.

> If the goal is genuine Q1 novelty, the **minimum honest path** is: restore the original MMPA/FGFF/EDB/RDB faithfully (so the base is the real MAFF-Net), **then** add boundary supervision (+1 more new idea) and prove the delta by ablation. Selling the current simplified build as "improved MAFF-Net" is not viable.

---

## 5. Methodology Subsections (proposed)

1. **Preliminaries** — frozen SAM2 Hiera-L encoder, multi-scale features.
2. **Adapted multi-scale perception** — describe MMPA *as actually implemented*, honestly (or restore original first).
3. **Frequency-guided fusion** — describe FGFF *as implemented* (or restore).
4. **Decoder & deep supervision** — EDB/decoder + 4-head supervision.
5. **Boundary-Aware Deep Supervision** ← the contribution subsection (formula + warmup).
6. **(If restored) Reconstruction self-supervision.**
7. **Total loss & training.**

---

## 6. Figures To Create

| # | Figure | Why |
|---|--------|-----|
| F1 | Overall architecture **matching the code** | Current paper-style figure would misrepresent the impl |
| F2 | MMPA block (as implemented) | Differs from paper Fig. 2/3 |
| F3 | FGFF block (as implemented) | Differs from paper Fig. 4 |
| F4 | **Boundary-supervision pipeline** (Sobel on prob & GT, multi-scale) | New contribution |
| F5 | Qualitative best/avg/difficult predictions (already produced by report generator) | Standard |
| F6 | Loss-curve / ablation bar charts | Standard |

---

## 7. Tables Required

| # | Table | Content |
|---|-------|---------|
| T1 | SOTA comparison | vs U-Net, Att-UNet, UNet++, **official TransUNet (now migrated)**, original MAFF-Net, recent 2023–24 SOTA, on ISIC 2016/2017/2018 (test, mean±std) |
| T2 | Component ablation | baseline → +MMPA → +FGFF → +EDB → +boundary (+recon) |
| T3 | Boundary-loss λ sweep | {0, .001, .01, .05, .1} |
| T4 | Loss-composition ablation | Dice / Dice+BCE / Dice+MSE(recon) / Dice+boundary / full |
| T5 | Efficiency | params (trainable %), FLOPs, latency — vs original's 14.8M / 6.5% / 0.08s |

---

## 8. Ablation Studies Required

1. **Boundary loss:** on/off, λ sweep, warmup vs always-on, Sobel vs Laplacian.
2. **Loss composition:** Dice-only vs +BCE vs +recon-MSE vs +boundary vs full.
3. **Reconstruction self-supervision:** off (current) vs on (restored).
4. **MMPA variant:** SE-MMPA (ours) vs dynamic-adaptor MMPA (original) — to justify whichever you keep.
5. **FGFF variant:** concat (ours) vs cross-modal β (original).
6. **EDB attention:** on/off.
7. **Encoder tap:** FPN-256 vs raw-trunk multi-channel.
8. **Multi-seed** (≥3) mean±std for every headline number.

---

## 9. Experimental Validation Required

- **Test-split** metrics (not val) on ISIC 2016/2017/2018 — ISIC 2017 test masks now exist via the re-split (`SPLIT_AUDIT.md`).
- Metrics: Dice, IoU, Accuracy, Sensitivity, Specificity **+ HD95 / Boundary-IoU** (mandatory to support a boundary-loss claim).
- ≥3 seeds, mean±std, paired significance test vs original MAFF-Net config.
- SOTA baselines incl. the **migrated official TransUNet** and recent 2023–24 methods.
- Match (or fairly state deviation from) the original's training budget (batch 12 / 100 epochs) — current batch 2 / 20–40 epochs is **not comparable** to the paper's numbers.
- Efficiency table vs the paper's 14.8M trainable / 6.5% / 0.08s.

---

## Evidence References
- Original paper §3.1 MMPA (bottleneck-32, softmax experts, Γ=1e-7), §3.2 FGFF (grayscale FFT, freq transformer, cross-modal β), §3.3 EDB+RDB, §3.4 loss (Dice+MSE), §4.2 (sam2_hiera_large, batch 12, 100 ep, AdamW 1e-3), §4.5 ablation groups
- `models/maffnet.py:39-76` (MMPA=SE), `:82-113` (FGFF=concat), `:119-132` (plain decoder), `:138-180` (RDB), `:350-351` (dec1 zero-skip), `:362-364` (rec computed)
- `training/trainer.py:68-113` (loss = Dice+BCE+boundary, **no recon MSE**; RDB output `_rec` discarded)
- companion: `ARCHITECTURE_AUDIT.md`, `NOVELTY_AUDIT.md`, `PREPROCESSING_AUDIT.md`, `SPLIT_AUDIT.md`
