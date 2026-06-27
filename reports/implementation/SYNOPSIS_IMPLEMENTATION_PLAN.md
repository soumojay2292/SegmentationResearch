# Synopsis Implementation — Architecture Gap Analysis

**Reference:** approved synopsis *"Stabilised MAFFNet with SAM-based Encoder"* (§1.4 Architecture + Proposed Methodology, Fig. 1.1, §1.3.1 Objectives).
**Compared against:** current stable implementation (`models/maffnet.py`, `training/trainer.py`, `train.py`, `train_all.py`).
**Mode:** analysis only. No code modified, no source file created, nothing executed. `maffnet.py` is **not** overwritten.

> **Constraint for any future build:** the synopsis variant must be a **new** model (recommended `models/stabilized_maffnet.py`) registered under a new `--model` key. The current restored MAFFNet stays untouched.

---

## Synopsis Architecture (reference)

Dual-path design: input → **SAM encoder** → two **parallel** pathways:
- **Spatial path** — preserves semantic context, routed through gated fusion + projected skips to the decoder.
- **Frequency path** — **stabilised FFT → high-pass filter → inverse FFT** → boundary-sensitive spatial map.
- A **feature-alignment module** (instance norm + 1×1 channel projection) aligns SAM features and frequency features.
- **Lightweight gated fusion** (sigmoid-gated, linear complexity) merges the two aligned streams.
- **Projected skip connections** to the decoder; auxiliary **Sobel boundary supervision**; Dice/IoU + boundary objective; numerically stable mixed-precision training. Baselines include **SGT-Net**.

---

## Module-by-Module Classification

| # | Module | Classification |
|---|--------|----------------|
| 1 | SAM2 Encoder | ✅ Already Implemented |
| 2 | Spatial Branch | ✅ Already Implemented (via MMPA+FGFF) |
| 3 | Frequency Branch | ⚠️ Partially Implemented |
| 4 | FFT Processing (fp32 stabilised) | ✅ Already Implemented |
| 5 | High-pass Filtering | ❌ Missing |
| 6 | Feature Alignment | ❌ Missing |
| 7 | Lightweight Gated Fusion | 🔁 Needs Replacement |
| 8 | Decoder | ✅ Already Implemented |
| 9 | Skip Connections | ⚠️ Partially Implemented |
| 10 | Boundary Supervision | ✅ Already Implemented |
| 11 | Loss Functions | ✅ Already Implemented |
| 12 | Training Pipeline | ✅ Already Implemented |

**Net:** the synopsis's four signature components — high-pass, inverse FFT (within the frequency branch), feature alignment, gated fusion — are the gap. The encoder, fp32-FFT stabilisation, decoder, boundary supervision, loss, and stable training are already in place.

---

## Per-Module Detail (mismatches)

### 1. SAM2 Encoder — ✅ Already Implemented
- **Current:** frozen SAM2 Hiera-Large image encoder (`models/maffnet.py` `SAM2Encoder`, params `requires_grad_(False)`).
- **Synopsis:** SAM-based frozen encoder backbone.
- **Mismatch:** none (SAM2 satisfies/exceeds "SAM"). No action.

### 2. Spatial Branch — ✅ Already Implemented
- **Current:** SAM2 features → MMPA adaptation → FGFF spatial sub-branch (dilated convs) → decoder, preserving semantic context.
- **Synopsis:** spatial path preserving semantic context routed to the decoder.
- **Mismatch:** structural only — current spatial route passes through MMPA/FGFF rather than a plain pass-through. Acceptable; no required change for synopsis conformance.

### 3. Frequency Branch — ⚠️ Partially Implemented
- **Current:** FGFF computes a **per-level FFT magnitude** spectrum (`torch.abs(rfft2(x.float()))`), uses it as a spatial-attention gate; **no separate parallel frequency stream, no high-pass, no inverse FFT**.
- **Synopsis:** a **separate** frequency path: stabilised FFT → high-pass → **inverse FFT** → boundary-sensitive spatial feature map.
- **Required modification:** build a distinct frequency branch that high-pass-filters the spectrum and reconstructs a spatial map via `irfft2`, instead of magnitude-as-attention.
- **Files to edit:** new `models/stabilized_maffnet.py`.
- **Difficulty:** **Medium**.

### 4. FFT Processing — ✅ Already Implemented
- **Current:** FFT isolated to fp32 inside the bf16 pipeline (`rfft2(x.float())`), the synopsis's Objective-2 stabilisation.
- **Synopsis:** stabilised FFT (float32) preventing NaN.
- **Mismatch:** none. Reuse this approach in the new branch.

### 5. High-pass Filtering — ❌ Missing
- **Current:** none; the full magnitude spectrum is used.
- **Synopsis:** explicit high-pass spectral filter retaining edge-encoding coefficients.
- **Required modification:** apply a radial high-pass mask (learnable or fixed cutoff) to the complex spectrum before reconstruction: `X_hp = X * M_highpass(cutoff)`.
- **Files to edit:** new `models/stabilized_maffnet.py`.
- **Difficulty:** **Low–Medium**.

### 6. Feature Alignment — ❌ Missing
- **Current:** none. (Input ImageNet normalisation exists for the encoder, but no SAM↔frequency **feature**-alignment module.)
- **Synopsis:** dedicated module — instance normalisation + 1×1 channel projection — aligning SAM encoder outputs and frequency-branch features before fusion.
- **Required modification:** add `InstanceNorm2d` + `Conv2d(1×1)` projection on each stream so activation statistics match prior to fusion.
- **Files to edit:** new `models/stabilized_maffnet.py`.
- **Difficulty:** **Low**.

### 7. Lightweight Gated Fusion — 🔁 Needs Replacement
- **Current:** FGFF fusion `beta * (freq_attn * spatial_fused) + x` (per-channel `beta` scaling + frequency spatial-attention) — not a two-stream sigmoid gate.
- **Synopsis:** lightweight **sigmoid-gated** fusion at **linear** spatial complexity, learning a spatially varying weighting between aligned spatial and frequency streams.
- **Required modification:** replace the FGFF fusion with `g = σ(Conv([F_spatial, F_freq]))`, `F_out = g·F_spatial + (1−g)·F_freq`, operating on the aligned streams.
- **Files to edit:** new `models/stabilized_maffnet.py`.
- **Difficulty:** **Medium**.

### 8. Decoder — ✅ Already Implemented
- **Current:** U-Net-style decoder (`DecoderBlock` ×3 + skip-less finest refine) + 1×1 segmentation heads.
- **Synopsis:** convolutional decoder receiving fused features and skips.
- **Mismatch:** none of substance; reusable by the new model.

### 9. Skip Connections — ⚠️ Partially Implemented
- **Current:** decoder concatenates encoder/FGFF features as skips; no separate **projection** layer on the skips.
- **Synopsis:** **projected** skip connections (1×1) to preserve spatial detail.
- **Required modification:** add a 1×1 projection on each skip before concatenation.
- **Files to edit:** new `models/stabilized_maffnet.py`.
- **Difficulty:** **Low**.

### 10. Boundary Supervision — ✅ Already Implemented
- **Current:** Sobel-based boundary deep supervision (BDS) + edge-weighted reconstruction (EWR) in `training/trainer.py` (`BoundaryLoss`, now numerically stabilised).
- **Synopsis:** auxiliary Sobel edge map providing boundary supervision; boundary-aware learning as the primary objective.
- **Mismatch:** none — current implementation meets and exceeds (dual boundary supervision). Reusable as-is.

### 11. Loss Functions — ✅ Already Implemented
- **Current:** Dice + BCE + Sobel boundary (BDS) + edge-weighted reconstruction (EWR).
- **Synopsis:** Dice/IoU optimisation with boundary-aware supervision.
- **Mismatch:** none (current has extra BCE/EWR; harmless supersets). Reusable.

### 12. Training Pipeline — ✅ Already Implemented
- **Current:** AdamW, bf16 autocast (no GradScaler), grad-clip 1.0, early stopping, non-finite guard, fp32 FFT — numerically stable mixed-precision on standard hardware.
- **Synopsis:** stable, mixed-precision-compatible training without specialised hardware.
- **Mismatch:** none for training. **Evaluation gap:** synopsis lists an **SGT-Net** baseline, which is not implemented.
- **Files to edit (eval only):** `train_all.py` baseline list (+ a new SGT-Net model file if added).
- **Difficulty:** **Medium** (SGT-Net baseline integration).

---

## Gap Summary

| Synopsis component | Status | Difficulty |
|--------------------|:------:|:----------:|
| High-pass filtering | Missing | Low–Medium |
| Inverse FFT (frequency branch) | Missing/Partial | Medium |
| Feature alignment module | Missing | Low |
| Sigmoid gated fusion | Needs Replacement | Medium |
| Projected skip connections | Partial | Low |
| SGT-Net baseline | Missing | Medium |
| SAM2 encoder, fp32 FFT, decoder, boundary supervision, loss, stable training | Already done | — |

---

## Recommended Implementation Order (for the future build — not performed)

1. **New file `models/stabilized_maffnet.py`** — assemble: frozen SAM2 encoder (reuse `SAM2Encoder`) → spatial branch + **frequency branch (fp32 FFT → high-pass → irfft2)** → **feature-alignment** (InstanceNorm + 1×1) → **sigmoid gated fusion** → decoder (reuse `DecoderBlock`) with **projected skips** → segmentation heads.
2. **Register** the new model under a new key (e.g. `--model stabilized_maffnet`) in `train.py` and `train_all.py` (a few build-branch lines; do **not** alter the existing `maffnet` branch).
3. **Reuse** `training/trainer.py` (`MAFFNetTrainer`, boundary supervision, stable training) unchanged.
4. **(Optional, synopsis baseline)** add SGT-Net and append to the `train_all.py` baseline list.
5. Validate with the existing one-batch overfit + smoke gates before full ISIC runs.

**Total estimated effort:** ~2–3 days implementation + compute for ISIC 2016/2017/2018 runs. The current restored MAFFNet remains the stable fallback throughout.

---

## Constraints Honoured
- No code modified; `models/maffnet.py` not overwritten.
- No source file created (only this analysis document).
- Nothing executed.

## Evidence References
- Synopsis §1.4 (dual-path: stabilised FFT, high-pass, inverse FFT, feature alignment, gated fusion, projected skips), §1.3.1 Objectives, Fig. 1.1, Table 1.1 (SGT-Net baseline)
- `models/maffnet.py` (SAM2Encoder, MMPA, FGFF magnitude fusion — no high-pass/iFFT/alignment/gated-fusion), `training/trainer.py` (boundary supervision, stable bf16 training)
- `RESEARCH_ALIGNMENT_REPORT.md`, `RESTORATION_REPORT.md`, `FGFF_NUMERICAL_ANALYSIS_REPORT.md`
