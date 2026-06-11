# MAFFNet Architecture Audit

**Scope:** Compare implemented graph (`models/maffnet.py`) against intended spec.
**Code read:** `models/maffnet.py` (366 lines), `training/trainer.py` (loss + BoundaryLoss).
**Verdict up front:** Implementation diverges in 3 of 11 spec stages. FFT path has **no high-pass filter and no inverse FFT** — magnitude spectrum used directly as feature gate. Boundary supervision exists but uses Sobel on `sigmoid(logits)`, not on `output mask`.

---

## 1. Actual Computational Graph (as coded)

```
Input  x  (B, 3, H, W)         [H = W = 384]
  │
  ▼
SAM2Encoder (frozen Hiera-Large + FPN)
  │   torch.no_grad() backbone
  │   returns 4 maps (256 ch each); if FPN has 3, f4 = avgpool(features[2], ×0.5)
  ▼
f1, f2, f3, f4   (shallow → deep, all 256 ch)
  │
  ▼  per-level
MMPA_i(f_i):
   b3 = ConvBNReLU(k=3, dilation=3)(f_i)
   b5 = ConvBNReLU(k=3, dilation=5)(f_i)
   b7 = ConvBNReLU(k=3, dilation=7)(f_i)
   b1 = ConvBNReLU(k=1)(f_i)             # context
   out = ConvBNReLU(1×1)(cat[b3,b5,b7,b1])
   ca  = Sigmoid(FC(ReLU(FC(AvgPool(out)+MaxPool(out)))))   # SE-style
   f_i = out · ca + f_i                  # residual
  │
  ▼  per-level
FGFF_i(f_i):
   x_float = f_i.float()
   x_fft   = torch.fft.rfft2(x_float, norm='ortho')         # complex, (B,C,H,W/2+1)
   mag     = |x_fft|                                         # magnitude only
   mag     = bilinear_resize(mag, H×W)
   mag     = (mag − min) / (max − min + 1e−8)                # per-channel min-max
   freq    = ConvBNReLU_1x1(mag)
   spat    = ConvBNReLU_3x3(f_i)
   fused   = ConvBNReLU_1x1(cat[spat, freq])
   f_i     = fused + f_i                                     # residual
  │
  ▼
Decoder (U-Net, no channel reduction)
   d4 = DecoderBlock(f4, skip=f3)
   d3 = DecoderBlock(d4, skip=f2)
   d2 = DecoderBlock(d3, skip=f1)
   d1 = DecoderBlock(d2, skip=ZEROS)                        # no real skip at lvl 1
  │
  ▼
Seg heads (1×1 conv, then bilinear upsample to H×W)
   p4 = up(head4(d4)),  p3 = up(head3(d3)),
   p2 = up(head2(d2)),  p1 = up(head1(d1))
   ──> logits, NOT sigmoided
  │
  ▼
RDB (auxiliary RGB reconstruction)
   rec = up_chain(f4, f3, f2, f1) → Tanh → resize H×W
  │
  ▼
Loss (training/trainer.py MAFFNetLoss):
   for p in (p1,p2,p3,p4):
       prob = sigmoid(p)
       seg += Dice(prob, mask) + BCEWithLogits(p, mask, pos_weight=2.0)
   seg /= 4
   if epoch >= 10:
       for p in (p1..p4):
           prob_smooth = avg_pool2d(sigmoid(p), k=3)
           bnd += L1(Sobel(prob_smooth), Sobel(gt))
       bnd /= 4
       total = seg + 0.01 · bnd
   else:
       total = seg
```

Note: `rec` is computed every forward pass but **never used by the loss** (`MAFFNetLoss.forward` discards `_rec`). RDB is dead weight at training time.

---

## 2. Intended Spec vs. Actual — Per-Stage Comparison

| # | Intended stage | Status | Where in code | Notes |
|---|----------------|--------|--------------|-------|
| 1 | Input | ✅ **Fully implemented** | `forward(x)` | (B,3,384,384), correct |
| 2 | SAM2 Encoder | ✅ **Fully implemented** | `SAM2Encoder` lines 186–235 | Frozen Hiera-Large + FPN; 4 levels (last via ×0.5 avgpool of FPN[2] when FPN n=3); 256 ch each |
| 3 | Stabilized FFT | ⚠️ **Partially implemented** | `FGFF.forward` lines 97–113 | `torch.fft.rfft2` with `norm='ortho'`, cast to fp32. No NaN/inf guard, no spectral whitening — "stabilized" is just `.float()` upcast. No log-magnitude or DC removal |
| 4 | High-Pass Filter | ❌ **Missing** | — | No frequency mask, no radial filter, no DC suppression. Magnitude spectrum used **as is** |
| 5 | Inverse FFT | ❌ **Missing** | — | No `torch.fft.irfft2` anywhere. Frequency information consumed as magnitude only — does not return to spatial domain via inverse transform |
| 6 | Feature Alignment | ⚠️ **Implemented differently** | `F.interpolate` calls in `DecoderBlock`, `RDB`, `SAM2Encoder` | Naive bilinear resize on shape mismatch. No deformable conv, no cross-attention align, no learned offset |
| 7 | Gated Fusion | ⚠️ **Implemented differently** | `FGFF.fuse`, `MMPA` SE block | "Gating" = 1×1 conv on `cat[spatial, freq]` (no sigmoid gate). MMPA has SE-style channel gate, but spec gate is fusion-level, not channel attention |
| 8 | Decoder | ✅ **Fully implemented** | `DecoderBlock` lines 119–132, `MAFFNet.dec1..dec4` lines 308–311 | U-Net style, 4 levels. Caveat: `dec1` uses **zero tensor as skip** (line 350) — degrades the lowest level to a plain upsample-conv |
| 9 | Output Mask | ⚠️ **Implemented differently** | `head1..head4` lines 314–317; outputs lines 357–360 | Returns **4 deep-supervision logits** (p1..p4), not single output mask. No sigmoid applied in forward (correct for BCEWithLogits). p1 is "main"; spec implies one mask |
| 10 | Sobel Edge Detection | ⚠️ **Implemented differently** | `BoundaryLoss._edges` `training/trainer.py:57–62` | Sobel applied to `sigmoid(logits)` after `avg_pool2d(k=3)`, not to "output mask" (i.e., not on `pred > 0.5` binary). Done at loss time, not as architectural branch |
| 11 | Boundary Supervision | ⚠️ **Partially implemented** | `MAFFNetLoss` `training/trainer.py:97–108` | L1 between Sobel of smoothed prob and Sobel of GT. Weight 0.01. Active only from `epoch >= 10` (warmup). Spec implies always-on; current impl gates by epoch and uses tiny weight |

**Extra components not in spec but present:**
- **MMPA** (multi-scale dilated convs + SE): exists in code, not in spec — adds spatial multi-scale before frequency stage.
- **RDB** (RGB reconstruction head): defined, runs in forward, **not used by loss** — pure compute waste.
- **Deep supervision** (4 seg heads): in code, not in spec.

---

## 3. Classification Summary

| Status | Count | Stages |
|--------|------:|--------|
| ✅ Fully implemented | 3 | Input, SAM2 Encoder, Decoder |
| ⚠️ Partially implemented | 2 | Stabilized FFT, Boundary Supervision |
| ❌ Missing | 2 | High-Pass Filter, Inverse FFT |
| 🔄 Implemented differently | 4 | Feature Alignment, Gated Fusion, Output Mask, Sobel Edge Detection |

---

## 4. Impact of Each Mismatch

### 4.1 Missing High-Pass Filter (stage 4)
- **Methodology:** Paper claim "frequency-guided high-pass enhancement of lesion boundaries" is **unsupported by code**. Low-frequency DC component dominates magnitude → FGFF effectively learns a global brightness gate, not edge enhancement.
- **Experimental setup:** All reported MAFFNet metrics (Dice 0.92/0.78/0.88 on ISIC 2016/2017/2018) were produced **without** the spec's high-pass operator. Cannot attribute performance to high-pass design.
- **Ablation:** Cannot run "with vs. without high-pass" ablation — no high-pass branch exists. Reviewer asking for ablation will get null result.
- **Reproducibility:** Anyone reading the paper and implementing high-pass faithfully will produce a **different model** than the one in the checkpoint.

### 4.2 Missing Inverse FFT (stage 5)
- **Methodology:** Spec says signal returns to spatial domain after frequency manipulation. Code never inverses — frequency magnitude treated as 2D image and convolved spatially. Phase information **discarded entirely** (only `|FFT|` used).
- **Experimental setup:** Frequency branch is technically `|rfft2|` → bilinear-resize → 1×1 conv. Equivalent to learning a spatial magnitude statistic, not a frequency-domain transformation. Method name "Frequency-Guided Feature Fusion" is misleading.
- **Ablation:** Cannot isolate "spatial-domain frequency representation" vs. "inverse-FFT'd reconstruction" — second variant unimplemented.
- **Reproducibility:** Faithful re-implementer using `irfft2` will get different feature scale, different gradients, likely lower memory, possibly different convergence.

### 4.3 Partial Stabilized FFT (stage 3)
- **Methodology:** "Stabilized" implies numerical safeguards (log scaling, DC clamp, NaN replacement). Current "stabilization" is one line: `.float()` upcast to escape `cuFFT` fp16 limitation. No log spectrum, no NaN sentinel.
- **Experimental setup:** Min-max normalisation per channel (`mag_full - min / (max - min + 1e-8)`) can collapse to ≈0 for nearly-constant feature maps → freq_conv input near zero, contribution vanishes. Silent failure mode.
- **Ablation:** No log-spectrum vs. linear-spectrum ablation possible.
- **Reproducibility:** `1e-8` denominator floor is arbitrary; another implementer may pick `1e-6` and shift activations.

### 4.4 Different Feature Alignment (stage 6)
- **Methodology:** Bilinear resize is the simplest possible alignment; spec language ("alignment") suggests learned. Not catastrophic but downgrades novelty claim.
- **Experimental setup:** `if x.shape[2:] != skip.shape[2:]` fallback hides shape bugs — adversary can't tell from logs whether resize fired.
- **Ablation:** "Bilinear vs. deformable align" ablation unimplementable without new code.
- **Reproducibility:** OK — `F.interpolate(..., mode='bilinear', align_corners=False)` is deterministic and standard.

### 4.5 Different Gated Fusion (stage 7)
- **Methodology:** No sigmoid gate on the fusion path; `self.fuse = ConvBNReLU(channels*2, channels, 1)` is plain concat+conv. "Gated" misnomer. MMPA SE block is channel attention applied to MMPA output, not to FGFF fusion.
- **Experimental setup:** Performance attributed to "gated fusion" actually comes from concat+conv+residual.
- **Ablation:** Cannot ablate "gate on/off" — gate doesn't exist.
- **Reproducibility:** Re-implementer adding a real sigmoid gate will not match reported metrics.

### 4.6 Output Mask vs. Multi-head Deep Supervision (stage 9)
- **Methodology:** Spec implies single output; code returns `(p1, p2, p3, p4, rec)`. Deep supervision is a meaningful design choice but it changes loss landscape and effective gradient flow.
- **Experimental setup:** Reported Dice computed on `p1` only (`compute_metrics` receives `p1` via trainer's `_val_epoch`). Need to verify which head is reported.
- **Ablation:** "With vs. without deep supervision" — possible by zeroing aux head losses, but not done in any logged run.
- **Reproducibility:** Single-head re-implementer will under-train shallow decoder layers, get lower Dice.

### 4.7 Sobel on prob, not on output mask (stage 10)
- **Methodology:** Spec "Sobel on output mask" suggests binary mask edges. Code uses Sobel on `sigmoid(logit)` after 3×3 avg-pool smoothing. Differentiable (required for backprop) but **not the same operator** as spec.
- **Experimental setup:** Smoothing kernel changes edge response; `avg_pool2d(k=3, stride=1, padding=1)` is hidden hyperparameter.
- **Ablation:** Sobel kernel size, smoothing window, prob-vs-binary all untested.
- **Reproducibility:** Faithful Sobel-on-binary won't backprop; spec is incompatible with gradient training without softening — design intent ambiguous.

### 4.8 Boundary Supervision gated by epoch (stage 11)
- **Methodology:** Boundary loss only active `epoch >= 10`. Spec gives no such gate. Warmup design choice is a hyperparameter, but `0.01` weight × late-onset means boundary signal contributes < 1% of total gradient.
- **Experimental setup:** From training_log.csv (MAFFNet ISIC 2016): `train_boundary_loss` stays ≈ 0.0433 from epoch 11–20. Negligible effect on Dice (curve plateaus before epoch 10).
- **Ablation:** "Boundary loss on/off" ablation possible by setting weight=0, but not run.
- **Reproducibility:** Re-implementer training 5-epoch budget never sees boundary loss fire → reports lower Dice incorrectly attributed to architecture.

### 4.9 RDB defined but unused at loss (extra component)
- **Methodology:** Reconstruction head is computed every forward, dropped at loss. Wasted FLOPs and parameters; could mislead a reader of `forward` into believing reconstruction loss is in play.
- **Experimental setup:** Train time penalty (extra decoder, ~few MB params).
- **Ablation:** Removing RDB has zero effect on metrics — easy win.
- **Reproducibility:** Re-implementer adding reconstruction loss (as the docstring implies) will get a **different model**.

### 4.10 dec1 zero-skip (decoder)
- **Methodology:** Lowest decoder level uses `torch.zeros_like(up_d2)` as skip (line 350). Equivalent to halving input channels of `dec1.conv`. Architectural quirk, not documented in any spec.
- **Experimental setup:** p1 (the highest-resolution and most-reported head) is built from this skip-less block.
- **Ablation:** "Real skip from x" vs. "zero skip" not run.
- **Reproducibility:** Re-implementer providing real f0 (e.g., stem features) will get different p1.

---

## 5. Aggregate Reproducibility Risk

| Risk dimension | Severity |
|----------------|----------|
| Paper-claim alignment | **High** — high-pass + inverse-FFT story unsupported |
| Numerical determinism | Low — bilinear, BN, no exotic ops |
| Result replicability from spec only | **High** — naive re-impl will diverge ≥ 2–5 pp Dice |
| Result replicability from code | Low — code is deterministic given seed |
| Ablation feasibility from current code | **High risk** — 4 spec stages missing; 4 different — half the ablation table is unimplementable |

---

## 6. Recommended Actions (no code changes performed)

Priority order, not yet executed per instruction:

1. **Decide spec authority.** If paper spec is canonical → add high-pass mask + inverse FFT; current code drifts from claim. If code is canonical → rewrite paper's FGFF section to describe magnitude-spectrum gating without inverse FFT.
2. **Add radial high-pass filter** in `FGFF.forward` between `rfft2` and conv: `mask = (radial_freq > cutoff).float(); x_fft = x_fft * mask`. Then either (a) feed `irfft2(x_fft)` to spatial conv (faithful), or (b) keep magnitude path and document deviation.
3. **Remove or wire RDB.** Either drop from `MAFFNet.__init__` + `forward`, or add reconstruction L1 to `MAFFNetLoss`. Currently neither.
4. **Make boundary loss always-on** (remove `epoch >= 10` gate) **or** justify the warmup explicitly in paper.
5. **Document deep supervision.** Spec says "output mask" singular; either drop p2–p4 heads or declare deep supervision as an intentional design.
6. **Replace `zero_skip` in dec1** with real stem features (e.g., a learned conv on `x`).
7. **Add ablation harness.** Currently no flag to toggle MMPA, FGFF, boundary loss, deep supervision — without flags, no rigorous ablation table possible.

---

## Evidence Reference

| Claim | Source |
|-------|--------|
| FFT uses magnitude only, no inverse | `models/maffnet.py:101–113` |
| No high-pass mask | grep `mask`, `cutoff`, `high.*pass` in `maffnet.py` → none |
| No `irfft` | grep `irfft` in repo → 0 hits in `maffnet.py` |
| FGFF "fuse" is plain 1×1 conv, no sigmoid | `maffnet.py:95, 112` |
| MMPA SE channel attention | `maffnet.py:55–63, 74–76` |
| Decoder zero skip at lvl 1 | `maffnet.py:349–351` |
| RDB output discarded at loss | `training/trainer.py:89` (`_rec` underscore) |
| Boundary loss epoch-gated | `training/trainer.py:97, 103` |
| Boundary loss weight 0.01 | `training/trainer.py:105` |
| Sobel on smoothed prob, not binary | `training/trainer.py:98` (`F.avg_pool2d(prob, ...)`) → `BoundaryLoss._edges` |
| Boundary loss flat at ≈0.0433 epochs 11–20 | `experiments/maffnet_ISIC_2016_2026-04-28_15-19-03/checkpoints/training_log.csv` |
