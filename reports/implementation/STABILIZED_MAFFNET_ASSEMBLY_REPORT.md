# Stabilized MAFFNet — Assembly Report

**Task:** assemble the synopsis architecture from the approved reusable modules into a NEW model. `models/maffnet.py` and `training/trainer.py` not modified. No training, smoke test, or benchmark executed (static `py_compile` only → OK).

**File created:** `models/stabilized_maffnet.py` → class `StabilizedMAFFNet`.

---

## 1. Full Architecture Diagram

```
Input  x  (B,3,384,384)
   │  ImageNet normalise
   ▼
Frozen SAM2 (Hiera-L) encoder  ──►  f1,f2,f3,f4   (256-ch; strides 4/8/16/32)
   │
   ├──────────────── per encoder scale i ────────────────┐
   │   spatial stream:   FeatureAlignment_s(f_i)          │
   │   frequency stream: FrequencyBranch(f_i) → FeatureAlignment_f
   │                         │                            │
   │                         ▼                            │
   │                   GatedFusion_i(F_s, F_f)  ──► fused_i (g1..g4)
   └──────────────────────────────────────────────────────┘
   │
   ├─ projected skips:  s1=1x1(g1), s2=1x1(g2), s3=1x1(g3)
   ▼
Decoder (reused DecoderBlock):
   d4 = dec4(g4, s3)  →  d3 = dec3(d4, s2)  →  d2 = dec2(d3, s1)  →  d1 = up+conv(d2)
   │                                                                │
   ├─ heads: head4(d4),head3(d3),head2(d2),head1(d1) → upsample → p1..p4 (B,1,384,384)
   └─ RDB(g4,g3,g2,g1) → upsample → rec (B,3,384,384)
   ▼
Output  (p1, p2, p3, p4, rec)   ── identical interface to MAFFNet
```

---

## 2. Tensor Dimensions Through the Network (B, 384×384 input, C=256)

| Stage | Tensor(s) | Shape |
|-------|-----------|-------|
| input | x | (B, 3, 384, 384) |
| encoder | f1 / f2 / f3 / f4 | (B,256,96,96) / (…,48,48) / (…,24,24) / (…,12,12) |
| spatial align | align_s(f_i) | same as f_i |
| frequency branch | freq(f_i) | same as f_i (resolution preserved) |
| frequency align | align_f(freq(f_i)) | same as f_i |
| gated fusion | g1 / g2 / g3 / g4 | (B,256,96,96) / (…,48,48) / (…,24,24) / (…,12,12) |
| projected skips | s1 / s2 / s3 | (B,256,96,96) / (…,48,48) / (…,24,24) |
| decoder | d4 / d3 / d2 / d1 | (B,256,24,24) / (…,48,48) / (…,96,96) / (…,192,192) |
| heads (pre-upsample) | head4..head1 | (B,1,24,24)/(…,48,48)/(…,96,96)/(…,192,192) |
| seg outputs | p1, p2, p3, p4 | (B, 1, 384, 384) each |
| reconstruction | rec | (B, 3, 384, 384) |

Resolutions match the existing MAFFNet exactly; output interface `(p1,p2,p3,p4,rec)` is preserved.

---

## 3. Files Created
- `models/stabilized_maffnet.py` — the assembled model (new).
- `STABILIZED_MAFFNET_ASSEMBLY_REPORT.md` — this report.

*(The three reusable modules `modules/feature_alignment.py`, `modules/frequency_branch.py`, `modules/gated_fusion.py` were created in earlier tasks and are imported unchanged.)*

## 4. Files Modified
- **None.** `models/maffnet.py` and `training/trainer.py` untouched (timestamps verified). `train.py` not modified — see §"train.py compatibility".

## 5. Reused Components
| Component | Source | Use |
|-----------|--------|-----|
| `SAM2Encoder` / `SAMEncoder` | `models/maffnet.py` | frozen backbone |
| `DecoderBlock`, `ConvBNReLU` | `models/maffnet.py` | decoder |
| `RDB` | `models/maffnet.py` | auxiliary reconstruction |
| `FeatureAlignment` | `modules/feature_alignment.py` | spatial + frequency alignment |
| `FrequencyBranch` | `modules/frequency_branch.py` | high-pass spectral stream |
| `GatedFusion` | `modules/gated_fusion.py` | spatial–frequency fusion |
| `MAFFNetTrainer` + boundary supervision | `training/trainer.py` | training (via identical output interface — no change needed) |

## 6. New Components (in this assembly)
- The dual-stream-per-scale wiring (spatial align ∥ frequency branch+align → gated fusion) across all 4 encoder scales.
- Projected skip connections: three 1×1 convs (`skip_proj`) on the fused features.
- ImageNet input normalisation (same pattern as MAFFNet) for the frozen encoder.
- Constructor `StabilizedMAFFNet(encoder=None, checkpoint=None, cutoff=0.1)` mirroring MAFFNet.

---

## 7. FLOPs and Parameter Overhead Estimate (C=256, 384², B=1)

**Trainable parameters (new model):**
| Block | Params |
|-------|-------:|
| Frozen SAM2 backbone | 212.70 M (frozen, 0 trainable) |
| FeatureAlignment ×8 (4 spatial + 4 freq) | ~0.526 M |
| GatedFusion ×4 | ~0.525 M |
| FrequencyBranch ×4 | 0 (fixed mask) |
| skip_proj ×3 (1×1) | ~0.197 M |
| Decoder (DecoderBlock ×3 + dec1) | ~6.49 M |
| RDB | ~3.26 M |
| heads ×4 | ~0.001 M |
| **Total trainable** | **≈ 11.0 M** |
| **Total (incl. frozen)** | **≈ 223.7 M** |

**Overhead from the new synopsis modules** (alignment + fusion + skip_proj + freq): **≈ +1.25 M trainable params**, **+~4.2 G FLOPs** (FeatureAlignment 1×1 convs ~1.6 G, GatedFusion ~1.6 G, skip_proj ~0.8 G, FrequencyBranch FFTs ~0.2 G).

**Net vs the current restored MAFFNet** (trainable ≈ 28.98 M): StabilizedMAFFNet **drops** MMPA (~0.08 M / ~25.7 G) and the heavy dilated FGFF (~19.14 M / ~10 G) and **adds** the lightweight modules above. Result: **trainable ≈ 11.0 M (down ~18 M)** and **lower forward FLOPs** than the current MAFFNet. So the synopsis assembly is *lighter*, not heavier, than the stable baseline.

*(Estimates by hand from layer shapes; exact counts require a model build, which was not run per the no-execution constraint.)*

---

## train.py Compatibility (interface only; not wired)

- Constructor matches: `StabilizedMAFFNet(checkpoint=args.checkpoint)` parallels `MAFFNet(checkpoint=args.checkpoint)`.
- Output matches: 5-tuple `(p1,p2,p3,p4,rec)` → `MAFFNetTrainer` and its boundary supervision consume it unchanged.
- **Not done (1-line wiring, deferred):** add `"stabilized_maffnet"` to the `--model` choices and a build branch in `train.py` / `train_all.py`. This is the only remaining integration step; it edits `train.py` (not in the do-not-modify list) but was left out to keep this task to assembly only.

---

## Constraints Honoured
- New model file only; `models/maffnet.py` and `training/trainer.py` **not modified**.
- Reuses DecoderBlock, SAM2 encoder, RDB, and the existing trainer/boundary supervision.
- Spatial resolutions and the output interface preserved.
- No training, smoke test, or benchmark executed (static `py_compile` only).

## Evidence References
- `models/stabilized_maffnet.py` (assembled model)
- `modules/feature_alignment.py`, `modules/frequency_branch.py`, `modules/gated_fusion.py` (reused)
- `models/maffnet.py` (SAM2Encoder, DecoderBlock, RDB — imported, unchanged)
- `SYNOPSIS_IMPLEMENTATION_PLAN.md` (build order this follows)
