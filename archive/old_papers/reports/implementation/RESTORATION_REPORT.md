# MAFFNet Restoration Report

**Phase:** restoration only (faithful to original MAFF-Net). **Not** the research-contribution phase.
**Constraint honoured:** no redesign, no new modules, training pipeline touched only where restoration required it.
**Result:** all 5 tasks done; compile + smoke + shape + param checks pass.

---

## 1. Tasks Completed

| # | Task | Status | What was done |
|---|------|:------:|---------------|
| 1 | Restore **Dynamic MMPA** | ✅ | Replaced static SE-MMPA with the paper's adaptor: LN → project C→32 → 3 depthwise dilated experts (d=3,5,7) → WetGeneration (GAP→FC32→16→3→softmax dynamic weights) → Γ-scaled (1e-7) bottleneck residual → project 32→C → adaptor residual |
| 2 | Restore **FGFF cross-modal** | ✅ | Replaced concat-fusion with eq.9: spatial branch (1×1 → 4 dilated convs d=1,3,5,7 → 3×3) + frequency spatial-attention `σ(Conv3×3([Avg;Max](mag)))` + cross-modal `β ⊙ (freq_attn ⊗ spatial_fused)` + residual |
| 3 | Connect **RDB reconstruction** to loss | ✅ | `MAFFNetLoss` now takes the input image and adds `0.01 · MSE((rec+1)/2, image)`; trainer passes `image=imgs` at both call sites. Previously `rec` was discarded |
| 4 | Remove **decoder zero-skip** | ✅ | Deleted `torch.zeros_like` skip; finest level is now a skip-less refine (`Upsample → DoubleConv`) |
| 5 | Verify **ImageNet/SAM normalization** | ✅ (was missing → added) | SAM2 `image_encoder` is called directly, bypassing SAM2's internal `forward_image` normalization → input was raw [0,1]. Added ImageNet/SAM buffers `mean=[.485,.456,.406] std=[.229,.224,.225]`, applied to backbone input in `forward` |

---

## 2. Files Modified

| File | Change |
|------|--------|
| `models/maffnet.py` | MMPA class (dynamic adaptor); FGFF class (cross-modal); `dec1` skip-less; norm buffers + in-forward normalization |
| `training/trainer.py` | `MAFFNetLoss.forward(+image)` adds reconstruction MSE; both criterion calls pass `image=imgs` |

**Not modified:** `train.py`, `train_all.py` (build `MAFFNet(checkpoint=...)` — unchanged API), `utils/report_generator.py`, `utils/benchmark_report.py`, `utils/aggregate_benchmark.py`, `models/sam2_loader.py`, other models, dataloaders.

---

## 3. Verification

| Check | Result |
|-------|--------|
| `py_compile` maffnet.py + trainer.py | ✅ OK |
| Zero-skip removed | ✅ `NO_ZERO_SKIP` (grep) |
| MMPA dynamic (γ, WetGeneration, proj_back) | ✅ present |
| FGFF cross-modal (β, freq_attn) | ✅ present |
| Normalization buffers + applied | ✅ `img_mean/img_std`, `x_norm` to backbone |
| Build `MAFFNet(checkpoint=...)` | ✅ SAM2 loaded |
| **Output shapes** | ✅ p1–p4 `(1,1,384,384)`, rec `(1,3,384,384)` |
| **Loss finite + recon active** | ✅ ep0 `seg=1.546, rec=0.087`; ep11 finite (boundary+recon) |
| **Parameters** | total **241.7M**, trainable **28.98M (12.0%)**, frozen 212.7M |
| Compatibility (trainer/report/benchmark/aggregate) | ✅ 5-tuple output unchanged; CSV schema unchanged (`rec_loss` folded into total, not a new column) |

---

## 4. Faithfulness Notes & Scope Boundaries (honest)

**Restored faithfully (mechanism-level):** dynamic MMPA adaptor, cross-modal FGFF, RDB reconstruction MSE — these now match the paper's design intent.

**Known deviations still present (outside the 5 restoration tasks — flagged, not fixed here):**
1. **Trainable params 28.98M (12.0%) vs paper 14.8M (6.5%).** Cause: we tap the **SAM2 FPN neck (256-ch uniform)** instead of raw Hiera channels `[144,288,576,1152]`, and the **decoder is full-width (no channel reduction)** + heavy RDB. The *modules* are faithful; the *param budget* is not. (R4 EDB attention and channel-reduction were **not** in this task's list.)
2. **Loss is currently Dice + BCE + boundary + recon-MSE**, whereas the paper is **Dice + recon-MSE only**. Task 3 only asked to *add* the recon term; BCE and the Sobel boundary loss (a prior research add) were **not removed** because that was not in the 5 tasks. → To be reconciled in the contribution phase (boundary loss is the planned contribution; BCE should be ablated or dropped for paper-faithful comparison).
3. **Frequency stream (R3)** and **EDB channel-attention skip (R4)** were **not** restored — they are P1 in the roadmap, not part of this task's 5 items. FGFF still derives frequency from the feature-map FFT (not the paper's separate grayscale-image freq-transformer stream).
4. **Training budget** unchanged (batch 2 / 20–40 ep) ≠ paper's batch 12 / 100 ep — a comparability issue to fix before reporting paper-vs-ours numbers.

**MMPA param note:** our dynamic MMPA is smaller per-module than the paper's 1.7M because our C=256 (FPN) vs the paper's up-to-1152 raw channels; mechanism identical.

---

## 5. Not Done (by instruction — no research improvements)

- No boundary-loss changes, no edge-weighted reconstruction, no new modules.
- No EDB attention (R4), no frequency-transformer stream (R3).
- No training-budget change, no normalization of the reconstruction target beyond [0,1] mapping.

These belong to the **contribution phase** (`FINAL_MAFFNET_ARCHITECTURE.md` STEP 2).

---

## 6. Recommended Next Step (not executed)

Run one real epoch to confirm the restored model trains end-to-end and `rec_loss` decreases:
```
python train.py --model maffnet --dataset ISIC_2016 --epochs 1 --exp maffnet_restore_smoke --checkpoint checkpoints/sam2_hiera_large.pt
```
(Note: MAFFNet uses `MAFFNetTrainer`, which does not emit `metrics.json`; that is pre-existing and unrelated to this restoration.)

---

## Evidence References
- `models/maffnet.py` — MMPA (dynamic adaptor), FGFF (cross-modal), `dec1_up/dec1_conv` (skip-less), `img_mean/img_std` + `x_norm`
- `training/trainer.py` — `MAFFNetLoss.forward(image=...)` recon MSE; `criterion(..., image=imgs)` ×2
- smoke: total 241.7M / trainable 28.98M (12%); shapes p1–p4 (1,1,384,384) + rec (1,3,384,384); loss finite, rec_loss=0.087
- `FINAL_MAFFNET_ARCHITECTURE.md` (roadmap), `MAFFNET_DESIGN.md`, original paper §3.1–3.4
