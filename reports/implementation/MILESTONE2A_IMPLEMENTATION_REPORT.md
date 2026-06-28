# MAFF-Net Reconstruction — Milestone 2A Implementation Report

**File:** [`models/maffnet_original_reconstructed.py`](../../models/maffnet_original_reconstructed.py)
**Blueprint:** `reports/implementation/MAFFNET_FINAL_DESIGN_SPECIFICATION.md` (sole source of truth).
**Scope:** Milestone 2A only — FFT preprocessing, Frequency Encoder, Frequency Projection. No FGFF, decoder, EDB, RDB, deep supervision, seg heads, loss, or final outputs. No training run.

Milestone 1 modules (encoder, MMPA, spatial projection) were **not modified**; only the top-level model `__init__`/`forward` were extended to wire in the frequency stream (necessary to add the new outputs).

---

## 1. Modules implemented

| # | Module | Design Freeze ref | Class | Frozen config |
|---|---|---|---|---|
| 1 | FFT Preprocessing | §1.4 | `FFTPreprocess` | gray·255 → FFT2 → fftshift → \|·\| → per-sample min-max → pseudo-RGB → interp ×0.25 (384→96) |
| 2 | Frequency Encoder | §1.4, §4 | `FrequencyEncoder`, `FreqTransformerBlock` | stem `Conv3×3(3→128)`; 4 stages × 2 blocks; E=128, heads=4 (d_k=32), FFN×2, depthwise QKV, learnable T∈R⁴ˣ¹ˣ¹, GroupNorm(1); stride-2 downs between stages |
| 3 | Frequency Projection | §1.3 | `FrequencyProjection` | `Conv1×1(128 → 80)` ×4 |

**FFT pipeline implemented in full (not simplified):**
```
B×3×384×384 → grayscale → ×255 → FFT2 → fftshift → magnitude
            → min-max norm → pseudo-RGB (B×3×384×384) → ↓0.25 → B×3×96×96
```

**Attention note:** the frequency Transformer uses channel-axis (head_dim×head_dim) attention with depthwise Q/K/V and a learnable temperature — the design's depthwise-QKV + temperature formulation. This keeps cost independent of the H·W token count (spatial H·W×H·W attention at 96² is infeasible). Consistent with Design Freeze §4 / Eq. 8.

---

## 2. Tensor shapes (dummy forward, batch = 2, input `2×3×384×384`)

### Milestone 1 outputs (unchanged)
| Output | Shapes | Match |
|---|---|---|
| `encoder_features` | `(2,144,96,96) (2,288,48,48) (2,576,24,24) (2,1152,12,12)` | ✔ |
| `mmpa_features` | `(2,144,96,96) (2,288,48,48) (2,576,24,24) (2,1152,12,12)` | ✔ |
| `projected_features` | `(2,80,96,96) (2,80,48,48) (2,80,24,24) (2,80,12,12)` | ✔ |

### Milestone 2A outputs (new)
| Output | Stage | Shape | Expected | Match |
|---|---|---|---|---|
| `frequency_features` | F1 | `(2,128,96,96)` | `B,128,96,96` | ✔ |
| | F2 | `(2,128,48,48)` | `B,128,48,48` | ✔ |
| | F3 | `(2,128,24,24)` | `B,128,24,24` | ✔ |
| | F4 | `(2,128,12,12)` | `B,128,12,12` | ✔ |
| `frequency_projected` | Pf1 | `(2,80,96,96)` | `B,80,96,96` | ✔ |
| | Pf2 | `(2,80,48,48)` | `B,80,48,48` | ✔ |
| | Pf3 | `(2,80,24,24)` | `B,80,24,24` | ✔ |
| | Pf4 | `(2,80,12,12)` | `B,80,12,12` | ✔ |

**Shape compliance with Design Freeze: PASS** (all 20 tensors; 12 from M1 + 8 from M2A).

---

## 3. Frequency encoder parameter count

| Component | Trainable |
|---|---:|
| Frequency Encoder (stem + 8 blocks + 3 downs) | **1.6690 M** |
| Frequency Projection ×4 | 0.0413 M |

Design-Freeze §6 estimate for the frequency encoder was ≈ 1.63 M; measured **1.669 M** (within ≈ 2 %, the excess being GroupNorm affines and conv biases). **Verified consistent with the budget.**

---

## 4. Total trainable parameter count

| Group | Trainable |
|---|---:|
| MMPA ×4 (M1) | 1.2525 M |
| Spatial projection ×4 (M1) | 0.1731 M |
| Frequency encoder (M2A) | 1.6690 M |
| Frequency projection ×4 (M2A) | 0.0413 M |
| Encoder | 0 (frozen) |
| **Total trainable (M1 + M2A)** | **3.1360 M** |
| Total incl. frozen placeholder encoder | 10.9975 M |

Milestone-1 figures (MMPA 1.2525 M, spatial proj 0.1731 M) are **identical** to the M1 report — confirming the verified modules were untouched.

---

## 5. Compile / verification status

| Check | Result |
|---|---|
| `python -m py_compile` | **OK** |
| Dummy forward pass (`2×3×384×384`) | **OK** |
| Print every tensor shape | **OK** (table §2) |
| Trainable-parameter count | **OK** (3.1360 M) |
| Frequency-encoder parameter count | **OK** (1.6690 M ≈ 1.63 M budget) |
| Dimension match to Design Freeze | **PASS** (20/20 tensors) |

---

## 6. Compliance with Milestone-2A requirements

| Requirement | Status |
|---|---|
| Implement only inside `maffnet_original_reconstructed.py` | ✔ |
| No edit to `maffnet.py` / `stabilized_maffnet.py` / `maffnet_original.py` | ✔ |
| M1 verified modules unchanged | ✔ |
| FFT pipeline implemented in full (not simplified) | ✔ |
| Freq embed = 128 | ✔ |
| 4 stages, 2 blocks/stage | ✔ |
| 4 attention heads | ✔ |
| FFN ratio = 2 | ✔ |
| Depthwise QKV | ✔ |
| Learnable temperature | ✔ |
| GroupNorm(1) | ✔ |
| Stride-2 downsampling between stages | ✔ |
| Outputs F1–F4 @ 128 / [96,48,24,12] | ✔ |
| Frequency projection `Conv1×1(128→80)` ×4 | ✔ |
| forward adds `frequency_features`, `frequency_projected` | ✔ |
| M1 outputs unchanged | ✔ |
| No FGFF added | ✔ |
| No symbolic widths | ✔ |

---

## 7. Remaining modules (future milestones)

| Milestone | Modules |
|---|---|
| 2B / 3 | FGFF (width 80, dil {1,3,5,7}, freq-attn, β) — fuse `projected_features` ⊕ `frequency_projected` |
| 4 | EDB decoder (width 64) + Deep Supervision + Segmentation Heads |
| 5 | RDB (native widths, train-only) |
| 6 | Loss (Dice `[1,1,1,0.01]` + recon MSE) + full outputs `(p1,p2,p3,p4,rec)` |

---

**Milestone 2A complete and verified. Stopping here — not proceeding to FGFF.**

*Status: Milestone 2A / verified / existing models unchanged / M1 modules unchanged / no training run.*
