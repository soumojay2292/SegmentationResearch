# MAFF-Net Reconstruction — Milestone 1 Implementation Report

**File created:** [`models/maffnet_original_reconstructed.py`](../../models/maffnet_original_reconstructed.py)
**Blueprint:** `reports/implementation/MAFFNET_FINAL_DESIGN_SPECIFICATION.md` (sole source of truth).
**Scope:** Milestone 1 only — Frozen SAM Encoder, MMPA, Projection Layers. No decoder, no segmentation output, no loss. No training run.

---

## 1. Modules implemented

| # | Module | Design Freeze ref | Class | Frozen config |
|---|---|---|---|---|
| 1 | Frozen SAM Encoder | §1.1 | `FrozenSAMEncoder` (+ `_DummyHieraTrunk` fallback) | out `[144,288,576,1152]` @ strides 4/8/16/32; `requires_grad=False` |
| 2 | MMPA (×4) | §1.2 | `MMPA`, `PerceptionNet`, `WeightGenerator` | proj `C→32`; 3 full-3×3 perceptionNets, dil {3,5,7}; WetGeneration on native `C`; softmax; `Γ∈R³²` init `1e-7`; recon `32→C`, residual |
| 3 | Projection Layers | §1.3 | `SpatialProjection` | `Conv1×1(C_i → 80)` for all 4 stages |

**Deliberately NOT implemented** (later milestones): Frequency Encoder, FGFF, EDB, RDB, Deep Supervision, Segmentation Heads, Loss, Decoder.

`forward()` returns exactly:
```python
{
  "encoder_features":  [X1, X2, X3, X4],
  "mmpa_features":     [S1, S2, S3, S4],
  "projected_features":[Ps1, Ps2, Ps3, Ps4],
}
```
No decoder, no segmentation output, no loss — per scope.

---

## 2. Tensor shapes (dummy forward, batch = 2, input `2×3×384×384`)

| Output | Stage | Shape | Expected (Design Freeze) | Match |
|---|---|---|---|---|
| `encoder_features` | X1 | `(2,144,96,96)` | `B,144,96,96` | ✔ |
| | X2 | `(2,288,48,48)` | `B,288,48,48` | ✔ |
| | X3 | `(2,576,24,24)` | `B,576,24,24` | ✔ |
| | X4 | `(2,1152,12,12)` | `B,1152,12,12` | ✔ |
| `mmpa_features` | S1 | `(2,144,96,96)` | `B,144,96,96` | ✔ |
| | S2 | `(2,288,48,48)` | `B,288,48,48` | ✔ |
| | S3 | `(2,576,24,24)` | `B,576,24,24` | ✔ |
| | S4 | `(2,1152,12,12)` | `B,1152,12,12` | ✔ |
| `projected_features` | Ps1 | `(2,80,96,96)` | `B,80,96,96` | ✔ |
| | Ps2 | `(2,80,48,48)` | `B,80,48,48` | ✔ |
| | Ps3 | `(2,80,24,24)` | `B,80,24,24` | ✔ |
| | Ps4 | `(2,80,12,12)` | `B,80,12,12` | ✔ |

**Shape compliance with Design Freeze: PASS** (all 12 tensors).

---

## 3. Parameter count

| Group | Trainable |
|---|---:|
| MMPA ×4 | 1.2525 M |
| Projection ×4 | 0.1731 M |
| Encoder | 0 (frozen) |
| **Total trainable (Milestone 1)** | **1.4257 M** |
| Total (incl. frozen placeholder encoder) | 9.2872 M |

**MMPA vs paper budget.** Measured MMPA = **1.25 M**; paper states ≈ 1.7 M. This exactly reproduces the value computed in the Dimension Reconstruction (§1.5: "WetGen on native C + full perception = 1.25 M"). The paper's 1.7 M carries a ≈ 0.45 M increment that the article never itemises; it is not recoverable from the text and is logged as a known, paper-driven residual rather than fixed by inflating widths. No deviation from the frozen spec was introduced to chase it.

> Note: the encoder figure (9.29 M total) reflects the lightweight **placeholder** trunk used when the SAM2 checkpoint is absent. With the real frozen SAM2 Hiera-Large trunk the frozen count rises to ≈ 227 M; trainable params are unaffected.

---

## 4. Compile / verification status

| Check | Result |
|---|---|
| `python -m py_compile` | **OK** |
| Dummy forward pass (`2×3×384×384`) | **OK** (no error) |
| Tensor-shape print | **OK** (table §2) |
| Trainable-parameter count | **OK** (1.4257 M) |
| Dimension match to Design Freeze | **PASS** (all 12 tensors) |

---

## 5. Compliance with Milestone-1 requirements

| Requirement | Status |
|---|---|
| New file only `models/maffnet_original_reconstructed.py` | ✔ |
| No edit to `maffnet.py` / `stabilized_maffnet.py` / `maffnet_original.py` | ✔ (untouched) |
| Self-contained (no import from existing models) | ✔ |
| MMPA proj `C→32` | ✔ |
| 3 perception branches, dil {3,5,7}, full 3×3 | ✔ |
| WetGeneration on native encoder channels | ✔ |
| Softmax weighting | ✔ |
| Γ init `1e-7` | ✔ |
| Residual reconstruction back to `C` | ✔ |
| Spatial projection `Conv1×1(C→80)` ×4 | ✔ |
| No symbolic widths (all frozen integers) | ✔ |
| forward returns only the 3-key dict | ✔ |
| No decoder / seg output / loss | ✔ |

---

## 6. Remaining modules (future milestones)

| Milestone | Modules |
|---|---|
| 2 | Frequency Encoder (E=128, 2 blocks/stage, 4 heads, FFN×2) + frequency projection `Conv1×1(128→80)` |
| 3 | FGFF (width 80, dil {1,3,5,7}, freq-attn, β) |
| 4 | EDB decoder (width 64) + Deep Supervision + Segmentation Heads |
| 5 | RDB (native widths, train-only) |
| 6 | Loss (Dice `[1,1,1,0.01]` + recon MSE) + full `forward` outputs `(p1,p2,p3,p4,rec)` |

---

**Milestone 1 complete and verified. Stopping here — not proceeding to Milestone 2.**

*Status: Milestone 1 / verified / existing models unchanged / no training run.*
