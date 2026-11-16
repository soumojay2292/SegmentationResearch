# MAFF-Net Reconstruction — Milestone 3 Implementation Report

**File:** [`models/maffnet_original_reconstructed.py`](../../models/maffnet_original_reconstructed.py)
**Blueprint:** `reports/implementation/MAFFNET_FINAL_DESIGN_SPECIFICATION.md` (sole source of truth).
**Scope:** Milestone 3 only — EDB Decoder, Skip Connections, Decoder Feature Pipeline. No Deep Supervision, Segmentation Heads, RDB, Loss, or Training. No training run.

Milestones 1 / 2A / 2B were **not modified**; only the top-level model `__init__`/`forward` were extended to add the EDB decoder and its output key.

---

## 1. Modules implemented

| Module | Design Freeze ref | Class | Frozen config |
|---|---|---|---|
| EDB (×3) | §1.6, §5 / Eqs. 10-11 | `EDB`, `DoubleConv` | decoder width 64; bilinear upsample; `Concat(skip)` → `DoubleConv(in→64→64)` (Conv3×3→BN→ReLU ×2); spatial-attn residual `x+γ·(σ(Conv7×7([avg_c;max_c]))·x)`, `γ` init 0 |

**Decoder feature pipeline** (deepest → finest), consuming FGFF outputs `G1..G4`:
```
D4 = G4                    (bottleneck)            80 × 12×12
D3 = EDB3(G4↑, G3)         skip G3 (80)            64 × 24×24
D2 = EDB2(D3↑, G2)         skip G2 (80)            64 × 48×48
D1 = EDB1(D2↑, G1)         skip G1 (80)            64 × 96×96
decoder_features = [D1, D2, D3, D4]   (finest → coarsest)
```

**Frozen items honoured:** decoder width 64 ✔ · skip widths 80 (G1/G2/G3) ✔ · upsampling bilinear `align_corners=False` ✔ · convolution ordering Conv→BN→ReLU ×2 ✔ · normalization BatchNorm2d ✔ · activation ReLU ✔ · feature fusion concat + spatial-attn residual ✔.

> **Decoder outputs only** — no segmentation maps are produced (deep supervision / seg heads deferred to Milestone 4). `D4` is the un-decoded bottleneck (`G4`, 80-ch) that head `p4` will later read; `D1..D3` are the 64-ch EDB outputs that heads `p1..p3` will read.

---

## 2. Tensor shapes (dummy forward, batch = 2, input `2×3×384×384`)

### Previous outputs (M1 + M2A + M2B) — unchanged
| Output | Shapes | Match |
|---|---|---|
| `encoder_features` | `(2,144,96²) (2,288,48²) (2,576,24²) (2,1152,12²)` | ✔ |
| `mmpa_features` | `(2,144,96²) (2,288,48²) (2,576,24²) (2,1152,12²)` | ✔ |
| `projected_features` | `(2,80,96²) … (2,80,12²)` | ✔ |
| `frequency_features` | `(2,128,96²) … (2,128,12²)` | ✔ |
| `frequency_projected` | `(2,80,96²) … (2,80,12²)` | ✔ |
| `fgff_features` | `(2,80,96²) … (2,80,12²)` | ✔ |

### Milestone 3 output (new)
| Output | Stage | Shape | Expected | Match |
|---|---|---|---|---|
| `decoder_features` | D1 | `(2,64,96,96)` | `B,64,96,96` | ✔ |
| | D2 | `(2,64,48,48)` | `B,64,48,48` | ✔ |
| | D3 | `(2,64,24,24)` | `B,64,24,24` | ✔ |
| | D4 | `(2,80,12,12)` | `B,80,12,12` (bottleneck) | ✔ |

**Shape compliance with Design Freeze: PASS** (all 28 tensors; 24 prior + 4 decoder).

---

## 3. Decoder parameter count

| Component | Trainable |
|---|---:|
| EDB decoder ×3 | **0.3697 M** |

Design-Freeze §6 estimate for EDB was ≈ 0.40 M → measured **0.370 M** (within ≈ 8 %; the small shortfall is fewer BatchNorm affine/bias parameters than the rough estimate). **Verified consistent with the budget.**

---

## 4. Total parameter count

| Group | Trainable |
|---|---:|
| MMPA ×4 (M1) | 1.2525 M |
| Spatial projection ×4 (M1) | 0.1731 M |
| Frequency encoder (M2A) | 1.6690 M |
| Frequency projection ×4 (M2A) | 0.0413 M |
| FGFF ×4 (M2B) | 1.9489 M |
| EDB decoder ×3 (M3) | 0.3697 M |
| Encoder | 0 (frozen) |
| **Total trainable (M1+M2A+M2B+M3)** | **5.4545 M** |
| Total incl. frozen placeholder encoder | 13.3161 M |

All prior-milestone figures are **identical** to their reports — verified modules untouched.

---

## 5. Compile / verification status

| Check | Result |
|---|---|
| `python -m py_compile` | **OK** |
| Dummy forward pass (`2×3×384×384`) | **OK** |
| Print every tensor shape | **OK** (table §2) |
| Decoder parameter count | **OK** (0.3697 M ≈ 0.40 M budget) |
| Total parameter count | **OK** (5.4545 M) |
| Output-dimension match to Design Freeze | **PASS** (28/28 tensors) |

---

## 6. Compliance with Milestone-3 requirements

| Requirement | Status |
|---|---|
| Implement only inside `maffnet_original_reconstructed.py` | ✔ |
| No edit to `maffnet.py` / `stabilized_maffnet.py` / `maffnet_original.py` | ✔ |
| M1 / M2A / M2B verified modules unchanged | ✔ |
| EDB decoder, width 64 | ✔ |
| Skip connections (G3/G2/G1, width 80) | ✔ |
| Upsampling / conv ordering / norm / activation / fusion frozen | ✔ |
| Uses FGFF outputs G1–G4 as decoder inputs | ✔ |
| Outputs decoder features only (no segmentation maps) | ✔ |
| forward adds `decoder_features`; previous outputs unchanged | ✔ |
| No deep supervision / seg heads / RDB / loss / training | ✔ |
| No symbolic widths | ✔ |

---

## 7. Remaining modules (future milestones)

| Milestone | Modules |
|---|---|
| 4 | Deep Supervision + Segmentation Heads (p4←D4, p3←D3, p2←D2, p1←D1; `Conv1×1→↑384²`) |
| 5 | RDB (native widths, train-only) |
| 6 | Loss (Dice `[1,1,1,0.01]` + recon MSE) + full outputs `(p1,p2,p3,p4,rec)` |

---

**Milestone 3 complete and verified. Stopping here — not proceeding to Deep Supervision.**

*Status: Milestone 3 / verified / existing models unchanged / M1+M2A+M2B modules unchanged / no training run.*
