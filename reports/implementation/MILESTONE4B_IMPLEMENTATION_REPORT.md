# MAFF-Net Reconstruction — Milestone 4B Implementation Report

**File:** [`models/maffnet_original_reconstructed.py`](../../models/maffnet_original_reconstructed.py)
**Blueprint:** `MAFFNET_FINAL_DESIGN_SPECIFICATION.md` + `RDB_DESIGN_SPECIFICATION.md` (sources of truth).
**Scope:** Milestone 4B only — Reconstruction Decoder Branch (RDB). No deep supervision, seg heads, loss, training, or final integration. No training run; no loss computed.

Milestones 1 / 2A / 2B / 3 were **not modified**; only the model `__init__`/`forward` were extended to add the RDB and its output key.

---

## 1. Modules implemented

| Module | Spec ref | Class | Frozen config |
|---|---|---|---|
| RDB | RDB spec §7 / Eq. 12 / Freeze §1.7 | `RDB` | inputs = 4 post-MMPA spatial features (native, deep→shallow); per stage depthwise `ConvTranspose2d(C→C,k4,s2,p1,groups=C)` → `Concat(skip)` → `Conv3×3((C+C_skip)→C//2)` → `ReLU`; final `ConvTranspose2d(144→3,k4,s2,p1)` → interpolate to 384²; no normalization |

**Pipeline (as built):**
```
post-MMPA spatial (deep→shallow): s1=1152@12² s2=576@24² s3=288@48² s4=144@96²
  s1 ─ deconv(1152→1152, dw) →24² ─ cat s2(576) →1728 ─ Conv3×3→576 →ReLU   [w1: 576@24²]
  w1 ─ deconv(576→576,  dw) →48² ─ cat s3(288) →864  ─ Conv3×3→288 →ReLU    [w2: 288@48²]
  w2 ─ deconv(288→288,  dw) →96² ─ cat s4(144) →432  ─ Conv3×3→144 →ReLU    [w3: 144@96²]
  w3 ─ deconv(144→3) →192² ─ interpolate →384²                              [rec: 3@384²]
```

**Output:** `reconstruction = f_rec ∈ B×3×384×384`. No loss is computed (returns the tensor only).

---

## 2. Reconstruction tensor + shapes (dummy forward, batch = 2)

| Output | Shape | Expected | Match |
|---|---|---|---|
| `reconstruction` | `(2,3,384,384)` | `B,3,384,384` | ✔ |

All previous outputs (encoder / mmpa / projected / frequency / frequency_projected / fgff / decoder — 28 tensors) **unchanged**.

**Shape compliance with Design Freeze: PASS** (29/29 tensors incl. reconstruction).

---

## 3. RDB parameter count

| Component | Trainable |
|---|---:|
| RDB | **11.7995 M** |

RDB-spec §5 / §7.3 estimated the faithful native + 3×3-fuse RDB at **≈ 11.8 M** → measured **11.7995 M**, an **exact match to the specification's k3 estimate**.

**vs paper target (7.8 M):** the paper reports ≈ 7.8 M. The RDB-spec §5/§6 already flagged this: matching 7.8 M requires the **unspecified `groups_i`** of the transposed convs and/or a lighter fuse kernel — neither derivable from the paper. The implemented module is the **structurally faithful** native-width, depthwise-transpose, 3×3-fuse form (11.8 M). Closing to 7.8 M is a tuning step deferred until the authors' grouping is known; it is logged, not silently forced.

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
| RDB (M4B) | 11.7995 M |
| Encoder | 0 (frozen) |
| **Total trainable (M1+M2A+M2B+M3+M4B)** | **17.2541 M** |
| Total incl. frozen placeholder encoder | 25.1156 M |

> **Budget note.** The Design-Freeze projected ≈ 13.7 M assuming RDB ≈ 7.8 M. The faithful RDB measured 11.8 M (k3 native), so the running total is 17.25 M. The entire ≈ 3.5 M overshoot is the RDB groups/kernel ambiguity (RDB-spec §5). All other modules match their budgets. Prior-milestone figures are identical to their reports — verified modules untouched.

---

## 5. Compile / verification status

| Check | Result |
|---|---|
| `python -m py_compile` | **OK** |
| Dummy forward pass (`2×3×384×384`) | **OK** |
| Print every tensor shape | **OK** |
| RDB parameter count | **OK** (11.7995 M = spec k3 estimate) |
| Total trainable parameter count | **OK** (17.2541 M) |
| Reconstruction output size | **PASS** (`2×3×384×384`) |

---

## 6. Compliance with Milestone-4B requirements

| Requirement | Status |
|---|---|
| Implement only inside `maffnet_original_reconstructed.py` | ✔ |
| No edit to other models / verified modules | ✔ |
| Inputs = 4 post-MMPA spatial features, deep→shallow | ✔ |
| Pipeline ConvTranspose → Concat → Conv3×3 → ReLU per stage | ✔ |
| Final ConvTranspose 144→3 → Interpolate → 384² | ✔ |
| Output `reconstruction` `B×3×384×384` | ✔ |
| No loss computed (tensor only) | ✔ |
| forward adds `reconstruction`; previous outputs unchanged | ✔ |
| No deep supervision / seg heads / loss / training / integration | ✔ |
| No symbolic widths | ✔ |

---

## 7. Remaining modules (future milestones)

| Milestone | Modules |
|---|---|
| 5 | Deep Supervision + Segmentation Heads (p4←D4, p3←D3, p2←D2, p1←D1; `Conv1×1→↑384²`) |
| 6 | Loss (Dice `[1,1,1,0.01]` + `MSE(rec, x)`) + final integration → outputs `(p1,p2,p3,p4,rec)`; RDB train-only gating |

---

**Milestone 4B complete and verified. Stopping here — not proceeding to Deep Supervision.**

*Status: Milestone 4B / verified / existing models unchanged / M1+M2A+M2B+M3 modules unchanged / no loss, no training run.*
