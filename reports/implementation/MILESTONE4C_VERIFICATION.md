# MAFF-Net Reconstruction — Milestone 4C Verification (RDB Candidate C5)

**File:** [`models/maffnet_original_reconstructed.py`](../../models/maffnet_original_reconstructed.py)
**Change applied:** Candidate **C5** from `RDB_PARAMETER_ANALYSIS.md` — deepest fuse convolution `Conv3×3(1728→576)` groups `1 → 2`. **No other layer modified.** No training run.

---

## 1. Modification (single layer)

| Layer | Before | After |
|---|---|---|
| `rdb.fuse_convs.0.0` (deepest fuse) | `Conv2d(1728, 576, k3, groups=1)` | `Conv2d(1728, 576, k3, **groups=2**)` |
| `rdb.fuse_convs.1.0` | `groups=1` | `groups=1` (unchanged) |
| `rdb.fuse_convs.2.0` | `groups=1` | `groups=1` (unchanged) |
| all transposes, `to_rgb`, all other modules | — | unchanged |

Verified group settings post-edit: `fuse_convs.0.0.groups = 2` (weight `(576, 864, 3, 3)`), `fuse_convs.1.0.groups = 1`, `fuse_convs.2.0.groups = 1`.

**Unchanged (confirmed):** tensor shapes · outputs · topology · channel widths · reconstruction pipeline.

---

## 2. Verification results

| Check | Result |
|---|---|
| `python -m py_compile` | **OK** |
| Dummy forward (`2×3×384×384`) | **OK** |
| Tensor verification (all 8 output keys) | **PASS** (29/29 shapes) |
| Reconstruction output | `(2,3,384,384)` ✔ |
| Shape compliance with Design Freeze | **PASS** |

All prior outputs (`encoder_features`, `mmpa_features`, `projected_features`, `frequency_features`, `frequency_projected`, `fgff_features`, `decoder_features`) — shapes unchanged.

---

## 3. Parameter count

### RDB
| | Params | M |
|---|---:|---:|
| **Previous count** (groups=1) | 11,799,507 | 11.7995 |
| **New count** (C5, groups=2 deepest) | 7,320,531 | 7.3205 |
| **Difference** | −4,478,976 | −4.4790 (−38.0 %) |
| **Paper target** | ≈ 7,800,000 | ≈ 7.8 |
| **New vs target** | −479,469 | −0.48 (93.8 % of target) |

### Total trainable
| | M |
|---|---:|
| Previous total | 17.2541 |
| **New total** | **12.7751** |
| Difference | −4.4790 |

(Other modules unchanged: MMPA 1.2525 · spatial proj 0.1731 · freq encoder 1.6690 · freq proj 0.0413 · FGFF 1.9489 · EDB 0.3697.)

---

## 4. Compliance

| Item | Status |
|---|---|
| Only deepest fuse modified | ✔ |
| groups 1 → 2 on `fuse_convs.0` | ✔ |
| No other layer changed | ✔ |
| Tensor shapes unchanged | ✔ |
| Outputs unchanged (8 keys) | ✔ |
| Topology unchanged | ✔ |
| Channel widths unchanged | ✔ |
| Reconstruction pipeline unchanged | ✔ |
| Eq. 12 preserved (native widths, `C_out=C_in/2`, k3, grouped transpose, concat, ReLU, final 3-ch deconv + interp) | ✔ |
| Classification | **Paper-compatible** (grouping unstated, no equation broken) |
| RDB vs paper budget | 7.32 M vs ≈ 7.8 M → within the paper's "about 7.8 M" tolerance (−6.2 %) |

> No clean integer-group option lands closer than 7.32 M without a bottleneck or fractional grouping; C5 is the closest single-knob, equation-preserving match.

**Verified modules untouched:** `maffnet.py`, `stabilized_maffnet.py`, `maffnet_original.py`.

---

## 5. Remaining modules (future milestones)

| Milestone | Modules |
|---|---|
| 5 | Deep Supervision + Segmentation Heads (p4←D4, p3←D3, p2←D2, p1←D1; `Conv1×1→↑384²`) |
| 6 | Loss (Dice `[1,1,1,0.01]` + `MSE(rec,x)`) + integration → `(p1,p2,p3,p4,rec)`; RDB train-only gating |

---

**Milestone 4C complete and verified. Stopping after verification.**

*Status: Milestone 4C / C5 applied / verified / one-layer change / existing models unchanged / no training run.*
