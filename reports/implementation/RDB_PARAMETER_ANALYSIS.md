# RDB Parameter Attribution Analysis

**Subject:** RDB in [`models/maffnet_original_reconstructed.py`](../../models/maffnet_original_reconstructed.py).
**Observed:** 11.7995 M trainable. **Paper:** ≈ 7.8 M. **Goal:** attribute the gap, estimate reduction levers, find the smallest equation-faithful change to ≈ 7.8 M.
**Constraint honoured:** analysis only — no code edited, nothing implemented, no training. Counts are exact (`named_parameters()`); candidates are arithmetic.

---

## 1. Layer-by-layer breakdown (exact)

| Layer | Shape | Params | % of RDB |
|---|---|---:|---:|
| up_convs.0 (deconv 1152→1152, dw) | `(1152,1,4,4)`+b | 19,584 | 0.17 % |
| **fuse_convs.0 (Conv3×3 1728→576)** | `(576,1728,3,3)`+b | **8,958,528** | **75.92 %** |
| up_convs.1 (deconv 576→576, dw) | `(576,1,4,4)`+b | 9,792 | 0.08 % |
| fuse_convs.1 (Conv3×3 864→288) | `(288,864,3,3)`+b | 2,239,776 | 18.98 % |
| up_convs.2 (deconv 288→288, dw) | `(288,1,4,4)`+b | 4,896 | 0.04 % |
| fuse_convs.2 (Conv3×3 432→144) | `(144,432,3,3)`+b | 560,016 | 4.75 % |
| to_rgb (deconv 144→3) | `(144,3,4,4)`+b | 6,915 | 0.06 % |
| **RDB total** | | **11,799,507** | 100 % |

---

## 2. Dominant contributor

- **fuse_convs.0 alone = 8.96 M = 75.9 %** of the entire RDB.
- The **three dense Conv3×3 fuses = 11.76 M = 99.65 %**.
- All transposed convs + `to_rgb` together = **41,187 params (0.35 %)** — already depthwise, effectively free.

**Conclusion:** RDB cost ≈ the fuse Conv3×3 layers, overwhelmingly the deepest one. Any parameter reduction must target the fuse convs; the transposes are already minimal.

---

## 3. Reduction-lever estimates

Native widths and `C_out = C_in/2` held fixed (Eq. 12). Fuse stages: `1728→576, 864→288, 432→144`. Transposes + to_rgb = 0.041 M throughout.

| Lever | Mechanism | Effect on RDB total |
|---|---|---|
| **Grouped Conv3×3 (fuse)** | divide fuse params by `g` (where divisible) | **Largest, controllable.** g2-all → 5.92 M; g4-all → 2.98 M; g2-deepest-only → **7.32 M** |
| **Bottleneck projections** | replace dense 3×3 with 1×1-reduce → 3×3 | Moderate-large. Deepest bottleneck (1×1 1728→576, then 3×3 576→576) → 6.82 M. Adds a layer not in Eq. 12 |
| **Reduced fusion widths** | shrink `C_out` below `C_in/2`, or project inputs to non-native widths | Large, but **changes `C_out=C_in/2` and/or native inputs** → contradicts Eq. 12 / §3.3.2 |
| **Grouped transpose variants** | change transpose `groups_i` | **≈ zero savings available** — already depthwise (0.041 M total). Un-grouping only *increases* (e.g. deconv0 groups=1 → 21.2 M) |

The transpose lever is exhausted (already minimal); the fuse-conv group lever is the effective one.

---

## 4. Candidate sweep (arithmetic)

| # | Configuration | RDB total | Δ vs 7.8 M |
|---|---|---:|---:|
| C1 | k3, groups=1 (current) | 11.800 M | +4.0 |
| C2 | k1, groups=1 | 1.349 M | −6.5 |
| C3 | k3, groups=2 (all fuses) | 5.921 M | −1.9 |
| C4 | k3, groups=4 (all fuses) | 2.982 M | −4.8 |
| **C5** | **k3, groups=2 on deepest fuse only** | **7.321 M** | **−0.48** |
| C6 | deepest fuse k1, others k3 | 3.837 M | −4.0 |
| C7 | bottleneck on deepest (1×1→3×3) | 6.823 M | −1.0 |
| C8 | k3, groups=2 on deepest + mid | 6.201 M | −1.6 |

**C5 is the closest to the paper's ≈ 7.8 M and the smallest change** (a single `groups=2` on `fuse_convs.0`). No integer-groups combination lands exactly on 7.8 M; since the paper says "about 7.8 million", **7.32 M is within tolerance**.

---

## 5. Smallest equation-faithful modification → ≈ 7.8 M

**Recommended: C5 — set `groups=2` on the deepest fuse `Conv3×3` (`fuse_convs.0`).**

- One-line change: `nn.Conv2d(1728, 576, 3, padding=1, groups=2)`.
- 11.80 M → **7.32 M** (≈ 7.8 M, within the paper's stated approximation).
- **Preserves every Eq. 12 element:** native input widths, `ConvTranspose(k4,s2,p1)` grouped, `Concat`, `C_out = C_in/2`, `Conv2d → ReLU`, final 3-ch deconv + interpolate.
- Touches only the single layer responsible for 76 % of the cost.

(If a closer match than 7.32 M is required, no clean integer-group option exists between 7.32 M and 11.80 M; reaching exactly 7.8 M would need fractional grouping or a small bottleneck, which the paper does not specify.)

---

## 6. Classification of every candidate

| Candidate | Class | Reason |
|---|---|---|
| C1 (k3, g1, 11.80 M) | **Paper-faithful** | Literal Eq. 12; kernel = 3×3 (default reading); groups=1 implied |
| C2 (k1, g1, 1.35 M) | **Paper-faithful** | Kernel is genuinely unspecified, so k1 is also literal; but far below 7.8 M |
| C3 (k3, g2-all, 5.92 M) | **Paper-compatible** | Grouping not stated but breaks no equation; widths & `C_out=C_in/2` intact |
| C4 (k3, g4-all, 2.98 M) | **Paper-compatible** | same; over-reduced |
| **C5 (k3, g2-deepest, 7.32 M)** | **Paper-compatible** | grouping unstated but Eq. 12-consistent; closest to budget |
| C6 (mixed kernels, 3.84 M) | **Paper-compatible** | kernel unspecified per stage; no eq broken |
| C7 (bottleneck deepest, 6.82 M) | **Paper-compatible** | adds a 1×1 layer not in Eq. 12, but does not contradict it |
| C8 (g2 deepest+mid, 6.20 M) | **Paper-compatible** | grouping; no eq broken |
| — reduce `C_out` below `C_in/2` | **Paper-breaking** | violates explicit `C_out = C_in/2` |
| — project inputs to non-native widths | **Paper-breaking** | inputs are "encoding-stage" native features (§3.3.2) |
| — drop concat / final 3-ch deconv / replace transpose with interpolate | **Paper-breaking** | removes explicit Eq. 12 operations |

**Note on faithfulness boundary:** pure kernel choice (k1↔k3) stays *faithful* but spans 1.35–11.80 M, never 7.8 M. Hitting ≈ 7.8 M requires **grouping the fuse conv** — an unstated-but-compatible detail (the paper itself uses "grouped" convolutions for the transposes). So the closest reproduction of the published budget is **paper-compatible, not strictly paper-faithful**. This is a paper-side under-specification, not an implementation defect.

---

## 7. Recommendation

| | |
|---|---|
| **Root cause** | 76 % of RDB is one dense `Conv3×3(1728→576)`; the paper's fuse kernel/groups are unspecified |
| **Smallest fix to ≈ 7.8 M** | `groups=2` on `fuse_convs.0` → **7.32 M** (C5) |
| **Class** | Paper-compatible (Eq. 12 fully preserved) |
| **If strict faithfulness preferred** | keep C1 (11.8 M) and log the budget gap as paper under-specification |

No code was changed. Decision (apply C5 vs keep C1) is deferred to the next milestone per the no-edit instruction.

---

*Status: RDB parameter analysis / read-only introspection + arithmetic / no code or file changed / existing models unchanged.*
