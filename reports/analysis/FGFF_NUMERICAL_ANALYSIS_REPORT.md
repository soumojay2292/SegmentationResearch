# FGFF Numerical Analysis Report

**Task:** verify the hypothesis that the non-finite gradient originates in the FGFF frequency branch (`torch.abs(rfft2(...))`).
**Method:** runtime instrumentation only (no repository edits). FFT-magnitude statistics, `torch.autograd.detect_anomaly` to name the exact backward operation, and a runtime `abs → sqrt(re²+im²+1e-8)` patch as a falsification test. bf16 pipeline, real ISIC-2016 batch.

---

## 1. Executive Summary

**The FGFF hypothesis is REJECTED.** The FFT magnitude is numerically clean (no near-zero coefficients), and patching `abs(rfft)` has **no effect** on the divergence. Anomaly detection identifies the actual operation: **`SqrtBackward0` in `BoundaryLoss._edges` at `training/trainer.py:62`** — the Sobel magnitude `(gx²+gy²).sqrt()`. Its gradient is `1/(2·sqrt(gx²+gy²))`, which is **infinite where the field is locally flat (gx=gy=0)**. This is invoked by **Boundary-aware Deep Supervision (BDS)** on the prediction (`_edge_mag(prob)`, line 168). Even though BDS is warmup-weighted to **zero** at epoch 0, the singular local Jacobian (`inf`) multiplied by the zero upstream gradient yields `0 · inf = NaN`, which propagates globally through the backward — corrupting every trainable parameter and surfacing first (by name order) as `mmpa1.gamma`.

The root cause is the **un-stabilized Sobel `sqrt`** in our BDS/EWR edge operator — not FGFF, not MMPA, not the original MAFF-Net.

---

## 2. Evidence

### (a) FFT magnitude statistics — `|rfft2|` per FGFF level (clean)
| Level | shape | min | max | mean | std | `\|z\|==0` | `\|z\|<1e-8` | `\|z\|<1e-10` | finite |
|-------|-------|----:|----:|-----:|----:|------:|--------:|---------:|:------:|
| fgff1 | (2,256,96,49) | 2.12e-05 | 55.70 | 4.54e-02 | 0.178 | 0 | 0 | 0 | ✅ |
| fgff2 | (2,256,48,25) | 4.35e-05 | 17.12 | 1.13e-01 | 0.167 | 0 | 0 | 0 | ✅ |
| fgff3 | (2,256,24,13) | 2.00e-04 | 124.9 | 5.39e-01 | 0.988 | 0 | 0 | 0 | ✅ |
| fgff4 | (2,256,12,7) | 1.38e-04 | 61.89 | 4.57e-01 | 0.894 | 0 | 0 | 0 | ✅ |

No zero-magnitude coefficients, none below 1e-8 or 1e-10; minimum magnitude ~2e-5. The `abs(rfft)` gradient `z̄/|z|` is therefore bounded (magnitude ≈ 1) everywhere here — **no singularity in FGFF**.

### (b) Anomaly detection — exact operation
```
Function 'SqrtBackward0' returned nan values in its 0th output.
Traceback of forward call that caused the error:
  training/trainer.py:168  acc = acc + wi * F.l1_loss(self._edge_mag(prob), sy)
  training/trainer.py:120  return self.boundary._edges(x)
  training/trainer.py:62   return (gx ** 2 + gy ** 2).sqrt().clamp(0, 1)
```
The first non-finite backward operation is the **`sqrt` in `BoundaryLoss._edges`** (line 62), reached through **BDS** `_edge_mag(prob)` (line 168) — *not* the FGFF `abs`.

### (c) Falsification test — FGFF `abs → sqrt(re²+im²+1e-8)`
| Variant | step 1 | step 2 |
|---------|--------|--------|
| Original FGFF (`abs`) | loss 1.5880, grads non-finite, optim updated | **NaN**, optim halted |
| Patched FGFF (`sqrt(re²+im²+1e-8)`) | loss 1.5880, grads non-finite, optim updated | **NaN**, optim halted |

**Identical.** Stabilizing the FGFF magnitude does nothing → FGFF is not the cause. (Note step-1 `grads_finite=False` in both: the non-finite grad is present already at step 1, consistent with the `SqrtBackward0` source firing on the first backward.)

---

## 3. Verdict

**Hypothesis (FGFF / `abs(rfft)` responsible): REJECTED.**

True root cause: the **Sobel magnitude `(gx²+gy²).sqrt()` in `BoundaryLoss._edges` (`trainer.py:62`)** has an infinite gradient at flat regions; called by **BDS** on the prediction, the `0·inf = NaN` interaction (BDS warmup weight is 0 at epoch 0, but the singular Jacobian still poisons the graph) produces a global non-finite gradient on the first backward.

---

## 4. Confidence Level

**≈ 98%.** Three independent lines converge: (i) anomaly detection names the exact op and file:line (`SqrtBackward0`, `trainer.py:62`); (ii) FFT statistics show no near-zero coefficients, so the `abs(rfft)` gradient is bounded; (iii) the proposed FGFF patch has literally zero effect on the failure. Residual 2% allows for a secondary contributor unmasked only after the Sobel `sqrt` is fixed.

---

## 5. Recommended Permanent Fix

The FGFF hypothesis is rejected, so **no FGFF change is recommended.** The actual fix targets the Sobel edge operator (not implemented here — diagnosis only):

1. **Primary — epsilon-stabilize the Sobel magnitude** in `BoundaryLoss._edges` (`trainer.py:62`):
   `(gx**2 + gy**2).sqrt()` → `(gx**2 + gy**2 + 1e-6).sqrt()`.
   This bounds the gradient at flat regions (gx=gy=0) and removes the `sqrt(0)` singularity. It fixes both BDS (edge-of-prediction) and any EWR boundary-map use of the same operator.
2. **Secondary (defensive)** — when the BDS warmup weight is 0, skip the BDS term entirely (do not build its graph) so a zeroed term cannot inject `0·inf = NaN`. The epsilon fix (1) is the root fix; this is belt-and-suspenders.

After applying (1), re-run the one-batch overfit and require iteration 2 finite, gradients finite, optimizer updating, and Dice → > 0.95.

---

## Codebase Status
- `models/maffnet.py`, `training/trainer.py` — **unchanged**. All instrumentation and patches were runtime-only and restored. No fix applied (per rules: the `abs(rfft)` patch was used only as a falsification test, and the evidence rejected it).

## Evidence References
- FFT stats (4 FGFF levels), anomaly `SqrtBackward0 @ trainer.py:62` (via `_edge_mag(prob)` line 168), orig-vs-patched overfit (identical NaN)
- `training/trainer.py:62` `BoundaryLoss._edges`; `:120` `_edge_mag`; `:168` BDS edge term
- supersedes the FGFF suspicion in `MMPA_COMPONENT_ABLATION_REPORT.md`
