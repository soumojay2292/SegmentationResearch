# Non-Finite Trace Report

**Trigger:** `RuntimeError: Non-finite loss detected at epoch 0` (the new MAFFNetTrainer guard).
**Mode:** diagnosis only. No code modified, nothing fixed. Instrumented the real bf16 pipeline (gamma init 1e-2, gamma excluded from weight decay, no GradScaler, clip 1.0) on real ISIC-2016 batches, train mode.

---

## Executive Finding

The first non-finite value is **not in the forward pass and not in any loss term**. The forward of the first iteration is **entirely finite**, including all loss components. The first NaN/Inf is a **backward-pass event**: the **gradient accumulated into `mmpa1.gamma`** (the MMPA Γ scaling vector) is non-finite on the very first `backward()`. The optimizer then writes that non-finite gradient into `mmpa1.gamma`, so from the second iteration the MMPA forward emits NaN and the loss becomes non-finite, firing the guard.

> **First operation producing a non-finite value:** the **backward of `p_enhanced = z + gamma * e_fused` in MMPA**, i.e. the gradient w.r.t. the `gamma` parameter — *not* any forward stage.

---

## Stage-by-Stage Forward Trace (iteration 1, clean weights) — ALL FINITE

| # | Stage | Shape | min | max | mean | std | finite |
|---|-------|-------|----:|----:|-----:|----:|:------:|
| 1 | input_image | (2,3,384,384) | 0.043 | 1.0 | 0.617 | 0.221 | ✅ |
| 2 | input_mask | (2,1,384,384) | 0 | 1 | 0.267 | 0.442 | ✅ (binary {0,1}) |
| 3 | SAM encoder f1 | (2,256,96,96) | −0.547 | 0.711 | −0.003 | 0.045 | ✅ |
| 4 | MMPA1 | (2,256,96,96) | −0.680 | 1.219 | 0.015 | 0.142 | ✅ |
| 4 | MMPA2 | (2,256,48,48) | −1.383 | 1.617 | 0.021 | 0.171 | ✅ |
| 4 | MMPA3 | (2,256,24,24) | −6.504 | 8.186 | 0.041 | 0.926 | ✅ |
| 4 | MMPA4 | (2,256,12,12) | −3.270 | 7.366 | 0.031 | 0.820 | ✅ |
| 5 | FGFF1 | (2,256,96,96) | −0.656 | 4.502 | 0.218 | 0.330 | ✅ |
| 5 | FGFF2 | (2,256,48,48) | −1.383 | 3.160 | 0.215 | 0.333 | ✅ |
| 5 | FGFF3 | (2,256,24,24) | −6.249 | 8.296 | 0.215 | 0.961 | ✅ |
| 5 | FGFF4 | (2,256,12,12) | −3.241 | 7.584 | 0.239 | 0.877 | ✅ |
| 6 | dec4 | (2,256,24,24) | 0 | 4.656 | 0.399 | 0.582 | ✅ |
| 6 | dec3 | (2,256,48,48) | 0 | 5.438 | 0.398 | 0.586 | ✅ |
| 6 | dec2 | (2,256,96,96) | 0 | 6.438 | 0.397 | 0.585 | ✅ |
| 6 | dec1_conv | (2,256,192,192) | 0 | 6.312 | 0.395 | 0.590 | ✅ |
| 7 | logit p1 | (2,1,384,384) | −1.344 | 1.332 | 0.067 | 0.317 | ✅ |
| 7 | logits p2–p4 | — | finite | | | | ✅ |
| 7 | rec (RDB) | (2,3,384,384) | −0.895 | 0.950 | 0.097 | 0.303 | ✅ |
| 8 | sigmoid(p1) | (2,1,384,384) | 0.207 | 0.791 | 0.516 | 0.077 | ✅ |

### Loss terms (iteration 1) — ALL FINITE
| Term | Value | finite |
|------|------:|:------:|
| 9 Dice (per head) | 0.687–0.707 | ✅ |
| 10 BCE (per head) | 0.883–0.935 | ✅ |
| 11 Boundary (BDS raw) | 0.113 | ✅ |
| EWR (edge-weighted recon) | 0.087 | ✅ |
| 12 Total seg | 1.605 | ✅ |

**Conclusion of forward trace:** the network and loss are numerically clean on a fresh model. The guard does not fire on iteration 1.

---

## Training-Loop Trace (where it actually breaks)

| Step | loss | loss finite | param non-finite (pre-step) | first non-finite grad (post-backward) |
|------|-----:|:-----------:|-----------------------------|---------------------------------------|
| 1 | 1.5880 | ✅ | none | **`mmpa1.gamma`** |
| 2 | NaN | ❌ | **`mmpa1.gamma`** | — (guard fires) |

### Stage trace on the failing step (step 2)
| Stage | finite |
|-------|:------:|
| backbone (frozen SAM2) | ✅ |
| **mmpa1** | ❌ **first corrupted forward stage** |
| mmpa2–4, fgff1–4, dec4/3/2, dec1_conv, head1, rdb | ❌ (all downstream of mmpa1) |
| loss sub | `{seg: nan, bds: nan, ewr: nan, rec: nan}` |

---

## Exact Causal Chain

1. **Iteration 1 forward:** finite everywhere (table above).
2. **Iteration 1 backward:** `∂L/∂(mmpa1.gamma)` is **non-finite**. This is the **first** non-finite value in the entire pipeline. It arises in the backward of the MMPA residual `p_enhanced = z + gamma * e_fused` (the gradient toward the broadcast scaling vector `gamma`, reduced over batch and spatial dimensions under bf16 autocast). Sibling MMPA parameters (`proj`, experts `e1/e2/e3`, `gen`, `proj_back`) have finite gradients at this step — the non-finiteness is specific to `gamma`.
3. **Optimizer step:** writes the non-finite gradient into `mmpa1.gamma`; gradient clipping (max-norm 1.0) cannot rescue a non-finite value, so `mmpa1.gamma` becomes NaN/Inf.
4. **Iteration 2 forward:** `backbone` is finite (frozen), but `mmpa1` computes `gamma * e_fused` with a non-finite `gamma` → `mmpa1` output non-finite → propagates through every downstream module → all four loss terms NaN → **guard raises `Non-finite loss detected at epoch 0`**.

---

## What This Rules In / Out

| Candidate | Status |
|-----------|:------:|
| Input images / masks | clean (finite, mask binary) |
| Frozen SAM2 encoder | clean (finite at every step) |
| MMPA/FGFF/decoder/RDB **forward** | clean on fresh weights |
| Dice / BCE / Boundary / EWR / total loss **forward** | clean on fresh weights |
| **`mmpa*.gamma` gradient (backward)** | **first non-finite — root** |
| Weight corruption propagating to mmpa1 forward (step 2+) | secondary effect |

The bf16 + GradScaler-removal fixes did **not** address this: they were aimed at fp16 overflow, but the actual fault is a **non-finite gradient into the MMPA `gamma` parameter**, which occurs even in bf16. Raising the `gamma` init from 1e-7 to 1e-2 did not change the location of the fault (still `mmpa1.gamma` gradient on the first backward).

---

## Severity
**CRITICAL** — training cannot proceed past iteration 1; the model never updates usefully before the guard halts it.

---

## Exact Source (answer to the task)
**First non-finite value = the gradient of `mmpa1.gamma` (MMPA Γ scaling vector) during the first backward pass**, originating in the backward of `p_enhanced = z + gamma * e_fused` in `models/maffnet.py`. The forward pass and all loss terms are finite; the fault is purely in the gradient computation for the `gamma` parameter, after which it corrupts the MMPA forward from iteration 2 onward.

*No fix applied, as instructed.*

---

## Evidence References
- `models/maffnet.py` MMPA: `gamma` parameter and `p_enhanced = z + self.gamma * e_fused`
- Instrumented runs: (a) single-iteration stage trace — all forward/loss finite; (b) training-loop trace — `first_nf_grad_after_bwd = mmpa1.gamma` at step 1, `mmpa1` forward non-finite at step 2
- `ROOT_CAUSE_ANALYSIS.md` (prior identification of the MMPA `gamma` instability)
