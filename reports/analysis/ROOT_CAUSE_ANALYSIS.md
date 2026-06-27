# Root Cause Analysis — MAFFNet Training Plateau

**Mode:** diagnosis only. No code modified, no fixes implemented. All conclusions backed by instrumented runs on the real model + SAM2 checkpoint (RTX/CUDA, fp32 and AMP).

---

## 1. Executive Summary

MAFFNet does **not train at all** under the real (AMP) pipeline. Every training step produces **non-finite gradients**, so `GradScaler` **skips every optimizer step** — measured cumulative parameter movement is exactly **0.0** over 8 steps while the loss scale decays 65536 → 256. Each epoch therefore evaluates the **untrained, randomly-initialized** trainable head on top of frozen SAM2, which is why train loss, val loss, Dice, and IoU are all flat and the "best epoch" is epoch 1.

The non-finite gradients originate in the restored **MMPA adaptor**: the first parameter to go non-finite (fp32) is **`mmpa*.gamma`** (the Γ scaling vector, initialized to `1e-7`). Under fp16 AMP the whole adaptor/decoder backward overflows fp16 on every step regardless of loss scale. Because **MAFFNetTrainer has no non-finite-loss guard**, the failure is silent and masquerades as a convergence plateau rather than crashing.

**Verdict:** implementation/numerical bug (not underfitting, not data, not the frozen backbone). **Severity: CRITICAL.**

---

## 2. Findings (mapped to requested checks)

### Check 1 — Trainable parameters (✅ correct)
| Module | Total | Trainable |
|--------|------:|----------:|
| encoder (SAM2 backbone) | 212.703M | **0.000M (frozen — correct)** |
| MMPA | 0.078M | 0.078M |
| FGFF | 19.140M | 19.140M |
| decoder | 6.492M | 6.492M |
| seg head | 0.001M | 0.001M |
| RDB | 3.264M | 3.264M |
| **TOTAL** | **241.68M** | **28.98M** (frozen 212.70M) |

Nothing accidentally frozen; the encoder is intentionally frozen, all heads/adaptors/decoder are trainable. **Not the cause.**

### Check 2 — Optimizer (✅ correct)
`AdamW`, `lr=1e-3`, `weight_decay=0.01`, over only `requires_grad` params (frozen encoder excluded). `LinearLR` 1.0→0.0. Configuration is correct. **Not the cause.** (Note: `gamma` is included with weight decay — minor concern, not the trigger.)

### Check 3 — Gradient flow (🔴 fault here)
At **init** (step 1) all module grad norms are finite: MMPA 0.319, FGFF 2.832, decoder 5.509, seg-head 0.903, RDB 0.090 — gradient reaches every trainable module. But from **step 2** the gradient of **`mmpa*.gamma`** becomes non-finite, and under AMP **140 grad tensors are non-finite every step from step 1**.

### Check 4 — Loss audit (✅ not the cause)
Init loss finite (1.5882); components `seg_loss=1.584, bds=0.126, ewr=0.077`. Matches the BcDS design (Dice + BCE + boundary + edge-weighted reconstruction). The loss is correct and finite at init; it only becomes NaN **after** weights are corrupted. (Separate, non-fatal note: §3.6 of the draft omits the BCE term that the code includes — a documentation inconsistency, not the plateau cause.)

### Check 5 — Forward pass (✅ at init, corrupts later)
At init `p1` is a real prediction (mean 0.033, **std 0.328 → not constant**), shapes `(2,1,384,384)`. No NaN at init. Forward becomes non-finite at step 3 in fp32 **after** `gamma` corrupts the weights. No bad detach / interpolation found in the forward itself.

### Check 6 — Dataset (✅ not the cause)
Masks load as binary `{0.0, 1.0}` after resize/threshold; ImageNet normalization applied inside the model; shapes correct. Data is clean.

### Check 7 — One-batch overfit (🔴 FAILS — the headline)
Two real ISIC-2016 images, 200 steps:
- **fp32, lr=1e-3, clip=1.0:** loss NaN at **step 3**.
- **fp32, lr=1e-4, no clip:** loss NaN at **step 3** (lower LR does **not** help → not an LR-magnitude problem).
- **AMP + GradScaler, lr=1e-3, clip=1.0 (real pipeline):** loss **frozen at exactly 1.5882, Dice 0.4533 for 120 steps**, parameters never move.

Expected Dice > 0.95; achieved a frozen 0.45. **The model cannot overfit a single batch.**

### Check 8 — Prediction audit (explained)
Under AMP the parameters never update, so predictions are **identical at every epoch**. "Best epoch 1" is an artifact: every epoch is the same untrained model, so the first evaluated epoch is recorded as best.

### Check 9 — Learning dynamics (conclusion)
Not underfitting-by-capacity (28.98M trainable is ample), not a data issue, not the frozen backbone (decoder/adaptors are trainable and receive gradient at init), not the LR (1e-4 fails identically), not the loss (finite at init). It is an **implementation/numerical-stability bug** in the MMPA `gamma` path, amplified by fp16 AMP and hidden by the missing finite-loss guard.

---

## 3. Evidence

| Evidence | Source |
|----------|--------|
| AMP: cumulative param movement = 0.000 across 8 steps; scale 65536→256; 140 non-finite grad tensors/step | scale-tracking run |
| fp32: step1 finite → step2 `mmpa3.gamma` grad non-finite (loss 1.0135) → step3 NaN | first-non-finite localization run |
| Overfit fails (NaN fp32 / frozen 0.45 AMP) | one-batch overfit run |
| `self.gamma = nn.Parameter(torch.full((1,bottleneck,1,1), 1e-7))`; `p_enhanced = z + self.gamma * e_fused` | `models/maffnet.py:77,87` |
| MAFFNetTrainer: `scaler.step`/`scaler.update` with **no `isfinite` guard** | `training/trainer.py:333-334,673-674` |
| SimpleTrainer **has** the guard (so U-Net-family fails loudly, MAFFNet fails silently) | `training/simple_trainer.py:142` |
| FGFF backward is finite (abs(rfft2) not the culprit) | FGFF isolation micro-test |
| Params: 241.68M total / 28.98M trainable / 212.70M frozen | param breakdown run |

---

## 4. Severity

| Issue | Severity |
|-------|:--------:|
| AMP skips every optimizer step → model never trains (plateau) | **CRITICAL** |
| MMPA `gamma=1e-7` produces non-finite gradients (root trigger) | **CRITICAL** |
| MAFFNetTrainer has no non-finite-loss guard → silent failure masquerading as plateau | **HIGH** |
| fp16 AMP overflow in adaptor/decoder backward (scale-independent) | **HIGH** |
| `gamma` included in weight decay; loss-doc/code BCE mismatch | **MEDIUM/LOW** |

---

## 5. Recommended Fixes (ranked by expected impact — NOT implemented)

1. **[CRITICAL] Switch AMP from fp16 to bf16** (`autocast("cuda", dtype=torch.bfloat16)`, drop `GradScaler`). bf16 has fp32-range exponent, so the per-step overflow that forces every step to be skipped disappears. This is the single highest-impact change and directly targets the "scale decays, nothing moves" failure. (Same fix class recommended earlier for the U-Net NaN.)
2. **[CRITICAL] Fix the MMPA `gamma` instability.** Re-initialise Γ to a stable value (e.g. `1e-2` or `1.0`) instead of `1e-7`, and/or `clamp` it, and exclude it from weight decay. The `1e-7` init gives no protection under AdamW (which is scale-invariant) and is the first parameter to diverge in fp32.
3. **[HIGH] Add a non-finite-loss/grad guard to `MAFFNetTrainer`** (mirroring `SimpleTrainer:142`). This will not fix training but will convert the silent plateau into an explicit error, preventing wasted runs and misdiagnosis.
4. **[HIGH] After (1)+(2), re-run the one-batch overfit test** and require Dice > 0.95 before any full training. Treat this as a gate.
5. **[MEDIUM] Verify gamma is in a no-weight-decay parameter group** and consider gradient clipping that is actually effective (current clip-to-1.0 cannot rescue inf/NaN, only large-finite values).
6. **[LOW] Reconcile the loss documentation** (§3.6 vs code BCE term) once training is restored.

**Note:** the earlier 04-28 MAFFNet runs that reached Dice ≈ 0.92 used the *pre-restoration* MMPA (no `gamma=1e-7` adaptor). The instability was introduced by the restored dynamic MMPA; fixes 1–2 target exactly that regression.

---

## Evidence References
- `models/maffnet.py:77,87` (MMPA gamma), `:128` (fp32 FFT)
- `training/trainer.py:333-334,673-674` (scaler step, no guard)
- `training/simple_trainer.py:142` (guard present for contrast)
- Instrumented runs: param breakdown; grad-norm audit; one-batch overfit (fp32/AMP); AMP scale-tracking; fp32 first-non-finite localization; FGFF isolation
- `RESTORATION_REPORT.md` (restored dynamic MMPA introduced gamma)
