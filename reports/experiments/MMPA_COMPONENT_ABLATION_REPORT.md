# MMPA Component Ablation Report (Diagnostic)

**Goal:** binary-search the MMPA sub-operations to find which one emits the first non-finite gradient.
**Method:** runtime monkey-patch of `MMPA.forward` (no repository edit — modification cannot persist). bf16 pipeline, AdamW (gamma no-decay), clip 1.0, one-batch overfit on 2 real ISIC-2016 images.

---

## Result — All Three MMPA Components Are Exonerated

| Experiment | Bypass | step-1 loss | step-1 finite | first non-finite grad | iter-2 finite | optimizer updates |
|------------|--------|------------:|:-------------:|-----------------------|:-------------:|-------------------|
| Baseline (full) | — | 1.5880 | ✅ | `mmpa1.gamma` | ❌ NaN | halted at step 2 |
| **Exp 1** | GroupNorm (LN) | 1.5837 | ✅ | `mmpa1.gamma` | ❌ NaN | halted at step 2 |
| **Exp 2** | softmax → uniform mean | 1.5881 | ✅ | `mmpa1.gamma` | ❌ NaN | halted at step 2 |
| **Exp 3** | dilated experts → identity | 1.5880 | ✅ | `mmpa1.gamma` | ❌ NaN | halted at step 2 |

Every variant behaves **identically**: finite iteration-1 forward/loss, `mmpa1.gamma` reported as the first non-finite gradient, NaN at iteration 2, optimizer halts. Removing GroupNorm, the softmax expert weighting, or the dilated experts has **no effect**.

---

## Interpretation — The Fault Is Not Inside MMPA's Sub-Operations

1. **GroupNorm, softmax weighting, and the dilated experts are all ruled out.** Bypassing each leaves the divergence unchanged. Combined with the earlier gamma ablation (also negative), **none of the MMPA-internal operations is the cause.**
2. **`mmpa1.gamma` is a reporting artifact, not the source.** `first_nf_grad` returns the **earliest parameter in iteration order** with a non-finite gradient. `mmpa1` is the first module holding trainable parameters, so whenever the backward produces a **global** non-finite gradient, `mmpa1.gamma` is reported first regardless of where the inf actually originates. The prior AMP trace already showed **~140 non-finite gradient tensors per step** — i.e., the non-finiteness is **global across the whole backward**, not localized to MMPA.
3. **Therefore the true source is a component that is common to all four runs and was never ablated here** — most plausibly the **FGFF frequency branch**. Its `torch.abs(rfft2(...))` (complex-magnitude) backward is singular where spectral coefficients approach zero: the gradient of `|z|` is `z̄/|z|`, which → ∞ as `|z| → 0`. The forward magnitude is finite, which is exactly the observed pattern (iteration-1 forward finite, iteration-1 backward non-finite). A single inf in the FGFF backward propagates to **every** upstream trainable parameter, producing the global non-finite gradient and the `mmpa1.gamma`-first artifact.
   *(Note: an isolated FGFF test on random input was finite earlier, but random input has no near-zero coefficients; real SAM2 features and the bf16 path can hit the singular region.)*

---

## Conclusion

The exact MMPA operation responsible for the first non-finite gradient is **none of the three tested** — GroupNorm, softmax expert weighting, and dilated experts are all exonerated. The non-finite gradient is **global**, and `mmpa1.gamma` is only the first-by-order symptom. The diagnosis must move **outside** the MMPA sub-graph; the leading remaining suspect is the **FGFF FFT-magnitude backward** (`abs(rfft2)` gradient singularity).

---

## Recommended Next Diagnostic (not performed here)
Ablate the FGFF frequency path on the first backward:
1. Bypass the frequency branch entirely (return `spatial_fused + x`, no FFT) → re-test iteration-2 finiteness.
2. If finite, narrow to `torch.abs(rfft2(...))`: replace `|z|` with `sqrt(z.real² + z.imag² + ε)` (ε-stabilized magnitude) **in a diagnostic patch only** and re-test.
Whichever makes iteration 2 finite and lets the optimizer update identifies the true source.

---

## Codebase Status
- `models/maffnet.py` — **unchanged**. All four experiments were runtime monkey-patches of `MMPA.forward`, restored at script end. No persisted modification.
- No fix applied — diagnostic only, as instructed.

## Evidence References
- Runs: full / gn_off / uniform / identity_experts — identical (step-1 finite, `mmpa1.gamma` first-nf-grad, step-2 NaN, optimizer halted)
- `NONFINITE_TRACE_REPORT.md` (`mmpa1.gamma` first-nf-grad), `GAMMA_ABLATION_REPORT.md` (gamma exonerated), prior AMP trace (~140 non-finite grad tensors/step → global)
