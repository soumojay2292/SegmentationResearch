# Gamma Ablation Report (Diagnostic)

**Experiment:** temporarily bypass the MMPA learnable `gamma` — replace `p_enhanced = z + gamma * e_fused` with `p_enhanced = z + e_fused` — and compare against the current implementation.
**Method:** runtime monkey-patch of `MMPA.forward` inside the diagnostic script only. **The repository file `models/maffnet.py` was never edited; the modification cannot persist.** Same bf16 pipeline (no GradScaler, clip 1.0, AdamW, gamma excluded from weight decay).

---

## Result — Gamma is NOT the root cause

| Run | step 1 | step 2 | outcome |
|-----|--------|--------|---------|
| **A. Current** (`z + gamma * e_fused`) | loss 1.5880, finite, dice 0.4527 | **NaN** | diverges |
| **B. Ablated** (`z + e_fused`, gamma bypassed) | loss 1.5878, finite, dice 0.4528 | **NaN** | diverges |
| **C. Ablated 5-epoch smoke** (64-img ISIC-2016 subset, bs=2) | epoch 0 mean_loss NaN | — | all 5 epochs NaN |

Bypassing `gamma` produced **identical behaviour**: finite at iteration 1, non-finite at iteration 2, and the 5-epoch smoke is NaN from the first epoch. Removing `gamma` does **not** restore training.

---

## Interpretation

1. **The earlier signal was misleading.** The non-finite-trace reported `mmpa1.gamma` as the first non-finite gradient, but that was only the **first parameter in iteration order** to show a non-finite grad — not the source. With `gamma` removed, some **other** MMPA parameter's gradient goes non-finite instead, and the model still diverges at step 2.
2. **The fault is upstream of `gamma`, in the shared MMPA sub-graph.** Both variants are finite on the iteration-1 forward and loss; the non-finite value appears in the **iteration-1 backward**, then the optimizer writes it into an MMPA parameter, and the iteration-2 forward propagates NaN from `mmpa1` onward. This pattern is independent of the `gamma` term.
3. **Therefore the previously-implemented gamma changes (init 1e-7 → 1e-2, gamma excluded from weight decay) do not address the real cause.** They neither moved nor removed the divergence.

**Conclusion:** `gamma` is **exonerated**. The instability lives elsewhere in MMPA's backward — the remaining candidates (not yet isolated) are the shared MMPA operations: the `GroupNorm` ("LN") layer, the depthwise dilated expert convolutions (`e1/e2/e3`), or the `gen` → `softmax` weight generator. The next diagnostic should ablate these sub-components individually (e.g., bypass `GroupNorm`, or replace the softmax-weighted expert sum with a plain mean) to localize which one emits the non-finite gradient on the first backward.

---

## Comparison Summary

| Aspect | Current (gamma) | Ablated (no gamma) |
|--------|:---------------:|:------------------:|
| iter-1 forward/loss finite | ✅ | ✅ |
| iter-2 loss finite | ❌ NaN | ❌ NaN |
| one-batch overfit reaches Dice>0.95 | ❌ | ❌ |
| 5-epoch subset smoke stable | not run (known NaN) | ❌ NaN all epochs |
| **Verdict** | diverges | **diverges identically** |

The two are indistinguishable → `gamma` has no causal role in the divergence.

---

## Status of the Codebase

- `models/maffnet.py` — **unchanged by this experiment** (ablation was a runtime monkey-patch, restored at end of script). The `gamma` initialization in the file remains `1e-2` from the earlier fix task; this ablation neither added nor removed any persisted code.
- No fix applied. This was a diagnostic comparison only, as instructed.

---

## Recommended Next Diagnostic (not performed here)
Localize the true source inside MMPA by ablating the shared path on the **first backward**:
1. Bypass `GroupNorm` (the `ln`) → re-test step-2 finiteness.
2. Replace softmax expert weighting with a uniform mean (remove `gen`/`softmax`) → re-test.
3. Disable the depthwise dilated experts (identity bottleneck) → re-test.
Whichever removal makes iteration 2 finite identifies the offending operation.

---

## Evidence References
- Runtime monkey-patch of `MMPA.forward` (no-gamma), diagnostic script
- Runs: A (gamma, NaN step 2), B (no-gamma overfit, NaN step 2), C (no-gamma 5-epoch subset smoke, NaN all epochs)
- `NONFINITE_TRACE_REPORT.md` (prior `mmpa1.gamma` first-non-finite-grad signal — now shown to be order-artifact, not cause)
