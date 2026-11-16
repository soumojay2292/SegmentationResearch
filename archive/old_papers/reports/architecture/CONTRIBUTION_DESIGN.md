# Contribution Design — Boundary-centric Dual Supervision (BDS+EWR)

**Phase:** research contribution. Base = the **frozen restored MAFFNet** (`RESTORATION_REPORT.md`). No further restoration.
**Single contribution:** **Boundary-centric Dual Supervision (BcDS)** — couple two boundary-aware signals driven by the *same* ground-truth lesion-boundary map:
1. **N1 — Boundary-aware Deep Supervision (BDS):** edge supervision on every segmentation head.
2. **N2 — Edge-weighted Reconstruction (EWR):** boundary-weighted reconstruction in the auxiliary RDB branch.
**Mode:** design only. No code.

> **Unifying idea.** One boundary prior `B(y)` derived from the GT mask drives *both* the segmentation output loss *and* the reconstruction loss. The segmentation branch is told *"match lesion edges"*; the reconstruction branch is told *"reconstruct the image most faithfully near lesion edges."* Two complementary gradients push the frozen-SAM2-adapted features to preserve boundary detail. This is the whole novelty — kept to **one coherent claim**.

---

## 1. Architecture Changes

**None structural. Supervision-graph only.** The restored model already emits everything needed:
- 4 segmentation logits `p1..p4` (deep supervision heads).
- 1 reconstruction `rec` (RDB, Tanh, [-1,1]).

BcDS changes **only the loss** (`MAFFNetLoss` in `training/trainer.py`). Therefore:
- **Zero new parameters**, **zero inference cost** (RDB already train-only; boundary terms are loss-side).
- The frozen SAM2 encoder, MMPA, FGFF, decoder, RDB are untouched.
- Gains are attributable to **supervision**, not capacity → clean ablation.

This is a deliberate scope choice: a loss-level contribution on a frozen base is small, defensible, and fully ablatable.

---

## 2. Mathematical Formulation

**Notation:** input image `x ∈ [0,1]^{B×3×H×W}`; GT mask `y ∈ {0,1}^{B×1×H×W}`; head logits `p_i`, `i=1..4`; reconstruction `rec ∈ [-1,1]^{B×3×H×W}`; Sobel operator `S(·)` (gradient magnitude); `σ` = sigmoid.

**Shared boundary prior** (computed once per batch from GT):
```
B(y) = clamp( |S(y)| , 0, 1 )                      # thin lesion-edge map
B_r(y) = Dilate_r( B(y) )                            # optional band of width r (r∈{0,1,3})
```
(`r=0` ⇒ thin edge; `r>0` ⇒ tolerance band. Alternative: morphological gradient `Dilate(y)−Erode(y)`.)

**N1 — Boundary-aware Deep Supervision:**
```
ℓ_bnd^(i) = || S(σ(p_i)) − S(y) ||_1                # per head, edge agreement
L_BDS = Σ_{i=1}^{4} w_i · ℓ_bnd^(i)                 # deep-supervision weights w
```
- `w = {1, 1, 1, 0.01}` (mirror the seg deep-supervision weights) or uniform — ablated.
- Optional pre-smoothing `σ(p_i) → AvgPool_{3×3}(σ(p_i))` (already in code) to stabilise Sobel on logits — ablated.
- Replaces the current ad-hoc single-weight, epoch-gated boundary term.

**N2 — Edge-weighted Reconstruction:**
```
rec01 = (rec + 1)/2                                  # [-1,1] → [0,1]
W = 1 + α · B_r(y)                                   # baseline 1 everywhere, boosted at edges
L_EWR = ( Σ W ⊙ (rec01 − x)^2 ) / ( Σ W )           # normalised weighted MSE
```
- `α ≥ 0` controls edge emphasis (`α=0` ⇒ plain MSE = restored base). Normalisation by `ΣW` keeps the scale stable across α.
- Upgrades the restored plain `MSE((rec+1)/2, x)`.

**Total objective** (with curriculum warmup on the boundary term):
```
λ_b(t) = λ_b^max · min(1, t / T_warm)               # linear ramp, t = epoch
L_total = L_seg  +  λ_b(t) · L_BDS  +  λ_r · L_EWR
```
- `L_seg` = the restored segmentation loss (Dice [+BCE], deep-supervised). BCE retention is a base decision to ablate.
- Defaults to sweep: `λ_b^max ∈ {0.01..0.1}`, `λ_r ∈ {0.01..0.1}`, `α ∈ {1..5}`, `T_warm ∈ {0, 5, 10}` epochs.

---

## 3. Implementation Steps (when approved — no code now)

All in `training/trainer.py` (loss); optional tiny helper. Model untouched.

1. **Boundary-map utility** — extend the existing `BoundaryLoss`/Sobel infra to expose `B(y)` (thin + dilated band, with morphological-gradient option).
2. **Refactor `MAFFNetLoss`:**
   - Replace the current epoch-gated single boundary term with `L_BDS` (per-head, weighted, warmup-ramped via `λ_b(t)`).
   - Replace plain recon MSE with `L_EWR` (edge-weighted, normalised).
   - Expose hyperparams: `λ_b_max, λ_r, α, T_warm, w, band_r, edge_source, smooth, use_bce`.
3. **Plumb hyperparams** through `MAFFNetTrainer.__init__` → config, with safe defaults (so `train.py`/`train_all.py` keep working with no signature change; pass via the existing `config` dict).
4. **Logging:** add `train_bds_loss`, `train_ewr_loss` CSV columns (DictReader-safe; report/benchmark read named columns → no break). Lets ablations track each term.
5. **Ablation switches:** boolean flags to disable BDS / EWR / warmup / band independently.
6. **No change** to `maffnet.py`, `train.py`, `train_all.py` build sites, report/benchmark/aggregate.

---

## 4. Ablation Plan

**Core (proves the "dual" claim):**
| Variant | BDS | EWR | Purpose |
|---------|:---:|:---:|---------|
| A — restored base | ✗ | ✗ (plain MSE) | reference |
| B — +BDS only | ✓ | ✗ | segmentation-edge effect |
| C — +EWR only | ✗ | ✓ | reconstruction-edge effect |
| **D — BcDS (full)** | ✓ | ✓ | the contribution |

**The claim survives only if `D > max(B, C)`** on boundary metrics. If `D ≈ B` (EWR adds nothing) → drop EWR, contribution shrinks to BDS alone. If `D ≈ C` → symmetric. This is the decisive test.

**Hyperparameter sweeps:**
- `λ_b^max ∈ {0, .01, .05, .1}`, `λ_r ∈ {0, .01, .05, .1}`, `α ∈ {0,1,3,5}`.
- Warmup `T_warm ∈ {0, 5, 10}` (always-on vs ramp).
- Boundary map: Sobel-thin vs morphological-gradient; band `r ∈ {0,1,3}`.
- Edge source for EWR: **GT-boundary** (primary, coherent with BDS) vs input-image gradient (alt).
- `w` uniform vs `{1,1,1,.01}`; smoothing on/off.

**Protocol:** ISIC 2016/2017/2018 **test** split, **≥3 seeds (mean±std)**, paired significance vs base A.

**Metrics:** Dice, IoU, Acc, Sens, Spec **+ HD95 + Boundary-IoU** (mandatory — a boundary contribution unsupported by boundary metrics is unfalsifiable).

---

## 5. Expected Advantages

| Advantage | Why |
|-----------|-----|
| Better low-contrast boundary delineation | Both losses concentrate gradient on lesion edges — the paper's stated failure mode |
| Zero added params / zero inference cost | Loss-only; RDB already train-only → no deployment penalty (preserves MAFFNet's 0.08s/img claim) |
| Clean attribution | No capacity change → measured gains come from supervision, ideal for ablation |
| Coherent single story | One boundary prior `B(y)` drives two branches → easy to state, diagram, and formalise |
| Synergy hypothesis | Reconstruction edge-weighting regularises the frozen-feature adaptation toward edges, complementing direct output-edge supervision |

---

## 6. Risks

| Risk | Severity | Mitigation |
|------|:--------:|-----------|
| **Dual collapses to single** (`D ≈ B` or `D ≈ C`) — the core claim fails | **High** | Decisive A/B/C/D ablation; if it fails, honestly reframe as single-term contribution (BDS) |
| **EWR marginal** (recon is auxiliary/train-only) | Med-High | α sweep; GT-boundary vs input-gradient; if still flat, drop EWR |
| Boundary terms destabilise early training | Med | Warmup ramp `λ_b(t)`, small λ, `(1+αB)` keeps base weight 1 everywhere |
| Over-focus on edges → interior under-segmentation | Med | `(1+αB)` (not pure B) preserves region loss; cap α; monitor Dice not just HD95 |
| Sobel on σ(logit) sensitive to logit scale | Low-Med | AvgPool smoothing (ablated); consider morphological gradient |
| HD95 noisy on tiny lesions | Low | report median + per-size quartile; Boundary-IoU as companion |
| Budget mismatch with paper (batch 2 / few epochs) | Med | match batch 12 / 100 ep (or state deviation) before headline numbers — else gains not comparable |
| Reviewer: "edge loss is well known" | Med | novelty is the **dual GT-boundary coupling across seg + reconstruction on a frozen-SAM2 adaptor**, proven by D>B,C — not the Sobel term itself |

---

## 7. Success Criteria (go/no-go for the paper)

- `D` beats `A` on **HD95 / Boundary-IoU** with significance (≥3 seeds) on ≥2 of 3 datasets.
- `D > max(B, C)` (justifies "dual").
- Dice not regressed vs `A` (boundary gain must not cost region accuracy).
- If any fails → reframe (BDS-only) or strengthen, do not overclaim.

---

## Evidence References
- Restored base: `RESTORATION_REPORT.md`; `training/trainer.py` (`MAFFNetLoss` — current Sobel boundary + recon MSE to be reformulated), `models/maffnet.py` (p1–p4 + rec outputs, unchanged)
- Roadmap: `FINAL_MAFFNET_ARCHITECTURE.md` STEP 2 (N1+N2)
- Original paper §3.3.2/§3.4 (RDB + Dice/MSE loss), §3.2 (frequency/edge motivation)
- Metric requirement: `MAFFNET_DESIGN.md` §9 (HD95/Boundary-IoU)
