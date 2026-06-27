# FINAL MAFFNet Architecture — Implementation Roadmap

**Inputs:** original MAFF-Net paper (Zhou et al., BSPC 2026), `MAFFNET_DESIGN.md`, current `models/maffnet.py` + `training/trainer.py`.
**Mode:** roadmap only. No code written. No redesign-from-scratch.

**Governing principle:** the cited base IS the real MAFF-Net, so the **base must equal the real MAFF-Net** (restore downgraded modules), and **novelty must sit on top** (boundary-centric supervision). You cannot claim simplifications as contributions.

---

## 1. Original Modules That MUST Be Restored

Current code downgraded these below the paper. Restore to the published design — they become the **faithful base**, not contributions.

| # | Module | Restore to (original) | Current (downgraded) | Priority |
|---|--------|-----------------------|----------------------|:--------:|
| R1 | **MMPA** (dynamic adaptor) | C→32 bottleneck, 3 depthwise dilated experts (d=3,5,7), **WetGeneration softmax dynamic weights**, residual w/ Γ=1e-7 (~1.7M params) | full-width dilated + static SE attn | **P0** (paper's #1 contribution) |
| R2 | **FGFF cross-modal fusion** | `β⊙(f_freq_attn ⊗ f_spatial_fused)`, multi-dilation spatial (d=1,3,5,7), learnable β | concat([spatial,freq])→1×1 | **P0** |
| R3 | **Frequency stream** | grayscale-image FFT → fftshift → magnitude → min-max → pseudo-RGB → lightweight **freq transformer** (DWConv-QKV, learnable temperature) | per-feature `rfft2` + 1×1 conv | **P1** (heavy; restore at least magnitude+freq-attn even if transformer trimmed — document) |
| R4 | **EDB channel-attention skip** | avg+max pool → conv → σ → `x + γ·(attn⊙x)` | plain concat skip | **P1** |
| R5 | **RDB reconstruction loss** | MSE(f_rec, input) as auxiliary, λ3=0.01 | RDB computed but **output discarded** | **P0** (free — already built, just wire it) |

> R1+R2+R5 are P0: without them the "base MAFF-Net" in the paper is fictional and reviewers will reject on reproducibility.

---

## 2. Modules That Should Remain Unchanged

Keep as-is — already correct or genuinely fine.

| Module | Status |
|--------|--------|
| **Frozen SAM2 Hiera-L encoder** | Correct (matches paper §4.2 `sam2_hiera_large`). Keep frozen. |
| **4-head deep supervision** | Matches paper (4 prediction maps, λ={1,1,1,0.01}). Keep. |
| **Inference pipeline** (RDB off at inference, seg-only) | Matches paper. Keep. |
| **Optimizer skeleton** (AdamW 1e-3, wd 0.01, linear decay) | Matches paper. Keep (but fix batch/epochs — see §5). |
| **EDB upsampling** (bilinear, no deconv checkerboard) | Matches paper. Keep. |

---

## 3. New Modules That SHOULD Be Added (the contribution)

These are the *only* things that make this a paper, not a re-implementation. Add **on top of the restored base**.

| # | New module | What it is | Why it is novel vs MAFF-Net | Effort |
|---|-----------|------------|------------------------------|:------:|
| N1 | **Boundary-Aware Deep Supervision** | Sobel-edge L1 between `σ(p_i)` and GT edges, summed over the 4 heads, curriculum warmup (epoch≥k) | MAFF-Net loss is **Dice + reconstruction-MSE only** — no edge/boundary term. This is a clean, defensible addition. | Low |
| N2 | **Edge-weighted reconstruction** (upgrade RDB) | Replace plain MSE recon with boundary-weighted recon (`MSE·(1+α·edge_map)`) | Couples the auxiliary recon branch to lesion boundaries → one coherent "boundary-centric supervision" story spanning both branches. Distinct from paper's plain MSE. | Low (RDB exists) |
| N3 (optional) | **Explicit high-pass gating in FGFF** | Learnable radial high-pass mask + cutoff on the magnitude spectrum before fusion | Gives the "frequency-guided" claim real teeth and a clean ablation; paper keeps full magnitude. Add only if N1+N2 prove insufficient. | Medium |

**Primary contribution = N1 + N2** (synergistic: "boundary-centric dual supervision on a SAM2-adapted segmenter"). N3 is a reserve if reviewers want more.

---

## 4. Modules That Should Be REMOVED / Fixed

| # | Remove / fix | Action | Reason |
|---|--------------|--------|--------|
| X1 | **dec1 zero-skip** (`torch.zeros_like` at finest decoder level) | Remove the zeros hack; use a real finest-level skip (or restore EDB skip) | Defect — wastes the highest-resolution level; hurts boundaries |
| X2 | **Static SE-MMPA** | Removed implicitly by R1 (replaced by dynamic adaptor) | It is the downgrade; cannot coexist with restored MMPA |
| X3 | **Concat-FGFF** | Removed implicitly by R2 | Same — the downgrade |
| X4 | **Dead RDB output** | Removed by R5 (wire recon into loss) | Currently pure wasted compute (~7.8M-param branch, no gradient) |
| X5 | **BCEWithLogits term** (pos_weight=2, ours-only) | Demote to *optional/ablated*, not core | Not in paper (Dice+MSE); keep only if ablation shows it beats Dice-only — else it muddies the MAFF-Net comparison |

Silent fixes (housekeeping, not paper-worthy): input **ImageNet/SAM normalization** (currently missing — required for the frozen encoder), anisotropic-resize note, seed control.

---

## 5. Modifications Required for Q1 Publication

Beyond architecture, these are mandatory for acceptance:

| Area | Requirement |
|------|-------------|
| **Faithful base** | R1–R5 restored so "MAFF-Net base" is real and reproducible |
| **Novelty** | N1+N2 implemented and shown to help by ablation |
| **Evaluation split** | Report **test** split (ISIC 2017 test masks now exist via re-split, `SPLIT_AUDIT.md`) — not val |
| **Boundary metrics** | Add **HD95 + Boundary-IoU** (mandatory to support a boundary-loss claim) alongside Dice/IoU/Sens/Spec |
| **Statistics** | ≥3 seeds, mean±std, paired significance vs base |
| **Training budget** | Match paper (**batch 12, 100 epochs**) or explicitly state the deviation — current batch 2 / 20–40 ep is **not comparable** to paper numbers |
| **Baselines** | U-Net, Att-UNet, UNet++, **migrated official TransUNet**, **original MAFF-Net**, ≥2 recent (2023–24) SOTA |
| **Ablations** | N1 on/off + λ sweep + warmup; N2 on/off; (Dice / Dice+MSE / Dice+boundary / full) |
| **Efficiency** | params (trainable %), FLOPs, latency vs paper's 14.8M / 6.5% / 0.08s |
| **Normalization** | apply ImageNet/SAM norm to the frozen encoder input (fix) |

---

## 6. Minimum Architectural Change Set for a Defensible Contribution

The smallest set that yields a real, reviewer-survivable contribution **without redesigning MAFFNet**:

```
STEP 1 (restore faithful base — not novelty, but mandatory):
   R1  dynamic MMPA adaptor (softmax experts, 32-bottleneck, Γ=1e-7)
   R2  cross-modal FGFF (β·freq_attn ⊗ spatial_fused, multi-dilation)
   R5  wire RDB reconstruction MSE into the loss

STEP 2 (the contribution — add on top):
   N1  boundary-aware deep supervision (Sobel L1, multi-scale, warmup)
   N2  edge-weighted reconstruction (upgrade R5's MSE)

STEP 3 (fixes that gate validity):
   X1  remove dec1 zero-skip → real finest-level skip
   NORM  add ImageNet/SAM input normalization
```

- **STEP 1** makes the base honest (else nothing is comparable).
- **STEP 2** is the paper's claim: *boundary-centric dual supervision* (segmentation edges + reconstruction edges) on a frozen-SAM2 adaptor segmenter — a coherent, ablatable, single-sentence novelty.
- **STEP 3** removes correctness defects that a reviewer would flag.

Everything else in `MAFFNET_DESIGN.md` (EDB attention R4, freq-transformer R3, high-pass N3, BCE) is **optional polish** — add only if reviewers demand more or if STEP 1–2 ablations come out weak.

**Defensible one-line claim:** *"We extend MAFF-Net with boundary-centric dual supervision — coupling Sobel-edge deep supervision in the segmentation heads with edge-weighted reconstruction in the auxiliary decoder — improving low-contrast boundary delineation on ISIC 2016/2017/2018, validated by HD95/Boundary-IoU and ablation."*

---

## Risk / Scope Notes
- **Restoration ≠ novelty:** R1–R5 only rebuild the cited base; do not present them as contributions.
- **Budget mismatch is a real threat:** current batch 2 / 20–40 ep cannot reproduce the paper's MAFF-Net numbers — fix or the whole comparison is invalid.
- **Boundary claim needs boundary metrics:** without HD95/Boundary-IoU, N1/N2 are unfalsifiable and will be rejected.
- **Keep it single-story:** one coherent contribution (boundary-centric dual supervision) beats three weak ones.

---

## Evidence References
- Paper §3.1 (MMPA dynamic adaptor), §3.2 (FGFF cross-modal, freq transformer), §3.3 (EDB attn, RDB), §3.4 (loss = Dice+MSE), §4.2 (sam2_hiera_large, batch 12, 100 ep)
- `models/maffnet.py:39-76,82-113,119-132,138-180,350-351,362-364`
- `training/trainer.py:68-113` (Dice+BCE+boundary, no recon MSE; `_rec` discarded)
- `MAFFNET_DESIGN.md`, `SPLIT_AUDIT.md`, `PREPROCESSING_AUDIT.md`, `MIGRATION_REPORT.md`
