# Experimental Gap Analysis — MAFFNet Framework

**Scope:** assess whether the current empirical setup (3 ISIC datasets × 6 baselines × 1 seed × val-only metrics) is sufficient for a Q1 venue. Identify what is missing and what to add.

**Verdict up front:** **Not Q1-ready.** Borderline Q3 with current evidence; Q2 reachable with ~2 months focused work; Q1 requires the full plan below (~4–6 months).

---

## 1. Q1 Sufficiency Audit

A Q1 medical-imaging venue (*Medical Image Analysis*, *IEEE TMI*, *IEEE TIP*, *Pattern Recognition*) typically demands:

| Requirement | Current state | Pass? |
|-------------|---------------|:-----:|
| ≥ 3 datasets with ≥ 1 cross-domain external test | 3 ISIC datasets, no external test | ❌ |
| ≥ 6–8 recent SOTA baselines (≥ 3 from last 2 years) | 6 baselines, only 2 from ≥ 2021 | ❌ |
| Multi-seed runs (≥ 3) with mean ± std | 1 seed | ❌ |
| Statistical significance test (Wilcoxon / paired t) | none | ❌ |
| Held-out **test** metrics, not val | val only (see `latest_experiment_report.md` §7) | ❌ |
| Full ablation on every claimed contribution | none — not even runnable (see `ABLATION_PLAN.md`) | ❌ |
| HD95 / ASSD boundary metrics | only Dice/IoU/Acc/Sens/Spec | ❌ |
| Params / FLOPs / latency table | none | ❌ |
| Cross-dataset generalisation (train on A, test on B) | none | ❌ |
| Failure-case analysis / qualitative gallery | basic prediction samples saved, not analysed | ⚠️ |
| Reproducibility artefacts (seeds, env, checkpoints) | none (see `REPRODUCIBILITY_CHECKLIST.md`) | ❌ |
| Theoretical justification for novel module | none | ❌ |

**Score: 0 / 12 fully passed.** Q1 reviewers will desk-reject or request major revision.

For a Q2 venue the bar drops to ~6/12 passed; Q3 to ~3/12.

---

## 2. Missing Baselines

Current six baselines covered: U-Net (2015), Attention U-Net (2018), UNet++ (2018), TransUNet (2021), SGT-Net (2022), MAFFNet (named in repo, ambiguous w.r.t. published priors).

### 2.1 What's missing by category

**A. Modern transformer / hybrid segmenters (Q1 expects 3+ from this group):**

| Baseline | Year | Why needed |
|----------|-----:|------------|
| Swin-Unet | 2021 | Standard transformer-UNet hybrid; every recent ISIC paper compares |
| TransFuse | 2021 | CNN + transformer parallel fusion, ISIC-targeted |
| MISSFormer | 2021 | Pure transformer, strong on medical seg |
| MedT (Medical Transformer) | 2021 | Gated axial attention — direct rival to gating claim |
| H2Former | 2023 | Hybrid hierarchical for medical seg |
| nnFormer | 2022 | Standard medical seg baseline at MICCAI |

**B. Boundary-aware methods (direct rivals to boundary supervision claim):**

| Baseline | Year | Why needed |
|----------|-----:|------------|
| BAT (Boundary-Aware Transformer) | 2021 | Explicitly compares against boundary loss methods |
| BA-Net | 2020 | Boundary-aware Net for skin lesion |
| FAT-Net | 2022 | Feature adaptive transformer for skin lesion |
| MS-RED | 2022 | Multi-scale residual encoder-decoder, ISIC SOTA at time |

**C. Frequency-domain methods (rivals to FGFF):**

| Baseline | Year | Why needed |
|----------|-----:|------------|
| FcaNet | 2021 | DCT-based channel attention — most direct freq rival |
| GFNet | 2021 | Global filter in frequency domain |
| AFNO (Adaptive Fourier Neural Op) | 2022 | Token mixing in frequency |
| FNO-style segmenter | 2022+ | Demonstrates real iFFT use |

**D. SAM / Foundation-model rivals (rivals to SAM2 use):**

| Baseline | Year | Why needed |
|----------|-----:|------------|
| SAMed | 2023 | SAM + LoRA for medical seg |
| SAM-Med2D | 2023 | SAM fine-tuned for 2D medical |
| MedSAM | 2024 | Foundation medical SAM |
| Adapter-based SAM | 2023 | Parameter-efficient SAM adaptation |

**E. nnU-Net (mandatory):**

| nnU-Net (2D variant) | 2018/2021 (self-configuring) | Acknowledged universal medical baseline; **omission alone can trigger reject** at MICCAI / MedIA |

### 2.2 Minimum baseline set for Q1

| Tier | Baselines |
|------|-----------|
| Classical (keep) | U-Net, Attention U-Net, UNet++ |
| Transformer hybrid (add 3) | TransUNet (keep), Swin-Unet, TransFuse, MISSFormer **or** H2Former |
| Boundary-aware (add 2) | BAT, FAT-Net **or** MS-RED |
| Frequency (add 1) | FcaNet **or** GFNet |
| Foundation-model (add 1) | SAMed **or** MedSAM |
| Universal medical (add 1) | nnU-Net 2D |
| **Total** | **≈ 11–12 baselines** |

Drop SGT-Net if reimplementation cost is high; keep if reference impl exists.

### 2.3 Critical: confound control via shared backbone

**Without this, the SAM2 advantage swamps every architectural claim.**
Add a *"SAM2 + plain U-Net decoder"* baseline (Tier 0 ablation). If this lifts all baselines by ~5 pp Dice, the paper's contribution shrinks to "SAM2 transfer works for ISIC" — which is not novel.

---

## 3. Recommended Additional Experiments

### 3.1 Statistical rigour

| Experiment | Spec | Effort |
|-----------|------|--------|
| Multi-seed | 3 seeds × all configs → mean ± std | ~3× compute |
| Significance | Paired Wilcoxon vs. each baseline, per dataset | trivial post-hoc |
| Bootstrap CI | 95 % CI on Dice via 1000-sample bootstrap of test predictions | trivial |
| Effect size | Cohen's *d* per dataset comparison | trivial |

### 3.2 Metric expansion

| Metric | Why add |
|--------|---------|
| HD95 (Hausdorff 95) | Standard boundary metric in MICCAI; current Dice/IoU don't capture edge quality |
| ASSD (Average Symmetric Surface Distance) | Companion to HD95 |
| Edge-F1 (5 px boundary band) | Direct test of boundary loss claim |
| NSD (Normalized Surface Dice) at τ = 1–3 px | Used by MedIA reviewers |
| Per-lesion-size-quartile Dice | Tiny lesions are the failure mode in dermoscopy |
| Calibration metrics (ECE, Brier) | Required at TMI for clinical-translation claims |

### 3.3 Robustness / stress tests

| Test | Spec |
|------|------|
| Noise robustness | Gaussian σ ∈ {5, 10, 20} added to test input |
| JPEG compression | quality ∈ {30, 50, 70} |
| Hair occlusion | Add synthetic hair via standard dermoscopy hair sim |
| Out-of-focus blur | Gaussian blur σ ∈ {1, 2, 3} px |
| Color jitter at test | ColorJitter(0.4, 0.4, 0.4, 0.2) |
| Adversarial | FGSM ε = 4/255, PGD-5 ε = 4/255 |
| Domain shift | Train ISIC 2018 → test PH2 / Dermofit (cross-domain) |

### 3.4 Ablation (full plan in `ABLATION_PLAN.md`)

Required minimum table for Q1:
- Backbone ablation (SAM2 vs. ResNet-50 vs. random)
- MMPA on/off
- FGFF on/off + high-pass on/off + iFFT on/off
- Boundary loss on/off + warmup on/off + λ sweep
- Gated fusion vs. concat vs. sum vs. cross-attention
- Deep supervision on/off
- 2³ pair-wise table

### 3.5 Efficiency reporting

| Item | Value to report |
|------|----------------|
| Trainable params (M) | per model |
| Total params (M) | per model |
| FLOPs (G) @ 384² | per model |
| Train-time per epoch (s) | per model on same GPU |
| Inference latency (ms) at B=1 | per model |
| Peak GPU memory at train (GB) | per model |
| Per-image inference energy (mJ) | nice-to-have, RAPL or NVML measurement |

See `COMPUTATIONAL_ANALYSIS.md` for the proposed table.

### 3.6 Qualitative analysis

| Required | Detail |
|----------|--------|
| Side-by-side qualitative gallery | ≥ 12 cases per dataset, all baselines + GT |
| Failure-case taxonomy | tiny lesions, multi-lesion, low-contrast, hair-occluded, ruler artefact |
| Feature visualisation | FGFF magnitude spectrum + decoder activation heatmaps |
| Grad-CAM or attention maps | from MMPA SE block |

### 3.7 Clinical relevance hook (optional but recommended at MedIA/TMI)

| Item | Value |
|------|-------|
| Dermatologist agreement study | inter-rater κ between MAFFNet and ≥ 2 dermatologists on ~50 cases |
| Downstream classification | feed predicted masks into a melanoma classifier; report Δ AUC |
| Decision-curve analysis | clinical-utility metric |

---

## 4. External Validation Strategies

Q1 medical imaging requires evidence the model generalises beyond the training distribution.

### 4.1 External datasets to include

| Dataset | Size | Use as | Why |
|---------|----:|--------|-----|
| **PH2** | 200 | external test only | Standard dermoscopy benchmark, different camera |
| **Dermofit** | 1 300 | external test only | Different annotation protocol, license-controlled |
| **HAM10000 (segm. subset)** | ~10 000 | held-out test or auxiliary train | Largest dermoscopy archive |
| **ISIC 2019/2020** (where masks exist) | varies | external test | Newer distribution shift |
| **DermNet NZ** | scraping required | qualitative only | Real-world web images, evaluates domain gap |
| **Internal hospital data** | varies | gold standard | Clinical translation evidence |

### 4.2 Cross-dataset protocols

| Protocol | Train | Test |
|----------|-------|------|
| **Within-dataset** | ISIC 2018 train | ISIC 2018 test |
| **Sibling-dataset** | ISIC 2017 train | ISIC 2018 test (and vice versa) |
| **External-zero-shot** | ISIC 2018 train | PH2 test (no fine-tune) |
| **External-few-shot** | ISIC 2018 train + 10 % PH2 fine-tune | PH2 test |
| **Leave-one-dataset-out** | ISIC 16+17 train | ISIC 18 test |

Report **at least** within-dataset + 1 external-zero-shot for Q1.

### 4.3 Robustness suites (already listed in §3.3)

Tie noise/JPEG/blur results to a "deployment readiness" subsection.

---

## 5. Prioritised Experiment Plan

Ranked by **(reviewer impact) / (compute + dev cost)**.

### Tier S — Must do, low-cost, high-impact (~1 week)

| # | Action | Cost | Why |
|---|--------|------|-----|
| 1 | Switch reported metric: val → **test split** | trivial | Current numbers are inflated relative to test |
| 2 | Add HD95 + ASSD metrics | trivial | Boundary claim needs boundary metric |
| 3 | Add Params / FLOPs / latency table | trivial | Refutes "more compute" rebuttal |
| 4 | Set seeds (3) and rerun current 6 baselines + MAFFNet → mean ± std | ~3× current compute | Multi-seed required |
| 5 | Paired Wilcoxon test on per-image Dice | trivial | Significance markers |
| 6 | nnU-Net 2D baseline | ~2 days (off-the-shelf) | Mandatory in medical seg |

### Tier A — Strong impact, medium cost (~3–4 weeks)

| # | Action | Cost | Why |
|---|--------|------|-----|
| 7 | Implement full FGFF spec (high-pass mask + iFFT) | 2–3 days dev | Without this, the freq-domain claim is empty |
| 8 | Implement ablation flags for MMPA, FGFF, Boundary | 1 day | Enables Tables 3–5 from `ABLATION_PLAN.md` |
| 9 | Run full ablation table (13 × 3 datasets × 3 seeds) | ~315 GPU-h | Q1 prerequisite |
| 10 | Add 3 transformer baselines: Swin-Unet, TransFuse, FAT-Net | ~1 week setup + train | Replace 2015–2018 baselines as headliners |
| 11 | Add 1 SAM-based baseline (SAMed or MedSAM) | ~3 days | Without this, SAM2 use looks unfairly compared |
| 12 | Cross-dataset zero-shot: train ISIC 2018 → test PH2 | ~1 day after data | External generalisation evidence |

### Tier B — Important but lower priority (~4–6 weeks)

| # | Action | Cost | Why |
|---|--------|------|-----|
| 13 | Add 1 boundary-aware (BAT) + 1 freq baseline (FcaNet) | ~1 week each | Direct rivals to specific claims |
| 14 | Robustness suite (noise, JPEG, blur, hair) | ~3 days | Differentiator at TMI / MedIA |
| 15 | Calibration metrics (ECE, Brier) | 1 day | Clinical translation hook |
| 16 | Per-lesion-size-quartile Dice | 1 day | Reveals tiny-lesion failure mode |
| 17 | Qualitative gallery + failure taxonomy | 3 days | Reviewer-friendly |
| 18 | Feature visualisation (FFT magnitude, attention maps) | 2 days | Sells the "frequency-guided" story |

### Tier C — Optional polish (~4 weeks)

| # | Action | Cost | Why |
|---|--------|------|-----|
| 19 | Dermatologist agreement study | ~6 weeks (logistics) | Big win for MedIA / TMI |
| 20 | Downstream classification (predicted mask → melanoma AUC) | ~1 week | Clinical relevance |
| 21 | Confound study: all baselines re-trained with SAM2 backbone | ~2 weeks compute | Refutes "all gain from backbone" critique |
| 22 | Adversarial robustness (FGSM, PGD-5) | ~3 days | Trend at TMI 2024+ |

---

## 6. Effort vs. Venue Matrix

| Venue | Min tiers needed | Approx total effort |
|-------|------------------|---------------------|
| Q3 (IEEE Access etc.) | S | ~1–2 weeks |
| Q2 (CMPB, BSPC, PRL) | S + A | ~6–8 weeks |
| Q1 (MedIA, TMI, IEEE TIP) | S + A + B | ~4–6 months |
| Q1 + clinical (MedIA) | S + A + B + C19/C20 | ~6–9 months |

Tier S alone is essentially free and turns the manuscript from "desk-reject material" into a defensible Q3 submission. Tier A is the highest ROI for upgrading to Q2. Tier B unlocks Q1 if the methodology gaps in `ARCHITECTURE_AUDIT.md` are also closed.

---

## 7. Compute Budget Estimate (Tier S + A + minimum B)

| Item | Runs | GPU-hours (RTX 3090 / A6000) |
|------|----:|----------------------------:|
| Re-run 7 models × 3 datasets × 3 seeds (Tier S) | 63 | ~170 |
| nnU-Net (auto-tuned) | 3 datasets × 1 seed | ~30 |
| Full ablation (Tier A item 9) | 13 × 3 × 3 = 117 | ~315 |
| Swin-Unet + TransFuse + FAT-Net + SAMed (Tier A 10–11) | 4 × 3 × 3 = 36 | ~100 |
| Cross-dataset PH2 test (Tier A 12) | 7 inference runs only | ~2 |
| Robustness suite (Tier B 14) | 6 corruptions × 3 datasets × 1 seed | ~10 (inference-only) |
| **Total** | **~225 runs** | **≈ 630 GPU-hours** |

On 4× A6000 → ~7 days wall-clock. On single 3090 → ~26 days.

---

## 8. Bottom Line

| Dimension | Current | Q1-ready |
|-----------|--------|---------|
| Datasets used | 3 ISIC | 3 ISIC + 1–2 external (PH2 / Dermofit) |
| Baselines | 6 (4 from ≤ 2018) | 11–12 incl. ≥ 3 from 2022+ |
| Seeds | 1 | 3 |
| Statistical test | none | paired Wilcoxon + 95 % CI |
| Metrics | Dice, IoU, Acc, Sens, Spec | + HD95, ASSD, NSD, edge-F1, calibration |
| Reported split | val | test |
| Ablation | none | full 13-row + pair-wise (`ABLATION_PLAN.md`) |
| Efficiency | none | params, FLOPs, latency, memory |
| Robustness | none | noise/JPEG/blur/hair |
| External validation | none | ≥ 1 cross-dataset zero-shot |
| Reproducibility | 1.4 / 5 (`REPRODUCIBILITY_CHECKLIST.md`) | ≥ 4 / 5 |
| Methodology gaps | high-pass + iFFT missing | implemented and ablated |

**Single line of advice:** before adding any baseline, implement the missing high-pass + iFFT path. Otherwise every new comparison is fought from a fundamentally weak methods position, and reviewers at any Q2/Q1 venue will note that the headline freq-domain claim is unsupported by code (see `ARCHITECTURE_AUDIT.md` §4.1–4.2).

---

## Evidence References

- `results/2026-04-28_15-19-03/summary.csv` — current 6-baseline × 3-dataset table (val only)
- `latest_experiment_report.md` §7 — val-not-test caveat
- `ARCHITECTURE_AUDIT.md` — methodology gaps (high-pass, iFFT)
- `NOVELTY_AUDIT.md` §5 — SAM2 confound rationale
- `ABLATION_PLAN.md` — full ablation design (referenced here)
- `REPRODUCIBILITY_CHECKLIST.md` — 1.4 / 5 baseline
- `COMPUTATIONAL_ANALYSIS.md` — params/FLOPs source
