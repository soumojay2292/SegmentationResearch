# Ablation Plan — Final MAFFNet Framework

**Scope:** systematic ablation of the seven declared components.
Final architecture assumed:
`Input → SAM2 Encoder → Stabilized FFT → High-Pass Filter → Inverse FFT → Feature Alignment → Gated Fusion → Decoder → Output → Sobel Edge → Boundary Supervision`.

Baselines used: 3 ISIC datasets (2016/2017/2018) per `dataset_split/`. Training budget: 20 epochs, batch 2, lr 1e-3, AdamW, AMP, img 384², seed = 3 fixed seeds {0, 1, 2} for mean ± std. Same trainer for all runs.

---

## 0. Master Protocol

| Setting | Value |
|---------|-------|
| Datasets | ISIC 2016 / 2017 / 2018, train/val/test splits frozen |
| Epochs | 20 (same as `results/2026-04-28_15-19-03`) |
| Seeds | 3 (0, 1, 2) → report mean ± std |
| Optim | AdamW, lr 1e-3, wd 0.01, LinearLR 1.0→0.0 |
| AMP | autocast fp16 + GradScaler |
| Metrics | Dice, mIoU, Acc, Sensitivity, Specificity, F1, HD95 (boundary), 95% bootstrap CI |
| Significance test | Paired Wilcoxon signed-rank vs. **Full** model, per dataset |
| Compute report | Params (M), FLOPs (G) @ 384², train-time/epoch (s), inference latency (ms) |
| Report split | **Test set** (not val) — fix metric leakage flagged in prior reports |

Full model = all 7 components on. Each ablation toggles one knob, keeps all others = full.

---

## 1. Component-Level Ablations

### A1. SAM2 Encoder
**Variants**
- `A1a` Replace SAM2 → ImageNet-pretrained ResNet-50 (channels [256,512,1024,2048] → 1×1 proj to 256)
- `A1b` Replace SAM2 → randomly-initialised UNet encoder ([64,128,256,512])
- `A1c` Keep SAM2 but **unfreeze last 2 stages** (LoRA-style fine-tune, lr 1e-5)

**Why remove:** isolate contribution of foundation-model transfer. Suspected to dominate the gain (see `NOVELTY_AUDIT.md` §5). Without this isolation the paper claim "MMPA/FGFF help" is confounded.

**Setup:** identical decoder, identical heads, identical loss. Adjust input projection only.

**Expected obs:**
- A1a: −4 to −7 pp Dice on ISIC 2016 (small data, biggest transfer benefit). Smaller gap (~−2 pp) on ISIC 2018 (more data).
- A1b: −8 to −15 pp Dice. Confirms pretraining matters at all.
- A1c: +0.5 to +1.5 pp Dice **or** training instability if backbone unfreezes too aggressively. Demonstrates frozen vs. tunable trade-off.

**Metrics affected:** Dice ↓↓, mIoU ↓↓, sensitivity ↓ (boundary capture), train-time/epoch ↑ for A1c.

---

### A2. Stabilized FFT
**Variants**
- `A2a` Remove `.float()` cast — run FFT in fp16 (likely cuFFT crash → expected NaN baseline)
- `A2b` Replace min-max normalisation with log-magnitude `log(1 + |X|)`
- `A2c` Remove per-channel normalisation entirely (use raw `|X|`)

**Why remove:** verify "stabilization" is necessary, not cosmetic. Currently the only "stabilization" is fp32 upcast — needs proof.

**Setup:** identical except FGFF's pre-conv normalisation block.

**Expected obs:**
- A2a: training fails or runs degrade to NaN within 1–2 epochs (cuFFT does not support fp16 well). Documents the necessity of fp32.
- A2b: ±0.3 pp Dice. Log-mag often comparable. If log-mag wins by > 1 pp, switch the default.
- A2c: −1 to −3 pp Dice. Unnormalised magnitudes have huge dynamic range → SE/conv saturates.

**Metrics affected:** Dice (small effect for A2b/A2c), training stability (A2a), val_loss curve smoothness.

---

### A3. High-Pass Filtering
**Variants**
- `A3a` Remove high-pass mask → use full spectrum (current code state, see `ARCHITECTURE_AUDIT.md` §4.1)
- `A3b` Replace with **low-pass** mask (sanity-check: should hurt)
- `A3c` Sweep radial cutoff ∈ {0.05, 0.1, 0.2, 0.3} of Nyquist
- `A3d` Replace radial mask with **learned** spectral mask (1×1 conv on (H, W/2+1) magnitude)

**Why remove:** high-pass is the central frequency-domain claim. Without ablation, reviewers will not believe lesion-boundary frequency content matters.

**Setup:** apply mask `M(u,v)` after `rfft2`, before any further processing, then iFFT (A3 depends on A4 being on). For A3a, `M=1` everywhere.

**Expected obs:**
- A3a: −1 to −2 pp Dice, +20–30% HD95 (boundary metric worsens most). This is the **headline ablation**.
- A3b: −3 to −5 pp Dice. Low-pass kills edges → confirms direction of effect.
- A3c: best cutoff likely 0.1–0.2 Nyquist. Sub-0.05 keeps DC → no effect; >0.3 loses lesion body.
- A3d: matches or beats fixed-cutoff by ≤ 1 pp; adds params; main novelty hook if A3a fails to show big drop.

**Metrics affected:** Dice ↓, HD95 ↑↑ (boundary), sensitivity at boundary pixels, F1 on edge-pixel subset (compute by morphological erosion of GT).

---

### A4. Inverse FFT
**Variants**
- `A4a` Remove iFFT — use `|filtered_spectrum|` directly as 2D feature (current code state)
- `A4b` Use `irfft2` but discard imaginary part of input (zero-phase filter)
- `A4c` Use real-spectrum reconstruction `irfft2(re·M)` (drop imaginary entirely upstream)

**Why remove:** justify spatial-domain reconstruction. Without iFFT, "frequency-domain" claim is vacuous (see `ARCHITECTURE_AUDIT.md` §4.2).

**Setup:** swap `mag = |X|` (current) for `x_rec = irfft2(X * M)`. Downstream conv unchanged.

**Expected obs:**
- A4a: −2 to −4 pp Dice. Largest single-component drop expected here. Justifies the iFFT path.
- A4b: −1 to −2 pp Dice. Phase information matters more than expected in dermoscopy textures.
- A4c: ≤ A4b. Sanity check.

**Metrics affected:** Dice, mIoU, **HD95 most**, sensitivity at fine-structure regions.

---

### A5. Feature Alignment
**Variants**
- `A5a` Replace alignment with naive bilinear interpolation (current code state)
- `A5b` Add learned deformable conv align (Dai et al. DCN v2)
- `A5c` Cross-scale attention alignment (Q from finer level, K/V from coarser)

**Why remove:** alignment claim needs proof beyond `F.interpolate`. Otherwise reviewer asks "why call it alignment".

**Setup:** insert align module between encoder and MMPA at every level fusion point.

**Expected obs:**
- A5a: −1 to −2 pp Dice. Bilinear is decent for ISIC's smooth lesions; small effect expected.
- A5b: +0.5 to +1.5 pp Dice. DCN adds ~0.3M params, +5–10% train time.
- A5c: +1 to +2 pp Dice; significantly more params/FLOPs.

**Metrics affected:** Dice (small), edge mIoU (medium), HD95 (small).

---

### A6. Gated Fusion
**Variants**
- `A6a` Replace gate with plain concat + 1×1 conv (current code state)
- `A6b` Sum fusion (`spat + freq`)
- `A6c` Sigmoid-gated fusion: `out = σ(g)·spat + (1−σ(g))·freq` with `g = Conv(cat[spat,freq])`
- `A6d` Cross-attention fusion (spatial as Q, freq as K/V)

**Why remove:** justify "gated" name. Currently no gate exists (see `NOVELTY_AUDIT.md`).

**Setup:** swap `self.fuse` in FGFF. Keep residual.

**Expected obs:**
- A6a: −0.5 to −1.5 pp Dice vs. full sigmoid-gated. Reviewer-targeted ablation.
- A6b: −1 to −2 pp Dice. Sum prevents the model from suppressing noisy freq channels.
- A6c: best of fixed-form gates.
- A6d: ≤ A6c by ≤ 1 pp; +params/FLOPs.

**Metrics affected:** Dice (modest), specificity (gate suppresses false-positive freq channels → fewer FP).

---

### A7. Boundary Supervision
**Variants**
- `A7a` Remove boundary loss entirely (`λ_bnd = 0`)
- `A7b` Always-on boundary loss from epoch 1 (no warmup)
- `A7c` Weight sweep `λ ∈ {0.001, 0.01, 0.05, 0.1, 0.5}` (current default 0.01)
- `A7d` Replace Sobel with **Laplacian** edge operator
- `A7e` Replace differentiable boundary loss with **BoundaryLoss** (Kervadec 2019, level-set form)

**Why remove:** current λ=0.01 + epoch≥10 gate makes contribution near zero (CSV evidence: `train_boundary_loss ≈ 0.0433` flat). If removing has no effect, the loss is dead weight and must be dropped or re-tuned.

**Setup:** modify `MAFFNetLoss.forward`. Keep seg losses unchanged.

**Expected obs:**
- A7a: ±0.3 pp Dice **at current λ=0.01**. Negative result is the publishable finding — current weight is too small.
- A7b: +0.2 to +0.5 pp Dice. Warmup is unnecessary.
- A7c: peak likely at λ = 0.05–0.1. λ ≥ 0.5 degrades Dice (boundary fights region loss).
- A7d: ≈ Sobel within noise; Laplacian sometimes slightly worse on noisy dermoscopy.
- A7e: +0.5 to +1.5 pp HD95 improvement, often −0.2 to −0.5 pp Dice trade-off.

**Metrics affected:** **HD95 most**, edge-pixel F1, sensitivity on small lesions; overall Dice modest.

---

## 2. Pair-Wise / Combinatorial Ablations (Critical for Q1/Q2)

Required to disentangle component interactions.

| Variant | Components active |
|---------|------------------|
| **Baseline** | SAM2 encoder + plain U-Net decoder + Dice+BCE only |
| `+MMPA` | Baseline + MMPA |
| `+FGFF` | Baseline + FGFF (full FFT path) |
| `+Bnd` | Baseline + Boundary loss |
| `+MMPA +FGFF` | both spatial + freq enhancements |
| `+MMPA +Bnd` | |
| `+FGFF +Bnd` | |
| **Full** | all components on |

Use this 2³ design to compute main effects + pairwise interactions (ANOVA-style on Dice).

---

## 3. Publication-Ready Ablation Table Template

Copy verbatim into the paper. Fill cells from runs.

### Table 3. Component ablation on ISIC 2016 / 2017 / 2018 test sets. Reported as mean ± std over 3 seeds. Bold = best per column. † = significantly different from Full (p < 0.05, paired Wilcoxon).

| # | Variant | SAM2 | sFFT | HP | iFFT | Align | Gate | Bnd | Params (M) | FLOPs (G) | ISIC 2016 Dice ↑ | ISIC 2017 Dice ↑ | ISIC 2018 Dice ↑ | Mean Dice ↑ | Mean mIoU ↑ | Mean HD95 ↓ |
|--:|--------|:----:|:----:|:--:|:----:|:-----:|:----:|:---:|----------:|---------:|---------------:|---------------:|---------------:|------------:|------------:|------------:|
| 0 | **Full (proposed)** | ● | ● | ● | ● | ● | ● | ● | ___ | ___ | **__.____ ± __.__** | **__.____ ± __.__** | **__.____ ± __.__** | **__.____** | **__.____** | **__.__** |
| 1 | w/o SAM2 (ResNet-50) | ○ | ● | ● | ● | ● | ● | ● | ___ | ___ | __.__ ± __.__† | __.__ ± __.__† | __.__ ± __.__† | __.__ | __.__ | __.__ |
| 2 | w/o SAM2 (random init) | ○ | ● | ● | ● | ● | ● | ● | ___ | ___ | __.__† | __.__† | __.__† | __.__ | __.__ | __.__ |
| 3 | SAM2 unfrozen (last 2) | ◐ | ● | ● | ● | ● | ● | ● | ___ | ___ | __.__ | __.__ | __.__ | __.__ | __.__ | __.__ |
| 4 | w/o sFFT stabilization | ● | ○ | ● | ● | ● | ● | ● | ___ | ___ | __.__ | __.__ | __.__ | __.__ | __.__ | __.__ |
| 5 | w/o High-Pass | ● | ● | ○ | ● | ● | ● | ● | ___ | ___ | __.__† | __.__† | __.__† | __.__ | __.__ | __.__† |
| 6 | Low-Pass (sanity) | ● | ● | LP | ● | ● | ● | ● | ___ | ___ | __.__† | __.__† | __.__† | __.__ | __.__ | __.__† |
| 7 | w/o iFFT (magnitude only) | ● | ● | ● | ○ | ● | ● | ● | ___ | ___ | __.__† | __.__† | __.__† | __.__ | __.__ | __.__† |
| 8 | w/o Feature Alignment | ● | ● | ● | ● | ○ | ● | ● | ___ | ___ | __.__ | __.__ | __.__ | __.__ | __.__ | __.__ |
| 9 | w/o Gated Fusion (concat) | ● | ● | ● | ● | ● | ○ | ● | ___ | ___ | __.__† | __.__† | __.__† | __.__ | __.__ | __.__ |
| 10 | Sum Fusion | ● | ● | ● | ● | ● | SUM | ● | ___ | ___ | __.__ | __.__ | __.__ | __.__ | __.__ | __.__ |
| 11 | w/o Boundary Supervision | ● | ● | ● | ● | ● | ● | ○ | ___ | ___ | __.__ | __.__ | __.__ | __.__ | __.__ | __.__† |
| 12 | Boundary always-on | ● | ● | ● | ● | ● | ● | A1 | ___ | ___ | __.__ | __.__ | __.__ | __.__ | __.__ | __.__ |
| 13 | Boundary λ=0.1 | ● | ● | ● | ● | ● | ● | ×10 | ___ | ___ | __.__ | __.__ | __.__ | __.__ | __.__ | __.__ |

Legend: ● = on, ○ = off, ◐ = partial, LP = low-pass, SUM = sum fusion, A1 = epoch 1 onset.

### Table 4. Pair-wise / cumulative ablation (mean over 3 datasets, 3 seeds).

| # | Variant | Components | Dice ↑ | ΔDice vs. Baseline | HD95 ↓ |
|--:|---------|-----------|-------:|-------------------:|-------:|
| B | Baseline (SAM2 + plain U-Net dec) | — | __.__ | — | __.__ |
| B1 | + MMPA | spatial | __.__ | +__._ | __.__ |
| B2 | + FGFF | freq | __.__ | +__._ | __.__ |
| B3 | + Bnd | edge | __.__ | +__._ | __.__ |
| B4 | + MMPA + FGFF | spatial + freq | __.__ | +__._ | __.__ |
| B5 | + MMPA + Bnd | spatial + edge | __.__ | +__._ | __.__ |
| B6 | + FGFF + Bnd | freq + edge | __.__ | +__._ | __.__ |
| F | Full (B + MMPA + FGFF + Bnd) | all | __.__ | +__._ | __.__ |

### Table 5. Boundary-weight sweep (ISIC 2018 only, 1 seed).

| λ_bnd | 0 | 0.001 | 0.01 (default) | 0.05 | 0.1 | 0.5 |
|-------|--:|------:|---------------:|----:|----:|----:|
| Dice ↑ | | | | | | |
| HD95 ↓ | | | | | | |

---

## 4. Per-Component Metric Sensitivity Matrix (qualitative)

| Component removed | Dice | mIoU | Sens. | Spec. | HD95 | edge-F1 | Params | FLOPs |
|-------------------|:----:|:----:|:-----:|:-----:|:----:|:-------:|:------:|:-----:|
| SAM2 → ResNet-50 | ↓↓↓ | ↓↓↓ | ↓↓ | ↓ | ↑↑ | ↓↓ | ↓ | ↓ |
| sFFT stabilization | NaN / ↓↓ | NaN / ↓↓ | — | — | — | — | = | = |
| High-Pass | ↓ | ↓ | ↓ | = | ↑↑ | ↓↓ | = | = |
| iFFT | ↓↓ | ↓↓ | ↓ | = | ↑↑ | ↓↓ | = | ↓ |
| Feature Alignment | ↓ | ↓ | = | = | ↑ | ↓ | ↓ | ↓ |
| Gated Fusion | ↓ | ↓ | = | ↓ | = | = | ↓ | ↓ |
| Boundary Sup. | ≈ | ≈ | ↓ | = | ↑↑ | ↓↓ | = | = |

`↓↓↓` = > 5 pp drop expected, `↓↓` = 2–5 pp, `↓` = 0.5–2 pp, `≈` = < 0.5 pp, `↑` = worse, `=` = unchanged.

---

## 5. Reporting Checklist

Before submission, verify:

- [ ] All 13 ablations × 3 datasets × 3 seeds = 117 training runs logged
- [ ] Each row has Params/FLOPs filled (use `fvcore` or `thop`)
- [ ] Significance markers (`†`) applied per paired Wilcoxon vs. Full
- [ ] HD95 computed via `medpy.metric.hd95` on binary masks at threshold 0.5
- [ ] Edge-F1 computed on GT mask boundary band (5-pixel morphological band)
- [ ] All hyperparameters except the ablated one held constant
- [ ] Seeds logged in `experiments/.../config.txt`
- [ ] Test (not val) split used for final numbers

---

## 6. Compute Budget Estimate

Single 20-epoch run on RTX 3090 / A6000 (batch 2, img 384²):

| Backbone | Approx min/epoch | Total/run (20 ep) | × 13 ablations × 3 datasets × 3 seeds |
|----------|----------------:|------------------:|-------------------------------------:|
| SAM2 frozen | 8 min | 2.7 h | ≈ **315 GPU-hours** |
| ResNet-50 | 4 min | 1.3 h | (subset of above) |

Plan: ~2 weeks single-GPU; ~3 days on 4× A6000.

---

## Evidence References

- `models/maffnet.py:39–113` — current MMPA + FGFF impl (basis for "current code state" baselines)
- `training/trainer.py:97–108` — boundary loss formulation + warmup gate
- `experiments/maffnet_ISIC_2016_2026-04-28_15-19-03/checkpoints/training_log.csv` — flat boundary-loss ≈ 0.0433 evidence
- `ARCHITECTURE_AUDIT.md` §4 — confirmation of missing High-Pass and iFFT in current build
- `NOVELTY_AUDIT.md` §5 — SAM2-confound rationale for A1 ablation
- `latest_experiment_report.md` — current full-model val metrics for sanity-check magnitudes
