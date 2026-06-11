# Novelty Audit — Proposed MAFFNet Framework

**Scope:** assess true novelty of the framework in this repo vs. five published baselines + the prior published "MAFFNet" name collision. Based on actual code (`models/maffnet.py`, `training/trainer.py`) and prior architecture/audit findings (see `ARCHITECTURE_AUDIT.md`).

**Note on naming:** "MAFFNet" already exists in skin-lesion segmentation literature (e.g., Multi-scale Attention Feature Fusion variants, 2022–2024). Reusing the same name is itself a reproducibility/citation risk — flagged here, not novelty.

---

## 1. Component-by-Component Comparison

Legend: ● present, ○ absent, ◐ partial / different.

| Component | U-Net | Att. U-Net | UNet++ | TransUNet | SGT-Net | Prior MAFFNet (lit.) | **This repo** |
|-----------|:-----:|:----------:|:------:|:---------:|:-------:|:---------:|:-------------:|
| Encoder–decoder backbone | ● CNN | ● CNN | ● CNN dense | ● CNN+ViT hybrid | ● CNN+Transformer | ● CNN | **● SAM2 Hiera-L (frozen, pretrained)** |
| Skip connections | ● plain | ● attention-gated | ● dense nested | ● plain | ● spatial-gated | ● multi-scale | ● plain (lvl 1 = zero-skip) |
| Multi-scale dilated conv (ASPP-like) | ○ | ○ | ○ | ○ | ◐ | ● | ● MMPA r=3,5,7 |
| Channel attention (SE-style) | ○ | ◐ (gate, not SE) | ○ | ○ | ◐ | ● | ● SE in MMPA |
| Spatial attention | ○ | ● additive gate | ○ | ● self-attn | ● gated | ◐ | ○ (none explicit) |
| Transformer | ○ | ○ | ○ | ● ViT bottleneck | ● gated transformer | ○ | ○ (SAM2 internal only) |
| Frequency-domain branch | ○ | ○ | ○ | ○ | ○ | ○ | ◐ FFT magnitude only (no high-pass, no iFFT) |
| Gated fusion | ○ | ● (attention) | ○ | ○ | ● (spatial gate) | ◐ | ○ (plain concat+conv) |
| Deep supervision | ○ | ○ | ● | ○ | ○ | ● | ● 4 heads (p1–p4) |
| Boundary / edge loss | ○ | ○ | ○ | ○ | ◐ | ◐ | ● Sobel L1, ε=0.01, epoch≥10 |
| Auxiliary reconstruction head | ○ | ○ | ○ | ○ | ○ | ○ | ● defined, **not used in loss** |
| Pretrained foundation backbone | ○ | ○ | ○ | ● ImageNet ViT | ● ImageNet | ○ | **● SAM2 Hiera-L** |

---

## 2. What Is Genuinely New vs. Recombined

### True novel contribution (this repo's actual delta over prior art):
**Combining a frozen SAM2 Hiera-Large encoder with a multi-scale dilated-conv + magnitude-spectrum gating decoder for skin lesion segmentation, with Sobel-based boundary regularisation activated mid-training.**

That sentence is the maximum honest novelty claim. Decompose:

| Atomic claim | Prior art | Novel? |
|--------------|-----------|--------|
| Use SAM2 as frozen encoder for medical seg | SAM/SAM2 adapter & medical-SAM work 2023–2024 (Med-SAM, SAM-Med2D, SAMed) | **Incremental** — SAM/SAM2 adaptation is a crowded sub-field |
| Multi-scale dilated convolutions at 4 levels | DeepLab v2/v3 ASPP, RFB-Net (2018), prior MAFFNet | **Incremental** — direct ASPP descendant; dilation choice (3,5,7) is hyperparameter, not method |
| SE-style channel attention | Hu et al. SENet 2018 | **Not novel** — textbook block |
| FFT magnitude as feature gate | FcaNet (Qin 2021), GFNet (Rao 2021), spectral attention papers | **Incremental** — magnitude-only spectral attention exists; no high-pass mask + no iFFT means contribution is weaker than published spectral methods |
| Sobel boundary L1 loss | Boundary-aware losses since 2018; Active Contour Loss; BoundaryLoss (Kervadec 2019) | **Not novel** — direct re-use |
| Warmup (epoch ≥ 10) for boundary loss | Curriculum learning, loss annealing | **Not novel** — standard trick |
| Auxiliary reconstruction head | Multi-task learning since FCN era | **Not novel + not wired** (loss ignores `rec`) |
| Deep supervision (4 heads) | Lee 2015 DSN; UNet++ uses it | **Not novel** |

### Net novelty grade: **incremental recombination**, not substantial.

No single block is new. The **combination** (SAM2 + ASPP-ish MMPA + FFT-magnitude FGFF + Sobel boundary loss) has not been published in this exact form to my knowledge, but each pair already exists somewhere.

---

## 3. Novelty by Category

### 3.1 Architectural novelty — **Low / Incremental**
- Backbone: pre-trained SAM2 reuse, no adapter design, no LoRA, fully frozen. Not different from naive feature-extraction baseline.
- MMPA = ASPP variant with rate set (3,5,7) + SE. Decade-old idea with new dilation knob.
- Decoder = vanilla U-Net. Zero-skip at lvl 1 is a quirk, not a method.
- **No new module is proposed end-to-end.**

### 3.2 Training strategy novelty — **Low**
- AdamW, linear LR decay, AMP, batch size 2, pos_weight=2. All standard.
- Two-phase loss (seg only → seg + boundary at epoch 10) = mild curriculum, weight 0.01. From the training_log.csv the boundary loss stays flat ≈ 0.043 — its effect on Dice is below noise.
- No new optimisation scheme, no new sampler, no self-supervised pretext, no semi-supervised use of unlabelled ISIC archive.

### 3.3 Frequency-domain novelty — **Weakest claim**
- Code does: `rfft2 → |·| → resize → minmax-norm → 1×1 conv → concat with spatial → 1×1 fuse → residual`.
- **No high-pass filter.** No radial mask, no DC removal. Spec claims it, code doesn't have it (see `ARCHITECTURE_AUDIT.md` §4.1).
- **No inverse FFT.** Phase discarded. Magnitude treated as 2D image, then spatial conv. Not a true frequency-domain operator.
- FcaNet, GFNet, AFNO already use richer spectral attention (DCT bands, learned spectral filters, full complex multiplication, then iFFT). This repo's FGFF is **weaker than 2021 baselines**.
- "Stabilized FFT" = one `.float()` upcast to avoid cuFFT fp16. Not a method.

### 3.4 Boundary optimization novelty — **Incremental**
- Sobel-edge L1 on `sigmoid(logits)` after 3×3 avg-pool. Differentiable boundary loss is standard since 2018 (Kervadec, Active Contour, BoundaryLoss).
- ε = 0.01 weight is so small that gradient contribution is negligible (CSV evidence: post-activation Dice barely moves).
- No learned boundary detector, no boundary-aware attention, no level-set parametrisation.

---

## 4. Substantial vs. Incremental Verdict

| Contribution | Substantial? | Reason |
|--------------|:------------:|--------|
| SAM2 frozen encoder for ISIC | ✗ Incremental | Foundation-model transfer is now standard; no adapter design |
| MMPA (dilated + SE) | ✗ Incremental | ASPP + SE recombination |
| FGFF | ✗ Incremental, **partially broken vs. claim** | Missing high-pass + iFFT; weaker than 2021 spectral baselines |
| Sobel boundary loss with warmup | ✗ Incremental | 2018-era loss + curriculum trick |
| Deep supervision (4 heads) | ✗ Not novel | UNet++ 2018 |
| Pretrained SAM2 + Sobel boundary combination | ◐ Mildly novel | Specific combination not yet published with this exact backbone |
| RDB (reconstruction head) | ✗ Not used | Defined but loss discards it |

**None of the contributions are substantial.** The framework is a coherent engineering recombination, **not** a methodological advance.

---

## 5. Empirical Strength (what could rescue a weak novelty story)

From `results/2026-04-28_15-19-03/summary.csv`:

| Dataset | Proposed Dice | Best baseline Dice (this repo) | Margin |
|---------|--------------:|------------------------------:|-------:|
| ISIC 2016 | 0.9216 | 0.8663 (Att-UNet) | **+5.5 pp** |
| ISIC 2017 | 0.7824 | 0.6630 (Att-UNet) | **+11.9 pp** |
| ISIC 2018 | 0.8796 | 0.8716 (TransUNet) | +0.8 pp |

**Caveats reviewers will hit hard:**
- U-Net baseline failed (NaN); fix was applied post-hoc. Original numbers invalid.
- TransUNet collapsed on ISIC 2017 (Dice 0.27) → not a fair baseline.
- All baselines trained from scratch; **only proposed model uses SAM2 pretraining**. The gain is therefore mostly **backbone effect**, not method.
- No comparison vs. published SOTA on ISIC: e.g., Swin-Unet, TransFuse, FAT-Net, BoundaryAware Transformer, MS-RED. Repo baselines are 2015–2021 era.
- No statistical test, no cross-validation, single split, no inference-time / FLOPs / param count table.
- Reported metric is **val**, not held-out test (see `latest_experiment_report.md` §7).

Strong empirical margin on ISIC 2016/2017 is the **only thing carrying the paper**, and most of it is attributable to SAM2.

---

## 6. Journal Suitability

Assessing in current state, before further work.

### Q3 journals (lower-tier, e.g., *IEEE Access*, *Computers in Biology & Medicine* low-IF tier, regional medical-AI journals)
**Verdict: Acceptable with revisions.** Engineering recombination + 3-dataset benchmark + SAM2 novelty (in skin-lesion sub-niche) is enough. Reviewers at this tier often accept solid empirical work without deep methodological novelty.

**Required before submit:**
- Fix U-Net baseline (done), report test (not val) metrics, add 2–3 recent SOTA baselines (Swin-Unet, TransFuse), add params/FLOPs table, statistical significance.

### Q2 journals (e.g., *Computer Methods and Programs in Biomedicine*, *Biomedical Signal Processing and Control*, *Pattern Recognition Letters*)
**Verdict: Borderline; not yet acceptable.** Needs a real methodological hook.

**Required:**
- Actually implement the high-pass + inverse-FFT path. Currently the "frequency-guided" claim is unsupported by code (see `ARCHITECTURE_AUDIT.md`).
- Add ablation table: backbone-only, +MMPA, +FGFF, +boundary loss. Currently impossible without code flags.
- Add SOTA baselines from 2022–2024 (Swin-Unet, TransFuse, FAT-Net, MS-RED, BAT).
- Cross-validation or at minimum mean ± std over 3 seeds.
- Honest accounting that backbone delta explains most of the gain (or train all baselines with SAM2 backbone to neutralise it).

### Q1 journals (e.g., *Medical Image Analysis*, *IEEE TMI*, *Pattern Recognition*, *IEEE TIP*)
**Verdict: Not acceptable in any version of the current framework.** Reasons:

1. **No methodological novelty.** Q1 reviewers want a new operator, new theory, new loss with proof, or new paradigm. This repo has none.
2. **Frequency-domain claim is empirically vacuous** without iFFT — Q1 reviewer will ask "what does FGFF do that 1×1 conv cannot" and the answer is "almost nothing".
3. **SAM2 use is generic** — no adapter, no prompt-engineering, no domain adaptation theory.
4. **Baselines are too old.** Q1 requires comparison vs. 2023–2024 SOTA on ISIC, including transformer-based and SAM-based competitors. Current baselines are 2015–2021.
5. **Single-domain validation.** ISIC only. Q1 medical imaging usually demands cross-dataset (PH2, Dermofit, Dermquest) generalisation or external clinical test.
6. **No theoretical justification.** No convergence argument, no inductive-bias argument, no information-theoretic argument for FGFF. Q1 expects at least one.

To reach Q1: invent a genuinely new module (e.g., learned spectral gate with iFFT and a proven property; or SAM2 adapter with parameter-efficient fine-tuning specific to dermoscopy; or boundary loss with theoretical guarantee), add ≥ 5 recent SOTA baselines, 4–5 datasets, ablation, statistical test, FLOPs/latency analysis. Effectively a different paper.

---

## 7. Bottom Line

| Item | Grade |
|------|-------|
| Architectural novelty | Low / incremental |
| Training strategy novelty | Low |
| Frequency-domain novelty | Weakest — claim not matched by code |
| Boundary optimization novelty | Incremental |
| Empirical strength | Moderate, but backbone confound |
| Reproducibility (code vs. spec) | High risk (see ARCHITECTURE_AUDIT) |
| Best-fit venue today | **Q3** |
| Reach Q2 with effort | Yes, ~1–2 months work |
| Reach Q1 | No — needs a new paper |

**Strongest realistic story:** "Frozen SAM2 + Sobel boundary regularisation for skin-lesion segmentation; SAM2 transfer accounts for most of the gain; we show how to integrate it cheaply."

**Weakest part to drop or rewrite:** the frequency-domain narrative. Either implement it faithfully (high-pass mask + iFFT + ablation) and re-train, or drop FGFF entirely and rename the network. Keeping the current FGFF + "frequency-guided" claim invites desk reject at Q1/Q2.

---

## Evidence References

- `models/maffnet.py:97–113` — FGFF code (magnitude only, no iFFT, no high-pass)
- `models/maffnet.py:39–76` — MMPA = ASPP+SE
- `training/trainer.py:97–108` — boundary loss epoch-gated, weight 0.01
- `results/2026-04-28_15-19-03/summary.csv` — Dice margins
- `ARCHITECTURE_AUDIT.md` — spec vs. impl gaps (high-pass / iFFT missing)
- `latest_experiment_report.md` — val-not-test metric caveat
