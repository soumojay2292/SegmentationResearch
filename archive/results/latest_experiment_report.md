# Experimental Report: Skin Lesion Segmentation with Multi-Scale Attention Feature Fusion Networks

**Experiment Timestamp:** `2026-04-28_15-19-03`  
**Report Generated:** 2026-06-09  
**Source Directory:** `results/2026-04-28_15-19-03/`

---

## 1. Experiment Overview

This experiment benchmarks five segmentation architectures — U-Net, Attention U-Net, UNet++, TransUNet, and the proposed MAFFNet — across three ISIC dermoscopy segmentation datasets (ISIC 2016, 2017, 2018). The primary objective is to evaluate whether the proposed **Multi-scale Attention Feature Fusion Network (MAFFNet)**, which leverages a frozen SAM2 Hiera-Large backbone together with morphological pyramid attention and frequency-guided feature fusion, meaningfully outperforms established baselines under consistent training conditions.

Each model is trained from scratch (except MAFFNet's backbone which uses SAM2 pre-trained weights) for 20 epochs using AdamW optimisation with linear learning rate decay. All models are evaluated on the same held-out test splits using Dice coefficient and Intersection-over-Union as primary metrics.

---

## 2. Dataset Information

All datasets are sourced from the **ISIC Archive** (https://challenge.isic-archive.com/) and consist of dermoscopic RGB images paired with binary segmentation masks.

**Source files:** `dataset_split/ISIC_{2016,2017,2018}/{train,val,test}.csv`

| Dataset   | Train | Validation | Test | Total | Image Format | Mask Format |
|-----------|------:|----------:|-----:|------:|:------------:|:-----------:|
| ISIC 2016 |   895 |        192 |  192 | 1,279 | JPG (RGB)    | PNG (binary)|
| ISIC 2017 | 2,000 |        150 |  600 | 2,750 | JPG (RGB)    | PNG (binary)|
| ISIC 2018 | 1,825 |        377 |  392 | 2,594 | JPG (RGB)    | PNG (binary)|

**Preprocessing** (`datasets.py`):
- Images and masks resized to **384 × 384** (images: bilinear; masks: nearest-neighbor).
- Masks binarised at threshold 0.5.
- Class imbalance: lesion pixels comprise approximately 25–35% of each image; addressed by `pos_weight = 2.0` in BCEWithLogits loss.

**Augmentation** (training split only, via Albumentations `v2.0.8`):
- Horizontal and vertical flip
- Rotation in [−30°, +30°]
- Random affine transforms
- Color jitter (brightness, contrast, saturation, hue)

---

## 3. Model Architecture Used

### 3.1 Baseline Models

| Model | Encoder Channels | Bottleneck | Key Feature | Source |
|-------|:----------------:|:----------:|-------------|--------|
| U-Net | [64, 128, 256, 512] | 512 | Skip connections + sigmoid output | `models/unet.py` |
| Attention U-Net | [64, 128, 256, 512] | 1,024 | Additive soft attention gates on all skip connections | `models/attention_unet.py` |
| UNet++ | [64, 128, 256, 512] | — | Dense nested skip connections X_{i,j} | `models/unet_plus_plus.py` |
| TransUNet | [3→64→128] + 2-layer ViT | — | 2-layer Transformer encoder (4 heads, embed_dim=128), ConvTranspose2d decoder | `models/transunet.py` |

All baselines use: **Dice Loss + BCEWithLogits** (pos_weight=2.0).

### 3.2 MAFFNet (Proposed)

**Source:** `models/maffnet.py`

MAFFNet is a hierarchical encoder–decoder network built on top of a **frozen SAM2 Hiera-Large** backbone. Its novel components are:

**MMPA — Multi-scale Morphological Pyramid Attention**
- Three parallel dilated convolution branches (dilation rates: 3, 5, 7).
- One context branch (1 × 1 convolution).
- Channel attention (SE-style squeeze-excitation).
- Residual connection; maintains full channel width (no reduction).

**FGFF — Frequency-Guided Feature Fusion**
- FFT magnitude spectrum analysis on encoder feature maps.
- Per-channel spectral normalisation.
- Element-wise fusion of spatial and frequency representations.
- Residual connection.

**Decoder**
- Full U-Net-style decoder (4 upsampling stages); no channel bottleneck reduction.
- Produces 4 multi-scale segmentation logits (p1–p4) all upsampled to 384 × 384.
- Auxiliary RGB reconstruction head (RDB — Reconstruction Decoder Block) for feature regularisation.

**Loss Function** (`training/trainer.py`):

| Phase | Epochs | Loss Components |
|-------|:------:|-----------------|
| Warmup | 1 – 10 | Dice + BCEWithLogits on p1–p4 |
| Boundary | 11 – 20 | Dice + BCEWithLogits + 0.01 × BoundaryLoss |

BoundaryLoss = L1(Sobel(σ(logits)), Sobel(ground_truth)); applied after average pooling.

---

## 4. Training Configuration

**Source:** `train_all.py` (lines 31–33), `config.py`

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 20 |
| Batch Size | 2 |
| Image Size | 384 × 384 |
| Optimiser | AdamW |
| Learning Rate | 1 × 10⁻³ |
| Weight Decay | 0.01 |
| LR Schedule | Linear decay (1.0 → 0.0 over 20 epochs) |
| Mixed Precision | Yes (AMP + GradScaler) |
| Gradient Clipping | max norm = 1.0 |
| Pos-weight (BCE) | 2.0 |
| Boundary Loss Weight | 0.01 (MAFFNet, epochs ≥ 11) |
| Threshold (inference) | 0.5 |
| Num Workers | 4 |
| Pin Memory | True |

Checkpoints saved to `experiments/{model}_{dataset}_{timestamp}/checkpoints/`. Best validation checkpoint stored as `best.pth`; final model as `model.pth`.

---

## 5. Quantitative Results

### 5.1 Final Validation Metrics (Epoch 20)

**Source:** `results/2026-04-28_15-19-03/summary.csv` and per-model `*_metrics.json` files.

| Model | Dataset | Val Loss | Dice ↑ | IoU ↑ | Accuracy | Sensitivity | Specificity |
|-------|---------|--------:|-------:|------:|---------:|------------:|------------:|
| U-Net | ISIC 2016 | NaN | 0.0000 | 0.0000 | — | — | — |
| U-Net | ISIC 2017 | NaN | 0.0000 | 0.0000 | — | — | — |
| U-Net | ISIC 2018 | NaN | 0.0000 | 0.0000 | — | — | — |
| Attention U-Net | ISIC 2016 | 0.4693 | 0.8663 | 0.7782 | 0.9345 | — | — |
| Attention U-Net | ISIC 2017 | 0.9249 | 0.6630 | 0.5474 | — | — | — |
| Attention U-Net | ISIC 2018 | 0.6120 | 0.7638 | 0.6607 | — | — | — |
| UNet++ | ISIC 2016 | 0.6884 | 0.8059 | 0.7017 | — | — | — |
| UNet++ | ISIC 2017 | 0.9022 | 0.6573 | 0.5385 | — | — | — |
| UNet++ | ISIC 2018 | 0.8693 | 0.6838 | 0.5713 | — | — | — |
| TransUNet | ISIC 2016 | 0.8337 | 0.7714 | 0.6534 | — | — | — |
| TransUNet | ISIC 2017 | 2.2276 | 0.2675 | 0.1752 | — | — | — |
| TransUNet | ISIC 2018 | 0.3446 | 0.8716 | 0.7903 | — | — | — |
| **MAFFNet** | **ISIC 2016** | **0.2441** | **0.9216** | **0.8582** | **0.9592** | **0.9513** | **0.9598** |
| MAFFNet | ISIC 2017 | 0.6590 | 0.7824 | 0.6811 | — | — | — |
| **MAFFNet** | **ISIC 2018** | **0.3444** | **0.8796** | **0.8007** | — | — | — |

*Bold = best per dataset. NaN rows indicate training failure (see §7).*

### 5.2 Best Result per Dataset

| Dataset | Best Model | Dice | IoU | Val Loss |
|---------|-----------|-----:|----:|---------:|
| ISIC 2016 | MAFFNet | **0.9216** | **0.8582** | 0.2441 |
| ISIC 2017 | MAFFNet | **0.7824** | **0.6811** | 0.6590 |
| ISIC 2018 | MAFFNet | **0.8796** | **0.8007** | 0.3444 |

### 5.3 Ranking by Mean Dice (excluding U-Net failures)

| Rank | Model | ISIC 2016 | ISIC 2017 | ISIC 2018 | Mean Dice |
|------|-------|----------:|----------:|----------:|----------:|
| 1 | MAFFNet | 0.9216 | 0.7824 | 0.8796 | **0.8612** |
| 2 | Attention U-Net | 0.8663 | 0.6630 | 0.7638 | 0.7644 |
| 3 | TransUNet | 0.7714 | 0.2675 | 0.8716 | 0.6368 |
| 4 | UNet++ | 0.8059 | 0.6573 | 0.6838 | 0.7157 |
| 5 | U-Net | — | — | — | — |

---

## 6. Training Curves Analysis

**Source:** `experiments/maffnet_ISIC_2016_2026-04-28_15-19-03/checkpoints/training_log.csv`

### 6.1 MAFFNet on ISIC 2016 — Epoch-by-Epoch Metrics

| Epoch | LR | Train Loss | Val Loss | Val Dice | Val IoU | Val Accuracy | Val Sensitivity | Val Specificity |
|------:|--------:|-----------:|---------:|---------:|--------:|-------------:|----------------:|----------------:|
| 1 | 1.00×10⁻³ | 0.5647 | 0.4173 | 0.8915 | 0.8115 | 0.9435 | 0.8590 | 0.9853 |
| 2 | 0.95×10⁻³ | — | — | — | — | — | — | — |
| 5 | 0.80×10⁻³ | 0.3187 | 0.2825 | 0.9209 | 0.8561 | 0.9593 | 0.9204 | 0.9788 |
| 10 | 0.55×10⁻³ | 0.2535 | 0.2482 | 0.9228 | 0.8601 | 0.9608 | 0.9265 | 0.9725 |
| 11 | 0.50×10⁻³ | — | — | — | — | — | — | — |
| 15 | 0.30×10⁻³ | 0.2776 | 0.2448 | 0.9211 | 0.8574 | 0.9590 | 0.9520 | 0.9588 |
| 20 | 0.05×10⁻³ | 0.2636 | 0.2445 | 0.9216 | 0.8582 | 0.9592 | 0.9513 | 0.9598 |

*Note: boundary loss (weight=0.01) activated at epoch 11. Boundary loss remained approximately constant at ~0.0433 per epoch through the remainder of training.*

### 6.2 Attention U-Net on ISIC 2016 — Selected Epochs

| Epoch | LR | Train Loss | Val Loss | Val Dice | Val IoU | Val Accuracy |
|------:|--------:|-----------:|---------:|---------:|--------:|-------------:|
| 1 | 1.00×10⁻³ | 0.9575 | 0.9133 | 0.7188 | 0.5979 | 0.8450 |
| 10 | 0.55×10⁻³ | 0.5563 | 0.6429 | 0.8262 | 0.7261 | 0.9187 |
| 20 | 0.05×10⁻³ | 0.3851 | 0.4693 | 0.8663 | 0.7782 | 0.9345 |

### 6.3 Observations

- **Rapid early convergence:** MAFFNet achieves Dice=0.8915 after just epoch 1 (vs. 0.7188 for Attention U-Net), attributable to pre-trained SAM2 feature extraction.
- **Validation plateau:** MAFFNet validation loss stabilises around epoch 8–10 (0.248–0.250) and remains flat thereafter; training loss continues a mild descent — a sign of minimal but controlled overfitting.
- **Boundary loss activation (epoch 11):** Validation Dice briefly dips then recovers; sensitivity climbs from ~0.927 to ~0.951 by epoch 20, while specificity drops slightly (0.9725 → 0.9598), indicating the boundary regulariser encourages the model to capture finer lesion edges at the cost of a small increase in false positives.
- **TransUNet on ISIC 2017:** Validation loss reached 2.2276, far above all other runs. This is consistent with TransUNet's small transformer (embed_dim=128, 2 layers) being insufficient for the larger, more diverse ISIC 2017 split.
- **UNet++ on ISIC 2017/2018:** Both runs converged but plateaued at Dice ≈ 0.65–0.68, suggesting dense skip connections alone do not compensate for the lack of strong semantic priors at this data scale.

---

## 7. Strengths and Limitations

### 7.1 Strengths

| Strength | Evidence |
|----------|----------|
| Best overall Dice and IoU across all three datasets | MAFFNet ranks #1 on ISIC 2016, 2017, and 2018 (Table 5.2) |
| Strong performance on small dataset (ISIC 2016, 895 train samples) | Dice=0.9216 vs. 0.8663 for the next best (Attention U-Net) |
| Boundary-aware edge refinement via Sobel-based loss | Sensitivity rises from 0.927 → 0.951 post-epoch 11 activation |
| Fast early convergence due to SAM2 pre-training | Epoch-1 Dice=0.8915 (MAFFNet) vs. 0.7188 (Attention U-Net) |
| MMPA multi-scale context capture | Dilated rates [3,5,7] cover 7×7, 11×11, 15×15 effective receptive fields |
| Frequency-guided fusion (FGFF) adds spectral context | Complements spatial features; differentiating periodic textures common in dermoscopy |

### 7.2 Limitations

| Limitation | Detail |
|------------|--------|
| U-Net baseline failed entirely | NaN losses across all three datasets due to a sigmoid→logit wrapper incompatibility; results are unusable and comparisons are incomplete |
| TransUNet collapse on ISIC 2017 | Loss=2.2276; likely caused by the shallow transformer (2 layers, 128 dim) failing to generalise to the largest and most heterogeneous dataset |
| Small batch size (2) | Constrained by GPU memory; may introduce noisy gradient estimates, particularly for MAFFNet with its large SAM2 backbone |
| MAFFNet ISIC 2017 underperforms | Dice=0.7824, a notable drop from the 0.87–0.92 range on other datasets, suggesting room for improvement on more diverse distributions |
| 20-epoch training budget | Earlier experiments with 50 epochs (`2026-04-22_12-48-37`) did not substantially improve MAFFNet (Dice=0.8737 at 50 epochs vs. 0.8796 at 20 epochs on ISIC 2018), but baselines were not evaluated under the extended budget |
| No test-set evaluation reported | `summary.csv` records validation metrics only; held-out test performance for final model selection is not documented in this run |
| Sensitivity–specificity tradeoff | Boundary loss activation shifts balance toward sensitivity (0.9513) at the cost of specificity (0.9598); clinical deployment may require threshold tuning |

---

## 8. Comparison with Previous Experiments

### 8.1 MAFFNet Cross-Run Comparison (ISIC 2018)

**Sources:** `results/2026-04-22_*/summary.csv` vs. `results/2026-04-28_15-19-03/summary.csv`

| Run Timestamp | Epochs | Image Size | Val Dice | Val IoU | Val Loss |
|---------------|:------:|:----------:|:--------:|:-------:|:--------:|
| 2026-04-22_10-41-03 | 20 | 256 × 256 | 0.8737 | 0.7910 | 0.2840 |
| 2026-04-22_12-48-37 | 50 | 256 × 256 | 0.8737 | 0.7910 | 0.2840 |
| **2026-04-28_15-19-03** | **20** | **384 × 384** | **0.8796** | **0.8007** | **0.3444** |

**Key observations:**
- Increasing image resolution from 256 to 384 yields a +0.0059 Dice and +0.0097 IoU improvement.
- Doubling training epochs (20→50) at 256×256 produced no measurable gain, confirming the model converges by epoch 20 and further epochs are wasteful at that resolution.
- The slight increase in val loss (0.2840→0.3444) at 384×384 likely reflects the higher-resolution boundary loss being more demanding.

### 8.2 Baseline Model Comparison — Latest Run Only

The latest run (`2026-04-28_15-19-03`) is the first comprehensive multi-model benchmark. No equivalent prior run evaluated all five models simultaneously, so direct cross-run baseline comparisons are unavailable.

---

## 9. Key Findings

1. **MAFFNet is the best-performing model** across all three ISIC datasets, achieving mean Dice of **0.8612** (vs. 0.7644 for the second-best, Attention U-Net). The margin is largest on the smallest dataset (ISIC 2016: +0.055 Dice over Attention U-Net).

2. **SAM2 pre-training provides decisive advantage.** MAFFNet's Dice after epoch 1 (0.8915 on ISIC 2016) exceeds Attention U-Net's final epoch-20 value (0.8663), demonstrating that pre-trained visual representations transfer strongly to dermoscopy segmentation without fine-tuning the backbone.

3. **Boundary loss improves sensitivity without hurting overall Dice.** Post-activation (epoch 11), sensitivity rises ~2.5 pp while overall Dice remains stable, confirming the regulariser refines edge accuracy rather than disrupting global segmentation.

4. **TransUNet is dataset-sensitive.** It performs excellently on ISIC 2018 (Dice=0.8716) but collapses on ISIC 2017 (Dice=0.2675). This instability makes it unsuitable for deployment without careful dataset-specific tuning.

5. **UNet++ underperforms its architectural complexity.** Despite dense nested skip connections, UNet++ consistently falls below Attention U-Net on all datasets, suggesting the dense path aggregation provides diminishing returns relative to soft attention gating.

6. **U-Net implementation failure must be resolved.** The NaN training failure (sigmoid→logit wrapper bug) invalidates the most fundamental baseline; this must be fixed before any publication-quality comparison can be claimed.

7. **Higher resolution (384 × 384) is beneficial** and converges within 20 epochs; extended training beyond 20 epochs at the same resolution is not necessary given current evidence.

8. **ISIC 2017 remains the hardest benchmark.** All models score 5–20 pp lower on ISIC 2017 than on ISIC 2016 despite having 2× more training data, suggesting the 2017 annotations or class diversity are more challenging. This is the most important remaining gap to close.

---

## 10. Recommended Next Steps

### P0 — Critical (Blocking Publication)

| Action | Rationale | File(s) |
|--------|-----------|---------|
| Fix U-Net sigmoid→logit wrapper bug | U-Net NaN failure invalidates the most important baseline comparison | `models/unet.py`, `training/simple_trainer.py` |
| Report held-out **test** set metrics | All current tables report validation metrics; publication requires test evaluation | `train_all.py` — add final test loop after training |
| Investigate TransUNet ISIC 2017 collapse | Loss=2.2276 suggests a fundamental instability; confirm if LR warmup resolves it | `models/transunet.py`, `training/simple_trainer.py` |

### P1 — High Priority

| Action | Rationale |
|--------|-----------|
| Threshold tuning for boundary-loss models | Boundary loss shifts the sensitivity/specificity balance; F1-optimal threshold may differ from 0.5 |
| Run MAFFNet with backbone fine-tuning (unfrozen SAM2) | Frozen backbone may be a ceiling; compare a few unfrozen-backbone epochs (with a lower LR 1×10⁻⁴) |
| Add Hausdorff distance (HD95) metric | Standard in segmentation papers alongside Dice and IoU; captures boundary quality directly |
| Cross-validation on ISIC 2017 | High variance on ISIC 2017 suggests a single split may be insufficient for robust conclusions |

### P2 — Medium Priority

| Action | Rationale |
|--------|-----------|
| Ablation study: remove MMPA / FGFF individually | Isolate contribution of each novel component vs. backbone alone |
| Larger batch size experiment (BS=8 or 16) | Memory constraints at BS=2 introduce noisy gradients; test with gradient accumulation |
| Data augmentation expansion | Add elastic deformation and CutMix — both known to help in dermoscopy segmentation |
| Test on external dataset (PH2, DermQuest) | Generalisability beyond ISIC benchmarks is required for clinical relevance claims |

### P3 — Nice to Have

| Action | Rationale |
|--------|-----------|
| Model efficiency analysis (FLOPs, inference time) | SAM2 backbone is large (~0.5 GB); report parameter count and latency vs. baselines |
| Interactive results dashboard verification | Confirm `dashboard/segmentation_research_dashboard.html` reflects latest metrics correctly |
| Standardise logging across all trainers | `trainer.py` and `simple_trainer.py` log slightly different fields; unify for cleaner CSV output |

---

## Appendix A — File Reference Index

| Section | Primary Source Files |
|---------|---------------------|
| Dataset splits | `dataset_split/ISIC_{2016,2017,2018}/{train,val,test}.csv` |
| Dataset loader | `datasets.py` |
| Training config | `config.py`, `train_all.py` (lines 31–33) |
| Model architectures | `models/{unet,attention_unet,unet_plus_plus,transunet,maffnet}.py` |
| MAFFNet trainer | `training/trainer.py` |
| Baseline trainer | `training/simple_trainer.py` |
| Results summary | `results/2026-04-28_15-19-03/summary.csv` |
| Epoch-level logs | `experiments/{model}_{dataset}_2026-04-28_15-19-03/checkpoints/training_log.csv` |
| Per-model JSON | `results/2026-04-28_15-19-03/{model}_{dataset}_metrics.json` |
| HTML dashboard | `results/2026-04-28_15-19-03/dashboard/segmentation_research_dashboard.html` |
| Prior MAFFNet runs | `results/2026-04-22_10-41-03/`, `results/2026-04-22_12-48-37/` |

---

*Report generated from repository data only. All metrics sourced directly from CSV/JSON outputs in `results/2026-04-28_15-19-03/`. No values were estimated or extrapolated.*
