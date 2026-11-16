# 5. Results

> Structure only. All numeric cells are placeholders (`—`) to be filled from the experiments; no values are entered here. Metrics are reported on the test split as mean ± standard deviation over the random seeds. Boundary-specific metrics are listed only if implemented at evaluation time.

## 5.1 Quantitative Comparison

*Table 2. Comparison on the ISIC 2016 test set (mean ± std). Best per column to be highlighted after results are available.*

| Model | Dice | IoU | Accuracy | Sensitivity | Specificity |
|-------|:----:|:---:|:--------:|:-----------:|:-----------:|
| U-Net | — | — | — | — | — |
| Attention U-Net | — | — | — | — | — |
| UNet++ | — | — | — | — | — |
| TransUNet | — | — | — | — | — |
| MAFF-Net (restored) | — | — | — | — | — |
| MAFF-Net + BcDS (ours) | — | — | — | — | — |

*Table 3. Comparison on the ISIC 2017 test set (mean ± std).*

| Model | Dice | IoU | Accuracy | Sensitivity | Specificity |
|-------|:----:|:---:|:--------:|:-----------:|:-----------:|
| U-Net | — | — | — | — | — |
| Attention U-Net | — | — | — | — | — |
| UNet++ | — | — | — | — | — |
| TransUNet | — | — | — | — | — |
| MAFF-Net (restored) | — | — | — | — | — |
| MAFF-Net + BcDS (ours) | — | — | — | — | — |

*Table 4. Comparison on the ISIC 2018 test set (mean ± std).*

| Model | Dice | IoU | Accuracy | Sensitivity | Specificity |
|-------|:----:|:---:|:--------:|:-----------:|:-----------:|
| U-Net | — | — | — | — | — |
| Attention U-Net | — | — | — | — | — |
| UNet++ | — | — | — | — | — |
| TransUNet | — | — | — | — | — |
| MAFF-Net (restored) | — | — | — | — | — |
| MAFF-Net + BcDS (ours) | — | — | — | — | — |

<!-- Analysis of the quantitative comparison to be written after results are available. -->

## 5.2 Qualitative Comparison

*Figure 6. Qualitative comparison on representative dermoscopic cases (input, ground truth, and predictions for each model). [Figure to be added to `Figures/`.]*

*Figure 7. Best, average, and difficult cases for the proposed model, ordered by per-image Dice. [Figure to be added to `Figures/`.]*

<!-- Analysis of the qualitative comparison to be written after figures are available. -->

## 5.3 Ablation Study

*Table 5. Component ablation of Boundary-centric Dual Supervision (mean ± std). A: restored base; B: + BDS; C: + EWR; D: full BcDS.*

| Variant | BDS | EWR | Dice | IoU | Sensitivity | Specificity |
|---------|:---:|:---:|:----:|:---:|:-----------:|:-----------:|
| A — restored base | ✗ | ✗ | — | — | — | — |
| B — + BDS | ✓ | ✗ | — | — | — | — |
| C — + EWR | ✗ | ✓ | — | — | — | — |
| D — BcDS (full) | ✓ | ✓ | — | — | — | — |

*Table 6. Sensitivity to the boundary-loss weight λ_b^max.*

| λ_b^max | 0 | 0.01 | 0.05 | 0.1 |
|---------|:-:|:----:|:----:|:---:|
| Dice | — | — | — | — |
| IoU | — | — | — | — |

*Table 7. Sensitivity to the reconstruction weight λ_r and edge-emphasis factor α.*

| Setting | λ_r = 0.01 | λ_r = 0.05 | λ_r = 0.1 |
|---------|:----------:|:----------:|:---------:|
| α = 1 | — | — | — |
| α = 3 | — | — | — |
| α = 5 | — | — | — |

*Table 8. Effect of the boundary-term warmup (T_warm) and boundary band width.*

| Setting | T_warm = 0 | T_warm = 5 | T_warm = 10 |
|---------|:----------:|:----------:|:-----------:|
| band width = 0 | — | — | — |
| band width = 1 | — | — | — |
| band width = 3 | — | — | — |

<!-- Analysis of the ablation results to be written after results are available. -->

## 5.4 Discussion

<!-- Discussion of the results to be written after the quantitative comparison, qualitative comparison, and ablation study are complete. No analysis is written at this stage. -->
