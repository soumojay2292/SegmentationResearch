# Performance Comparison Tables (Markdown)

> **Data provenance (read first).**
> - **Baseline rows (U-Net, Attention U-Net, UNet++, TransUNet)** — actual **test-set** results from `experiments/*/metrics.json` (single run; no multi-seed yet).
> - **Stabilized MAFFNet (ours)†** — actual final metrics from `experiments/stabilized_maffnet_ISIC*/summary.csv` (single run, 40 epochs, SAM2.1 Hiera-Large backbone).
> - **† The proposed-model values are validation-set metrics** (the trainer's reported summary), whereas baseline values are test-set. They are therefore **not directly comparable** and the proposed row is **not bolded**; a test-set evaluation of the proposed model is required before a like-for-like comparison.
> - **Attention U-Net / ISIC 2018** incomplete (no `metrics.json`) → `—`.
> - Metrics at threshold 0.5: Dice, IoU, Sensitivity (Sen), Specificity (Spe). **Bold = best test-set baseline per column.**

## Table I — ISIC 2016

| Method | Dice | IoU | Sen | Spe |
|--------|:----:|:---:|:---:|:---:|
| U-Net | 0.7874 | 0.6808 | 0.7649 | 0.9609 |
| Attention U-Net | **0.8466** | **0.7443** | **0.8780** | 0.9495 |
| UNet++ | 0.8141 | 0.7140 | 0.7996 | 0.9647 |
| TransUNet | 0.8365 | 0.7349 | 0.8077 | **0.9700** |
| Stabilized MAFFNet (ours)† | 0.9316 | 0.8746 | 0.9269 | 0.9803 |

## Table II — ISIC 2017

| Method | Dice | IoU | Sen | Spe |
|--------|:----:|:---:|:---:|:---:|
| U-Net | 0.7613 | 0.6604 | 0.7566 | 0.9659 |
| Attention U-Net | 0.8097 | 0.7159 | 0.8223 | 0.9737 |
| UNet++ | 0.7884 | 0.6959 | 0.7776 | 0.9784 |
| TransUNet | **0.8683** | **0.7843** | **0.8418** | **0.9870** |
| Stabilized MAFFNet (ours)† | 0.9101 | 0.8426 | 0.9193 | 0.9815 |

## Table III — ISIC 2018

| Method | Dice | IoU | Sen | Spe |
|--------|:----:|:---:|:---:|:---:|
| U-Net | 0.6866 | 0.5714 | 0.7766 | 0.8948 |
| Attention U-Net | — | — | — | — |
| UNet++ | 0.6978 | 0.5878 | 0.7692 | 0.9298 |
| TransUNet | **0.8259** | **0.7265** | **0.8929** | **0.9315** |
| Stabilized MAFFNet (ours)† | 0.8688 | 0.7926 | 0.8491 | 0.9861 |

**Notes for finalization:**
- Run a **test-set** evaluation of Stabilized MAFFNet so its row is comparable to the baselines (currently validation-set, marked †), then re-bold per column across all rows.
- Re-run Attention U-Net on ISIC 2018 to complete Table III.
- Add multi-seed mean ± std before submission; current values are single-run.
- Sources: `experiments/{attn_unet,unet,unetpp,transunet}_ISIC_*/metrics.json` (test); `experiments/stabilized_maffnet_ISIC*/summary.csv` (validation).
