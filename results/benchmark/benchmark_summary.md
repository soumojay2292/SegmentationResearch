# Benchmark Summary

Aggregated **11** completed experiments · generated 2026-06-21 14:36

Source: `experiments/*/metrics.json` · sorted Dataset → Model

## Master Results

| Model | Dataset | Best Epoch | Validation Dice | Test Dice | Validation IoU | Test IoU | Accuracy | Sensitivity | Specificity | Training Time | Early Stopped (Yes/No) | Experiment |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Attention U-Net | ISIC_2016 | 21 | 0.853 | 0.8466 | 0.7551 | 0.7443 | 0.9263 | 0.8891 | 0.9521 | ~0:38:36 | Yes | attn_unet_ISIC_2016 |
| TransUNet | ISIC_2016 | 15 | 0.8915 | 0.8736 | 0.8141 | 0.7896 | 0.9486 | 0.8785 | 0.9749 | ~4:45:51 | Yes | transunet_ISIC_2016 |
| U-Net | ISIC_2016 | 38 | 0.8145 | 0.7874 | 0.7102 | 0.6808 | 0.9064 | 0.797 | 0.9559 | ~0:35:39 | Yes | unet_ISIC_2016 |
| UNet++ | ISIC_2016 | 25 | 0.8357 | 0.8141 | 0.7385 | 0.714 | 0.9185 | 0.8367 | 0.9594 |  | Yes | unetpp_ISIC_2016 |
| Attention U-Net | ISIC_2017 | 16 | 0.784 | 0.8097 | 0.678 | 0.7159 | 0.9271 | 0.799 | 0.9674 | ~0:53:27 | Yes | attn_unet_ISIC_2017 |
| TransUNet | ISIC_2017 | 1 | 0.7743 | 0.7819 | 0.6548 | 0.6678 | 0.9145 | 0.8403 | 0.9426 | ~3:23:58 | Yes | transunet_ISIC_2017 |
| U-Net | ISIC_2017 | 38 | 0.6968 | 0.7613 | 0.5813 | 0.6604 | 0.8927 | 0.7227 | 0.9398 |  | Yes | experiment |
| UNet++ | ISIC_2017 | 33 | 0.7323 | 0.7884 | 0.6215 | 0.6959 | 0.9111 | 0.7388 | 0.9596 | ~1:39:09 | Yes | unetpp_ISIC_2017 |
| TransUNet | ISIC_2018 | 1 | 0.5273 | 0.5257 | 0.4035 | 0.3994 | 0.8747 | 0.4252 | 0.9883 | ~3:56:25 | Yes | transunet_ISIC_2018 |
| U-Net | ISIC_2018 | 37 | 0.6877 | 0.6866 | 0.5659 | 0.5714 | 0.8648 | 0.768 | 0.8907 | ~1:59:10 | Yes | unet_ISIC_2018 |
| UNet++ | ISIC_2018 | 9 | 0.6991 | 0.6978 | 0.5801 | 0.5878 | 0.8881 | 0.7567 | 0.9269 | ~1:01:37 | Yes | unetpp_ISIC_2018 |

## Best Model per Dataset (Validation Dice)

| Dataset | Best Model | Val Dice | Test Dice |
|---|---|---|---|
| ISIC_2016 | TransUNet | 0.8915 | 0.8736 |
| ISIC_2017 | Attention U-Net | 0.784 | 0.8097 |
| ISIC_2018 | UNet++ | 0.6991 | 0.6978 |

## Ranking — Mean Validation Dice

| Rank | Model | Mean Val Dice | # datasets |
|---|---|---|---|
| 1 | Attention U-Net | 0.8185 | 2 |
| 2 | UNet++ | 0.7557 | 3 |
| 3 | U-Net | 0.7330 | 3 |
| 4 | TransUNet | 0.7310 | 3 |

_Early-stopped runs: 11/11._

_Skipped (no metrics.json) — 55 folder(s): `attention_unet_ISIC_2016_2026-04-28_15-08-24`, `attention_unet_ISIC_2016_2026-04-28_15-19-03`, `attention_unet_ISIC_2017_2026-04-28_15-08-24`, `attention_unet_ISIC_2017_2026-04-28_15-19-03`, `attention_unet_ISIC_2018_2026-04-28_15-08-24`, `attention_unet_ISIC_2018_2026-04-28_15-19-03`, `attn_auto`, `attn_unet_run`, `baseline_test`, `boundary_25`, `boundary_final`, `boundary_refined`, `boundary_stable`, `boundary_test`, `final_baseline`, `maffnet_auto`, `maffnet_ISIC_2016_2026-04-28_15-08-24`, `maffnet_ISIC_2016_2026-04-28_15-19-03`, `maffnet_ISIC_2017_2026-04-28_15-08-24`, `maffnet_ISIC_2017_2026-04-28_15-19-03`, `maffnet_ISIC_2018_2026-04-28_15-08-24`, `maffnet_ISIC_2018_2026-04-28_15-19-03`, `maffnet_run`, `sanity_check`, `sanity_clean`, `sanity_final`, `sanity_final2`, `sanity_final_clean`, `test_boundary`, `transunet_ISIC_2016_2026-04-28_15-08-24`, `transunet_ISIC_2016_2026-04-28_15-19-03`, `transunet_ISIC_2017_2026-04-28_15-08-24`, `transunet_ISIC_2017_2026-04-28_15-19-03`, `transunet_ISIC_2018_2026-04-28_15-08-24`, `transunet_ISIC_2018_2026-04-28_15-19-03`, `transunet_run`, `unet_final_2016`, `unet_final_2017`, `unet_final_2018`, `unet_fix_smoke`, `unet_ISIC_2016_2026-04-28_15-08-24`, `unet_ISIC_2016_2026-04-28_15-19-03`, `unet_ISIC_2017_2026-04-28_15-08-24`, `unet_ISIC_2017_2026-04-28_15-19-03`, `unet_ISIC_2018_2026-04-28_15-08-24`, `unet_ISIC_2018_2026-04-28_15-19-03`, `unet_run`, `unetpp_auto`, `unetpp_ISIC_2016_2026-04-28_15-08-24`, `unetpp_ISIC_2016_2026-04-28_15-19-03`, `unetpp_ISIC_2017_2026-04-28_15-08-24`, `unetpp_ISIC_2017_2026-04-28_15-19-03`, `unetpp_ISIC_2018_2026-04-28_15-08-24`, `unetpp_ISIC_2018_2026-04-28_15-19-03`, `unetpp_run`._
