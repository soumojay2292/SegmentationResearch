# Final Canonical Benchmark Table — Paper 2

All values are **held-out TEST split** results read verbatim from `experiments/<run>/metrics.json`. No value in this table comes from `summary.csv` (which stores final-epoch *validation* metrics).

Test set sizes: ISIC 2016 n=192, ISIC 2017 n=200, ISIC 2018 n=392. All runs: 384x384 input, batch size 2, identical `dataset_split/` CSVs, identical metric function (`training.trainer.compute_metrics`, threshold 0.5).


## Primary benchmark

| Dataset | Model | Dice | mIoU | Accuracy | Precision&dagger; | Recall | F1&Dagger; |
|---|---|---|---|---|---|---|---|
| ISIC 2016 | U-Net | 0.7874 | 0.6808 | 0.8951 | 0.8113 | 0.7649 | 0.7874 |
| ISIC 2016 | TransUNet | 0.8365 | 0.7349 | 0.9161 | 0.8675 | 0.8077 | 0.8365 |
| ISIC 2016 | MAFF-Net (Paper-faithful Recon.) | 0.8989 | 0.8264 | 0.9456 | 0.8884 | 0.9096 | 0.8989 |
| ISIC 2016 | **MAFF-Net (Restored)** | **0.9183** | **0.8555** | **0.9541** | 0.8963 | **0.9414** | **0.9183** |
| ISIC 2016 | MAFF-Net (Proposed) | 0.9169 | 0.8517 | 0.9548 | 0.9118 | 0.9220 | 0.9169 |
| ISIC 2017 | U-Net | 0.7229 | 0.6152 | 0.9031 | 0.7246 | 0.7211 | 0.7229 |
| ISIC 2017 | TransUNet | 0.8683 | 0.7843 | 0.9561 | 0.8967 | 0.8418 | 0.8683 |
| ISIC 2017 | MAFF-Net (Paper-faithful Recon.) | 0.8734 | 0.7975 | 0.9588 | 0.8377 | 0.9123 | 0.8734 |
| ISIC 2017 | **MAFF-Net (Restored)** | **0.9116** | **0.8447** | **0.9700** | 0.9062 | **0.9171** | **0.9116** |
| ISIC 2017 | MAFF-Net (Proposed) | 0.9096 | 0.8421 | 0.9682 | 0.9123 | 0.9069 | 0.9096 |
| ISIC 2018 | U-Net | 0.6866 | 0.5714 | 0.8732 | 0.6154 | 0.7766 | 0.6866 |
| ISIC 2018 | TransUNet | 0.8259 | 0.7265 | 0.9259 | 0.7683 | 0.8929 | 0.8259 |
| ISIC 2018 | MAFF-Net (Paper-faithful Recon.) | 0.8655 | 0.7855 | 0.9514 | 0.8535 | 0.8779 | 0.8655 |
| ISIC 2018 | **MAFF-Net (Restored)** | **0.8844** | **0.8080** | **0.9584** | 0.8719 | **0.8973** | **0.8844** |
| ISIC 2018 | MAFF-Net (Proposed) | 0.8701 | 0.7900 | 0.9493 | 0.8669 | 0.8733 | 0.8701 |

Bold = highest Dice within that dataset.


## Supplementary baselines

Complete and valid, but outside the model set requested for the primary table.

| Dataset | Model | Dice | mIoU | Accuracy | Precision&dagger; | Recall | F1&Dagger; |
|---|---|---|---|---|---|---|---|
| ISIC 2016 | Attention U-Net | 0.8466 | 0.7443 | 0.9178 | 0.8173 | 0.8780 | 0.8466 |
| ISIC 2016 | UNet++ | 0.8141 | 0.7140 | 0.9055 | 0.8292 | 0.7996 | 0.8141 |
| ISIC 2017 | Attention U-Net | 0.8097 | 0.7159 | 0.9331 | 0.7974 | 0.8223 | 0.8097 |
| ISIC 2017 | UNet++ | 0.7884 | 0.6959 | 0.9291 | 0.7996 | 0.7776 | 0.7884 |
| ISIC 2018 | Attention U-Net | 0.7662 | 0.6641 | 0.9207 | 0.7078 | 0.8352 | 0.7662 |
| ISIC 2018 | UNet++ | 0.6978 | 0.5878 | 0.8912 | 0.6384 | 0.7692 | 0.6978 |

&dagger;/&Dagger; F1 is identical to Dice by construction in `compute_metrics` (both equal 2TP/(2TP+FP+FN)); it is reported for completeness, not as independent evidence. Precision is **not** computed anywhere in the repository and is derived from the exact identity P = 1/(2/Dice - 1/Recall); this is exact per batch but approximate on batch-averaged values, so it is marked derived. Recall is the `sensitivity` field (same definition, TP/(TP+FN)). `mIoU` is the foreground IoU TP/(TP+FP+FN), not a mean over classes.


## Source provenance

| Model | Dataset | Source metrics.json | n_test | Epochs | Best/ckpt epoch |
|---|---|---|---|---|---|
| U-Net | ISIC 2016 | `experiments/unet_ISIC_2016/metrics.json` | 192 | 40 | 38 |
| U-Net | ISIC 2017 | `experiments/unet_final_2017/metrics.json` | 200 | 20 | 13 |
| U-Net | ISIC 2018 | `experiments/unet_ISIC_2018/metrics.json` | 392 | 40 | 37 |
| TransUNet | ISIC 2016 | `experiments/transunet_ISIC_2016/metrics.json` | 192 | 40 | 7 |
| TransUNet | ISIC 2017 | `experiments/transunet_ISIC_2017/metrics.json` | 200 | 40 | 13 |
| TransUNet | ISIC 2018 | `experiments/transunet_ISIC_2018/metrics.json` | 392 | 40 | 10 |
| MAFF-Net (Paper-faithful Recon.) | ISIC 2016 | `experiments/maffnet_original_ISIC2016_final/metrics.json` | 192 | n/r | 37 |
| MAFF-Net (Paper-faithful Recon.) | ISIC 2017 | `experiments/maffnet_original_ISIC2017_final/metrics.json` | 200 | n/r | 38 |
| MAFF-Net (Paper-faithful Recon.) | ISIC 2018 | `experiments/maffnet_original_ISIC2018_final/metrics.json` | 392 | n/r | 35 |
| MAFF-Net (Restored) | ISIC 2016 | `experiments/maffnet_ISIC2016_final/metrics.json` | 192 | n/r | 32 |
| MAFF-Net (Restored) | ISIC 2017 | `experiments/maffnet_ISIC2017_final/metrics.json` | 200 | n/r | 25 |
| MAFF-Net (Restored) | ISIC 2018 | `experiments/maffnet_ISIC2018_final/metrics.json` | 392 | n/r | 39 |
| MAFF-Net (Proposed) | ISIC 2016 | `experiments/stabilized_maffnet_ISIC2016/metrics.json` | 192 | n/r | 38 |
| MAFF-Net (Proposed) | ISIC 2017 | `experiments/stabilized_maffnet_ISIC2017/metrics.json` | 200 | n/r | 31 |
| MAFF-Net (Proposed) | ISIC 2018 | `experiments/stabilized_maffnet_ISIC2018/metrics.json` | 392 | n/r | 32 |
| Attention U-Net | ISIC 2016 | `experiments/attn_unet_ISIC_2016/metrics.json` | 192 | 40 | 21 |
| Attention U-Net | ISIC 2017 | `experiments/attn_unet_ISIC_2017/metrics.json` | 200 | 40 | 16 |
| Attention U-Net | ISIC 2018 | `experiments/attention_unet_ISIC2018_final/metrics.json` | 392 | 40 | 26 |
| UNet++ | ISIC 2016 | `experiments/unetpp_ISIC_2016/metrics.json` | 192 | 40 | 25 |
| UNet++ | ISIC 2017 | `experiments/unetpp_ISIC_2017/metrics.json` | 200 | 40 | 33 |
| UNet++ | ISIC 2018 | `experiments/unetpp_ISIC_2018/metrics.json` | 392 | 40 | 9 |
