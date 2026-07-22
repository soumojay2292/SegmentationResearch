# Boundary-Sensitive Metrics — Paper 2

Held-out **TEST** split. Computed from the existing canonical checkpoints by inference only — no model was retrained, modified, or regenerated.

Implementations: **HD95** and **ASSD** from `medpy.metric.binary` (MedPy 0.5.2); **Boundary IoU** from the official formulation of Cheng et al., CVPR 2021 (dilation ratio 0.02 of the image diagonal). Predictions use the same preprocessing (384x384) and threshold (0.5) as the existing region evaluation.


## Boundary metrics

| Dataset | Model | HD95 (px) ↓ | ASSD (px) ↓ | Boundary IoU ↑ | n valid / n |
|---|---|---|---|---|---|
| ISIC 2016 | U-Net | 61.976 | 22.527 | 0.3195 | 192 / 192 |
| ISIC 2016 | Attention U-Net | 50.465 | 18.221 | 0.2436 | 192 / 192 |
| ISIC 2016 | UNet++ | 48.244 | 18.341 | 0.3283 | 190 / 192 |
| ISIC 2016 | TransUNet | 43.779 | 16.368 | 0.2898 | 192 / 192 |
| ISIC 2016 | MAFF-Net (Paper-faithful Recon.) | 24.678 | 10.053 | 0.4101 | 191 / 192 |
| ISIC 2016 | MAFF-Net (Restored) | **21.979** | **8.336** | **0.4518** | 192 / 192 |
| ISIC 2016 | MAFF-Net (Proposed) | 23.329 | 8.510 | 0.4514 | 192 / 192 |
| ISIC 2017 | U-Net | 60.572 | 23.027 | 0.3560 | 194 / 200 |
| ISIC 2017 | Attention U-Net | 40.745 | 17.004 | 0.3749 | 198 / 200 |
| ISIC 2017 | UNet++ | 42.561 | 15.603 | 0.4244 | 197 / 200 |
| ISIC 2017 | TransUNet | 26.750 | 9.710 | 0.4585 | 200 / 200 |
| ISIC 2017 | MAFF-Net (Paper-faithful Recon.) | 24.210 | 9.409 | 0.4806 | 200 / 200 |
| ISIC 2017 | MAFF-Net (Restored) | **16.788** | **6.565** | 0.5255 | 200 / 200 |
| ISIC 2017 | MAFF-Net (Proposed) | 16.869 | 6.645 | **0.5277** | 200 / 200 |
| ISIC 2018 | U-Net | 72.867 | 29.454 | 0.2958 | 388 / 392 |
| ISIC 2018 | Attention U-Net | 47.746 | 19.645 | 0.3035 | 385 / 392 |
| ISIC 2018 | UNet++ | 63.710 | 25.348 | 0.2672 | 383 / 392 |
| ISIC 2018 | TransUNet | 41.836 | 15.092 | 0.3406 | 392 / 392 |
| ISIC 2018 | MAFF-Net (Paper-faithful Recon.) | 23.136 | 9.606 | 0.4491 | 390 / 392 |
| ISIC 2018 | MAFF-Net (Restored) | **20.570** | **8.199** | **0.4701** | 392 / 392 |
| ISIC 2018 | MAFF-Net (Proposed) | 22.554 | 9.240 | 0.4611 | 391 / 392 |

Bold = best per dataset per metric.


HD95 and ASSD are **error** measures in pixels of the 384x384 evaluation grid (lower is better); Boundary IoU is an **overlap** measure in [0,1] (higher is better). No physical voxel spacing is available because the ISIC images are resized from heterogeneous native resolutions, so unit spacing is used and the distances are **not** in millimetres. They are directly comparable across the models in this table, which all share the same grid, but not with distances reported in millimetres elsewhere in the literature. HD95 and ASSD are undefined when either the prediction or the ground truth is empty; such images are excluded from the distance means and counted separately (n_valid).


## Dispersion and degenerate cases

| Dataset | Model | HD95 mean ± std | HD95 median | ASSD mean ± std | ASSD median | BIoU mean ± std | empty pred | empty GT |
|---|---|---|---|---|---|---|---|---|
| ISIC 2016 | U-Net | 61.976 ± 59.493 | 44.290 | 22.527 ± 27.352 | 13.101 | 0.3195 ± 0.2379 | 0 | 0 |
| ISIC 2016 | Attention U-Net | 50.465 ± 47.203 | 27.232 | 18.221 ± 17.564 | 12.580 | 0.2436 ± 0.1660 | 0 | 0 |
| ISIC 2016 | UNet++ | 48.244 ± 51.785 | 23.527 | 18.341 ± 23.023 | 9.068 | 0.3283 ± 0.2241 | 2 | 0 |
| ISIC 2016 | TransUNet | 43.779 ± 43.858 | 22.544 | 16.368 ± 17.954 | 10.317 | 0.2898 ± 0.2054 | 0 | 0 |
| ISIC 2016 | MAFF-Net (Paper-faithful Recon.) | 24.678 ± 29.434 | 14.213 | 10.053 ± 14.326 | 6.087 | 0.4101 ± 0.2228 | 1 | 0 |
| ISIC 2016 | MAFF-Net (Restored) | 21.979 ± 28.302 | 12.239 | 8.336 ± 11.558 | 4.897 | 0.4518 ± 0.2187 | 0 | 0 |
| ISIC 2016 | MAFF-Net (Proposed) | 23.329 ± 29.128 | 13.171 | 8.510 ± 11.128 | 5.358 | 0.4514 ± 0.2198 | 0 | 0 |
| ISIC 2017 | U-Net | 60.572 ± 62.205 | 28.653 | 23.027 ± 29.302 | 10.724 | 0.3560 ± 0.2627 | 6 | 0 |
| ISIC 2017 | Attention U-Net | 40.745 ± 46.573 | 17.514 | 17.004 ± 24.808 | 7.416 | 0.3749 ± 0.2487 | 2 | 0 |
| ISIC 2017 | UNet++ | 42.561 ± 49.260 | 17.088 | 15.603 ± 20.983 | 6.720 | 0.4244 ± 0.2694 | 3 | 0 |
| ISIC 2017 | TransUNet | 26.750 ± 33.860 | 12.042 | 9.710 ± 12.099 | 5.284 | 0.4585 ± 0.2373 | 0 | 0 |
| ISIC 2017 | MAFF-Net (Paper-faithful Recon.) | 24.210 ± 34.718 | 11.662 | 9.409 ± 13.745 | 4.790 | 0.4806 ± 0.2494 | 0 | 0 |
| ISIC 2017 | MAFF-Net (Restored) | 16.788 ± 21.917 | 9.110 | 6.565 ± 7.944 | 3.720 | 0.5255 ± 0.2363 | 0 | 0 |
| ISIC 2017 | MAFF-Net (Proposed) | 16.869 ± 21.622 | 9.311 | 6.645 ± 8.494 | 3.668 | 0.5277 ± 0.2407 | 0 | 0 |
| ISIC 2018 | U-Net | 72.867 ± 64.810 | 53.142 | 29.454 ± 32.159 | 17.464 | 0.2958 ± 0.2591 | 4 | 0 |
| ISIC 2018 | Attention U-Net | 47.746 ± 50.835 | 21.541 | 19.645 ± 24.590 | 9.432 | 0.3035 ± 0.2216 | 7 | 0 |
| ISIC 2018 | UNet++ | 63.710 ± 59.092 | 39.972 | 25.348 ± 28.946 | 14.459 | 0.2672 ± 0.2300 | 9 | 0 |
| ISIC 2018 | TransUNet | 41.836 ± 45.868 | 19.977 | 15.092 ± 19.053 | 8.890 | 0.3406 ± 0.2314 | 0 | 0 |
| ISIC 2018 | MAFF-Net (Paper-faithful Recon.) | 23.136 ± 29.895 | 12.581 | 9.606 ± 14.308 | 4.889 | 0.4491 ± 0.2456 | 2 | 0 |
| ISIC 2018 | MAFF-Net (Restored) | 20.570 ± 26.237 | 10.885 | 8.199 ± 10.648 | 4.674 | 0.4701 ± 0.2391 | 0 | 0 |
| ISIC 2018 | MAFF-Net (Proposed) | 22.554 ± 30.261 | 12.033 | 9.240 ± 14.319 | 4.960 | 0.4611 ± 0.2432 | 1 | 0 |

## Provenance and pipeline-identity check

`Dice (recomputed)` is accumulated with the same `training.trainer.compute_metrics` call used by the region evaluation, on the same batches that produced the boundary metrics. Agreement with the recorded `metrics.json` value confirms the boundary metrics were computed on the same predictions.

All 21 runs agree within a tolerance of 0.001 (max absolute deviation 4.26e-04). The recorded Dice was produced under mixed-precision (bf16) autocast, whereas this boundary pass recomputes in fp32; the sub-1e-3 differences are attributable to that precision change, not to different predictions or a different threshold.

| Model | Dataset | Checkpoint | Dice (recorded) | Dice (recomputed) | Match | Time (s) |
|---|---|---|---|---|---|---|
| U-Net | ISIC 2016 | `experiments/unet_ISIC_2016/model.pth` | 0.7874 | 0.7874 | YES | 20.6 |
| Attention U-Net | ISIC 2016 | `experiments/attn_unet_ISIC_2016/model.pth` | 0.8466 | 0.8466 | YES | 22.4 |
| UNet++ | ISIC 2016 | `experiments/unetpp_ISIC_2016/model.pth` | 0.8141 | 0.8141 | YES | 25.5 |
| TransUNet | ISIC 2016 | `experiments/transunet_ISIC_2016/model.pth` | 0.8365 | 0.8366 | YES | 30.7 |
| MAFF-Net (Paper-faithful Recon.) | ISIC 2016 | `experiments/maffnet_original_ISIC2016_final/model.pth` | 0.8989 | 0.8987 | YES | 19.8 |
| MAFF-Net (Restored) | ISIC 2016 | `experiments/maffnet_ISIC2016_final/model.pth` | 0.9183 | 0.9183 | YES | 33.6 |
| MAFF-Net (Proposed) | ISIC 2016 | `experiments/stabilized_maffnet_ISIC2016/model.pth` | 0.9169 | 0.9167 | YES | 32.6 |
| U-Net | ISIC 2017 | `experiments/unet_final_2017/model.pth` | 0.7229 | 0.7229 | YES | 17.9 |
| Attention U-Net | ISIC 2017 | `experiments/attn_unet_ISIC_2017/model.pth` | 0.8097 | 0.8096 | YES | 18.9 |
| UNet++ | ISIC 2017 | `experiments/unetpp_ISIC_2017/model.pth` | 0.7884 | 0.7884 | YES | 21.9 |
| TransUNet | ISIC 2017 | `experiments/transunet_ISIC_2017/model.pth` | 0.8683 | 0.8684 | YES | 23.6 |
| MAFF-Net (Paper-faithful Recon.) | ISIC 2017 | `experiments/maffnet_original_ISIC2017_final/model.pth` | 0.8734 | 0.8731 | YES | 16.2 |
| MAFF-Net (Restored) | ISIC 2017 | `experiments/maffnet_ISIC2017_final/model.pth` | 0.9116 | 0.9117 | YES | 29.7 |
| MAFF-Net (Proposed) | ISIC 2017 | `experiments/stabilized_maffnet_ISIC2017/model.pth` | 0.9096 | 0.9096 | YES | 28.3 |
| U-Net | ISIC 2018 | `experiments/unet_ISIC_2018/model.pth` | 0.6866 | 0.6866 | YES | 37.7 |
| Attention U-Net | ISIC 2018 | `experiments/attention_unet_ISIC2018_final/model.pth` | 0.7662 | 0.7662 | YES | 39.0 |
| UNet++ | ISIC 2018 | `experiments/unetpp_ISIC_2018/model.pth` | 0.6978 | 0.6978 | YES | 49.6 |
| TransUNet | ISIC 2018 | `experiments/transunet_ISIC_2018/model.pth` | 0.8259 | 0.8255 | YES | 41.4 |
| MAFF-Net (Paper-faithful Recon.) | ISIC 2018 | `experiments/maffnet_original_ISIC2018_final/model.pth` | 0.8655 | 0.8655 | YES | 34.3 |
| MAFF-Net (Restored) | ISIC 2018 | `experiments/maffnet_ISIC2018_final/model.pth` | 0.8844 | 0.8845 | YES | 59.1 |
| MAFF-Net (Proposed) | ISIC 2018 | `experiments/stabilized_maffnet_ISIC2018/model.pth` | 0.8701 | 0.8702 | YES | 57.2 |
