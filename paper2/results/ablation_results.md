# Architecture Ablation — Proposed MAFF-Net (ISIC 2018 TEST)

Leave-one-out ablation. **Full Proposed** = the canonical `experiments/stabilized_maffnet_ISIC2018` checkpoint (reused, not retrained). Each ablated run disables exactly one architectural component under the identical canonical training configuration (40 epochs, AdamW, LinearLR→0, bf16, BcDS λ 0.05, best-val-Dice selection). Region metrics from `metrics.json['test']`; boundary metrics (HD95/ASSD/Boundary IoU) from the Step-7 MedPy + Cheng-et-al. pipeline. HD95/ASSD in px (↓ better); Dice/IoU/Acc/Boundary IoU ↑ better.


## Absolute metrics

| Configuration | Dice ↑ | IoU ↑ | Accuracy ↑ | HD95 ↓ | ASSD ↓ | Boundary IoU ↑ |
|---|---|---|---|---|---|---|
| Full Proposed | 0.8701 | 0.7900 | 0.9493 | 22.554 | 9.240 | 0.4611 |
| - Frequency Branch | 0.8708 | 0.7868 | 0.9496 | 23.574 | 9.546 | 0.4416 |
| - Feature Alignment | 0.8649 | 0.7811 | 0.9502 | 23.676 | 9.869 | 0.4326 |
| - Gated Fusion | 0.8725 | 0.7907 | 0.9520 | 25.546 | 9.488 | 0.4508 |

## Contribution of each module (Full − ablated; positive = the module helps)

| Disabled module | ΔDice | ΔIoU | ΔAccuracy | ΔHD95 (px) | ΔASSD (px) | ΔBoundary IoU |
|---|---|---|---|---|---|---|
| - Frequency Branch | -0.0007 | +0.0032 | -0.0003 | +1.020 | +0.306 | +0.0196 |
| - Feature Alignment | +0.0052 | +0.0089 | -0.0009 | +1.121 | +0.629 | +0.0286 |
| - Gated Fusion | -0.0024 | -0.0007 | -0.0027 | +2.992 | +0.248 | +0.0103 |

ΔHD95/ΔASSD are (ablated − Full): a positive value means removing the module made the boundary error worse, i.e. the module reduces boundary error. All other Δ are (Full − ablated).
