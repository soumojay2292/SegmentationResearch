# Paper Outline

Complete hierarchy for the Q1 manuscript, mapped to the workspace folders.
Structure only — no content. Subsections mirror the project design docs
(`FINAL_MAFFNET_ARCHITECTURE.md`, `CONTRIBUTION_DESIGN.md`, `MAFFNET_DESIGN.md`).

Each section keeps the three-way distinction explicit where relevant:
[O] original MAFF-Net · [R] restored implementation · [C] our contribution.

```
00_Front_Matter/
   ├─ Title
   ├─ Abstract
   ├─ Keywords
   └─ Highlights

01_Introduction/
   └─ 1. Introduction          (problem, gap, contribution statement)

02_Related_Work/
   ├─ 2.1 CNN-based Segmentation
   ├─ 2.2 Transformer-based Segmentation
   ├─ 2.3 SAM and Parameter-Efficient Adaptation
   ├─ 2.4 Frequency-Domain Learning
   └─ 2.5 Boundary-Aware Segmentation

03_Methodology/
   ├─ 3.0 Overview (dual-stream encoder–decoder)
   ├─ 3.1 Frozen SAM2 Encoder                       [O/R]
   ├─ 3.2 Mixed Multi-scale Perception Adaptor (MMPA) [O/R]
   ├─ 3.3 Frequency-Guided Feature Fusion (FGFF)      [O/R]
   ├─ 3.4 Decoder and Deep Supervision               [O/R]
   ├─ 3.5 Boundary-centric Dual Supervision          [C]
   │     ├─ 3.5.1 Boundary-aware Deep Supervision (BDS)
   │     └─ 3.5.2 Edge-weighted Reconstruction (EWR)
   ├─ 3.6 Loss Function                              [O/R + C]
   └─ 3.7 Training Strategy

04_Experimental_Setup/
   ├─ 4.1 Datasets (ISIC 2016 / 2017 / 2018)
   ├─ 4.2 Implementation Details
   └─ 4.3 Evaluation Metrics

05_Results/
   ├─ 5.1 Comparison with Baselines (U-Net, Attention U-Net, UNet++, TransUNet, MAFF-Net)
   ├─ 5.2 Ablation Studies (A/B/C/D; BcDS hyperparameters)
   ├─ 5.3 Qualitative Results
   └─ 5.4 Efficiency Analysis

06_Discussion/
   ├─ 6.1 Analysis of Results
   └─ 6.2 Limitations and Threats to Validity

07_Conclusion/
   └─ 7. Conclusion and Future Work

(unnumbered)
   Acknowledgements
   References  (→ References/)
   Appendix
```

## Planned Figures (→ `Figures/`)
- F1 Overall architecture (code-accurate)
- F2 MMPA · F3 FGFF · F4 Decoder / deep supervision
- F5 Boundary-centric Dual Supervision pipeline (BDS + EWR sharing B(y))
- F6 Qualitative best/average/difficult panels
- F7 Ablation / loss-curve charts

## Planned Tables (→ `Tables/`)
- T1 Baseline comparison (ISIC 2016/2017/2018, test, mean±std)
- T2 Component ablation (A base / B BDS / C EWR / D full)
- T3 BcDS hyperparameter sweeps (λ_bds, λ_ewr, α, warmup, band)
- T4 Loss-composition ablation
- T5 Efficiency (params / FLOPs / latency)

## Planned Equations (→ `Equations/`)
- E1 SAM2 multi-scale features
- E2 MMPA adaptor (bottleneck, experts, softmax weights, residual)
- E3 FGFF cross-modal fusion
- E4 Deep-supervised segmentation loss
- E5 Boundary-aware deep supervision (BDS)
- E6 Edge-weighted reconstruction (EWR)
- E7 Total objective with warmup

## Notes
- Numbered folders sort in reading order.
- Front matter, Acknowledgements, References, Appendix are unnumbered (Elsevier/Q1 style).
- This outline is the structural source of truth; update if sections change.
