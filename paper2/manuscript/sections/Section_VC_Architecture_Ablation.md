# Section V-C. Architecture Ablation

> Standalone extract of Section V-C from paper2/manuscript/Paper2_Manuscript.md.
> All values are held-out ISIC 2018 TEST metrics taken from paper2/results/ablation_results.csv.

---

The three components introduced in this work — the Frequency Branch, the Feature Alignment Module, and the Gated Fusion Module — were examined by the leave-one-out ablation described in Section IV-D, in which each component is disabled in turn while the remainder of the network and the training configuration are held fixed. The study was conducted on ISIC 2018. Table III reports both the region-based and the boundary-based measures for the complete model and for the three ablated configurations, so that the two families of measures may be considered together.

**TABLE III.** Leave-one-out architecture ablation on the ISIC 2018 test subset. HD95 and ASSD are expressed in pixels of the 384×384 evaluation grid, and lower values indicate closer agreement; higher Dice, IoU, accuracy, and Boundary IoU values indicate closer agreement. The best value in each column is shown in bold.

| Configuration | Dice | IoU | Accuracy | HD95 (px) | ASSD (px) | Boundary IoU |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Full Proposed MAFF-Net | 0.8701 | 0.7900 | 0.9493 | **22.554** | **9.240** | **0.4611** |
| without Frequency Branch | 0.8708 | 0.7868 | 0.9496 | 23.574 | 9.546 | 0.4416 |
| without Feature Alignment Module | 0.8649 | 0.7811 | 0.9502 | 23.676 | 9.869 | 0.4326 |
| without Gated Fusion Module | **0.8725** | **0.7907** | **0.9520** | 25.546 | 9.488 | 0.4508 |

With respect to the boundary-based measures, the complete Proposed MAFF-Net records the best value of the four configurations on all three measures, obtaining the lowest HD95 at 22.554 pixels, the lowest ASSD at 9.240 pixels, and the highest Boundary IoU at 0.4611. Disabling any one of the three components degrades all three measures simultaneously. Removing the Frequency Branch increases the HD95 by 1.020 pixels and the ASSD by 0.306 pixels and reduces the Boundary IoU by 0.0196; removing the Feature Alignment Module increases the HD95 by 1.121 pixels and the ASSD by 0.629 pixels and reduces the Boundary IoU by 0.0286; and removing the Gated Fusion Module increases the HD95 by 2.992 pixels and the ASSD by 0.248 pixels and reduces the Boundary IoU by 0.0103. No single component accounts for the whole of the difference, and the ordering of the components depends on the measure considered: the largest increase in HD95 follows removal of the Gated Fusion Module, whereas the largest increase in ASSD and the largest reduction in Boundary IoU both follow removal of the Feature Alignment Module.

The region-based measures do not separate the four configurations in the same manner, and they do not favour the complete model. The Dice coefficient varies from 0.8649 to 0.8725 across the four configurations, a spread of 0.0076, and the complete model, at 0.8701, does not record the highest value. The configuration without the Gated Fusion Module records the highest Dice coefficient at 0.8725, the highest intersection-over-union at 0.7907, and the highest accuracy at 0.9520, while the complete model records the lowest accuracy of the four configurations at 0.9493. The only configuration whose Dice coefficient falls below that of the complete model is the one without the Feature Alignment Module, at 0.8649. The present evidence therefore does not support the proposition that these three components improve region overlap, and no such claim is made.

Considered together, the two families of measures indicate that the effect of the three components is confined to the fidelity of the predicted contour rather than to the extent of the predicted region. This pattern is consistent with the observation reported in Section V-B that region overlap is comparatively insensitive to displacements of the contour: a configuration may reproduce the lesion area to within a small difference in Dice coefficient while agreeing less closely with the reference contour. On the evidence of this study, the architectural contribution of the present work is accordingly to be understood as an improvement in boundary fidelity, and not as an improvement in region-overlap performance.

Three qualifications delimit the strength of these observations. Each configuration was trained once, with a single experimental seed, and no repeated runs and no statistical testing were performed; the differences reported in Table III are therefore observations on these particular runs and are not established as statistically significant. The magnitudes concerned are moreover small in relation to the per-image dispersion of the distance measures noted in Section V-B, which further limits the weight that individual differences can bear. Finally, the study was conducted on ISIC 2018 alone, so the consistency of these observations across the other two datasets is not established here.
