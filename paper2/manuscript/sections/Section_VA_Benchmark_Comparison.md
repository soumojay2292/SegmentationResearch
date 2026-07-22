# Section V-A. Benchmark Comparison

> Standalone extract of Section V-A from paper2/manuscript/Paper2_Manuscript.md.
> All values are held-out TEST metrics taken from paper2/results/final_benchmark_table.csv.

---

Table I reports the region-based segmentation performance of every evaluated model on the held-out test subsets of ISIC 2016, ISIC 2017, and ISIC 2018. The comparison comprises the four baseline networks — the U-Net, the Attention U-Net, the UNet++, and the TransUNet — together with the three members of the MAFF-Net family, namely the Paper-faithful Reconstruction, the Restored MAFF-Net, and the Proposed MAFF-Net. The Dice coefficient and the intersection-over-union are treated as the principal indicators, and the accuracy, sensitivity, and specificity are reported alongside them.

**TABLE I.** Region-based segmentation performance on the held-out test subsets of ISIC 2016, ISIC 2017, and ISIC 2018. The highest value in each column, within each dataset, is shown in bold.

| Dataset | Method | Dice | IoU | Accuracy | Sensitivity | Specificity |
|---|---|:---:|:---:|:---:|:---:|:---:|
| ISIC 2016 | U-Net | 0.7874 | 0.6808 | 0.8951 | 0.7649 | 0.9609 |
| ISIC 2016 | Attention U-Net | 0.8466 | 0.7443 | 0.9178 | 0.8780 | 0.9495 |
| ISIC 2016 | UNet++ | 0.8141 | 0.7140 | 0.9055 | 0.7996 | 0.9647 |
| ISIC 2016 | TransUNet | 0.8365 | 0.7349 | 0.9161 | 0.8077 | **0.9700** |
| ISIC 2016 | Paper-faithful Reconstruction | 0.8989 | 0.8264 | 0.9456 | 0.9096 | 0.9591 |
| ISIC 2016 | Restored MAFF-Net | **0.9183** | **0.8555** | 0.9541 | **0.9414** | 0.9545 |
| ISIC 2016 | Proposed MAFF-Net | 0.9169 | 0.8517 | **0.9548** | 0.9220 | 0.9689 |
| ISIC 2017 | U-Net | 0.7229 | 0.6152 | 0.9031 | 0.7211 | 0.9562 |
| ISIC 2017 | Attention U-Net | 0.8097 | 0.7159 | 0.9331 | 0.8223 | 0.9737 |
| ISIC 2017 | UNet++ | 0.7884 | 0.6959 | 0.9291 | 0.7776 | 0.9784 |
| ISIC 2017 | TransUNet | 0.8683 | 0.7843 | 0.9561 | 0.8418 | **0.9870** |
| ISIC 2017 | Paper-faithful Reconstruction | 0.8734 | 0.7975 | 0.9588 | 0.9123 | 0.9681 |
| ISIC 2017 | Restored MAFF-Net | **0.9116** | **0.8447** | **0.9700** | **0.9171** | 0.9807 |
| ISIC 2017 | Proposed MAFF-Net | 0.9096 | 0.8421 | 0.9682 | 0.9069 | 0.9842 |
| ISIC 2018 | U-Net | 0.6866 | 0.5714 | 0.8732 | 0.7766 | 0.8948 |
| ISIC 2018 | Attention U-Net | 0.7662 | 0.6641 | 0.9207 | 0.8352 | 0.9497 |
| ISIC 2018 | UNet++ | 0.6978 | 0.5878 | 0.8912 | 0.7692 | 0.9298 |
| ISIC 2018 | TransUNet | 0.8259 | 0.7265 | 0.9259 | 0.8929 | 0.9315 |
| ISIC 2018 | Paper-faithful Reconstruction | 0.8655 | 0.7855 | 0.9514 | 0.8779 | 0.9730 |
| ISIC 2018 | Restored MAFF-Net | **0.8844** | **0.8080** | **0.9584** | **0.8973** | 0.9730 |
| ISIC 2018 | Proposed MAFF-Net | 0.8701 | 0.7900 | 0.9493 | 0.8733 | **0.9768** |

Among the baseline networks, the U-Net records the lowest Dice coefficient on all three datasets, obtaining 0.7874, 0.7229, and 0.6866 on ISIC 2016, ISIC 2017, and ISIC 2018 respectively, with corresponding intersection-over-union values of 0.6808, 0.6152, and 0.5714. The relative ordering of the remaining baselines is not constant across the three datasets. On ISIC 2016 the Attention U-Net attains the highest baseline Dice coefficient of 0.8466, ahead of the TransUNet at 0.8365 and the UNet++ at 0.8141. On ISIC 2017 and ISIC 2018 the TransUNet attains the highest baseline Dice coefficient, at 0.8683 and 0.8259 respectively, followed by the Attention U-Net at 0.8097 and 0.7662 and the UNet++ at 0.7884 and 0.6978. The TransUNet is therefore the strongest baseline on two of the three datasets and the Attention U-Net on the remaining one.

All three members of the MAFF-Net family record higher Dice and intersection-over-union values than every baseline network on every dataset. The narrowest of these margins occurs on ISIC 2017, where the Paper-faithful Reconstruction attains a Dice coefficient of 0.8734 and an intersection-over-union of 0.7975 against 0.8683 and 0.7843 for the TransUNet. The widest of these margins occurs on ISIC 2016, where the lowest-scoring member of the family, the Paper-faithful Reconstruction, attains a Dice coefficient of 0.8989 against 0.8466 for the Attention U-Net, a difference of 0.0523; the corresponding difference on ISIC 2018 is 0.0396. Relative to the strongest baseline on each dataset, the Proposed MAFF-Net records higher Dice coefficients by 0.0703 on ISIC 2016, 0.0413 on ISIC 2017, and 0.0442 on ISIC 2018.

Within the MAFF-Net family the ordering is consistent across all three datasets. The Restored MAFF-Net records the highest Dice coefficient and the highest intersection-over-union of all evaluated models, obtaining Dice values of 0.9183, 0.9116, and 0.8844 and intersection-over-union values of 0.8555, 0.8447, and 0.8080 on ISIC 2016, ISIC 2017, and ISIC 2018 respectively. The Proposed MAFF-Net records the second-highest values on each dataset, with Dice coefficients of 0.9169, 0.9096, and 0.8701 and intersection-over-union values of 0.8517, 0.8421, and 0.7900. The differences in Dice coefficient between the Restored and the Proposed MAFF-Net are 0.0014 on ISIC 2016, 0.0020 on ISIC 2017, and 0.0143 on ISIC 2018. The Paper-faithful Reconstruction records the lowest Dice and intersection-over-union values within the family on all three datasets.

The remaining region-based measures do not follow the ordering of the Dice coefficient in every case. The Proposed MAFF-Net attains the highest accuracy of all models on ISIC 2016, at 0.9548 against 0.9541 for the Restored MAFF-Net, whereas the Restored MAFF-Net attains the highest accuracy on ISIC 2017 and ISIC 2018, at 0.9700 and 0.9584. With respect to specificity, the Proposed MAFF-Net records higher values than the Restored MAFF-Net on all three datasets, at 0.9689 against 0.9545 on ISIC 2016, 0.9842 against 0.9807 on ISIC 2017, and 0.9768 against 0.9730 on ISIC 2018, the last of these being the highest specificity recorded on that dataset. The TransUNet nevertheless attains the highest specificity of all models on ISIC 2016 and ISIC 2017, at 0.9700 and 0.9870. With respect to sensitivity, the Restored MAFF-Net records the highest value within the family on all three datasets, at 0.9414, 0.9171, and 0.8973; on ISIC 2018 the TransUNet records a sensitivity of 0.8929, which exceeds that of both the Proposed MAFF-Net at 0.8733 and the Paper-faithful Reconstruction at 0.8779.

A consistent trend is observed across the three benchmarks. Every model records its lowest Dice coefficient and its lowest intersection-over-union on ISIC 2018, and every model except the TransUNet records its highest values on ISIC 2016, the TransUNet attaining its highest Dice coefficient and intersection-over-union on ISIC 2017. The spread between the strongest and weakest models is also widest on ISIC 2018, where the Dice coefficients of the evaluated models range from 0.6866 to 0.8844, compared with a range of 0.7874 to 0.9183 on ISIC 2016. Each reported value corresponds to a single training run per configuration, and the narrower differences recorded between individual models should be interpreted with corresponding caution. It should further be noted that the Paper-faithful Reconstruction was trained without pretrained SAM2 weights; the values reported for it therefore characterise the reconstructed architecture under the training conditions of this study and do not constitute a reproduction of the results reported for the Original MAFF-Net.
