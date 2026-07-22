# Section V-B. Boundary Evaluation

> Standalone extract of Section V-B from paper2/manuscript/Paper2_Manuscript.md.
> All values are held-out TEST metrics taken from paper2/results/boundary_metrics.csv.

---

In dermoscopic lesion segmentation the practical value of a prediction depends heavily on the fidelity with which the lesion margin is delineated, since the margin determines the measured extent of the lesion and is the region in which annotation and prediction are most likely to diverge. Boundary pixels, however, constitute only a small fraction of the image, and the region-overlap measures reported above are dominated by the comparatively large and homogeneous lesion interior. A contour displaced by a few pixels alters the region overlap only marginally, so a model may attain a high overlap score while still reproducing the margin imprecisely. Measures defined directly on the contour are therefore required in order to characterise this aspect of segmentation quality. Table II reports the three boundary-based measures defined in Section IV-C — the 95th-percentile Hausdorff distance, the average symmetric surface distance, and the Boundary IoU — for every evaluated model on the held-out test subsets of the three datasets.

**TABLE II.** Boundary-based segmentation performance on the held-out test subsets of ISIC 2016, ISIC 2017, and ISIC 2018. HD95 and ASSD are expressed in pixels of the 384×384 evaluation grid, and lower values indicate closer agreement; higher Boundary IoU values indicate closer agreement. The column *n*<sub>valid</sub> gives the number of images contributing to the distance measures. The best value in each column, within each dataset, is shown in bold.

| Dataset | Method | HD95 (px) | ASSD (px) | Boundary IoU | *n*<sub>valid</sub> / *n* |
|---|---|:---:|:---:|:---:|:---:|
| ISIC 2016 | U-Net | 61.976 | 22.527 | 0.3195 | 192 / 192 |
| ISIC 2016 | Attention U-Net | 50.465 | 18.221 | 0.2436 | 192 / 192 |
| ISIC 2016 | UNet++ | 48.244 | 18.341 | 0.3283 | 190 / 192 |
| ISIC 2016 | TransUNet | 43.779 | 16.368 | 0.2898 | 192 / 192 |
| ISIC 2016 | Paper-faithful Reconstruction | 24.678 | 10.053 | 0.4101 | 191 / 192 |
| ISIC 2016 | Restored MAFF-Net | **21.979** | **8.336** | **0.4518** | 192 / 192 |
| ISIC 2016 | Proposed MAFF-Net | 23.329 | 8.510 | 0.4514 | 192 / 192 |
| ISIC 2017 | U-Net | 60.572 | 23.027 | 0.3560 | 194 / 200 |
| ISIC 2017 | Attention U-Net | 40.745 | 17.004 | 0.3749 | 198 / 200 |
| ISIC 2017 | UNet++ | 42.561 | 15.603 | 0.4244 | 197 / 200 |
| ISIC 2017 | TransUNet | 26.750 | 9.710 | 0.4585 | 200 / 200 |
| ISIC 2017 | Paper-faithful Reconstruction | 24.210 | 9.409 | 0.4806 | 200 / 200 |
| ISIC 2017 | Restored MAFF-Net | **16.788** | **6.565** | 0.5255 | 200 / 200 |
| ISIC 2017 | Proposed MAFF-Net | 16.869 | 6.645 | **0.5277** | 200 / 200 |
| ISIC 2018 | U-Net | 72.867 | 29.454 | 0.2958 | 388 / 392 |
| ISIC 2018 | Attention U-Net | 47.746 | 19.645 | 0.3035 | 385 / 392 |
| ISIC 2018 | UNet++ | 63.710 | 25.348 | 0.2672 | 383 / 392 |
| ISIC 2018 | TransUNet | 41.836 | 15.092 | 0.3406 | 392 / 392 |
| ISIC 2018 | Paper-faithful Reconstruction | 23.136 | 9.606 | 0.4491 | 390 / 392 |
| ISIC 2018 | Restored MAFF-Net | **20.570** | **8.199** | **0.4701** | 392 / 392 |
| ISIC 2018 | Proposed MAFF-Net | 22.554 | 9.240 | 0.4611 | 391 / 392 |

The distance measures are undefined when a prediction contains no positive pixels, and the affected images are excluded from the corresponding means and accounted for in the column *n*<sub>valid</sub>. Such cases arise only for a minority of images and occur predominantly among the baseline networks: the UNet++ produces an empty prediction for nine images of ISIC 2018 and the Attention U-Net for seven, whereas the TransUNet and the Restored MAFF-Net produce no empty prediction on any dataset and the Proposed MAFF-Net produces one, on ISIC 2018. The Boundary IoU is defined for all images and is computed over the complete test subsets.

All three members of the MAFF-Net family record lower HD95 values, lower ASSD values, and higher Boundary IoU values than every baseline network on each of the three datasets. The separation is present for all three measures simultaneously and on every dataset, rather than for an individual measure or an individual benchmark. Considered against the strongest baseline for each measure, the Proposed MAFF-Net reduces the HD95 from 43.779 to 23.329 pixels on ISIC 2016, from 26.750 to 16.869 pixels on ISIC 2017, and from 41.836 to 22.554 pixels on ISIC 2018, and reduces the ASSD from 16.368 to 8.510, from 9.710 to 6.645, and from 15.092 to 9.240 pixels on the same datasets. Its Boundary IoU exceeds the highest baseline value by 0.1231 on ISIC 2016, by 0.0692 on ISIC 2017, and by 0.1205 on ISIC 2018.

Within the MAFF-Net family, the Restored MAFF-Net records the lowest HD95 and the lowest ASSD on all three datasets, at 21.979 and 8.336 pixels on ISIC 2016, 16.788 and 6.565 pixels on ISIC 2017, and 20.570 and 8.199 pixels on ISIC 2018. The corresponding values for the Proposed MAFF-Net are 23.329 and 8.510, 16.869 and 6.645, and 22.554 and 9.240 pixels. For the Boundary IoU the ordering is not uniform: the Proposed MAFF-Net records the highest value of all evaluated models on ISIC 2017, at 0.5277 against 0.5255 for the Restored MAFF-Net, while the Restored MAFF-Net records the higher value on ISIC 2016, at 0.4518 against 0.4514, and on ISIC 2018, at 0.4701 against 0.4611. The Paper-faithful Reconstruction records the highest distances and the lowest Boundary IoU within the family on all three datasets.

The orderings induced by the boundary measures do not coincide with those induced by region overlap, which indicates that region overlap alone does not fully characterise the quality of the predicted contour. On ISIC 2016 the Attention U-Net records the highest Dice coefficient among the baseline networks, yet it records the lowest Boundary IoU of all evaluated models on that dataset, at 0.2436, below the value of 0.3195 obtained by the U-Net, which records the lowest Dice coefficient on the same dataset. A comparable disagreement is observed on ISIC 2018, where the UNet++ records a higher Dice coefficient than the U-Net, at 0.6978 against 0.6866, but a lower Boundary IoU, at 0.2672 against 0.2958. Two models may therefore attain similar region overlap while differing appreciably in the agreement of their contours, and the boundary measures provide information that the region measures do not.

Three qualifications apply to the values in Table II. Each value corresponds to a single training run of the configuration concerned, and no statistical testing was performed; the differences reported between closely ranked models, in particular those within the MAFF-Net family, should accordingly be regarded as observations on these runs rather than as established effects. The per-image dispersion of the distance measures is moreover large in relation to the differences between neighbouring models: the standard deviation of the HD95 is of the same order as its mean for every model, ranging from 21.6 to 30.3 pixels for the members of the MAFF-Net family and from 33.9 to 64.8 pixels for the baseline networks. Finally, as noted in Section IV-C, HD95 and ASSD are expressed in pixels of the 384×384 evaluation grid rather than in physical units, so they are comparable among the models evaluated here but not with distances reported in millimetres elsewhere.
