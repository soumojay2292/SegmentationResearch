# Figure Captions and Numbering (IEEE)

Final captions for the Experimental Results and Method sections. Source files
are listed for each figure. Architecture schematics (Fig. 1–3) are author-drawn;
training/validation curves and qualitative panels (Fig. 4–8) are produced from
the completed Stabilized MAFFNet experiments.

| IEEE Fig. | Source file | Caption |
|-----------|-------------|---------|
| Fig. 1 | `Figures/fig1_architecture.pdf` | **Fig. 1.** Overall architecture of the proposed Stabilized MAFFNet. A frozen SAM2 encoder produces multi-scale features that are processed by two parallel streams: a spatial stream and a frequency stream that applies a single-precision FFT, a radial high-pass filter, and an inverse FFT. The two streams are aligned by instance normalization and a 1×1 projection and combined by a lightweight sigmoid gated fusion. Projected skip connections feed a convolutional decoder with deep supervision, and an auxiliary reconstruction branch regularizes the encoder features. |
| Fig. 2 | `Figures/fig2_frequency_branch.pdf` | **Fig. 2.** Frequency branch. The input feature map is transformed by a single-precision two-dimensional FFT, multiplied by a radial high-pass mask, and reconstructed by an inverse FFT, isolating boundary-sensitive high-frequency content while preserving the spatial resolution. |
| Fig. 3 | `Figures/fig3_gated_fusion.pdf` | **Fig. 3.** Lightweight gated fusion. A 1×1 convolution over the concatenated spatial and frequency features produces a per-pixel, channel-wise gate g = σ(Conv[F_s ‖ F_f]); the output g · F_s + (1 − g) · F_f is a convex combination at linear spatial complexity, without attention. |
| Fig. 4 | `experiments/stabilized_maffnet_ISIC2016/loss_curve.png` | **Fig. 4.** Training and validation loss of Stabilized MAFFNet on ISIC 2016 over 40 epochs. The training loss (solid) and validation loss (dashed) decrease smoothly and stabilize, indicating stable convergence under mixed-precision training. |
| Fig. 5 | `experiments/stabilized_maffnet_ISIC2016/iou_curve.png` | **Fig. 5.** Validation Dice and IoU of Stabilized MAFFNet on ISIC 2016 over 40 epochs. Both metrics rise and plateau; the highest-Dice checkpoint is retained for evaluation. |
| Fig. 6 | `experiments/stabilized_maffnet_ISIC2016/samples/` | **Fig. 6.** Qualitative segmentation results of Stabilized MAFFNet on representative ISIC 2016 cases. From left to right: input dermoscopic image, ground-truth mask, and predicted mask. |
| Fig. 7 | `Figures/fig7_qualitative_comparison.pdf` | **Fig. 7.** Qualitative comparison across methods on representative ISIC cases. Columns: input image, ground truth, U-Net, Attention U-Net, UNet++, TransUNet, and the proposed Stabilized MAFFNet. The proposed method better preserves lesion margins on diffuse and irregular boundaries. |
| Fig. 8 | `Figures/fig8_ablation_chart.pdf` | **Fig. 8.** Ablation of the proposed components. Dice and IoU as the frequency branch, feature alignment, and gated fusion are added cumulatively to the frozen-SAM2 base. |

Notes: export final figures as vector PDF at IEEE column width with embedded fonts; verify every `\ref{fig:…}` resolves. Per-dataset training curves are also available under `experiments/stabilized_maffnet_ISIC2017/` and `experiments/stabilized_maffnet_ISIC2018/` if additional convergence plots are required.
