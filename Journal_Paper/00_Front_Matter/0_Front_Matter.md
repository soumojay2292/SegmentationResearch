# Front Matter

## Title

**Boundary-Centric Dual Supervision for Skin Lesion Segmentation with a Frozen SAM2 Encoder**

*Alternative titles (for supervisor selection):*
- *Boundary-Centric Dual Supervision: Extending a Frequency-Guided SAM2 Segmentation Framework for Dermoscopic Lesion Margins*
- *Restoring and Extending MAFF-Net with Boundary-Centric Dual Supervision for Skin Lesion Segmentation*

## Abstract

Accurate skin lesion segmentation from dermoscopic images depends primarily on the fidelity of the predicted lesion boundary, yet overlap-based training objectives weight all pixels equally and tend to produce spatially smooth predictions that misrepresent the margin. This work builds on MAFF-Net, a frequency-guided segmentation framework that adapts a frozen Segment Anything Model encoder. We first restore the original MAFF-Net components—a frozen SAM2 image encoder, a mixed multi-scale perception adaptor, a frequency-guided feature fusion module, and an auxiliary reconstruction decoder—into a faithful and reproducible base, with the fast Fourier transform isolated to single precision so that training is numerically stable under automatic mixed precision. This restored model reproduces the published design and serves as the experimental base; it is not claimed as a contribution. On this base we propose Boundary-Centric Dual Supervision, a supervision-level extension that derives a single boundary prior from the ground-truth mask and applies it in two complementary ways: boundary-aware deep supervision, which enforces edge agreement at the decoder's multi-scale prediction heads, and edge-weighted reconstruction, which steers the auxiliary reconstruction toward lesion-boundary regions. Both terms act only on the training objective and leave the inference-time model unchanged. We evaluate the restored base and the proposed supervision on the ISIC 2016, 2017, and 2018 benchmarks against U-Net, Attention U-Net, UNet++, and TransUNet, using Dice, IoU, sensitivity, and specificity, and assess each component through a controlled ablation. *[Quantitative findings to be inserted from Section 5 once experiments are finalized; no results are stated at this stage.]*

## Index Terms

Skin lesion segmentation, dermoscopy, Segment Anything Model (SAM2), frequency-guided feature fusion, boundary-aware learning, deep supervision, auxiliary reconstruction, ISIC benchmark.

---

Status:
Draft v1

Dependencies:
- Experimental Results
- Supervisor Feedback
- Final Figures
- Final Tables
