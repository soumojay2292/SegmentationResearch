# References

> Draft v1 reference list for the citation keys [1]–[15] used in the drafted sections, plus additions recommended in `REVIEW_REPORT.md` (SAM2, boundary-loss prior art, ViT/ResNet for the TransUNet backbone, and a source for the melanoma statistics in Section 1). Entries marked **[VERIFY]** require confirmation of exact authors, venue, volume, and pages against the original source before submission; no metadata has been invented for these. A machine-readable `.bib` will be produced from this list.

## Cited in current draft ([1]–[15])

[1] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," in *Proc. MICCAI*, 2015, pp. 234–241.

[2] J. Chen, Y. Lu, Q. Yu, X. Luo, E. Adeli, Y. Wang, L. Lu, A. L. Yuille, and Y. Zhou, "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation," *arXiv:2102.04306*, 2021.

[3] O. Oktay, J. Schlemper, L. L. Folgoc, M. Lee, M. Heinrich, K. Misawa, K. Mori, S. McDonagh, N. Y. Hammerla, B. Kainz, B. Glocker, and D. Rueckert, "Attention U-Net: Learning Where to Look for the Pancreas," in *Proc. MIDL*, 2018. (*arXiv:1804.03999*)

[4] N. Codella, V. Rotemberg, P. Tschandl, M. E. Celebi, S. Dusza, D. Gutman, B. Helba, A. Kalloo, K. Liopyris, M. Marchetti, H. Kittler, and A. Halpern, "Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)," *arXiv:1902.03368*, 2019. **[VERIFY — pair with P. Tschandl et al., "The HAM10000 dataset," Scientific Data, 2018, if the HAM10000 masks are used]**

[5] B. N. Patro, V. P. Namboodiri, and V. S. Agneeswaran, "SpectFormer: Frequency and Attention is What You Need in a Vision Transformer," *arXiv:2304.06446*, 2023. **[VERIFY authors/venue]**

[6] *SGT-Net*: "Spectral Generalized Transformer for skin lesion segmentation." **[VERIFY — authors, title, venue, year]**

[7] *BGFF-UNet*: boundary-guided feature-fusion U-Net for dermoscopic segmentation. **[VERIFY — authors, title, venue, year]**

[8] *FastTrans-Net*: efficient hybrid transformer for medical image segmentation. **[VERIFY — authors, title, venue, year]**

[9] Frequency-based masked image modeling for self-supervised medical image segmentation. **[VERIFY — authors, title, venue, year]**

[10] Y. Zhou, H. Gong, R. Qiu, S. Wei, Z. Li, and W. Zhang, "MAFF-Net: SAM-powered Mixed Multi-scale Perception Adaptation and frequency-guided feature fusion for robust skin lesion segmentation," *Biomedical Signal Processing and Control*, vol. 113, art. no. 109222, 2026.

[11] D. Gutman, N. C. F. Codella, E. Celebi, B. Helba, M. Marchetti, N. Mishra, and A. Halpern, "Skin Lesion Analysis toward Melanoma Detection: A Challenge at the International Symposium on Biomedical Imaging (ISBI) 2016, hosted by ISIC," *arXiv:1605.01397*, 2016.

[12] N. C. F. Codella, D. Gutman, M. E. Celebi, B. Helba, M. A. Marchetti, S. W. Dusza, A. Kalloo, K. Liopyris, N. Mishra, H. Kittler, and A. Halpern, "Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the 2017 ISBI, hosted by ISIC," in *Proc. IEEE ISBI*, 2018, pp. 168–172.

[13] Z. Zhou, M. M. Rahman Siddiquee, N. Tajbakhsh, and J. Liang, "UNet++: A Nested U-Net Architecture for Medical Image Segmentation," in *Proc. DLMIA/ML-CDS Workshop, MICCAI*, 2018, pp. 3–11.

[14] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollár, and R. Girshick, "Segment Anything," in *Proc. IEEE/CVF ICCV*, 2023, pp. 4015–4026.

[15] Frequency-aware feature fusion for medical image segmentation. **[VERIFY — authors, title, venue, year]**

## Additions recommended by the review (to be assigned numbers)

[16] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson, E. Mintun, J. Pan, K. V. Alwala, N. Carion, C.-Y. Wu, R. Girshick, P. Dollár, and C. Feichtenhofer, "SAM 2: Segment Anything in Images and Videos," *arXiv:2408.00714*, 2024. *(encoder actually used: SAM2 Hiera-Large)*

[17] H. Kervadec, J. Bouchtiba, C. Desrosiers, É. Granger, J. Dolz, and I. Ben Ayed, "Boundary loss for highly unbalanced segmentation," *Medical Image Analysis*, vol. 67, art. 101851, 2021. *(prior boundary-supervision work; cite in Section 3.4)*

[18] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale," in *Proc. ICLR*, 2021. *(ViT; TransUNet baseline backbone)*

[19] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. IEEE CVPR*, 2016, pp. 770–778. *(ResNet; TransUNet hybrid backbone)*

[20] Source for the melanoma incidence and stage-specific survival statistics quoted in Section 1 (e.g., a cancer-statistics report). **[ADD/VERIFY — primary epidemiological source required]**

---

## Notes for finalization
- Resolve all **[VERIFY]** entries [6], [7], [8], [9], [15] against the original drafts' intended sources; if any cannot be confirmed, remove the claim or replace the citation.
- Section 1's clinical statistics ([20]) require a primary source; do not leave them uncited.
- Add [16] (SAM2) wherever the encoder is described; the drafts currently cite only SAM [14].
- Cite [17] in Section 3.4 to position Boundary-aware Deep Supervision relative to existing boundary losses.
- If the supervisor selects the synopsis methodology (Path A), add the SGT-Net baseline [6] to the experiments and confirm its citation.
- Renumber sequentially once additions are merged; verify every in-text [n] resolves to an entry.

---

Status:
Draft v1

Dependencies:
- Experimental Results
- Supervisor Feedback
- Final Figures
- Final Tables
