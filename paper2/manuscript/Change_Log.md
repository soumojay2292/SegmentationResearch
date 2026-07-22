# Change Log — Final Manuscript Writing Phase

Manuscript: `paper2/manuscript/Paper2_Manuscript.md` (canonical)
Scope of this phase: complete the remaining front matter and the reference list.

---

## 1. Sections modified

| Section | Action |
|---|---|
| Title | **Written** (was `[TITLE — placeholder]`) |
| Abstract | **Written** (was `[Abstract — placeholder]`) |
| Keywords | **Written** (was `[Keywords — placeholder]`) |
| References | **Written** (was `[References — placeholder]`) — 19 entries |
| IV-A. Datasets | **Consistency fix only** — three citation anchors `[17] [18] [19]` added at the point where the datasets are named. No prose altered. |

Sections I, II, III, V, VI, VII were **not** modified. No completed prose was regenerated or rewritten.

---

## 2. Files updated

- `paper2/manuscript/Paper2_Manuscript.md` — front matter + references + one consistency fix
- `paper2/manuscript/Abstract.md` — **created** (standalone title / abstract / keywords)
- `paper2/manuscript/Change_Log.md` — **created** (this file)

---

## 3. Words added

| Item | Words |
|---|---|
| Title | 15 |
| Abstract | 241 (target 180–250) |
| Keywords | 8 terms |
| References | 19 entries (~640 words of bibliographic text) |
| **Total prose added** | **~900** |

Manuscript total: ~17,900 words across Sections I–VII.

---

## 4. References added

19 entries, all cited in the body; **zero orphans and zero dangling markers** (verified
programmatically). Numbering follows IEEE order of first appearance (verified).

| # | Work | First cited |
|---|---|---|
| 1 | Long et al., FCN, CVPR 2015 | II-A |
| 2 | Ronneberger et al., U-Net, MICCAI 2015 | II-A |
| 3 | Zhou et al., UNet++, DLMIA/MICCAI 2018 | II-A |
| 4 | Hu et al., Squeeze-and-Excitation, CVPR 2018 | II-B |
| 5 | Oktay et al., Attention U-Net, MIDL 2018 | II-B |
| 6 | Dosovitskiy et al., ViT, ICLR 2021 | II-C |
| 7 | Chen et al., TransUNet, arXiv 2021 | II-C |
| 8 | Cao et al., Swin-Unet, ECCVW 2022 | II-C |
| 9 | Kirillov et al., Segment Anything, ICCV 2023 | II-D |
| 10 | Ravi et al., SAM 2, arXiv 2024 | II-D |
| 11 | Ma et al., MedSAM, Nature Communications 2024 | II-D |
| 12 | Chi et al., Fast Fourier Convolution, NeurIPS 2020 | II-E |
| 13 | Rao et al., Global Filter Networks, NeurIPS 2021 | II-E |
| 14 | Kervadec et al., Boundary loss, Medical Image Analysis 2021 | II-F |
| 15 | Cheng et al., Boundary IoU, CVPR 2021 | II-F |
| 16 | Zhou et al., MAFF-Net, BSPC 113 (2026) 109222 | II-G |
| 17 | Gutman et al., ISIC 2016 challenge | IV-A |
| 18 | Codella et al., ISIC 2017 challenge, ISBI 2018 | IV-A |
| 19 | Codella et al., ISIC 2018 challenge | IV-A |

Count is 19 rather than the nominal 10–15 because the four implemented baselines, the
SAM2 backbone, the Boundary IoU measure, the Original MAFF-Net, and the three datasets
are all *used* in the work and must each be attributed.

---

## 5. Assumptions made

1. **Title, keywords, and dataset citations were completed** in addition to the Abstract
   and References. All were outstanding placeholders or attribution gaps; the phase brief
   was to "complete the remaining manuscript writing." The title in particular is an
   editorial choice and is offered as a proposal.
2. **Abstract written as a single paragraph**, the prevailing IEEE/Elsevier convention,
   following the requested six-step logical flow without sub-headings.
3. **References formatted in IEEE style** in Markdown (no LaTeX, no BibTeX), consistent
   with the instruction to defer all formatting conversion.
4. **Section I left uncited.** Adding citation anchors to the Introduction would renumber
   the entire list, because IEEE numbers by order of first appearance and Section I
   precedes Section II. This is deferred deliberately (see §8).

---

## 6. Repository information used

- `Journal_Paper/References/References.md` — the authoritative in-repository reference
  draft. Used to obtain the **verified** MAFF-Net citation and the three ISIC dataset
  citations. Entries in that file marked `[VERIFY]` (SGT-Net, BGFF-UNet, FastTrans-Net,
  and several unattributed frequency-learning entries) were **not** used, as their
  metadata is unconfirmed.
- `reports/architecture/MAFFNET_DESIGN.md`, `reports/implementation/*` — cross-checked
  the MAFF-Net authorship and article number in four independent files.
- `train.py`, `models/` — confirmed the identity of the four implemented baselines and
  the SAM2 Hiera-Large backbone, so that each is cited.
- `paper2/results/*.csv` — no new values used; the Abstract deliberately contains no
  numerical results.

---

## 7. Consistency fixes applied

1. **Corrected a fabricated citation (important).** Reference [16] was first drafted with
   invented author initials ("Z. Zhou, Y. Wang, W. Li") and an invented title
   ("Multi-scale attention feature fusion network"). This was caught before completion and
   replaced with the citation verified in the repository:
   *Y. Zhou, H. Gong, R. Qiu, S. Wei, Z. Li, and W. Zhang, "MAFF-Net: SAM-powered mixed
   multi-scale perception adaptation and frequency-guided feature fusion for robust skin
   lesion segmentation," Biomedical Signal Processing and Control, vol. 113, art. no.
   109222, 2026.*
2. **Dataset attribution gap closed.** The manuscript used three public datasets without
   citing any of them. Anchors [17]–[19] were added in IV-A. Appending them as new numbers
   preserves order-of-first-appearance and renumbers nothing.
3. **Abstract trimmed** from 258 to 241 words to meet the 180–250 requirement.
4. **Citation integrity verified programmatically**: 19 cited, 19 listed, no orphans, no
   dangling markers, first-appearance order correct.

---

## 8. Outstanding before submission (not addressed in this phase)

1. **Figures do not exist as files.** Figs. 1–3 (architecture, frequency branch, gated
   fusion) are author-drawn schematics that have never been produced. Fig. 4 (qualitative
   comparison) is assemblable without new inference from `experiments/*/samples/`, whose
   inputs and ground truths are byte-identical across models. Tracked as C2/C4 in
   `paper2/audits/21_Paper_Asset_Audit.txt`.
2. **Section I carries no citations.** Anchors are required for the epidemiology and
   dermoscopy claims and for the architectures named in its fourth paragraph. Adding them
   **will renumber the entire reference list** and should be done in one pass before the
   list is frozen.
3. **Verify all bibliographic metadata** for entries [1]–[15] against the original sources.
   These were composed from model knowledge, not from the repository; the works are real
   and standard, but page ranges, workshop-versus-journal venue for [3], and arXiv
   identifiers for [7] and [10] should be confirmed.
4. **Remaining scientific gaps** (unchanged, all disclosed in the manuscript): single-seed
   experiments with no statistical analysis; no ablation of the boundary-supervision terms;
   evaluation confined to three ISIC benchmarks.

---

## 9. Manuscript state

| Element | Status |
|---|---|
| Title, Abstract, Keywords | Complete |
| Sections I–VII | Complete |
| Tables I–III | Complete, values verified against canonical CSVs |
| References [1]–[19] | Complete, no orphans or dangling markers |
| Figures 1–4 | **Referenced but not produced** |

Ready for the reviewer audit and scientific consistency audit. Not yet ready for the
manuscript freeze, pending the figure assets and the Section I citation pass.
