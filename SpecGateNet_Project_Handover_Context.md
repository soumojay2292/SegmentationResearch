# SpecGate-Net — Project Handover / Context Document

*Generated as a portable context transfer for continuing this project on another
machine or Claude Code session. Read-only reconstruction; no manuscript or code
was modified while writing this. Facts not recoverable from the project are
explicitly marked **"Not available from project context."***

Repository root (this machine): `C:\Users\Soumojay\Desktop\New_Sm_P\SegmentationResearch`
Paper subtree: `paper2/`
Date of handover: 2026-09-03.

---

## 1. Project objective and current manuscript status

**Objective.** An IEEE Transactions–style journal manuscript ("Paper 2") on
skin-lesion segmentation. It introduces **SpecGate-Net** (Spatial–Spectral Gated
Fusion Network), a boundary-focused refinement of the fusion stage of the
published **MAFF-Net** architecture, and argues methodologically that
region-overlap metrics alone are insufficient for margin-focused methods.

**Current status: verified, camera-ready, 20 pages, 6-model experimental setup.**
- LaTeX compiles clean: **20 pages, 33 references**, no `??`, no undefined
  references, no overfull boxes (only 3 minor underfull-hbox warnings).
- The manuscript was migrated from a 7-model setup to a **final 6-model setup**
  by removing the "Paper-faithful Reconstruction" from the reader-facing paper;
  statistics were **recomputed from per-image data**, not hand-edited.
- A final source package, a regenerated 6-model Word version, and a full project
  archive have been produced (Section 13, 14).

---

## 2. Complete paper/model story and final positioning

**Title:** *SpecGate-Net: Spatial–Spectral Gated Fusion for Boundary-Faithful
Skin Lesion Segmentation* (en-dash in "Spatial–Spectral").
**Authors:** Soumojay Karmakar, Lokesh Singh — Manipal Institute of Technology
Bengaluru, Manipal Academy of Higher Education, Manipal, India.

**Story.** A frozen SAM2 (Hiera-Large) encoder yields four multi-scale features.
SpecGate-Net replaces the single frequency-guided fusion route of MAFF-Net with a
**dual-stream** pathway: a spatial stream + a parameter-free frequency stream,
whose statistics are reconciled (Feature Alignment) and merged by a learned,
spatially varying gate (Gated Fusion). The frozen encoder, convolutional decoder,
and auxiliary reconstruction branch (RDB) are **retained from MAFF-Net** and not
claimed as contributions. Training adds Boundary-centric Dual Supervision (BDS):
Boundary-aware Deep Supervision + Edge-weighted Reconstruction.

**Final positioning (honest, deliberate):**
- SpecGate-Net attains **higher Dice, IoU, and boundary-based scores than the four
  baselines** across the three benchmarks (statistically significant).
- SpecGate-Net is **NOT** claimed superior to MAFF-Net [15]; the paper states it
  "is not established as an improvement over the MAFF-Net [15], which remains at
  least its equal and is significantly closer to the reference contours on the
  largest benchmark."
- No state-of-the-art claim; no literature-number comparison.
- Three stated contributions: (1) region+boundary combined evaluation reveals
  ranking disagreement; (2) SpecGate-Net beats the baselines; (3) controlled
  leave-one-out ablation isolating the three components' effect on boundary fidelity.

---

## 3. MAFF-Net [15] vs SpecGate-Net relationship

- **MAFF-Net [15]** = the published prior-work architecture (ref15: Y. Zhou et al.,
  *Biomedical Signal Processing and Control*, 113, 109222, 2026). In the paper it
  appears both as the cited prior work AND as an **evaluated reference control**:
  the published architecture **reproduced under this study's protocol**. The table
  captions carry a footnote: *"The MAFF-Net [15] rows report a reproduction of the
  published architecture trained and evaluated under the protocol of this study,
  not the results reported in [15]."*
- **SpecGate-Net** = the sole proposed model; a targeted dual-stream refinement of
  MAFF-Net's fusion stage. Same frozen encoder + decoder + RDB as MAFF-Net [15];
  differs only in the fusion pathway. Because both share the frozen encoder and
  differ only in fusion, differences between them are attributable to the fusion
  design.
- **Naming history:** the evaluated MAFF-Net reproduction was previously called
  "Restored MAFF-Net"; it was renamed to **"MAFF-Net [15]"** throughout. Internally
  in code/data it is still keyed as **"MAFF-Net (Restored)"** (see Section 9).

---

## 4. All major manuscript edits and decisions made (chronological)

1. Ablation delta fix (0.0196 → 0.0195).
2. Scoped baseline-superiority claims to "higher Dice, IoU, and boundary-based
   scores than the evaluated baselines" (never vs. the reproduction).
3. Positioning revisions (conceptual contrast vs. MAFF-Net MMPA/FGFF; value of the
   controlled comparison; methodological contribution of ranking disagreement).
4. Recast the two MAFF-Net reproductions as prior-work reproduction controls of
   the Original MAFF-Net [15]; single proposed model.
5. Regenerated **per-image metrics** (evaluation-only, reproducing the tables) and
   ran Friedman + Holm-adjusted Wilcoxon + rank-biserial + bootstrap CIs.
6. Added a compact contextual (qualitative) comparison with MAFF-Net [15].
7. Optimised figure captions; fixed figure alignment/routing (Figs 1–3).
8. Moved metric definitions (Accuracy/Dice/IoU/Sensitivity/Specificity) into a
   3-column **Table I** (Metric | Expression | Purpose); kept ASSD/HD95 as display
   equations (8)/(9).
9. Expanded references to **33** (verified DOIs); citation-completeness pass added
   deep-supervision, Instance Normalization, AdamW citations.
10. Renamed proposed model **"Proposed MAFF-Net" → "SpecGate-Net"** (full name
    "Spatial–Spectral Gated Fusion Network" at first use in abstract + intro).
11. Terminology standardization: **"Original MAFF-Net" → "MAFF-Net"** for the
    published model; then bound "MAFF-Net framework/fusion stage" phrases to
    MAFF-Net~\cite{ref15}.
12. Renamed the evaluated reproduction **"Restored MAFF-Net" → "MAFF-Net [15]"**
    everywhere (tables, prose, figures) + provenance footnotes.
13. Title changed to the current title (Section 2).
14. Figure 1 boxes enlarged (padding/readability); Figure 3 F_s/F_f labels aligned.
15. **Removed "Paper-faithful Reconstruction" from the reader-facing paper**
    (final 6-model setup) and **recomputed 6-model statistics** (Section 8).

All edits preserved scientific content, equations, tables, numbers, citations, and
experiments. Standing constraint throughout: **never fabricate results, citations,
equations, predictions, sources, or similarity percentages.**

---

## 5. Final section structure and important content decisions

Sections (IEEEtran, numbered I–VII), each an `\input` file in
`paper2/latex/sections/`:

| # | Section (file) | Notes |
|---|---|---|
| — | `abstract.tex` | Full model name introduced once; bold IEEE abstract |
| I | Introduction (`introduction.tex`) | Threefold contributions; MAFF-Net components "not claimed as contributions" |
| II | Related Work (`related_work.tex`) | Surveys U-Net/attention/nested/transformer/SAM/frequency/boundary methods |
| III | Methodology (`methodology.tex`) | III-A Overall Framework, Frozen Backbone (III-C), Frequency Branch, Feature Alignment, Gated Fusion, Decoder & Reconstruction Pathway, Boundary-centric Dual Supervision, Loss |
| IV | Experiments/Setup (`experiments.tex`) | Datasets, protocol, **Table I metrics**, ASSD/HD95 eqs, baselines, ablation design, **Statistical Analysis** subsection |
| V | Results (`results.tex`) | V-A Benchmark (Table II), V-B Boundary (Table III), V-C Statistical Significance (Table IV), V-D Architecture Ablation (Table V), V-E Qualitative (Figure 4) |
| VI | Discussion (`discussion.tex`) | Component interpretation, boundary-vs-region, frozen backbone, Limitations |
| VII | Conclusion (`conclusion.tex`) | Honest positioning; no SOTA claim |

Content decisions: single proposed model; superiority only over the 4 baselines;
ablation is SpecGate-Net-internal (not vs. reproductions); qualitative panel shows
6 models (baselines + MAFF-Net [15] + SpecGate-Net).

---

## 6. Equations, tables, figures, captions — current status

**Equations (display, numbered):** (1) Dice loss, (2) BCE loss (split env),
(3) L_seg, (4a) edge operator S(m), (4b) L_BDS, (5) λ_B(t) warm-up, (6) L_EWR,
(7) total L, (8) ASSD, (9) HD95. All present and correct (tags 1,2,3,4a,4b,5,6,7,8,9).
Region metric definitions live in **Table I** (as fractions), not as numbered
display equations.

**Tables (final):**
- **Table I** — `tab:metrics` (`experiments.tex`): Metric | Expression | Purpose;
  5 rows (Accuracy, Dice, IoU (Jaccard), Sensitivity, Specificity) with `\dfrac`.
- **Table II** — `tab:benchmark` (`results.tex`): Region-based; **6 rows/dataset
  block** × 3 datasets; cols Dice, IoU, Accuracy, Sensitivity, Specificity.
- **Table III** — `tab:boundary`: Boundary-based; 6 rows/block; cols HD95(px),
  ASSD(px), Boundary IoU, n_valid/n.
- **Table IV** — `tab:stats`: **5 columns** (Dataset, Metric, median Δ, r, p_Holm)
  — SpecGate-Net vs MAFF-Net [15] only; baselines summarized in caption.
- **Table V** — `tab:ablation`: Leave-one-out ablation, 4 configs; **unchanged by
  the 6-model migration**.

**Figures (vector):** Figure1 Architecture, Figure2 Frequency Branch, Figure3
Feature Alignment & Gated Fusion, Figure4 Multi-dataset qualitative. All included
from `paper2/latex/figures/*.pdf`; captions optimized. Figure 4 caption's
"Paper-faithful … not shown" clause was removed (image already excludes it).

Caption formatting: `\captionsetup{font=footnotesize}` (8 pt) applied globally.

---

## 7. Final 6-model experimental setup

Six evaluated models:
1. U-Net · 2. Attention U-Net · 3. UNet++ · 4. TransUNet (baselines)
5. **MAFF-Net [15]** (prior-work reproduction / reference control)
6. **SpecGate-Net** (proposed)

Common protocol: input 384×384; frozen SAM2.1 Hiera-Large encoder; batch size 2;
AdamW (lr 1e-3, weight decay 1e-2, linear decay to 0); up to 40 epochs; mixed
precision (bfloat16); grad-clip norm 1.0. SpecGate-Net + MAFF-Net [15] trained with
the BDS composite objective; the four baselines with Dice + class-weighted BCE and
early stopping (patience 10). Best checkpoint by validation Dice; evaluated on the
held-out **test** split. (The removed Paper-faithful Reconstruction was a
from-scratch reproduction trained without pretrained encoder weights.)

---

## 8. Statistical methodology and final recomputed statistics

**Methodology.** Per-image region + boundary metrics; images are the unit of
analysis. For each (dataset, metric): a **Friedman** omnibus over the 6 models;
where rejected, **Wilcoxon signed-rank** of SpecGate-Net vs each of the other 5,
with **Holm** correction over those 5 comparisons; matched-pairs **rank-biserial**
effect size r; **percentile bootstrap** 95% CI of the median paired difference
(10,000 resamples). HD95/ASSD use **complete-case pairing** (images where both
models produced a non-empty prediction). Two-sided, α = 0.05. RNG seed 20260826.

**Final 6-model numbers (recomputed from per-image data; verified in the paper):**
- Friedman: max p across all dataset·metric = **4.104×10⁻³⁹** → paper states
  "all p < 10⁻³⁸".
- SpecGate-Net vs the 4 baselines: **all significant**, max p_Holm = **1.304×10⁻¹⁰**
  (paper bound "≤ 1.4×10⁻¹⁰"), min |r| = **0.538** (paper bound "|r| ≥ 0.53").
- SpecGate-Net vs MAFF-Net [15]: indistinguishable on ISIC 2016 & 2017 (p_Holm ≥ 0.23,
  |r| ≤ 0.10); on ISIC 2018 four significant differences favour MAFF-Net [15]
  (Dice, IoU, HD95, ASSD), Boundary IoU not significant.
- **Table IV vs-MAFF-Net cells** (median Δ / r / p_Holm), by dataset:
  - 2016: Dice +0.0009/0.00/0.962 · IoU +0.0017/0.00/0.968 · HD95 −0.0598/−0.02/0.789 · ASSD −0.1204/−0.03/0.721 · B-IoU +0.0100/+0.08/0.357
  - 2017: Dice +0.0007/+0.05/0.580 · IoU +0.0011/+0.05/0.550 · HD95 −0.0711/+0.04/0.631 · ASSD −0.0465/−0.04/0.613 · B-IoU +0.0055/+0.10/0.238
  - 2018: Dice −0.0029/−0.14/0.015* · IoU −0.0053/−0.14/0.017* · HD95 +0.4843/+0.23/<10⁻³* · ASSD +0.1473/+0.16/0.006* · B-IoU −0.0104/−0.10/0.089

Note: median Δ and r are pairwise → unchanged from the 7-model run; only the
Holm-adjusted p-values changed (family shrank 6→5 comparisons; Holm became less
conservative → no significant result flipped).

Output files: `paper2/results/stats6/friedman.csv`,
`paper2/results/stats6/posthoc_proposed.csv` (final 6-model). The 7-model
`paper2/results/stats/` artifacts are retained as historical provenance.

---

## 9. Datasets, checkpoints, evaluation pipeline, result files

**Datasets:** ISIC 2016 / 2017 / 2018, at `dataset_split/ISIC_{2016,2017,2018}/`
with `{train,val,test}/` (images, masks) and `{train,val,test}.csv`.
Raw/unsplit data: `dataset_raw/`, `datasets/` (helpers/scripts).

**Frozen backbone weights:** `checkpoints/sam2.1_hiera_large.pt` (~898 MB).

**Model → code → checkpoint mapping** (internal keys → paper name → files):
| Paper name | Internal key | Model file / class | Final checkpoints |
|---|---|---|---|
| SpecGate-Net | MAFF-Net (Proposed) | `models/specgate_net.py` / `SpecGateNet` | `experiments/stabilized_maffnet_ISIC{2016,2017,2018}/model.pth` |
| MAFF-Net [15] | MAFF-Net (Restored) | `models/maffnet.py` / `MAFFNet` | `experiments/maffnet_ISIC{2016,2017,2018}_final/model.pth` |
| (removed; archived) Paper-faithful Recon. | MAFF-Net (Paper-faithful Recon.) | `archive/models_obsolete/maffnet_original_reconstructed.py` / `MAFFNetOriginalReconstructed` | `archive/experiments/maffnet_original_ISIC{2016,2017,2018}_final/model.pth` |
| U-Net / Attn U-Net / UNet++ / TransUNet | same | `models/{unet,attention_unet,unet_plus_plus,transunet}.py` | see below |

Exact per-model checkpoint/metrics.json paths (incl. baselines, whose dir names are
inconsistent) are authoritative in **`paper2/results/final_benchmark_table.csv`**
(`source_metrics_json` column), e.g. baselines: `unet_ISIC_2016`, `unet_final_2017`,
`unet_ISIC_2018`; `attn_unet_ISIC_2016/2017`, `attention_unet_ISIC2018_final`;
`unetpp_ISIC_{2016,2017,2018}`; `transunet_ISIC_{2016,2017,2018}`.

**Entry point:** `train.py` (trains + auto-evaluates on test → `experiments/<exp>/metrics.json`).
It defines its own inline `SegDataset` (img 384, sigmoid>0.5, primary head p1).
`--model` ∈ {specgate_net, maffnet, attention_unet, unetpp, unet, transunet}.

**Evaluation-only per-image regeneration:** `analysis/eval_perimage.py` (in the
archive; original at scratchpad) — reuses `train.SegDataset` + the exact predict
path; region metrics via project `compute_metrics`; boundary via **MedPy**
`hd95`/`assd` (unit spacing, 384 grid) + **Boundary IoU** (Cheng et al., CVPR'21,
band = 0.02×diagonal). Writes `paper2/results/perimage/<model>__<dataset>.csv`.

**Key result files:**
- `paper2/results/perimage/*.csv` — per-image metrics (21 CSVs; includes recon as raw data).
- `paper2/results/stats6/{friedman,posthoc_proposed}.csv` — final 6-model stats.
- `paper2/results/final_benchmark_table.{csv,md,tex}` — region benchmark (Tables II).
- `paper2/results/boundary_metrics.{csv,md,tex}` — boundary benchmark (Table III).
- `paper2/results/_boundary_raw.json`, `_ablation_boundary_raw.json` — raw boundary means.
- `paper2/audits/` — 20+ audit logs documenting the pipeline (e.g. `20_Boundary_Evaluation.txt`).

Note: `final_benchmark_table.*` and `boundary_metrics.*` internal artifacts still
contain the recon/Paper-faithful rows (raw research record); the reader-facing
tables in the LaTeX are 6-model.

---

## 10. Ablation setup and results (Table V, §V-D — unchanged)

Leave-one-out on **ISIC 2018** (largest/most heterogeneous). Each of the three
components disabled in turn while the rest of the network + training config are held
fixed; the full model is the canonical SpecGate-Net checkpoint (reused, not retrained).

| Configuration | Dice | IoU | Accuracy | HD95 (px) | ASSD (px) | Boundary IoU |
|---|---|---|---|---|---|---|
| Full SpecGate-Net | 0.8701 | 0.7900 | 0.9493 | **22.554** | **9.240** | **0.4611** |
| without Frequency Branch | 0.8708 | 0.7868 | 0.9496 | 23.574 | 9.546 | 0.4416 |
| without Feature Alignment Module | 0.8649 | 0.7811 | 0.9502 | 23.676 | 9.869 | 0.4326 |
| without Gated Fusion Module | **0.8725** | **0.7907** | **0.9520** | 25.546 | 9.488 | 0.4508 |

Finding: disabling any one component degrades **all three boundary measures**;
region measures are not maximized by the full model (no region-overlap claim). B-IoU
reductions: −0.0195 (Freq), −0.0285 (FA), −0.0103 (GF). Ablation code:
`models/specgate_net_ablation.py` (class `SpecGateNetAblation`); ablation checkpoints
`experiments/ablation_no_{frequency,alignment,gated_fusion}/`.

---

## 11. Citation / reference status

- `paper2/latex/references.bib` — **33 entries**; every entry cited exactly once
  by first-appearance (verified 33 cited == 33 defined, no undefined, none uncited,
  contiguous [1]–[33] in the PDF).
- **Numbering is by IEEEtran first-appearance order, NOT by the `refNN` key digits.**
  Map (key → number): ref1..ref18 → 1..18; ref24→19, ref25→20, ref26→21, ref27→22,
  ref19→23, ref28→24, ref29→25, ref31→26, ref32→27, ref20→28, ref21→29, ref22→30,
  ref33→31, ref23→32, ref30→33.
- ref15 = MAFF-Net (Zhou et al., BSPC 113:109222, 2026). ref10 = SAM 2. ref12 =
  Boundary IoU (Cheng et al., CVPR'21). ref30 = Metrics Reloaded; ref29 = metric
  pitfalls. ref31 Instance Normalization, ref32 Deeply-Supervised Nets, ref33 AdamW.

---

## 12. Figure-generation workflow and important scripts

Figures are hand-built **SVG** emitted by Python, converted to PDF via **PyMuPDF
(fitz)** `convert_to_pdf()`. Generators (in the archive `02_code/figure_scripts/`;
originals in the session scratchpad):
- `gen_fig1.py` → Figure1_Architecture (SVG+PDF)
- `gen_fig2.py` → Figure2_FrequencyBranch
- `gen_fig3.py` → Figure3_FeatureAlignment_GatedFusion
- `gen_fig4_multi.py` → Figure4_MultiDataset (embeds dermoscopy sample PNGs from
  `experiments/<model>_ISIC*/samples/`)

Each writes to `paper2/figures/*.svg` and `paper2/figures/*.pdf`; the PDFs are then
copied into `paper2/latex/figures/`. Figure 4 columns are exactly: Input, Ground
Truth, U-Net, Attention U-Net, UNet++, TransUNet, MAFF-Net [15], SpecGate-Net.

---

## 13. Word / PDF / source-package status

- **Final PDF:** `paper2/latex/main.pdf` (20 pages) and copies in the packages.
- **Final LaTeX source package (self-contained):** `paper2/submission/`
  (main.tex, sections/, figures/, references.bib, main.bbl, IEEEtran.cls/.bst,
  main.pdf, README_SUBMISSION.txt). Zipped: `paper2/SpecGateNet_submission.zip`.
- **Final editable Word doc (6-model):** produced by the LaTeX→Word pipeline
  (`build_tables.js` native tables → `preprocess.py` → pandoc → `assemble.py`);
  delivered inside the archive at
  `SpecGateNet_FinalPackage/01_paper/SpecGateNet_Paper.docx` (native merged tables,
  editable OMML equations, 4 figures at 300 DPI, two-column IEEE styling, numbered
  headings). The earlier standalone tables-only docx and any 7-model Word doc are
  superseded.
- **Full project archive:** `SpecGateNet_FinalPackage/` and
  `SpecGateNet_FinalPackage.zip` (~19 MB, 101 files) — see Section 14.

---

## 14. Important project directories and exact file paths

Base: `C:\Users\Soumojay\Desktop\New_Sm_P\SegmentationResearch\`

- `paper2/latex/` — **working manuscript (source of truth)**: `main.tex`,
  `sections/*.tex` (8), `references.bib`, `figures/Figure1–4.pdf`, `main.pdf`.
- `paper2/submission/` — clean self-contained LaTeX package (+ IEEEtran.cls/.bst, main.bbl).
- `paper2/figures/` — vector figure sources (`Figure1–4.{svg,pdf}`).
- `paper2/results/` — data: `perimage/`, `stats6/` (final), `stats/` (7-model, historical),
  `final_benchmark_table.*`, `boundary_metrics.*`, `_boundary_raw.json`, `_ablation_boundary_raw.json`.
- `paper2/audits/` — pipeline/QA audit logs.
- `models/`, `modules/`, `training/`, `datasets/`, `utils/`, `train.py`, `requirements.txt` — code.
- `experiments/` — training runs + checkpoints (`<exp>/model.pth`, `metrics.json`, `config.txt`).
- `dataset_split/` — the ISIC splits; `checkpoints/sam2.1_hiera_large.pt` — SAM2 weights.
- `SpecGateNet_FinalPackage/` — the packaged handover archive:
  `01_paper/` (latex_source/, SpecGateNet_Paper.docx, SpecGateNet_FinalPaper_20pp.pdf),
  `02_code/`, `03_data/`, `04_figures/`, `README.md`.
- Session scratchpad (machine-specific, may not transfer):
  `C:\Users\Soumojay\AppData\Local\Temp\claude\C--Users-Soumojay-Desktop-New-Sm-P\d9396353-c9a1-46c9-b57a-e755af7cb994\scratchpad\` — holds the original
  `stats.py`, `stats6.py`, `eval_perimage.py`, `gen_fig*.py`, and `docx_work/`
  (build_tables.js, preprocess.py, assemble.py). **These are also copied into
  `SpecGateNet_FinalPackage/02_code/`, which is the portable copy.**

---

## 15. Files intentionally archived / removed and why

- `paper2/_archive_superseded_20260830/` — superseded Markdown manuscript draft,
  old Figure5/Figure6 (extra metric bar charts, not used), and old Figure4 variants.
  Archived (moved, reversible), not hard-deleted, because there is no git history.
- Reader-facing **"Paper-faithful Reconstruction"** removed from the manuscript
  (final 6-model) and its code/checkpoints moved to `archive/` during the repository
  cleanup. They are **retained** (`archive/models_obsolete/maffnet_original_reconstructed.py`,
  `archive/experiments/maffnet_original_ISIC*_final/`) for data reproducibility; the
  model is not active code and is not reported.
- Superseded 7-model Word doc / standalone tables docx: superseded by the 6-model
  Word doc in the archive.
- Excluded from packages (size/regenerable): model checkpoints, `sam2.1_hiera_large.pt`,
  raw ISIC images.

---

## 16. Current verified build / page count / QA

- LaTeX compiles from `paper2/latex/` and from `paper2/submission/`: **20 pages,
  33 references**, no `??`, no undefined references, no missing files, no overfull
  boxes; 3 minor underfull-hbox warnings (`main.bbl:38`, `methodology:125–126`,
  `results:117–118`) — cosmetic only.
- Terminology sweep: **0** occurrences of "Paper-faithful", "Restored",
  "Proposed MAFF-Net", "two reproductions", "seven models" in sources or PDF.
- Table IV / §V-C statistics **match `paper2/results/stats6/` exactly**.
- Citations 33/33; contiguous [1]–[33]. Equations 10 tags present. Figures 1–4 present.
- RDB / "auxiliary reconstruction branch" (architecture) intact and distinct from
  the removed model.
- Word doc verified via Microsoft Word render: 6 native tables, ~83 OMML equations,
  4 figures, no Paper-faithful/Restored.

---

## 17. Known limitations / caveats (as stated in the manuscript)

- **Single training run per configuration** (one seed); the significance tests treat
  images as the unit and characterise within-run dispersion, **not** run-to-run
  variability. Repetition over several seeds is the stated top priority.
- Ablation is on **ISIC 2018 only**; consistency across the other two benchmarks is
  unestablished.
- The BDS boundary supervision terms were active in every configuration and **not
  ablated** (their individual contribution unmeasured).
- No configuration fine-tunes the encoder (freeze-vs-fine-tune not assessed).
- Evaluation confined to three ISIC dermoscopic benchmarks (one modality/protocol).
- No numerical comparison to the literature; MAFF-Net [15]'s published numbers use
  different splits/protocol/backbone and are not directly comparable.
- SpecGate-Net does **not** surpass MAFF-Net [15] (honest positioning retained).
- Qualitative examples are from the validation split (illustrative only).

---

## 18. What must NOT be changed

- **Do not** claim SpecGate-Net superior to MAFF-Net [15] (MAFF-Net [15] is at least
  its equal / better on the majority of measures). Superiority is only over the four
  baselines.
- **Do not** make a state-of-the-art claim or add a numeric literature comparison.
- **Do not** fabricate results, citations, equations, predictions, sources, or
  similarity percentages. All statistics must come from `stats6.py` on the per-image
  data.
- **Do not** re-introduce "Paper-faithful Reconstruction", "Restored", "Proposed
  MAFF-Net", or a 7-model setup into the reader-facing paper.
- **Do not** rename or alter Original MAFF-Net [15] as prior work; keep SpecGate-Net
  as the sole proposed model.
- **Do not** modify canonical model files or checkpoints (the reported numbers depend
  on them).
- Preserve all equations, tables, numbers, captions, references, and IEEE formatting.
- Reference numbering is auto (first-appearance); do not hand-number.

---

## 19. Remaining tasks / recommended next steps

- **Multi-seed repetition** (≥3 seeds) with interval estimates — the manuscript's own
  stated top priority; would let the small SpecGate-vs-MAFF-Net differences be
  assessed across runs.
- Extend the **ablation to ISIC 2016/2017** to establish cross-dataset consistency.
- **Ablate the BDS terms** (BDS / EWR) to quantify their individual contribution.
- Optional: a fine-tuned-encoder configuration to assess freeze vs. fine-tune.
- Target-venue formatting pass (e.g., IEEE PDF eXpress / arXiv) if/when submitting.
- Nothing is outstanding for the current manuscript itself — it is verified/complete.

---

## 20. Exact commands / workflows

Absolute tool paths on this machine (adjust on a new system):
- tectonic: `C:\Users\Soumojay\miniconda3\envs\tex\Library\bin\tectonic.exe`
- Python (has fitz/scipy/medpy/python-docx/pypandoc): `C:\Users\Soumojay\miniconda3\envs\segmentation\python.exe`
- Node (docx-js): `node` (v24)
- pandoc: bundled via pypandoc at
  `C:\Users\Soumojay\miniconda3\envs\segmentation\Lib\site-packages\pypandoc\files\pandoc.exe`

**Compile the PDF** (from `paper2/latex/` or `paper2/submission/`):
```bash
tectonic main.tex
# or: pdflatex main; bibtex main; pdflatex main; pdflatex main   (main.bbl included in submission/)
```

**Verify pages / references / no ?? (Python + fitz):**
```python
import fitz, re
d = fitz.open("main.pdf"); full = "\n".join(p.get_text() for p in d)
print(d.page_count, "??" in full, max(int(x) for x in re.findall(r"\[(\d{1,2})\]", full)))
```

**Train / evaluate a model:**
```bash
python train.py --model specgate_net --dataset ISIC_2018 --data_root dataset_split \
  --epochs 40 --batch_size 2 --img_size 384 --lr 1e-3 \
  --checkpoint checkpoints/sam2.1_hiera_large.pt --exp specgate_ISIC2018
# --model in {specgate_net, maffnet, attention_unet, unetpp, unet, transunet}
```

**Regenerate per-image metrics, then statistics:**
```bash
python analysis/eval_perimage.py        # -> paper2/results/perimage/*.csv
python analysis/stats6.py               # -> paper2/results/stats6/{friedman,posthoc_proposed}.csv (FINAL 6-model)
# analysis/stats.py is the historical 7-model version (provenance only)
```

**Regenerate figures:**
```bash
python figure_scripts/gen_fig1.py       # ... gen_fig2.py gen_fig3.py gen_fig4_multi.py
# outputs SVG+PDF to paper2/figures/ ; copy the PDFs into paper2/latex/figures/
```

**Regenerate the Word document (LaTeX -> Word):**
```bash
cd word_build
node build_tables.js                    # native merged tables -> SpecGateNet_Tables.docx
python preprocess.py                    # LaTeX body -> body.tex + meta.json (resolves \cite/\ref)
pandoc -f latex -t docx --mathml body.tex -o body.docx
python assemble.py                      # -> SpecGateNet_Paper.docx (splices tables, inserts figures, IEEE styling)
# render/verify via Microsoft Word COM (Windows) or LibreOffice
```

**Environment:** `pip install -r requirements.txt` (PyTorch, numpy, scipy, medpy,
opencv/scipy for Boundary IoU, python-docx + pypandoc for the Word pipeline; SAM2
via `src/sam2`). Boundary metrics need **MedPy** (pulls SimpleITK).

---

### Provenance note
This document was reconstructed from the project files and the work performed in the
Claude Code session. Items that could not be verified from the project are marked
"Not available from project context." No manuscript or code files were modified while
producing this handover.
