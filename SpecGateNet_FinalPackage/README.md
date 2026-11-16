# SpecGate-Net — Final Project Archive

**Paper:** *SpecGate-Net: Spatial–Spectral Gated Fusion for Boundary-Faithful Skin Lesion Segmentation*
**Authors:** Soumojay Karmakar, Lokesh Singh (Manipal Institute of Technology Bengaluru, MAHE)

This archive is the verified **6-model** version of the study. Obsolete
7-model / "Restored" / "Paper-faithful Reconstruction" manuscript versions are
**excluded**. Scientific content, statistics, equations, tables, figures,
references, and manuscript formatting are unchanged from the verified manuscript.

## Models reported in the paper (6)
- **Baselines (4):** U-Net, Attention U-Net, UNet++, TransUNet
- **Prior work:** **MAFF-Net [15]** — the published MAFF-Net architecture,
  reproduced under this study's protocol as a reference control
- **Proposed:** **SpecGate-Net** (sole proposed model)

---

## Folder structure

```
SpecGateNet_FinalPackage/
├── README.md                         (this file)
├── 01_paper/
│   ├── latex_source/                 Final LaTeX source (self-contained, compiles)
│   │   ├── main.tex                  IEEEtran journal document
│   │   ├── sections/                 8 body sections (\input by main.tex)
│   │   ├── figures/                  Figure1–4 (PDF, vector)
│   │   ├── references.bib            33 references
│   │   ├── main.bbl                  pre-compiled bibliography
│   │   ├── IEEEtran.cls / .bst       document class + bib style (bundled)
│   │   ├── main.pdf                  reference compiled output (20 pp)
│   │   └── README_SUBMISSION.txt     build notes for the source package
│   ├── SpecGateNet_Paper.docx        Final editable Word version (6-model)
│   └── SpecGateNet_FinalPaper_20pp.pdf   Final 20-page PDF
├── 02_code/
│   ├── train.py                      Single training/eval entry point
│   ├── models/                       SpecGate-Net + baselines + reproductions
│   │   ├── stabilized_maffnet.py     ← SpecGate-Net (StabilizedMAFFNet)
│   │   ├── maffnet.py                ← MAFF-Net [15] reproduction (MAFFNet)
│   │   ├── maffnet_original_reconstructed.py   from-scratch reproduction (not reported; kept for data reproducibility)
│   │   ├── unet.py / attention_unet.py / unet_plus_plus.py / transunet.py   baselines
│   │   └── sam2_loader.py            frozen SAM2 Hiera-Large loader
│   ├── modules/                      frequency_branch, feature_alignment, gated_fusion
│   ├── training/                     trainer.py (MAFFNetTrainer), simple_trainer.py
│   ├── datasets/                     dataset utilities (train.py also defines its own SegDataset)
│   ├── utils/                        metrics.py, loss.py, reporting helpers
│   ├── analysis/                     eval_perimage.py, stats6.py (FINAL 6-model), stats.py (historical 7-model)
│   ├── figure_scripts/              gen_fig1.py … gen_fig4_multi.py (vector figure generators)
│   ├── word_build/                  build_tables.js, preprocess.py, assemble.py (LaTeX→Word pipeline)
│   └── requirements.txt
├── 03_data/
│   ├── stats_6model/                friedman.csv, posthoc_proposed.csv  (backs Table IV, §V-C)
│   ├── perimage/                    per-image metrics, one CSV per model×dataset
│   └── benchmark/                   final_benchmark_table.{csv,md}, boundary_metrics.{csv,md}  (back Tables II–III)
└── 04_figures/                      Figure1–4 vector sources (.svg + .pdf)
```

> **Not included (size / regenerable):** model checkpoints (`experiments/…/model.pth`,
> several GB), the frozen SAM2 weights (`checkpoints/sam2.1_hiera_large.pt`), raw ISIC
> image data, and superseded drafts. See "Reproduce" below for how to regenerate them.

---

## Reproduce the paper

### 1. Compile the PDF (LaTeX)
From `01_paper/latex_source/` (TeX Live or Tectonic; `main.bbl` is included so bibtex is optional):
```
tectonic main.tex          # or:  pdflatex main; bibtex main; pdflatex main; pdflatex main
```
Produces the 20-page `main.pdf`.

### 2. Open the Word version
`01_paper/SpecGateNet_Paper.docx` — native editable tables and equations (Word/LibreOffice).

### 3. Environment for the code
```
pip install -r 02_code/requirements.txt        # PyTorch, numpy, scipy, medpy, etc.
```
The frozen backbone requires the SAM2.1 Hiera-Large checkpoint
(`sam2.1_hiera_large.pt`); place it at `checkpoints/` and pass via `--checkpoint`.

### 4. Train / evaluate a model
```
python 02_code/train.py --model stabilized_maffnet --dataset ISIC_2018 \
       --data_root dataset_split --epochs 40 --batch_size 2 --img_size 384 \
       --lr 1e-3 --checkpoint checkpoints/sam2.1_hiera_large.pt --exp specgate_ISIC2018
```
`--model` ∈ {stabilized_maffnet (SpecGate-Net), maffnet (MAFF-Net [15]),
attention_unet, unetpp, unet, transunet}. Aggregate metrics are written to
`experiments/<exp>/metrics.json`.

### 5. Regenerate per-image metrics and statistics
```
python 02_code/analysis/eval_perimage.py      # → 03_data/perimage/*.csv
python 02_code/analysis/stats6.py             # → 03_data/stats_6model/{friedman,posthoc_proposed}.csv
```
`stats6.py` is the **final 6-model** analysis (Friedman + Holm-adjusted Wilcoxon +
rank-biserial effect size + bootstrap CIs). `stats.py` is the historical 7-model
version, retained for provenance only.

### 6. Regenerate the figures
```
python 02_code/figure_scripts/gen_fig1.py     # … gen_fig4_multi.py  → SVG + PDF in 04_figures/
```

### 7. Regenerate the Word document (optional)
`02_code/word_build/` converts the LaTeX to the native Word doc:
`build_tables.js` (native tables) → `preprocess.py` → pandoc → `assemble.py`.

---

## Verification status
- LaTeX source compiles with no errors, undefined references, or missing files → 20 pages, 33 references.
- Table IV / §V-C statistics match `03_data/stats_6model/` exactly (Friedman all p<10⁻³⁸;
  SpecGate-Net vs baselines all p_Holm ≤ 1.4×10⁻¹⁰, |r| ≥ 0.53).
- No reader-facing reference to "Paper-faithful Reconstruction", "Restored", or a 7-model setup.
