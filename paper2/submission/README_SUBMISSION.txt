SpecGate-Net — IEEE Manuscript Submission Source Package
========================================================

Title : SpecGate-Net: Spatial-Spectral Gated Fusion for Boundary-Faithful
        Skin Lesion Segmentation
Class : IEEEtran (journal)
Output: 20 pages, 33 references, 4 figures, 5 tables.

CONTENTS
--------
  main.tex                 Main document (\documentclass[journal]{IEEEtran})
  references.bib           BibTeX database (33 entries)
  main.bbl                 Pre-compiled bibliography (IEEEtran.bst output)
  IEEEtran.cls             Document class (bundled for self-containment)
  IEEEtran.bst            BibTeX style (bundled for self-containment)
  sections/               Body text, one file per section (\input by main.tex):
      abstract.tex  introduction.tex  related_work.tex  methodology.tex
      experiments.tex  results.tex  discussion.tex  conclusion.tex
  figures/                Vector figures (PDF) referenced by main.tex:
      Figure1_Architecture.pdf
      Figure2_FrequencyBranch.pdf
      Figure3_FeatureAlignment_GatedFusion.pdf
      Figure4_MultiDataset.pdf
  main.pdf                 Reference compiled output (20 pages)

BUILD
-----
The bibliography is pre-compiled (main.bbl included), so the document can be
built with the standard four-step sequence:

    pdflatex main
    bibtex   main        (optional; main.bbl already provided)
    pdflatex main
    pdflatex main

Required LaTeX packages (all standard TeX Live): fontenc, inputenc, cite,
amsmath, amssymb, graphicx, booktabs, multirow, subcaption (loads caption),
url, hyperref. No custom style files beyond the bundled IEEEtran.cls/.bst.

NOTES
-----
- Self-contained: compiles with no undefined references, missing files, or
  build errors; reproduces the 20-page main.pdf.
- Model set (6): U-Net, Attention U-Net, UNet++, TransUNet (baselines);
  MAFF-Net [15] (published prior-work architecture, reproduced under this
  study's protocol as a reference control); SpecGate-Net (proposed).
- Research/working artifacts (checkpoints, per-image results, analysis
  scripts, superseded drafts) are intentionally excluded from this package.
