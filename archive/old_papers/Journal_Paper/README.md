# Journal Paper Workspace

Documentation-only workspace for the Q1 journal paper. The paper restores the
original **MAFF-Net** (Zhou et al., BSPC) implementation and extends it with our
**Boundary-centric Dual Supervision** contribution (boundary-aware deep
supervision + edge-weighted reconstruction).

**No source code, training, data, or experiment outputs live here** — only the
manuscript scaffold and its supporting assets. No paper content is written yet.

The manuscript must always keep three things distinct:
1. **Original MAFF-Net** components (as published).
2. **Restored** implementation (our faithful re-implementation of the original).
3. **Our contribution** (Boundary-centric Dual Supervision).

## Folder Guide

| Folder | Purpose |
|--------|---------|
| `00_Front_Matter/` | Title, abstract, keywords, highlights, author/affiliation notes. |
| `01_Introduction/` | Introduction section drafts. |
| `02_Related_Work/` | Related-work drafts (CNN, transformer, SAM/adaptation, frequency, boundary). |
| `03_Methodology/` | Method drafts: SAM2 encoder, MMPA, FGFF, decoder/deep supervision, the contribution (BDS + EWR), loss, training. |
| `04_Experimental_Setup/` | Datasets, implementation details, evaluation metrics, protocol. |
| `05_Results/` | Comparison, ablation, qualitative, efficiency results drafts. |
| `06_Discussion/` | Discussion, limitations, threats to validity. |
| `07_Conclusion/` | Conclusion and future work. |
| `Figures/` | Final figure image files (PNG/PDF/SVG) referenced by the manuscript. |
| `Tables/` | Table sources (CSV/Markdown/LaTeX); raw numbers from experiments. |
| `Equations/` | Equation sources (LaTeX snippets) for reuse across sections. |
| `References/` | Bibliography (`.bib`), citation exports, PDFs of cited works. |
| `Assets/` | Diagram sources (`.drawio`/`.svg`), notes, style files — anything not a final figure/table/reference. |

## Workspace Files

| File | Purpose |
|------|---------|
| `PAPER_OUTLINE.md` | Complete section/subsection hierarchy mapped to folders. Structural source of truth. |
| `SECTION_STATUS.md` | Per-section drafting status tracker (Not started / Drafting / Review / Done). |
| `JOURNAL_WORKSPACE_REPORT.md` | Summary of what this workspace contains. |

## Conventions

- One folder per major section (numbered for reading order).
- Empty folders kept with `.gitkeep` until drafts are added.
- Figures/Tables/Equations are referenced from sections, not duplicated as prose.

## Out of Scope

Training code, models, datasets, benchmark/report utilities, experiment outputs — these remain in the repository root and are **not** modified by this workspace.
