# IEEE LaTeX project — Paper 2

Clean IEEE `IEEEtran` project for the manuscript. **Content migrated** from
`../manuscript/Paper2_Manuscript.md`: all seven body sections plus the abstract,
14 numbered equations (1–4a/4b–13), 3 tables (I–III), 6 figures (1–6), and the
23 references.

## Structure
```
latex/
  main.tex            % document class, packages, front matter, \input list, bibliography
  references.bib      % BibTeX database (placeholder entries for now)
  README.md           % this file
  sections/
    abstract.tex  introduction.tex  related_work.tex  methodology.tex
    experiments.tex  results.tex  discussion.tex  conclusion.tex
  figures/            % figure sources (PDF/PNG/EPS) — add here; \graphicspath points to it
  tables/             % standalone table .tex fragments (optional)
  assets/             % misc. assets
  output/             % build products (PDF/aux) when compiling with -output-directory
```

## The IEEEtran class and BibTeX style
`IEEEtran.cls` and `IEEEtran.bst` are **not bundled** in this folder. They are the
official IEEE files and are provided automatically by every LaTeX distribution
and by Overleaf, so `\documentclass{IEEEtran}` and `\bibliographystyle{IEEEtran}`
resolve without any local copy:

- **Overleaf:** compiles as-is (Overleaf ships IEEEtran). Upload this folder or
  start from Overleaf's "IEEE" template. No extra files needed.
- **TeX Live:** included in the `ieeetran` package (usually already installed);
  if not: `tlmgr install ieeetran`.
- **MiKTeX:** installed on first use automatically, or via the MiKTeX Package
  Manager (`ieeetran`).

If a **fully self-contained** project is required (e.g. for archival), copy the
official files into this folder so they sit next to `main.tex`:
```
kpsewhich IEEEtran.cls   # prints the path in your distribution
kpsewhich IEEEtran.bst
```
and copy both here. They were intentionally **not** hand-written, because only
the official IEEE files produce correct IEEE formatting.

## Build
From inside `latex/`:
```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```
or, in one command:
```bash
latexmk -pdf main.tex
```
To keep build products out of the source tree:
```bash
latexmk -pdf -outdir=output main.tex
```

## Status
- Full manuscript content migrated and self-consistent (structurally validated:
  balanced braces/environments; every `\input` target exists; all 23 `\cite`
  keys are defined in `references.bib` and every defined key is cited; equations,
  figures, and tables cross-reference via `\label`/`\ref`/`\eqref`). All source
  files are pure ASCII (accented author names use LaTeX escapes).
- **Not compiled in this environment:** no TeX distribution is installed on the
  build machine, so the PDF was not produced here. The project is written to
  compile with the standard `pdflatex + bibtex + pdflatex x2` sequence on any TeX
  Live / MiKTeX system and on Overleaf.
- **Layout note:** wide floats use `figure*`/`table*` (two-column span). The
  qualitative figure (Fig. 4) float is placed at the top of the Results section
  so that auto-numbering yields Fig. 4 (qualitative), Fig. 5 (region metrics),
  Fig. 6 (boundary metrics); table column spacing (`\tabcolsep`, `\footnotesize`)
  may be fine-tuned once compiled.

## Packages configured (main.tex)
`fontenc` (T1), `inputenc` (utf8), `cite`, `amsmath`, `amssymb`, `graphicx`,
`booktabs`, `multirow`, `subcaption`, `url`, and `hyperref` (loaded last).
No algorithm package is loaded — the manuscript contains no algorithm blocks.
