# Ablation Study Tables (Markdown)

> **Status: STRUCTURE ONLY — no ablation experiments have been run.**
> Every numeric cell is a placeholder (`—`) to be filled from completed runs.
> No values are invented. Two ablation designs are provided; use whichever matches
> the final paper direction (Stabilized-MAFFNet component ablation, and/or the
> boundary-supervision A/B/C/D ablation).

## Table IV — Stabilized MAFFNet Component Ablation (test, mean ± std)

Cumulative addition of the synopsis modules on the frozen-SAM2 base.

| # | Configuration | Freq. Branch | Feat. Align | Gated Fusion | Dice | IoU | Sen | Spe |
|---|---------------|:------------:|:-----------:|:------------:|:----:|:---:|:---:|:---:|
| B | Base (SAM2 + decoder) | ✗ | ✗ | ✗ | — | — | — | — |
| B1 | + Frequency Branch | ✓ | ✗ | ✗ | — | — | — | — |
| B2 | + Feature Alignment | ✓ | ✓ | ✗ | — | — | — | — |
| F | Full (+ Gated Fusion) | ✓ | ✓ | ✓ | — | — | — | — |

## Table V — High-pass Cutoff Sensitivity (ISIC 2016, test)

| cutoff | 0.05 | 0.10 | 0.20 | 0.30 |
|--------|:----:|:----:|:----:|:----:|
| Dice | — | — | — | — |
| IoU | — | — | — | — |

## Table VI — Boundary Supervision Ablation (optional; if the BcDS direction is reported)

A: base; B: + Boundary-aware Deep Supervision (BDS); C: + Edge-weighted Reconstruction (EWR); D: full.

| Variant | BDS | EWR | Dice | IoU | Sen | Spe |
|---------|:---:|:---:|:----:|:---:|:---:|:---:|
| A — base | ✗ | ✗ | — | — | — | — |
| B — + BDS | ✓ | ✗ | — | — | — | — |
| C — + EWR | ✗ | ✓ | — | — | — | — |
| D — full (BcDS) | ✓ | ✓ | — | — | — | — |

**Notes for finalization:**
- Run each configuration with identical protocol; report mean ± std over ≥3 seeds.
- The claim for Table VI requires **D > max(B, C)** to justify the dual coupling.
- Add a boundary-specific metric (e.g., HD95 / Boundary-IoU) to support boundary claims; not yet computed by the metric code.
