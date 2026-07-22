# FINAL IMPLEMENTATION AUDIT — Version 2 (Strategic Extension)

**This document extends, and does not replace,** [`FINAL_IMPLEMENTATION_AUDIT.md`](FINAL_IMPLEMENTATION_AUDIT.md) (v1). v1 remains the authoritative, verified repository audit and is **untouched**. v2 appends only the strategic engineering documentation requested for journal preparation. No code, no repository file, and no v1 finding was modified. All facts reused here are traceable to v1 and its cited files.

**New sections (v2 only):** Model Lineage · Official Model Naming Convention · Model Taxonomy · Architecture Evolution · Paper Writing Guidance · Journal Terminology Lock · Traceability Summary.

---

## MODEL LINEAGE

```
                    ┌────────────────────────────────────────────┐
                    │  ORIGINAL MAFF-Net  (Zhou et al., BSPC 2026)│  ← external, published paper
                    │  SAM2 + MMPA + FGFF + RDB + deep supervision │     (no code; the reference)
                    └───────────────────────┬────────────────────┘
                                            │  reproduce exactly
                                            ▼
        ┌───────────────────────────────────────────────────────────────┐
        │  PAPER-FAITHFUL RECONSTRUCTION                                 │
        │  models/maffnet_original_reconstructed.py  (primary)          │
        │  models/maffnet_original.py                (deprecated twin)  │
        │  Purpose: prove the published design reproduces from the text │
        └───────────────────────┬───────────────────────────────────────┘
                                │  branch A: engineering hardening (same philosophy)
                                ▼
        ┌───────────────────────────────────────────────────────────────┐
        │  RESTORED MAFF-Net                                             │
        │  models/maffnet.py                                            │
        │  Same modules (MMPA/FGFF/RDB); numerical stability added in   │
        │  the trainer (bf16, eps-Sobel, non-finite guard, no GradScaler)│
        └───────────────────────┬───────────────────────────────────────┘
                                │  branch B: research contributions (new modules)
                                ▼
        ┌───────────────────────────────────────────────────────────────┐
        │  PROPOSED MAFF-Net                                            │
        │  models/stabilized_maffnet.py                                 │
        │  NEW: FeatureAlignment ∥ FrequencyBranch → GatedFusion;       │
        │  trained with Boundary-aware Deep Supervision + Edge-weighted │
        │  Reconstruction (BcDS, in MAFFNetLoss)                        │
        └───────────────────────────────────────────────────────────────┘
```

**Why each version exists**
- **Original MAFF-Net** — the published baseline being studied; the target of faithful reproduction.
- **Paper-faithful Reconstruction** — a from-the-text re-implementation to demonstrate reproducibility of the paper's design (milestone-built, dimensionally reconstructed where the paper under-specified widths).
- **Restored MAFF-Net** — a runnable, numerically stable MAFF-Net that keeps the original module philosophy but fixes the training instabilities (root-caused in `ROOT_CAUSE_ANALYSIS.md` / `FGFF_NUMERICAL_ANALYSIS_REPORT.md`).
- **Proposed MAFF-Net** — the research contribution: replaces monolithic fusion with explicit spatial/frequency alignment + gated fusion, trained under boundary-centric dual supervision.

**Preserved / changed / untouched**
| Version | Preserved | Changed | Untouched (frozen) |
|---|---|---|---|
| Paper-faithful Reconstruction | all paper modules (MMPA/FGFF/EDB/RDB) | reconstructed widths (paper under-specified) + RDB C5 grouping | SAM2 encoder (frozen) |
| Restored MAFF-Net | MMPA/FGFF/DecoderBlock/RDB | numerical stability in trainer/loss | SAM2 encoder (frozen) |
| Proposed MAFF-Net | SAM2 encoder, DecoderBlock, RDB (reused) | fusion path → FeatureAlignment + FrequencyBranch + GatedFusion | SAM2 encoder (frozen) |

**Implementation evolution vs research contribution (the key separation)**
- **Implementation evolution** (no scientific novelty): Paper-faithful Reconstruction, numerical stabilization in Restored MAFF-Net.
- **Research contribution** (novel): FeatureAlignment, FrequencyBranch, GatedFusion (Proposed model) + BDS + EWR (loss). These are the only items the paper should claim as new.

---

## OFFICIAL MODEL NAMING CONVENTION

One convention, applied everywhere (repository, paper, experiments, figures, tables, methodology). This **supersedes** the ambiguous "Stabilized MAFFNet" label currently in the experiments and `paper/tables/performance_tables.md` (see v1 §13.1).

| Official Paper Name | Repository filename | Internal class | Experiment folder prefix | Performance-table row |
|---|---|---|---|---|
| **Original MAFF-Net** | — (external paper) | — | — | (cited, not run) |
| **Paper-faithful Reconstruction** | `models/maffnet_original_reconstructed.py` | `MAFFNetOriginalReconstructed` | `maffnet_original_*` | "MAFF-Net (paper-faithful)" |
| **Restored MAFF-Net** | `models/maffnet.py` | `MAFFNet` | `maffnet_*` | "MAFF-Net (restored)" |
| **Proposed MAFF-Net** | `models/stabilized_maffnet.py` | `StabilizedMAFFNet` | `stabilized_maffnet_*` | "Proposed MAFF-Net (ours)" |

**Rule:** the file `stabilized_maffnet.py` and its `stabilized_maffnet_*` experiments denote the **Proposed MAFF-Net**. In all paper text, the term "Stabilized" is **retired**; the model is called "Proposed MAFF-Net". Repository files are **not renamed** (that would break checkpoints/experiment paths — documentation-only reconciliation). `models/maffnet_original.py` is **deprecated** and is not referenced by the paper.

---

## MODEL TAXONOMY

| Repository File | Internal Name | Paper Name | Architecture Type | Status | Training | Experiments | Journal Usage |
|---|---|---|---|---|---|---|---|
| `models/maffnet_original_reconstructed.py` | `MAFFNetOriginalReconstructed` | Paper-faithful Reconstruction | SAM2+MMPA+FreqTransformer+FGFF+EDB+RDB | Complete (M1–M6) | via `MAFFNetTrainer` (BcDS loss, see v1 §3a) | `maffnet_original_ISIC{16,17,18}_final` | Reproducibility baseline / ablation reference |
| `models/maffnet_original.py` | `MAFFNetOriginal` | (deprecated) | full paper repro, ~185 M | Deprecated / orphan | none | none | Not cited |
| `models/maffnet.py` | `MAFFNet` | Restored MAFF-Net | MMPA+FGFF+RDB | Complete | `MAFFNetTrainer` | `maffnet_ISIC{16,17,18}_final` | Restored-baseline row / stability discussion |
| `models/stabilized_maffnet.py` | `StabilizedMAFFNet` | **Proposed MAFF-Net** | FeatureAlignment ∥ FrequencyBranch → GatedFusion + RDB | Complete | `MAFFNetTrainer` (BcDS) | `stabilized_maffnet_ISIC{16,17,18}` | **Main proposed method** |
| baselines (`unet`, `unet_plus_plus`, `attention_unet`, `transunet`) | resp. classes | U-Net / UNet++ / Attn U-Net / TransUNet | comparison baselines | Complete | `SimpleTrainer` | multiple `*_final` | Comparison tables |
| `models/transunet_legacy.py` | legacy | (deprecated) | old TransUNet | Deprecated | n/a | n/a | Not cited (`MIGRATION_REPORT.md`) |

(Statuses and experiment presence carried verbatim from v1 §11; not re-audited.)

---

## ARCHITECTURE EVOLUTION

Component-level trace across the four stages. **P**=preserved · **R**=restored (re-implemented, same intent) · **M**=modified · **N**=newly introduced · **F**=frozen/untouched.

| Component | Original Paper | Paper-faithful Reconstruction | Restored MAFF-Net | Proposed MAFF-Net |
|---|---|---|---|---|
| SAM2/Hiera encoder | core | F (frozen) | F (frozen) | F (frozen) |
| MMPA (multi-scale adapter) | core | R (proj-32, dil 3/5/7, WetGen) | R (dilated r=3,5,7 + weighting) | — (replaced by alignment path) |
| FGFF (freq-guided fusion) | core | R (dil 1/3/5/7 + β) | R (FFT-magnitude gated) | — (replaced) |
| Spatial/frequency separation | implicit | via FGFF | via FGFF | **N** FeatureAlignment ∥ FrequencyBranch |
| Fusion mechanism | FGFF | FGFF | FGFF | **N** GatedFusion (σ-gate) |
| Decoder | EDB | R (EDB, width 64) | R (DecoderBlock) | P (reused DecoderBlock) |
| RDB reconstruction | core | R (native, C5 group) | R | P (reused) |
| Deep supervision | yes | yes (p1–p4) | yes | yes |
| Boundary supervision (BDS) | no | (loss, if used) | **N** in MAFFNetLoss | **N** in MAFFNetLoss |
| Edge-weighted reconstruction (EWR) | no | (loss, if used) | **N** in MAFFNetLoss | **N** in MAFFNetLoss |
| Numerical stabilization | n/a | inherited (trainer) | **M/N** (bf16, eps-Sobel, guard) | inherited |

**Narrative:** the published design is first reproduced faithfully; the same modules are then restored into a stable trainer; finally the fusion sub-system is replaced by the three new modules and the objective is upgraded with BDS+EWR. Encoder stays frozen throughout.

---

## PAPER WRITING GUIDANCE

| Model | When to reference | Why it exists | Which paper/section uses it | How reviewers should read it |
|---|---|---|---|---|
| **Original MAFF-Net** | Related Work / method motivation | the published baseline | Intro, Related Work | the prior art being extended; cite Zhou et al. |
| **Paper-faithful Reconstruction** | Reproducibility / ablation base | shows the paper reproduces from text | Experimental Setup (repro note); optional ablation base "Base (SAM2+dec)" | a controlled re-implementation, not a contribution; establishes a fair starting point |
| **Restored MAFF-Net** | Baseline row + stability discussion | a runnable, stable MAFF-Net | Results (baseline), Discussion (numerical stability) | engineering baseline; improvements here are stability, not novelty |
| **Proposed MAFF-Net** | Main method + all headline results | the research contribution | Methodology (Section III), Results, Ablations | the novel method: FeatureAlignment + FrequencyBranch + GatedFusion under BcDS |

**Caution for authors (from v1):** Proposed-model headline numbers are currently **validation-set** (v1 §11.4) — run test-set before comparison claims. The paper-faithful model was trained with **BcDS**, not its own paper-loss (v1 §3a) — frame accordingly. Choose one methodology draft (v1 §13.2).

---

## JOURNAL TERMINOLOGY LOCK

These terms are **frozen**. No synonyms or alternates permitted in any manuscript, figure, table, or caption.

| Locked term | Definition | Do NOT call it |
|---|---|---|
| **Original MAFF-Net** | the published architecture (Zhou et al., BSPC 113, 2026) | "baseline MAFF-Net", "vanilla" |
| **Paper-faithful Reconstruction** | `maffnet_original_reconstructed.py` re-implementation | "original code", "our MAFF-Net copy" |
| **Restored MAFF-Net** | `maffnet.py`, stabilized restoration | "fixed MAFF-Net", "stable MAFF-Net" |
| **Proposed MAFF-Net** | `stabilized_maffnet.py`, the contribution | ~~"Stabilized MAFFNet"~~ (retired), "gated MAFF-Net" |
| **Feature Alignment Module** | InstanceNorm → 1×1 → ReLU harmonizer | "alignment block", "norm-proj" |
| **Frequency Branch** | fp32 FFT → radial high-pass → iFFT | "FFT module", "spectral encoder" |
| **Gated Fusion Module** | g = σ(Conv[Fs‖Ff]); gFs+(1−g)Ff | "attention fusion", "gate block" |
| **Boundary-aware Deep Supervision (BDS)** | Sobel-edge L1 across heads | "edge loss", "contour supervision" |
| **Edge-weighted Reconstruction (EWR)** | W=1+αB weighted MSE | "weighted recon", "boundary MSE" |
| **Boundary-centric Dual Supervision (BcDS)** | BDS + EWR combined objective | "dual loss" |
| **MMPA** | Mixed Multi-scale Perception Adaptor | "pyramid attention" |
| **FGFF** | Frequency-Guided Feature Fusion | "freq fusion" |
| **RDB** | Reconstruction Decoder Block | "recon head" |

---

## TRACEABILITY SUMMARY (master matrix)

Contribution → module → implementation → experiment → paper section → figure → equation → table. Figure/equation/table cells reference `paper/` assets; **[TBD]** where the asset is not yet produced (v1 §9, §10, §14).

| Contribution | Module | Implementation file | Experiment | Paper Section | Figure | Equation | Table |
|---|---|---|---|---|---|---|---|
| Feature alignment | FeatureAlignment | `modules/feature_alignment.py` | `stabilized_maffnet_ISIC*` | Sec III-D | Fig. (feature align) **[TBD]** | `section_III` (7)–(9) | ablation `paper/tables/ablation_tables` |
| Frequency processing | FrequencyBranch | `modules/frequency_branch.py` | `stabilized_maffnet_ISIC*` | Sec III-C | Fig. 2 (freq branch) | (2)–(6) | ablation (cutoff) |
| Gated fusion | GatedFusion | `modules/gated_fusion.py` | `stabilized_maffnet_ISIC*` | Sec III-E | Fig. 3 (gated fusion) | (10)–(11) | ablation (components) |
| Boundary supervision | BDS (MAFFNetLoss) | `training/trainer.py` | `stabilized_maffnet_ISIC*` / `maffnet_*` | Sec III-G | Fig. (boundary sup) **[TBD]** | (15)–(18) | ablation (BcDS) |
| Edge-weighted recon | EWR (MAFFNetLoss) | `training/trainer.py` | same | Sec III-G | — | (20) | ablation (BcDS) |
| Overall proposed arch | StabilizedMAFFNet | `models/stabilized_maffnet.py` | `stabilized_maffnet_ISIC*` | Sec III-A | Fig. 1 (overall) | (1),(21) | performance_tables (proposed row) |
| Paper-faithful repro | MAFFNetOriginalReconstructed | `models/maffnet_original_reconstructed.py` | `maffnet_original_*_final` | Setup (repro) | Fig. 1 (shared) | — | ablation base |
| Restored baseline | MAFFNet | `models/maffnet.py` | `maffnet_*_final` | Results (baseline) | — | — | performance_tables (restored row) |
| Numerical stability | trainer fixes | `training/trainer.py` | all MAFFNet runs | Discussion | Fig. (training pipeline) **[TBD]** | — | — |

**Open traceability gaps (carried from v1, not new findings):** feature-alignment / boundary-supervision / training-pipeline figures are **[TBD]**; equations exist only in `section_III` (not in `Journal_Paper/Equations/`); proposed table row currently labelled "Stabilized" and validation-only.

---

## Version 2 status

- v1 (`FINAL_IMPLEMENTATION_AUDIT.md`) — authoritative, verified, unchanged.
- v2 — strategic reference for manuscripts. Naming convention + terminology lock resolve the v1 §13.1 collision at the documentation level (no file renames).
- No code, architecture, or experiment was modified in producing v2.

*This Version 2 document is the permanent naming/terminology/traceability reference for all journal manuscripts. Technical verification remains governed by v1.*
