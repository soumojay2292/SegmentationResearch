# PUBLICATION ENGINEERING REFERENCE

**Internal engineering manual for manuscript preparation. Not part of any paper.**
Single source of truth for writing journal manuscripts from this repository. Consistent with [`FINAL_IMPLEMENTATION_AUDIT_v2.md`](FINAL_IMPLEMENTATION_AUDIT_v2.md) (naming/terminology) and [`FINAL_IMPLEMENTATION_AUDIT.md`](FINAL_IMPLEMENTATION_AUDIT.md) (verified findings). No repository audit was re-run; no code, terminology, or finding changed. Unverifiable items are marked **[MVR]** (Manual Verification Required).

Terminology follows the v2 lock: **Original MAFF-Net** (paper) · **Paper-faithful Reconstruction** (`maffnet_original_reconstructed.py`) · **Restored MAFF-Net** (`maffnet.py`) · **Proposed MAFF-Net** (`stabilized_maffnet.py`). "Stabilized" is retired in paper text.

---

## 1. Project Identity

| Field | Statement |
|---|---|
| Research objective | Improve dermoscopic skin-lesion segmentation by adding explicit spatial/frequency feature separation and gated fusion on a frozen-SAM2 backbone, under boundary-centric supervision. |
| Research question | Does decoupling spatial and frequency streams with feature alignment + a lightweight gated fusion (plus boundary-aware dual supervision) improve boundary segmentation over the original MAFF-Net fusion? |
| Baseline architecture | **Original MAFF-Net** (Zhou et al., BSPC 113, 2026): frozen SAM2 + MMPA + FGFF + RDB + deep supervision. Reproduced as **Paper-faithful Reconstruction** and stabilized as **Restored MAFF-Net**. |
| Proposed architecture | **Proposed MAFF-Net** (`stabilized_maffnet.py`): SAM2 → {Feature Alignment ∥ Frequency Branch → Feature Alignment} → Gated Fusion → decoder + RDB; trained with BcDS (BDS + EWR). |
| Publication objective | One Q1 journal manuscript presenting Proposed MAFF-Net, with restored + faithful variants and standard baselines for comparison. |

---

## 2. Model Lineage

| Stage | Why it exists | Preserved | Modified | Newly introduced | Purpose in project |
|---|---|---|---|---|---|
| **Original MAFF-Net** (paper) | prior art being extended | — | — | — | reference/baseline (cited, not run) |
| **Paper-faithful Reconstruction** (`maffnet_original_reconstructed.py`) | prove design reproduces from text | all paper modules | reconstructed widths + RDB C5 grouping | — | reproducibility base / ablation "Base" |
| **Restored MAFF-Net** (`maffnet.py`) | runnable, numerically stable MAFF-Net | MMPA/FGFF/DecoderBlock/RDB | numerical stability (bf16, eps-Sobel, non-finite guard — in trainer) | — (module-level) | stability baseline row |
| **Proposed MAFF-Net** (`stabilized_maffnet.py`) | research contribution | SAM2 encoder, DecoderBlock, RDB (reused) | fusion path replaced | **Feature Alignment, Frequency Branch, Gated Fusion**; BDS + EWR (loss) | main method |

Encoder (SAM2/Hiera) is **frozen** at every stage. Implementation evolution (repro + stabilization) is distinct from research contribution (3 fusion modules + BcDS).

---

## 3. Official Model Terminology (frozen)

| Repository file | Python class | Internal architecture | Paper terminology | Experiment naming | Performance-table naming |
|---|---|---|---|---|---|
| `models/maffnet_original_reconstructed.py` | `MAFFNetOriginalReconstructed` | SAM2+MMPA+FreqTransformer+FGFF+EDB+RDB | **Paper-faithful Reconstruction** | `maffnet_original_ISIC*_final` | "MAFF-Net (paper-faithful)" |
| `models/maffnet.py` | `MAFFNet` | MMPA+FGFF+RDB | **Restored MAFF-Net** | `maffnet_ISIC*_final` | "MAFF-Net (restored)" |
| `models/stabilized_maffnet.py` | `StabilizedMAFFNet` | Align∥FreqBranch→GatedFusion+RDB | **Proposed MAFF-Net** | `stabilized_maffnet_ISIC*` | "Proposed MAFF-Net (ours)" |
| `models/maffnet_original.py` | `MAFFNetOriginal` | full paper repro (~185 M) | (deprecated) | none | not used |
| baselines | `UNet`/`UNetPlusPlus`/`AttentionUNet`/`TransUNet` | — | U-Net / UNet++ / Attn U-Net / TransUNet | `unet_*`, `unetpp_*`, `attention_unet_*`, `transunet_*` | resp. |

**Rule:** experiment folder `stabilized_maffnet_*` and table label must be read/written as **Proposed MAFF-Net**. Files are not renamed.

---

## 4. Research Contribution Boundary (no overlap)

| Bucket | Items | Owner |
|---|---|---|
| **Existing (Original MAFF-Net)** | frozen SAM2 encoder, MMPA, FGFF, RDB, deep supervision | Zhou et al. — cite, never claim |
| **Engineering restoration** | numerical stability: bf16 autocast (no GradScaler), eps-Sobel `sqrt(gx²+gy²+1e-6)`, non-finite guard, gamma excluded from weight decay | `training/trainer.py` — describe as engineering, not novelty |
| **Research contribution (Proposed)** | Feature Alignment Module, Frequency Branch (high-pass), Gated Fusion Module, Boundary-aware Deep Supervision (BDS), Edge-weighted Reconstruction (EWR) | this work — the only claimable novelty |

Nothing in "Existing" or "Engineering restoration" may be claimed as a scientific contribution.

---

## 5. Evidence Matrix

| Claim | Repository evidence | Implementation evidence | Experimental evidence | Figure | Table | Paper section | Status |
|---|---|---|---|---|---|---|---|
| Feature alignment harmonizes spatial/freq stats | `modules/feature_alignment.py` | `FEATURE_ALIGNMENT_IMPLEMENTATION.md` | ablation [MVR — cells placeholder] | Fig (align) [TBD] | ablation_tables | III-D | Impl ✔ / results [MVR] |
| Frequency branch isolates boundary content | `modules/frequency_branch.py` | `FREQUENCY_BRANCH_IMPLEMENTATION.md` | `stabilized_maffnet_*` | Fig 2 | ablation | III-C | Impl ✔ / results [MVR] |
| Gated fusion improves over concat/attention at lower cost | `modules/gated_fusion.py` | `GATED_FUSION_IMPLEMENTATION.md` | ablation [MVR] | Fig 3 | ablation + complexity | III-E | Impl ✔ / results [MVR] |
| BcDS sharpens boundaries | `training/trainer.py::MAFFNetLoss` | `CONTRIBUTION_DESIGN.md`, `CONTRIBUTION_IMPLEMENTATION_REPORT.md` | `stabilized_maffnet_*`/`maffnet_*` | Fig (boundary) [TBD] | ablation (BcDS) | III-G | Impl ✔ / results [MVR] |
| Proposed > baselines on ISIC | performance_tables | assembly+integration reports | proposed **val** vs baseline **test** | Fig 7 | performance_tables | V | **[MVR] not comparable yet (val≠test)** |
| Numerical stability enables convergence | trainer fixes | `ROOT_CAUSE_ANALYSIS.md`, `FGFF_NUMERICAL_ANALYSIS_REPORT.md` | `maffnet_*` trains w/o NaN | — | — | Discussion | ✔ |

---

## 6. Experiment Provenance

| Dataset | Experiment folder | Checkpoint | Metrics (val, ISIC2016 Dice) | Result files | Journal usage |
|---|---|---|---|---|---|
| ISIC2016/17/18 | `maffnet_original_ISIC{16,17,18}_final` (config=`maffnet_original_reconstructed`) | `model.pth` | 0.9124 (2016) | config.txt, summary.csv, curves, samples, report.html | Paper-faithful repro / ablation base |
| ISIC2016/17/18 | `maffnet_ISIC{16,17,18}_final` (config=`maffnet`) | `model.pth` + checkpoints/ | 0.9345 (2016) | same | Restored baseline row |
| ISIC2016/17/18 | `stabilized_maffnet_ISIC{16,17,18}` (config=`stabilized_maffnet`) | `model.pth` | 0.9316 (2016) | same | **Proposed main results** |
| ISIC2016/17/18 | `unet_*`, `unetpp_*`, `attention_unet_*`, `transunet_*` | `model.pth` + `metrics.json` (test) | see performance_tables | metrics.json, summary.csv | Comparison baselines |

**Provenance caveats (from v1 §11):**
- Folder `maffnet_original_*` ← model `maffnet_original_reconstructed` (name misleading; trust config.txt).
- Proposed/restored/faithful metrics are **validation** (summary.csv); baselines are **test** (metrics.json) → **not comparable** until proposed test run exists. **[MVR]**
- Checkpoint loadability into current code (post-C5 + adapter prefix) **[MVR]** — test `load_state_dict` before citing.
- Per-run `training_log.csv` not in `experiments/` (trainer writes to `checkpoints/`, shared). Curves survive as PNG.

---

## 7. Module Traceability

| Module | Purpose | Design goal | File | Experiment | Section | Figure | Equation | Class |
|---|---|---|---|---|---|---|---|---|
| SAM2 encoder | backbone features | transfer priors, frozen | maffnet.py / all | all | III-B | Fig 1 | (1) | Existing |
| MMPA | multi-scale adapt | scale-adaptive context | maffnet.py, *_reconstructed | maffnet_*, faithful | (repro/restored method) | — | — | Existing |
| FGFF | freq-guided fusion | edge enhancement | maffnet.py, *_reconstructed | maffnet_*, faithful | (repro/restored) | — | — | Existing |
| Feature Alignment | stat harmonization | align streams pre-fusion | modules/feature_alignment.py | stabilized_* | III-D | Fig [TBD] | (7)-(9) | **New** |
| Frequency Branch | high-pass FFT | boundary-sensitive rep | modules/frequency_branch.py | stabilized_* | III-C | Fig 2 | (2)-(6) | **New** |
| Gated Fusion | σ-gated combine | linear-cost fusion | modules/gated_fusion.py | stabilized_* | III-E | Fig 3 | (10)-(11) | **New** |
| RDB | aux reconstruction | feature stability | maffnet.py (reused) | all MAFFNet | III-F | Fig [TBD] | (12) | Existing (reused) |
| BDS | boundary supervision | edge accuracy | trainer.py MAFFNetLoss | stabilized_*, maffnet_* | III-G | Fig [TBD] | (15)-(18) | **New** |
| EWR | edge-weighted recon | boundary consistency | trainer.py MAFFNetLoss | same | III-G | — | (20) | **New** |

---

## 8. Figure Manifest

Source of existing captions: `paper/figures/figure_captions.md`. Architecture schematics author-drawn; source PDFs absent (`Journal_Paper/Figures/` = `.gitkeep`).

| Fig | Purpose | Required inputs | Source | Status | Section | Verification |
|---|---|---|---|---|---|---|
| 1 | Overall architecture (Proposed) | schematic | author-drawn | **[TBD]** produce PDF | III-A | — |
| 2 | Frequency Branch | schematic | author-drawn | **[TBD]** | III-C | — |
| 3 | Gated Fusion | schematic | author-drawn | **[TBD]** | III-E | — |
| (add) | Feature Alignment | schematic | author-drawn | **[TBD] missing** | III-D | — |
| (add) | Boundary Supervision (BDS/EWR) | schematic | author-drawn | **[TBD] missing** | III-G | — |
| 4 | Training/val loss curve | `stabilized_maffnet_ISIC2016/loss_curve.png` | run output | ✔ exists | V | verify axes |
| 5 | Val Dice/IoU curve | `.../iou_curve.png` | run output | ✔ exists | V | — |
| 6 | Qualitative (Proposed) | `.../samples/` | run output | ✔ exists | V | — |
| 7 | Qualitative comparison (all methods) | per-method preds | multi-run | **[TBD] assemble** | V | — |
| 8 | Ablation chart | ablation runs | **[MVR] ablation not run** | **[TBD]** | V | — |

**Note:** task figure numbering (Fig2=Feature Align, Fig3=Freq, Fig4=Gated, Fig5=Boundary) **differs** from `figure_captions.md`. Reconcile numbering before drafting (v1 §9).

---

## 9. Table Manifest

| Table | Content | Status | Evidence |
|---|---|---|---|
| Performance comparison | Dice/IoU/Sen/Spe, 3 datasets | Present, **proposed row = val (†)** | `paper/tables/performance_tables.{md,tex}` |
| Ablation (components) | Base→+Freq→+Align→+Gate | **Structure only, placeholders** | `paper/tables/ablation_tables.*` — **[MVR] runs needed** |
| Ablation (cutoff) | high-pass cutoff sweep | placeholder | ablation_tables — **[MVR]** |
| Ablation (BcDS) | BDS/EWR on/off | placeholder | ablation_tables — **[MVR]** |
| Computational complexity | params/FLOPs/inference | **[TBD]** | `COMPUTATIONAL_ANALYSIS.md` (partial) |
| Training configuration | optimizer/lr/epochs/bs | derivable | config.txt, trainer defaults |
| Dataset summary | splits per dataset | Present | `DATASET_AUDIT.md`, `SPLIT_AUDIT.md` |
| Qualitative comparison | image/GT/preds | figure (Fig 7) | [TBD] |

---

## 10. Equation Manifest

Numbered equations exist only in `paper/sections/section_III_proposed_methodology.md` (1)–(21). `Journal_Paper/Equations/` empty. Do not derive; map only.

| Eq ID | Purpose | Variables | Module | Figure | Implementation | Status |
|---|---|---|---|---|---|---|
| (1) | encoder features | X_i | SAM2 | Fig 1 | encoder | ✔ |
| (2)-(6) | frequency branch (FFT/mask/iFFT) | F, M, ρ | Frequency Branch | Fig 2 | frequency_branch.py | ✔ |
| (7)-(9) | feature alignment (IN, 1×1) | μ,σ,W_p | Feature Alignment | Fig [TBD] | feature_alignment.py | ✔ |
| (10)-(11) | gated fusion | g, F_s, F_f | Gated Fusion | Fig 3 | gated_fusion.py | ✔ |
| (12)-(14) | decoder/skips/heads | S_i, D_j, p_k | DecoderBlock+heads | Fig 1 | maffnet.py | ✔ |
| (15)-(18) | Dice + Sobel BDS | Dice, S(·), λ_B(t) | MAFFNetLoss | Fig [TBD] | trainer.py | ✔ |
| (19) | BDS warmup | λ_B(t) | loss schedule | — | trainer.py | ✔ |
| (20) | EWR weighted MSE | W=1+αB | MAFFNetLoss | — | trainer.py | ✔ |
| (21) | total objective | L | loss | — | trainer.py | ✔ |

**[MVR]** the `Journal_Paper/03_Methodology` draft uses a **different** equation set (MMPA/FGFF/BcDS) — pick one direction (v1 §13.2).

---

## 11. Paper Mapping

| | **Paper 1 (defined)** | **Paper 2** |
|---|---|---|
| Purpose | Present Proposed MAFF-Net for skin-lesion segmentation | **Not scoped — [MVR]** (no second manuscript defined in repo/synopsis) |
| Baseline | U-Net, UNet++, Attn U-Net, TransUNet | — |
| Comparison | + Restored MAFF-Net, + Paper-faithful Reconstruction | — |
| Novelty | Feature Alignment + Frequency Branch + Gated Fusion + BcDS | — |
| Experiments | `stabilized_maffnet_*` (main), `maffnet_*`, `maffnet_original_*`, baselines | — |
| Figures | 1–8 (per §8) | — |
| Tables | performance + ablation + complexity + dataset + config | — |

Only one manuscript is supported by the current repository. Do not fabricate a second. If Paper 2 is later scoped, extend this section.

---

## 12. Reviewer Risk Register

| Issue | Risk | Supporting evidence | Mitigation | Status |
|---|---|---|---|---|
| Novelty vs original MAFF-Net | High | modules are new (feature align/gated fusion); MMPA/FGFF are not | frame novelty strictly on the 3 modules + BcDS; §4 boundary | Ready |
| Fair comparison (val vs test) | **High** | proposed=val, baselines=test (v1 §11.4) | run proposed on **test**; re-tabulate | **Open [MVR]** |
| Experimental validation depth | High | ablation tables are placeholders | run component/cutoff/BcDS ablations | **Open [MVR]** |
| Architecture justification | Med | design reports exist | cite ablations once run | Pending ablations |
| Computational complexity | Med | partial `COMPUTATIONAL_ANALYSIS.md`; gated fusion linear cost argued in section_III | add params/FLOPs table | **Open [MVR]** |
| Reproducibility | Med | seeds/splits in `SPLIT_AUDIT.md`, `REPRODUCIBILITY_CHECKLIST.md`; checkpoint compat unverified | verify `load_state_dict`; publish configs | Partial |
| Clinical relevance | Low | dermoscopy ISIC benchmark | keep claims to segmentation metrics | Ready |
| Generalization | Med | 3 ISIC datasets only | acknowledge single-modality limit (as original paper does) | Ready |
| Paper-faithful trained with BcDS not paper loss | Med | v1 §3a | disclose, or retrain with paper loss if labeled "faithful metrics" | **Open [MVR]** |

---

## 13. Do-Not-Claim Register

| Unsafe claim | Reason | Safe alternative |
|---|---|---|
| "first to …" | not verified against literature | "we propose …" |
| "state-of-the-art" | proposed metrics are val, not test-comparable | "competitive with / improves over our baselines under matched settings" |
| "best" | no verified head-to-head test comparison | "higher validation Dice than the restored/faithful variants" |
| "clinically validated" | no clinical study | "on public ISIC benchmarks" |
| "superior" / "outperforms SOTA" | not established | "shows improved boundary metrics in our experiments" |
| "real-time" | inference timing [MVR] | omit or report measured latency |
| exact param match to paper (14.8 M) | reconstruction differs (v1) | "structurally faithful; parameter differences noted" |

---

## 14. Publication Readiness Checklist

| Item | Status |
|---|---|
| Architecture frozen | ✔ (design freeze + v2) |
| Terminology frozen | ✔ (v2 lock, §3) |
| Implementation verified | ✔ (v1; all 3 models compile + run) |
| Experiments complete | ⚠ present for 3 models + baselines; **proposed test-set missing** |
| Figures planned | ⚠ result curves exist; architecture/align/boundary schematics **[TBD]** |
| Tables planned | ⚠ performance done; **ablation placeholders** |
| Equation mapping complete | ✔ for section_III; **[MVR]** competing draft |
| Contribution boundary clear | ✔ (§4) |
| Evidence complete | ⚠ impl ✔; several results **[MVR]** |
| Reviewer risks identified | ✔ (§12) |
| **Publication ready** | **NO — requires: proposed test run, ablations, figures, one methodology direction, checkpoint verification** |

---

## 15. Master Traceability Matrix

| Architecture | Module | Implementation | Experiment | Figure | Equation | Table | Paper section | Evidence | Status |
|---|---|---|---|---|---|---|---|---|---|
| Proposed MAFF-Net | Feature Alignment | modules/feature_alignment.py | stabilized_maffnet_* | Fig [TBD] | (7)-(9) | ablation | III-D | FEATURE_ALIGNMENT_IMPLEMENTATION.md | Impl ✔ / results [MVR] |
| Proposed MAFF-Net | Frequency Branch | modules/frequency_branch.py | stabilized_maffnet_* | Fig 2 | (2)-(6) | ablation | III-C | FREQUENCY_BRANCH_IMPLEMENTATION.md | Impl ✔ / results [MVR] |
| Proposed MAFF-Net | Gated Fusion | modules/gated_fusion.py | stabilized_maffnet_* | Fig 3 | (10)-(11) | ablation+complexity | III-E | GATED_FUSION_IMPLEMENTATION.md | Impl ✔ / results [MVR] |
| Proposed MAFF-Net | BDS + EWR | training/trainer.py | stabilized_maffnet_* | Fig [TBD] | (15)-(20) | ablation(BcDS) | III-G | CONTRIBUTION_*_REPORT.md | Impl ✔ / results [MVR] |
| Proposed MAFF-Net | Overall | models/stabilized_maffnet.py | stabilized_maffnet_ISIC{16,17,18} | Fig 1 | (1),(21) | performance | III-A / V | STABILIZED_MAFFNET_ASSEMBLY_REPORT.md | val ✔ / test [MVR] |
| Restored MAFF-Net | MMPA/FGFF/RDB (+stability) | models/maffnet.py + trainer.py | maffnet_ISIC{16,17,18}_final | — | — | performance | V / Disc | RESTORATION_REPORT.md, ROOT_CAUSE_ANALYSIS.md | val ✔ |
| Paper-faithful Reconstruction | full paper repro | models/maffnet_original_reconstructed.py | maffnet_original_ISIC{16,17,18}_final | Fig 1 (shared) | — | ablation base | Setup | MILESTONE*/FINAL_IMPLEMENTATION_REPORT.md | val ✔ / loss=BcDS [MVR] |
| Baselines | U-Net/UNet++/Attn/TransUNet | models/*.py | unet_*/unetpp_*/attention_unet_*/transunet_* | Fig 7 | — | performance | V | performance_tables | test ✔ |

---

*This reference organizes only verified information (v1/v2 + repository + original paper + synopsis). No code, terminology, or finding was changed. Every open item is marked [MVR] or [TBD]. Update this file — not v1/v2 — as manuscript work resolves the open items.*
