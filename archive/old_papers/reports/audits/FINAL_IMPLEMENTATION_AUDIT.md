# FINAL IMPLEMENTATION AUDIT

**Role:** Senior Research Software Auditor / Reproducibility Reviewer.
**Nature:** verification against the current repository. No code, architecture, or report was modified. Every claim below is traceable to a file cited inline. Items that cannot be verified from the repository are marked **[MANUAL VERIFICATION REQUIRED]**.

Repository root: `C:/Users/Soumojay/Desktop/New_Sm_P/SegmentationResearch`.

---

## 1. Executive Summary

The repository contains **four** MAFF-Net model files but the project taxonomy names **three** architectures. The mapping is verifiable from code and experiment `config.txt` files, and it reveals a **critical naming collision** that must be resolved before journal writing:

| Project name (task) | Implementing file / class | Verified by |
|---|---|---|
| Original MAFF-Net (paper-faithful) | `models/maffnet_original_reconstructed.py::MAFFNetOriginalReconstructed` (primary) **and** `models/maffnet_original.py::MAFFNetOriginal` (older, orphaned) | class defs; `experiments/maffnet_original_ISIC*_final/config.txt` = `maffnet_original_reconstructed` |
| Stabilized MAFF-Net (engineering) | `models/maffnet.py::MAFFNet` | class defs; `experiments/maffnet_ISIC*_final/config.txt` = `maffnet` |
| Proposed MAFF-Net (synopsis contributions) | `models/stabilized_maffnet.py::StabilizedMAFFNet` | class defs (FeatureAlignment/FrequencyBranch/GatedFusion); `experiments/stabilized_maffnet_ISIC*/config.txt` = `stabilized_maffnet` |

**The file named `stabilized_maffnet.py` implements the model the task calls "Proposed MAFF-Net"**, and the experiments + paper tables label it **"Stabilized MAFFNet (ours)"**. Meanwhile the task's "Stabilized MAFF-Net" is `maffnet.py`. This terminology drift is the single most important issue in the repository.

Overall: all three models are implemented, compile, and have completed experiments. The repository is **NOT yet paper-ready** — it requires naming reconciliation, checkpoint-compatibility verification, resolution of two divergent methodology drafts, and a validation→test metric fix.

---

## 2. Repository Overview

**Model files** (`models/`): `maffnet.py`, `stabilized_maffnet.py`, `maffnet_original.py`, `maffnet_original_reconstructed.py`, plus baselines `unet.py`, `unet_plus_plus.py`, `attention_unet.py`, `transunet.py`, `transunet_legacy.py`, and helpers `sam2_loader.py`, `test_baselines.py`, `test_transunet.py`.

**Supporting modules** (`modules/`): `feature_alignment.py`, `frequency_branch.py`, `gated_fusion.py` (imported only by `stabilized_maffnet.py`).

**Training:** `train.py` (+ `train_all.py`), `training/trainer.py` (`MAFFNetTrainer`, `MAFFNetLoss`, `Trainer`), `training/simple_trainer.py`.

**Reports:** 50 `.md` under `reports/{implementation,architecture,analysis,audits,experiments}`.

**Paper:** two workspaces — `paper/` (sections/figures/tables) and `Journal_Paper/` (00–07 sections). See §12–13 for the divergence.

---

## 3. Original MAFF-Net Audit (`maffnet_original.py` + `maffnet_original_reconstructed.py`)

### 3a. `maffnet_original_reconstructed.py` (primary; milestone-built)
| Item | Status | Evidence |
|---|---|---|
| Encoder | ✔ `FrozenSAMEncoder` (frozen, dummy fallback), channels 144/288/576/1152 | class + `MILESTONE1_IMPLEMENTATION_REPORT.md` |
| MMPA | ✔ proj C→32, dilations {3,5,7}, WetGeneration, Γ=1e-7 | `MILESTONE1` |
| Frequency branch | ✔ FFT→magnitude→pseudo-RGB→FreqTransformer (E=128, 2 blk/stage, 4 heads) | `MILESTONE2A` |
| FGFF | ✔ dilations {1,3,5,7} + freq-attn + β + residual | `MILESTONE2B` |
| Decoder (EDB) | ✔ width 64, deep supervision | `MILESTONE3` |
| RDB | ✔ native widths, C5 grouped deepest fuse (7.32 M) | `MILESTONE4B/4C`, `RDB_PARAMETER_ANALYSIS.md` |
| Seg heads | ✔ 4 heads, Conv1×1→↑384² | `MILESTONE5` |
| Loss (own) | ✔ Dice[1,1,1,0.01] + 0.01·MSE | `FINAL_IMPLEMENTATION_REPORT.md` |
| Tensor flow | ✔ verified 33/33 shapes | `FINAL_IMPLEMENTATION_REPORT.md` §3 |
| Train pipeline | ✔ registered via `_MAFFNetOriginalAdapter` | `TRAIN_PIPELINE_INTEGRATION_REPORT.md` |
| Inference | ✔ eval returns p1 | code `forward` |
| Completeness | ✔ M1–M6 complete | milestone reports |

**Critical caveat — loss actually used in its experiments:** the training pipeline drives this model with `MAFFNetLoss` (BcDS: Dice+BCE+boundary+EWR), **not** its own paper-faithful Dice+MSE (`TRAIN_PIPELINE_INTEGRATION_REPORT.md` §3). Therefore the `maffnet_original_ISIC*_final` metrics reflect a **BcDS-trained** run, not a paper-loss run. **[MANUAL VERIFICATION REQUIRED]** if the paper claims paper-faithful loss for these numbers.

### 3b. `maffnet_original.py` (older faithful reconstruction) — **ORPHANED**
- Complete standalone model (`MAFFNetOriginal`, FreqTransformer + EDB + RDB, ~185 M trainable, `ORIGINAL_MAFFNET_IMPLEMENTATION_REPORT.md`).
- **Not imported by `train.py`** (verified: only `maffnet`, `stabilized_maffnet`, `maffnet_original_reconstructed` are registered).
- **No experiment folder** references it.
- Verdict: superseded by `maffnet_original_reconstructed.py`; retained but unused. **Flag: two "Original" files exist; the paper must cite only one.**

---

## 4. Stabilized MAFF-Net Audit (`maffnet.py`)

| Item | Status | Evidence |
|---|---|---|
| Encoder | ✔ `SAM2Encoder` (frozen) + `SAMEncoder` fallback | class defs |
| MMPA (preserved) | ✔ dilated conv r=3,5,7 + adaptive weighting | `maffnet.py:39` |
| FGFF (preserved) | ✔ FFT-magnitude gated fusion | `maffnet.py:96` |
| Decoder + RDB (preserved) | ✔ `DecoderBlock`, `RDB` | `maffnet.py:144,163` |
| Engineering / numerical improvements | ✔ but located in **`training/trainer.py`**, not the model: bf16 autocast (no GradScaler), non-finite guard, `BoundaryLoss._edges` eps `sqrt(gx²+gy²+1e-6)`, gamma excluded from weight decay | `trainer.py:64,337,341,279` |
| Training compatibility | ✔ `MAFFNetTrainer`, 5-tuple `(p1..p4,rec)` | `trainer.py:145` |
| Experiments | ✔ `maffnet_ISIC{2016,2017,2018}_final` (config `model: maffnet`) | config.txt |

**Note:** the "stabilization" (bf16 + eps-sqrt + non-finite guard) lives in the **trainer/loss**, shared by all MAFFNet-family models — it is not a property of `maffnet.py` alone. The BcDS loss (boundary + edge-weighted reconstruction) is also in `MAFFNetLoss` and is a **research contribution**, so `maffnet.py` experiments were trained with a contribution-bearing loss. **Novelty attribution must reflect this** (see §8).

---

## 5. Proposed MAFF-Net Audit (`stabilized_maffnet.py`)

| Item | Status | Evidence |
|---|---|---|
| Feature Alignment | ✔ `FeatureAlignment` (InstanceNorm→1×1→ReLU), per-stage spatial+freq | `stabilized_maffnet.py:26,62-63`; `modules/feature_alignment.py`; `FEATURE_ALIGNMENT_IMPLEMENTATION.md` |
| Frequency processing | ✔ `FrequencyBranch` (fp32 FFT, radial high-pass, iFFT) | `stabilized_maffnet.py:61`; `modules/frequency_branch.py`; `FREQUENCY_BRANCH_IMPLEMENTATION.md` |
| Gated fusion | ✔ `GatedFusion` g=σ(Conv[Fs‖Ff]) | `stabilized_maffnet.py:64`; `modules/gated_fusion.py`; `GATED_FUSION_IMPLEMENTATION.md` |
| Reconstruction branch | ✔ reused `RDB` | `stabilized_maffnet.py:87` |
| Decoder / skips | ✔ reused `DecoderBlock` + projected skips | `STABILIZED_MAFFNET_ASSEMBLY_REPORT.md` |
| Boundary supervision | ⚠ **not in the model** — provided by `MAFFNetLoss` (BcDS) in the trainer | `trainer.py:48` |
| Losses (as trained) | ✔ `MAFFNetLoss` (BcDS) | trainer |
| Tensor flow | ✔ SAM2 → {spatial align ∥ freq→align} → gated fusion → decoder+RDB → heads | class `forward` |
| Experiments | ✔ `stabilized_maffnet_ISIC{2016,2017,2018}` (config `model: stabilized_maffnet`) | config.txt |
| Completeness | ✔ assembled + integrated | `STABILIZED_MAFFNET_ASSEMBLY_REPORT.md`, `INTEGRATION_REPORT.md` |

**This is the model the task calls "Proposed".** Its file name, experiment folders, and `paper/tables/performance_tables.md` label it **"Stabilized MAFFNet (ours)"** — a naming conflict (see §13).

---

## 6. Module Verification Matrix

| Module | File | Purpose | Impl? | Verified? | Train? | Infer? | Ref OK? | Update? |
|---|---|---|---|---|---|---|---|---|
| SAM2Encoder | maffnet.py | frozen backbone | ✔ | ✔ | ✔ | ✔ | ✔ | — |
| MMPA (maffnet) | maffnet.py | multi-scale adapt | ✔ | ✔ | ✔ | ✔ | ✔ | — |
| FGFF (maffnet) | maffnet.py | freq-guided fusion | ✔ | ✔ | ✔ | ✔ | ✔ | — |
| DecoderBlock | maffnet.py | upsample decode | ✔ | ✔ | ✔ | ✔ | ✔ | — |
| RDB | maffnet.py | recon (train aux) | ✔ | ✔ | ✔ | ✖ (train-only) | ✔ | — |
| FeatureAlignment | modules/ | stat harmonize | ✔ | ✔ | ✔ | ✔ | ✔ (stabilized only) | — |
| FrequencyBranch | modules/ | high-pass FFT | ✔ | ✔ | ✔ | ✔ | ✔ | — |
| GatedFusion | modules/ | σ-gate fuse | ✔ | ✔ | ✔ | ✔ | ✔ | — |
| MAFFNetLoss (BcDS) | trainer.py | Dice+BCE+BDS+EWR | ✔ | ✔ | ✔ | n/a | ✔ | — |
| FreqTransformer / EDB / SegHead (reconstructed) | maffnet_original_reconstructed.py | paper repro | ✔ | ✔ | ✔ (as maffnet_original) | ✔ | ✔ | checkpoint compat (§11) |
| MAFFNetOriginal (whole) | maffnet_original.py | paper repro (old) | ✔ | ✔ | ✖ | ✖ | ✖ unregistered | **orphan — decide keep/remove** |
| `_MAFFNetOriginalAdapter` | train.py | dict/p1→5-tuple shim | ✔ | ✔ (compile) | ✔ | n/a | ✔ | — |

---

## 7. Tensor Flow Verification

- **maffnet.py:** Input 384² → SAM2 (4 scales) → MMPA×4 → FGFF skips → DecoderBlock×… → p1–p4 (+ RDB rec). ✔ matches `MAFFNET_DESIGN.md` / `RESTORATION_REPORT.md`.
- **stabilized_maffnet.py:** Input → SAM2 → per-scale {FeatureAlignment(spatial) ∥ FrequencyBranch→FeatureAlignment} → GatedFusion → projected skips → DecoderBlock → p1–p4 (+ RDB rec). ✔ matches `STABILIZED_MAFFNET_ASSEMBLY_REPORT.md` and `paper/sections/section_III_proposed_methodology.md`.
- **maffnet_original_reconstructed.py:** Input → FrozenSAM → MMPA → {spatial proj ∥ FFT→FreqTransformer→proj} → FGFF → EDB (deep sup) → p1–p4 (+ RDB rec, train-only). ✔ 33/33 shapes (`FINAL_IMPLEMENTATION_REPORT.md`).

All three match their intended designs. No tensor-flow defect found.

---

## 8. Novelty Classification Matrix

| Component | Class | Note |
|---|---|---|
| SAM2 frozen encoder | **Existing** | from SAM2; frozen in all three |
| MMPA (dilated + weighting) | **Existing/Modified** | original MAFF-Net module; `maffnet.py` is a restoration |
| FGFF (FFT-magnitude) | **Existing/Modified** | original MAFF-Net module |
| RDB reconstruction | **Existing** | original MAFF-Net module, reused |
| FeatureAlignment | **New** | synopsis contribution (Proposed) |
| FrequencyBranch (high-pass) | **New** | synopsis contribution (Proposed) |
| GatedFusion | **New** | synopsis contribution (Proposed) |
| Boundary-aware Deep Supervision (BDS) | **New** | in `MAFFNetLoss` |
| Edge-weighted Reconstruction (EWR) | **New** | in `MAFFNetLoss` |
| Numerical stabilization (bf16, eps-sqrt, non-finite guard) | **Modified/New (engineering)** | in trainer |
| `maffnet_original.py` (MAFFNetOriginal) | **Deprecated** | superseded by `_reconstructed`; unregistered, no experiments |
| `transunet_legacy.py` | **Deprecated** | replaced by `transunet.py` (`MIGRATION_REPORT.md`) |
| Empty run dirs (removed earlier) | **Removed** | prior cleanup (`REPOSITORY_RESTRUCTURE_REPORT.md`) |

**Ambiguity flag:** BDS/EWR are in the shared `MAFFNetLoss` and were applied to **all three** models' training runs. The paper must state clearly which model is credited with these contributions and whether the "Original/paper-faithful" run should have used them at all (it should not, per §3a).

---

## 9. Figure Traceability Matrix

Source of truth for existing captions: `paper/figures/figure_captions.md` (Fig. 1–8, for the Proposed/Stabilized model). The task's figure list differs in numbering:

| Task figure | Task label | `paper/figures/figure_captions.md` | Match? |
|---|---|---|---|
| Fig. 1 | Overall Architecture | Fig. 1 overall architecture | ✔ |
| Fig. 2 | Feature Alignment | Fig. 2 = **Frequency branch** | ✖ mismatch |
| Fig. 3 | Frequency Branch | Fig. 3 = **Gated fusion** | ✖ mismatch |
| Fig. 4 | Gated Fusion | Fig. 4 = loss curve | ✖ mismatch |
| Fig. 5 | Boundary Supervision | Fig. 5 = IoU curve | ✖ mismatch |
| Fig. 6 | Training Pipeline | Fig. 6 = qualitative samples | ✖ mismatch |

**No Feature-Alignment figure and no Boundary-Supervision figure exist** in `figure_captions.md`; architecture schematics (Fig. 1–3) are marked "author-drawn" and the source PDFs are **not present** (`Journal_Paper/Figures/` holds only `.gitkeep`). **[MANUAL VERIFICATION REQUIRED] — figure files must be produced; numbering must be reconciled with the paper's final figure plan.**

---

## 10. Equation Traceability Matrix

`Journal_Paper/Equations/` contains only `.gitkeep` (no equation source). `paper/sections/section_III_proposed_methodology.md` contains numbered equations (1)–(21) for the Proposed model. Mapping code → those equations:

| Eq (section_III) | Implementation |
|---|---|
| (1) encoder features | `stabilized_maffnet` SAM2Encoder |
| (2)–(6) frequency branch | `modules/frequency_branch.py` |
| (7)–(9) feature alignment | `modules/feature_alignment.py` |
| (10)–(11) gated fusion | `modules/gated_fusion.py` |
| (12)–(14) decoder/skips/heads | `maffnet.DecoderBlock` + heads |
| (15)–(21) loss (Dice+BCE+Sobel-BDS+EWR) | `trainer.MAFFNetLoss` |

`Journal_Paper/03_Methodology/3.1–3.6` contains a **different** equation set (MMPA/FGFF/BcDS). **Two equation drafts exist for two different models** (see §13). Placeholders otherwise required for the un-authored equation files.

---

## 11. Experiment Verification

| Model | Experiment folders | metrics.json | summary.csv | curves+samples | model.pth |
|---|---|---|---|---|---|
| maffnet (Stabilized) | `maffnet_ISIC{2016,2017,2018}_final` | ✖ (MAFFNetTrainer writes summary.csv, not metrics.json) | ✔ | ✔ | ✔ |
| stabilized_maffnet (Proposed) | `stabilized_maffnet_ISIC{2016,2017,2018}` | ✖ | ✔ | ✔ | ✔ |
| maffnet_original_reconstructed (Original) | `maffnet_original_ISIC{2016,2017,2018}_final` (config confirms model) | ✖ | ✔ | ✔ | ✔ |
| baselines (unet/unetpp/attn/transunet) | multiple `*_final` / `*_ISIC_*` | ✔ | ✔ | ✔ | ✔ |

Verified metric samples (ISIC 2016, validation): maffnet dice **0.9345**, stabilized_maffnet **0.9316**, maffnet_original_reconstructed **0.9124** (each `summary.csv`).

**Findings:**
1. **Folder-name vs config mismatch:** `maffnet_original_*` folders were produced by `maffnet_original_reconstructed` (config.txt). Folder naming is misleading; the paper must cite the config-verified model.
2. **No per-experiment `training_log.csv`** in `experiments/` — `MAFFNetTrainer` writes it to `save_dir` (`checkpoints/`), shared/overwritten across runs. Per-run training curves survive only as `loss_curve.png`/`iou_curve.png`. **[MANUAL VERIFICATION REQUIRED]** if per-run epoch logs are needed for the paper.
3. **Checkpoint compatibility [CRITICAL]:** the C5 change altered `rdb.fuse_convs.0.0.weight` from `(576,1728,3,3)` to `(576,864,3,3)`, and the training adapter adds a `model.` prefix to state-dict keys. Any `maffnet_original_*` `model.pth` saved before C5 / without the adapter **will not load** into the current architecture without key remapping. **[MANUAL VERIFICATION REQUIRED] — attempt a `load_state_dict` before quoting these numbers as reproducible from current code.**
4. **Proposed metrics are validation-set, not test-set** (`paper/tables/performance_tables.md` marks them `†`), whereas baselines are test-set → **not directly comparable**.

---

## 12. Existing Report Validation

| Report group | Status | Note |
|---|---|---|
| `MILESTONE1–5`, `4B/4C`, `FINAL_IMPLEMENTATION_REPORT` | **Valid** | match current `maffnet_original_reconstructed.py` |
| `RDB_DESIGN_SPECIFICATION`, `RDB_PARAMETER_ANALYSIS`, `MAFFNET_FINAL_DESIGN_SPECIFICATION` | **Valid** | consistent with C5 code |
| `ORIGINAL_MAFFNET_IMPLEMENTATION_REPORT` | **Partially outdated** | describes `maffnet_original.py` (orphan), not the reconstructed/trained model |
| `FINAL_IMPLEMENTATION_REPORT.md` recon weight | **Check** | states λ_rec = 0.01 (milestone); `MAFFNET_FINAL_DESIGN_SPECIFICATION` §1.10 says 1.0 → **documented discrepancy, verify intended value** |
| `TRAIN_PIPELINE_INTEGRATION_REPORT` | **Valid** | matches current `train.py` |
| `STABILIZED_MAFFNET_ASSEMBLY_REPORT`, `INTEGRATION_REPORT`, `FEATURE_ALIGNMENT/FREQUENCY_BRANCH/GATED_FUSION` | **Valid** | match `stabilized_maffnet.py` + `modules/` |
| `RESTORATION_REPORT`, `MAFFNET_DESIGN`, `CONTRIBUTION_DESIGN` | **Valid** | match `maffnet.py` + `MAFFNetLoss` |
| Paths inside older reports | **Partially outdated** | pre-reorg root paths; content valid (`REPOSITORY_RESTRUCTURE_REPORT.md`) |

No fabricated report found. No report requires deletion. `ORIGINAL_MAFFNET_IMPLEMENTATION_REPORT` should be annotated as describing the orphan file.

---

## 13. Repository Consistency Check (mismatches)

1. **[CRITICAL] Naming collision.** `stabilized_maffnet.py` / `stabilized_maffnet_*` experiments / `performance_tables.md` "Stabilized MAFFNet (ours)" all denote the model the task calls **Proposed**. The task's **Stabilized** = `maffnet.py`. Paper, code, experiments, and tables use inconsistent names for the same models.
2. **[CRITICAL] Two methodology drafts.** `Journal_Paper/03_Methodology/3.1–3.6` describes MMPA/FGFF/**BcDS** (the `maffnet.py` direction). `paper/sections/section_III_proposed_methodology.md` describes FeatureAlignment/FrequencyBranch/**GatedFusion** (the `stabilized_maffnet.py` direction). The two paper workspaces describe **different architectures** as "the proposed method." (Consistent with `RESEARCH_ALIGNMENT_REPORT.md`.)
3. **Two "Original" files** (`maffnet_original.py` orphan vs `_reconstructed.py` trained).
4. **Folder vs config** naming (`maffnet_original_*` ← `maffnet_original_reconstructed`).
5. **Figure numbering** mismatch between task spec and `figure_captions.md` (§9).
6. **Loss used ≠ loss claimed** for the paper-faithful model (§3a): trained with BcDS, not its own Dice+MSE.
7. **Validation vs test** metrics for the proposed model (§11.4).
8. **Recon-weight** discrepancy 0.01 vs 1.0 (§12).

---

## 14. Paper Readiness Assessment

**Not ready.** Blocking items:
- Resolve the model-naming taxonomy (Original / Stabilized / Proposed) and apply it consistently to files, experiment folders, tables, and both paper workspaces.
- Choose **one** paper direction (BcDS `maffnet.py` vs GatedFusion `stabilized_maffnet.py`) or clearly frame the paper as a three-model study; retire the unused methodology draft.
- Verify checkpoint loadability (§11.3) before citing any trained metric as reproducible.
- Produce a **test-set** evaluation of the proposed model so its numbers are comparable to baselines; currently validation-only.
- Produce the architecture/feature/boundary figures (source files absent).
- Finalize equations (currently only `section_III` has them; `Journal_Paper/Equations/` empty).
- Decide on `maffnet_original.py` (keep as "faithful high-param variant" or remove from the narrative).

Ready elements: three models implemented and compiling; experiments exist for all three plus baselines; milestone + design reports are internally consistent; tensor flows verified.

---

## 15. Recommendations

1. Adopt a single naming map and rename experiment folders / table rows to match (documentation only; do **not** rename source files without a migration plan).
2. Add a one-paragraph "model taxonomy" note to `Journal_Paper/` fixing Original=`maffnet_original_reconstructed`, Stabilized=`maffnet`, Proposed=`stabilized_maffnet`.
3. Run `load_state_dict` smoke tests on the three `*_final` checkpoints against current code; record pass/fail.
4. Re-run the proposed model on the held-out **test** split; update `performance_tables`.
5. If the paper-faithful model must report a paper-loss run, retrain it with its own Dice+MSE (its experiments currently used BcDS).
6. Annotate `ORIGINAL_MAFFNET_IMPLEMENTATION_REPORT.md` as pertaining to the orphan file.
7. Generate the missing figures and equation files.

---

## 16. Final Verification Checklist

- [x] All model files located and classes identified (code-verified)
- [x] Experiment→model mapping verified via `config.txt`
- [x] Metrics read from `summary.csv` (not fabricated)
- [x] Naming collision identified and documented
- [x] Novelty matrix produced
- [x] Tensor flows verified against design reports
- [ ] Checkpoint loadability into current code — **NOT verified (manual)**
- [ ] Test-set metrics for proposed model — **absent**
- [ ] Single paper direction chosen — **unresolved**
- [ ] Figures / equations finalized — **absent**

---

## 17. Final Status

```
Original MAFF-Net         [ NEEDS UPDATE ]   (impl verified; orphan twin + checkpoint compat + loss-used≠claimed)
Stabilized MAFF-Net       [ NEEDS UPDATE ]   (impl verified; naming reconciliation required)
Proposed MAFF-Net         [ NEEDS UPDATE ]   (impl verified; named "Stabilized" in files/tables; val-only metrics)
Training Pipeline         [ VERIFIED ]       (train.py compiles; 3 models + baselines registered; maffnet_original.py intentionally unregistered)
Experiments               [ NEEDS UPDATE ]   (exist for all 3 + baselines; folder-vs-config naming + val≠test + checkpoint compat)
Implementation Reports    [ VERIFIED ]       (valid/consistent; annotate ORIGINAL_MAFFNET report + recon-weight note)
Repository                [ REQUIRES CHANGES ]
```

*This audit modified no code, no architecture, and no existing report. All findings are traceable to the cited files. Unverifiable items are flagged [MANUAL VERIFICATION REQUIRED].*
