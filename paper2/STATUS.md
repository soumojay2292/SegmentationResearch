# Paper 2 — Status

| Audit / Stage | Status |
|---|---|
| Synopsis→Implementation Compliance | PASS (canonical = `stabilized_maffnet.py`) |
| Methodology sections III-A … III-I | PASS (assembled, implementation-faithful) |
| **Experimental Pipeline Audit** | **PASS** (see `paper2/audits/11_Experimental_Pipeline_Audit.txt`) |
| **Experiment Asset Inventory** | **PASS** (inventory complete) — Results Readiness = **NOT READY** (see `paper2/audits/12_Experiment_Asset_Inventory.txt`) |
| **Results Completion Plan** | **READY** — 6 required tasks; critical path ~5–8 GPU-h, full (incl. ablation) ~20–40 GPU-h [est] (see `paper2/audits/13_Results_Completion_Plan.txt`) |
| **Checkpoint Verification** | **PASS** — 9/9 canonical checkpoints strict-load; no retraining required; Evaluation Ready = YES (see `paper2/audits/14_Checkpoint_Verification.txt`) |
| **Test Evaluation** | **PASS** — 9/9 canonical MAFF-Net runs evaluated on the held-out test split; `metrics.json` written for each; 0 failures (see `paper2/audits/15_Test_Evaluation_Report.txt`) |
| **U-Net 2017 Test Evaluation** | **PASS** — checkpoint loaded (strict), test evaluated, `experiments/unet_final_2017/metrics.json` created; no retraining (see `paper2/audits/16_Unet2017_Test_Evaluation.txt`) |
| **TransUNet Verification** | **PASS** — valid ISIC 2018 run already exists (test Dice 0.8259); **Step 4 retrain removed**; two audit-12 findings corrected (see `paper2/audits/17_TransUNet2018_Verification.txt`) |
| **Final Benchmark Table** | **PASS** — 15 canonical experiments (+6 supplementary) tabulated from `metrics.json["test"]` only; 0 values from `summary.csv` (see `paper2/audits/18_Final_Benchmark_Table.txt`, outputs in `paper2/results/`) |

| **Claim Consistency Audit** | **NEEDS REVISION** — 27 supported / 12 unsupported / 9 needing weaker wording; no boundary metric exists anywhere despite the paper's boundary-centric thesis (see `paper2/audits/19_Claim_Consistency_Audit.txt`) |
| **Boundary Evaluation** | **PASS** — 21/21 canonical checkpoints evaluated (HD95, ASSD, Boundary IoU) by inference only; MAFF-Net family beats every baseline on all three boundary metrics on all three datasets; resolves audit-19 §0 (see `paper2/audits/20_Boundary_Evaluation.txt`, outputs in `paper2/results/boundary_metrics.*`) |
| **Paper Asset Audit** | **NEEDS ASSETS** — results layer freeze-ready; missing bibliography, architecture Figs 1–3, cross-method qualitative figure, dataset/config/complexity tables; module ablation not run (see `paper2/audits/21_Paper_Asset_Audit.txt`) |
| **Ablation Design** | **PASS** — minimum 3-run leave-one-out matrix (−Frequency / −Feature Alignment / −Gated Fusion) on ISIC 2018, ~5.5 GPU-h, reusing the existing Full checkpoint; non-invasive subclass, no code change; supports the 3 architectural claims but **not** BcDS efficacy (see `paper2/audits/22_Ablation_Design.txt`) |
| **Architecture Ablation** | **PASS** — 3 new checkpoints trained + evaluated (region + boundary) on ISIC 2018; canonical model/trainer untouched; Full Proposed best on all 3 boundary metrics; **modules help boundary fidelity, not region Dice** (single-seed, no significance) (see `paper2/audits/23_Ablation_Execution.txt`, outputs in `paper2/results/ablation_results.*`) |

## Repository Freeze: **RESULTS LAYER READY / FULL MANUSCRIPT NOT READY**

Freeze the numerical results now (benchmark + boundary tables are complete, canonical, TEST-split, mutually consistent). Do **not** declare a manuscript freeze until these are produced/decided (audit 21):

- **References — MISSING entirely.** No `.bib`, no bibliography; blocks Abstract/Intro/Related Work. Highest-priority unblock.
- **Architecture Figures 1–3 — MISSING.** No `Figures/` dir exists; only captions. Blocks III.
- **~~Module ablation — not run.~~ RESOLVED (Architecture Ablation, Step 23).** 3-run leave-one-out on ISIC 2018 now in `paper2/results/ablation_results.*`. Key finding: the three modules **improve boundary fidelity (HD95/ASSD/Boundary IoU), not region Dice** — Full Proposed is best on all three boundary metrics, but removing −Frequency or −Gated Fusion leaves Dice ~flat/marginally higher. Frame the contribution as boundary fidelity, not Dice. Single-seed → no significance claim. BcDS efficacy (loss ablation) still not run.
- **Three tables NEED work:** dataset summary (assemble), training-config/hyperparameter (stale — reflects old 20-epoch `train_all.py`, λ 0.01; reconcile to `train.py` 40-epoch λ 0.05), proposed-model complexity (current numbers describe the Restored/original model, not `stabilized_maffnet.py`).
- **Cross-method qualitative figure (Fig 7)** needs baseline prediction dumps (figure-production pass, not a new experiment).
- **Random-seed policy** must be disclosed as a limitation (no seed anywhere; all results single-run).
- **Author decision still open:** Proposed-vs-Restored positioning (N1) — reinforced, not settled, by boundary metrics; gates the entire Results/Discussion framing.

## Repository State: **RESULTS READY** — Manuscript: **NEEDS CHANGES**

### Blocking claim issues (from audit 19)

- **~~No boundary metric is computed anywhere.~~ RESOLVED (Boundary Evaluation, Step 7).** HD95, ASSD and Boundary IoU now exist for all 21 canonical checkpoints in `paper2/results/boundary_metrics.*`. The MAFF-Net family beats every baseline on all three boundary metrics on all three datasets — the boundary-centric thesis is now measured, not just asserted. Note: boundary metrics reinforce rather than settle the Proposed-vs-Restored question (Restored has lower HD95/ASSD on all three datasets).
- **No ablation exists.** `paper/tables/ablation_tables.md` is an honest empty placeholder; the three architectural contributions have zero individual experimental support.
- **No seed anywhere** → every result is single-run; no "significant"/superiority claim is defensible.
- **`paper/tables/performance_tables.md` is superseded and partly unsourceable** — its U-Net ISIC 2017 Dice (0.7613) matches no `metrics.json` (canonical is 0.7229), its Attention U-Net 2018 "—" is stale, and its proposed rows are validation metrics. Replace with `paper2/results/final_benchmark_table.md`.
- **`Figures/` does not exist** — Fig. 1, 2, 3, 7, 8 have no source file (Fig. 4–6 assets do exist).
- Two III-B/III-G contradictions to fix: the reconstruction branch consumes **fused** (not encoder) features, and its output is **discarded** at inference rather than omitted from computation.

All benchmark values are held-out TEST metrics. Outstanding items are author decisions and disclosures, not missing data:

- **F1 (Critical, author decision):** on TEST Dice, **MAFF-Net (Restored) outperforms MAFF-Net (Proposed) on all three datasets** (+0.0014 / +0.0020 / +0.0143). The Proposed variant instead wins precision and specificity. A "best Dice" claim is not supported by the data.
- **F2 (Major):** Precision is not computed anywhere in the repository; the reported values are algebraically derived and labelled as such. Converting them to measured values needs one sub-GPU-hour re-evaluation accumulating global TP/FP/FN.
- **F3 (Major):** `f1` is identical to `dice` by construction — disclose or drop the column.
- **F4 (Major):** `mIoU` is foreground IoU, not a mean over classes.
- **F5 (Major):** Paper-faithful Reconstruction rows used the placeholder encoder (no pretrained SAM2) and must be labelled wherever shown.
- **F6/F7 (Minor):** U-Net ISIC 2017 trained 20 epochs vs 40 elsewhere; TransUNet ISIC 2018 diverged after its best epoch (results valid, curve needs explanation if plotted).

## Experimental Pipeline Audit — PASS

- Dataset pipeline reconstructed — PASS (ISIC 2016 895/192/192, 2017 1600/200/200, 2018 1825/377/392; 384², binarised masks).
- Training pipeline reconstructed — PASS (MAFFNetTrainer: AdamW/LinearLR/bf16; SimpleTrainer: Dice+BCE, early stopping).
- Evaluation pipeline reconstructed — PASS (best val Dice selection; baselines→test metrics.json, MAFF-Net family→val summary.csv).
- Metrics verified — PASS (acc, Dice, mIoU, sensitivity, specificity, F1 @0.5; `compute_metrics`).
- Baselines identified — PASS (repo: U-Net, Attn U-Net, UNet++, TransUNet; MAFF-Net family).
- Hardware verified — PASS at env level (Python 3.10.20, torch 2.7.1+cu118, CUDA 11.8, Windows); per-run hardware not recorded.
- Reproducibility verified — PASS (state documented; no training seed = Major).
- No unresolved Critical findings — PASS.

### Open items for manuscript attention (not blocking; documented in the audit)
- **M1 (Major):** no training random seed / determinism controls.
- **M2 (Major):** Proposed/Restored/Faithful metrics are validation-set; baselines are test-set → not directly comparable until a proposed test run exists.
- **M3 (Major):** SGT-Net and Swin-UNet (synopsis/paper baselines) not implemented in the repository.
- Minor: sparse per-run config logging; shared `training_log.csv`; no early stopping in MAFFNetTrainer.
- Editorial: per-experiment hardware not recorded.
