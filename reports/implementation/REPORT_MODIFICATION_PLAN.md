# Report Modification Plan

**Task:** upgrade reporting only. No change to training, models, datasets, losses, preprocessing, checkpoints, or experiment logic.
**Status:** PLAN — awaiting approval. No code modified yet.

---

## Current Reporting System (as found)

| Component | File | Behaviour |
|-----------|------|-----------|
| Single-experiment report | `utils/report_generator.py` → `generate_report(log_csv, summary_csv, image_dir, config)` | Writes **`results/dashboard/report.html`** (shared — **overwritten every run**). Title/architecture hardcoded "MAFFNet". Shows 1 sample only. No model/exp/timestamp. |
| PNG curves | `utils/report_generator.py` → `generate_training_curves` | Writes `results/dashboard/loss_curve.png`, `iou_curve.png` (MAFFNetTrainer copies these into its run_dir) |
| Benchmark dashboard | `train_all.py` → `_generate_dashboard(summary_csv, out_html)` | Bar chart of Dice/IoU from `summary.csv` → `results/dashboard/segmentation_research_dashboard.html` |
| Report callers | `training/simple_trainer.py`, `training/trainer.py` | Both call `generate_report(...)` with the same 4 args |

**Problems vs. requirements:** report overwrites history; hardcoded MAFFNet labels (wrong for UNet); no best/avg/difficult samples; benchmark lacks Sens/Spec, ranking, %improvement.

---

## Files That WILL Be Modified

### 1. `utils/report_generator.py` — rewrite single-experiment report
**Purpose:**
- Write report to the **experiment's own folder** (`run_dir/report.html`, derived from `image_dir.parent`) → **never overwrites historical reports**, preserves per-exp folder structure.
- **Self-contained HTML** (inline CSS, Chart.js via CDN) — no dependency on shared `style.css`, portable.
- Header auto-shows: **model name, dataset, experiment name, training timestamp** (from `log_csv` mtime), **best epoch**.
- Metric cards: **Dice, IoU, Sensitivity, Specificity, Accuracy** (read from best-dice row of `training_log.csv` — already present, no trainer change).
- Three plots: **train-loss vs epoch, val-loss vs epoch, val-Dice vs epoch** (Chart.js from log CSV).
- Qualitative samples categorized **Best / Average / Difficult** — computed by reading saved `gt_i.png` / `pred_i.png` pairs and scoring per-sample Dice **inside the report generator** (uses PIL+numpy; no trainer change).
- **Signature unchanged** `generate_report(log_csv, summary_csv, image_dir, config)` → both callers keep working.
- `generate_training_curves` keeps writing PNGs to `results/dashboard/` so MAFFNetTrainer's existing copy step still works (backward compat).
- Remove `webbrowser.open` auto-launch during batch runs? → keep but guarded (no behaviour change to training).

### 2. `utils/benchmark_report.py` — NEW FILE (benchmark dashboard)
**Purpose:**
- Aggregate completed experiments by scanning `experiments/*/checkpoints/training_log.csv`; parse `model`, `dataset`, `timestamp` from folder name `{model}_{dataset}_{timestamp}`; take the **best-Dice epoch row** → Dice/IoU/Sens/Spec/Acc. Works on **all historical experiments** (every run has `training_log.csv`).
- Compares **UNet, Attention UNet, UNet++, TransUNet, MAFFNet** across **ISIC 2016/2017/2018**.
- Outputs: comparative metric **table**, **separate plots** for Dice / IoU / Sensitivity / Specificity, **ranking by average Dice**, **% improvement vs. UNet baseline**.
- Self-contained HTML → **`results/benchmark/benchmark_<timestamp>.html`** (timestamped → **no overwrite of history**) + a convenience `benchmark_latest.html`.

### 3. `train_all.py` — minimal, report-section only
**Purpose:** in the **post-training dashboard block only** (the `try: ... _generate_dashboard ...` section near the end), add a call to `benchmark_report.generate_benchmark(...)`. Existing `_generate_dashboard` left intact (additive). **Training loop, model construction, data loading, metrics JSON, summary writing — all untouched.**

### 4. `training/simple_trainer.py` — report_config enrichment only
**Purpose:** add two keys to the `report_config` dict passed to `generate_report`: `"model": self.model_name`, `"exp": config.get("exp", "—")`. This dict is consumed **only** by the report generator. **No training change.**

### 5. `training/trainer.py` (MAFFNetTrainer) — report_config enrichment only
**Purpose:** same two keys (`"model": "maffnet"`, `"exp": ...`) added to its `report_config`. Report-only. **No training change.**

---

## Confirmation — Files That Will NOT Be Touched

- `models/*` (unet, attention_unet, unet_plus_plus, transunet, maffnet, sam2_loader) — no architecture change
- `datasets.py`, `datasets_hg.py`, `dataset_split/`, `dataset_raw/` — no data change
- Loss/metric code (`DiceLoss`, `BoundaryLoss`, `SimpleLoss`, `MAFFNetLoss`, `compute_metrics`) — untouched
- Preprocessing / augmentation in `train.py` / `train_all.py` `SegDataset` — untouched
- Checkpoints, `best_*.pth`, `model.pth`, `metrics.json` writing — untouched
- Training loops, optimiser, scheduler, early stopping, AMP — untouched
- `config.py`, `split_dataset.py`, `prepare_isic.py`, `scripts/*` — untouched

**Training behaviour invariant.** The only trainer edits are two added keys in report-config dicts (consumed solely by the report generator). Everything that affects weights, metrics, or saved artifacts stays identical.

---

## Backward Compatibility & Safety

| Requirement | How met |
|-------------|---------|
| Preserve folder structure | single report → `run_dir/report.html`; benchmark → `results/benchmark/` |
| No overwrite of historical reports | per-exp report + timestamped benchmark file |
| Backward compat with existing experiments | benchmark reads universal `training_log.csv`; `generate_report` signature unchanged; old `_generate_dashboard` kept |
| Publication-ready style | self-contained HTML, clean light research-dashboard theme, embedded plots |
| MAFFNet curve-copy still works | `generate_training_curves` still writes to `results/dashboard/` |

---

## Deliverables After Implementation
1. This plan (done).
2. Exact list of modified files.
3. Confirmation training behaviour unchanged (+ `py_compile` verification, no training run).
