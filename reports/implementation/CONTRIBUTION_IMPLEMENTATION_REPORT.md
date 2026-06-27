# Contribution Implementation Report — Boundary-centric Dual Supervision (BcDS)

**Implements:** `CONTRIBUTION_DESIGN.md` — N1 Boundary-aware Deep Supervision (BDS) + N2 Edge-weighted Reconstruction (EWR).
**Scope honoured:** supervision/loss pipeline **only**. MAFFNet architecture, encoder, decoder **unchanged**. No full training.
**Result:** compile + smoke + ablation-switch + warmup checks all pass.

---

## 1. What Was Implemented

| Item | Where | Detail |
|------|-------|--------|
| **BDS (N1)** | `MAFFNetLoss` | Per-head Sobel-edge L1 vs GT edges, deep-supervision weights `w={1,1,1,0.01}`, normalised by Σw, **warmup ramp** `λ_bds·min(1,t/t_warm)`, optional pred smoothing |
| **EWR (N2)** | `MAFFNetLoss` | Edge-weighted reconstruction `Σ W⊙(rec01−x)²/(ΣW·C)`, `W=1+α·B`, GT-boundary or input-edge source |
| **Shared boundary map** | `MAFFNetLoss._boundary_map` | `B(y)` from Sobel-thin or morphological gradient, optional dilation band `band_r` |
| **Hyperparameter switches** | `MAFFNetLoss.__init__` | `lambda_bds, lambda_ewr, alpha, t_warm, band_r, ds_weights, use_bds, use_ewr, use_bce, smooth_pred, bmap, ewr_edge_source, pos_weight` |
| **Switch passthrough** | `MAFFNetTrainer.__init__(loss_kwargs=None)` | Programmatic ablation without touching `train.py`/`train_all.py` |
| **Logging** | trainer CSV | `train_boundary_loss` = BDS; **new** `train_ewr_loss` column; both accumulated per epoch |

**Total objective:** `L = L_seg + λ_bds(t)·L_BDS + λ_ewr·L_EWR` (defaults = full BcDS).

---

## 2. Files Modified

| File | Change |
|------|--------|
| `training/trainer.py` | `MAFFNetLoss` reformulated to BcDS (+switches +boundary map); `MAFFNetTrainer.__init__` `+loss_kwargs`; `_train_epoch` accumulates `ewr_loss`; `_init_csv` + train-loop `_log_csv` add `train_ewr_loss` column |

**Not modified:** `models/maffnet.py` (architecture/encoder/decoder untouched), `train.py`, `train_all.py`, `utils/report_generator.py`, `utils/benchmark_report.py`, `utils/aggregate_benchmark.py`, dataloaders, other models.

---

## 3. Compatibility

| Subsystem | Status | Why |
|-----------|:------:|-----|
| `train.py` / `train_all.py` | ✅ | `MAFFNetTrainer(...)` called without `loss_kwargs` → full-BcDS defaults; no signature break |
| Model I/O | ✅ | Still consumes `(p1..p4, rec)`; no architecture change |
| Report generator | ✅ | Reads named columns; new `train_ewr_loss` appended at end is ignored by it |
| Benchmark / aggregate | ✅ | Read `val_*` by name from `training_log.csv` / `metrics.json`; extra column safe |
| CSV schema | ✅ | header 13 cols == row 13 values (verified) |
| Ablation access | ✅ | `MAFFNetTrainer(..., loss_kwargs={"use_ewr":False, ...})` |

---

## 4. Verification

### Compile
- `py_compile training/trainer.py models/maffnet.py` → **OK**
- CSV header columns = 13, row values = 13 → **symmetric**

### Smoke (built real MAFFNet w/ SAM2; forward on random [0,1] input)
- Output shapes: p1–p4 `(1,1,384,384)`, rec `(1,3,384,384)` ✅

### Ablation-switch matrix (ep0 / ep10 total loss; finite)
| Variant | ep0 | ep10 | bds | ewr | finite |
|---------|----:|-----:|----:|----:|:------:|
| **D full BcDS** | 1.5460 | 1.5905 | 0.8900 | 0.0869 | ✅ |
| B BDS only | 1.5416 | 1.5861 | 0.8900 | 0.0000 | ✅ |
| C EWR only | 1.5460 | 1.5460 | 0.0000 | 0.0869 | ✅ |
| A base | 1.5416 | 1.5416 | 0.0000 | 0.0000 | ✅ |
| warmup t_warm=0 | 1.5905 | 1.5905 | 0.8900 | 0.0869 | ✅ |
| edge_source=image | 1.5460 | 1.5905 | 0.8900 | 0.0870 | ✅ |
| bmap=morph | 1.5460 | 1.5905 | 0.8900 | 0.0869 | ✅ |

### Logic checks (all True)
- BDS active only when `use_bds`; EWR active only when `use_ewr`; base has neither.
- **Warmup verified:** with `t_warm=5`, BDS contributes 0 at ep0 (ramp) and full at ep10 → ep10 total > ep0 total.
- `t_warm=0` ⇒ always-on (ep0 == ep10).
- Numbers consistent: `A_ep0(1.5416) + λ_ewr·ewr(0.05·0.0869=0.0043) = 1.5460 = D_ep0` ✅; `+ λ_bds·bds(0.05·0.89=0.0445) = 1.5905 = D_ep10` ✅.
- `MAFFNetTrainer` exposes `loss_kwargs` ✅.

---

## 5. Defaults (full BcDS)

`lambda_bds=0.05, lambda_ewr=0.05, alpha=3.0, t_warm=5, band_r=1, ds_weights=(1,1,1,0.01), use_bds=True, use_ewr=True, use_bce=True, smooth_pred=True, bmap="sobel", ewr_edge_source="gt"`.

Ablation recipes (via `loss_kwargs`):
- **A base:** `use_bds=False, use_ewr=False, alpha=0`
- **B BDS-only:** `use_ewr=False`
- **C EWR-only:** `use_bds=False`
- **D full:** defaults
- sweeps: `lambda_bds`, `lambda_ewr`, `alpha`, `t_warm`, `band_r`, `bmap`, `ewr_edge_source`, `use_bce`.

---

## 6. Not Done (by instruction)

- No full training (only forward/loss smoke).
- No architecture/encoder/decoder change.
- No metric additions (HD95/Boundary-IoU) — belongs to the evaluation step.
- BCE still on by default (now a switch) — paper-faithful Dice-only is `use_bce=False`, to be decided during ablation.

---

## 7. Next Step (not executed)

Run the A/B/C/D ablation (≥3 seeds, test split, +HD95/Boundary-IoU) to test the core claim **D > max(B, C)**. Example:
```
python train.py --model maffnet --dataset ISIC_2016 --epochs 40 --exp maffnet_bcds_D --checkpoint checkpoints/sam2_hiera_large.pt
```
(For ablation variants, construct `MAFFNetTrainer(loss_kwargs=...)` — a thin CLI flag can be added later if desired.)

---

## Evidence References
- `training/trainer.py` — `MAFFNetLoss` (BcDS, switches, `_boundary_map`), `MAFFNetTrainer(loss_kwargs)`, `_train_epoch` ewr accumulation, CSV `train_ewr_loss`
- smoke: shapes OK; A/B/C/D matrix; warmup ramp; numeric consistency; loss_kwargs present
- `CONTRIBUTION_DESIGN.md` (spec), `RESTORATION_REPORT.md` (frozen base)
