# Training Pipeline Inspection — Why Restored MAFFNet Trains Faster

**Mode:** inspection only. No code modified.
**Headline:** batches/epoch = `ceil(N_train / batch_size)` with `batch_size=2`, `drop_last=False`. The drop **913 → 800** is purely a training-image-count change: **913 = 1825 images (ISIC 2018 train), 800 = 1600 images (ISIC 2017 re-split train)**. No augmentation removed, no dataloader change, backbone still frozen. A secondary per-step speedup comes from the lighter restored MMPA.

---

## Point-by-Point

**1. Total training images loaded** (= CSV rows; whole split loaded):
| Dataset | Train N |
|---------|--------:|
| ISIC 2016 | 895 |
| ISIC 2017 (re-split) | 1600 |
| ISIC 2018 | 1825 |
| ISIC 2017 (old, backup) | 2000 |

**2. Total validation images loaded:**
| Dataset | Val N |
|---------|------:|
| ISIC 2016 | 192 |
| ISIC 2017 (re-split) | 200 |
| ISIC 2018 | 377 |

**3. Batches per epoch** (`ceil(N/2)`, drop_last=False):
| Dataset | Train batches | Val batches |
|---------|-------------:|------------:|
| ISIC 2016 | 448 | 96 |
| ISIC 2017 (re-split) | **800** | 100 |
| ISIC 2018 | **913** | 189 |
| ISIC 2017 (old) | 1000 | — |

**4. drop_last** — **False** (not set anywhere → PyTorch default). Confirmed: `train_all.py:312-315` and `train.py:181-186` pass no `drop_last`. So the final partial batch (odd N, e.g. 1825) is kept → 913 not 912.

**5. Samples skipped for missing/corrupt masks** — **None.** The active loader `SegDataset` (`train_all.py:97-160`, `train.py:46-111`) reads every CSV row in `__getitem__` with no existence check; a missing file would raise, not skip. `len(dataset)` = CSV row count exactly. (The `except FileNotFoundError` / `continue` grep hits are in the dashboard generator and the per-model try/except in `run_all`, not in data loading. The legacy `datasets.py:ISICDataset` does skip, but it is unused.)

**6. Train/validation split changed?** — **Yes, ISIC 2017 only.** Re-split from 2000/150/600 → **1600/200/200** (seed 42; `SPLIT_AUDIT.md`, backup confirms old train=2000). ISIC 2016 and ISIC 2018 unchanged. This is the direct cause of the batch-count change for ISIC 2017 (1000 → 800).

**7. DataLoader configuration changed?** — **No.** `batch_size=2`, `shuffle=True` (train) / `False` (val), `pin_memory=True`, `num_workers=0` (`train_all.py`) or `args.workers=2` (`train.py`). Restoration touched only `models/maffnet.py` and the loss in `training/trainer.py`; it did not edit either DataLoader.

**8. Any augmentation removed?** — **No.** `SegDataset._augment` still applies hflip, vflip, rotation [−30°,30°], random affine, and ColorJitter on the train split; val/test still resize-only. Restoration did not touch the dataset/augmentation code.

**9. SAM2 backbone frozen or trainable?** — **Frozen.** `SAM2Encoder` / `SAMEncoder` call `p.requires_grad_(False)` on all encoder params (`maffnet.py:229,274`) and run the encoder under `torch.no_grad()` (`:232,277`). ~212.7M backbone params frozen.

**10. Optimizer receives all MAFFNet parameters?** — **No, by design.** `MAFFNetTrainer` builds `trainable = [p for p in model.parameters() if p.requires_grad]` and passes only those to AdamW (`trainer.py:269-270`). Frozen backbone excluded. Trainable ≈ **28.98M (12%)**: adaptors + FGFF + decoder + RDB + heads. Grad clipping likewise filters `requires_grad` (`:331`).

**11. Gradient flow reaches every trainable module?** — **Yes.** MMPA, FGFF, decoder, segmentation heads, and the RDB all receive gradient. Note the RDB now receives gradient because the restoration wired its reconstruction MSE into the loss; previously the RDB output was discarded and that branch got no gradient. The frozen encoder correctly receives none.

**12. Why batches dropped ~913 → ~800.** Batches/epoch depend only on `ceil(N_train / 2)`:
- **913 ⟺ 1825 training images** = ISIC 2018 train.
- **800 ⟺ 1600 training images** = ISIC 2017 **re-split** train.
- (Old ISIC 2017 = 2000 → 1000 batches.)

So the decrease is entirely a **training-set size change**, not a pipeline speedup. Either you compared an ISIC 2018 run (1825 → 913) against an ISIC 2017 re-split run (1600 → 800), or you compared ISIC 2017 before/after the re-split (2000 → 1600, i.e. 1000 → 800). In both cases fewer images → fewer steps/epoch. `drop_last`, batch size, workers, and augmentation are unchanged, so per-step behaviour is the same; the epoch is shorter only because it has fewer steps.

---

## Note on "much faster" (beyond batch count)

Two independent effects compound:
1. **Fewer steps/epoch** (Point 12): fewer training images after the re-split / dataset choice.
2. **Cheaper per step:** the restored MMPA is a lightweight bottleneck-32 depthwise adaptor (paper-faithful), much lighter than the pre-restoration full-width dilated + SE MMPA, so each forward/backward over the trainable modules does less work. The frozen backbone runs under `no_grad`, so its cost is forward-only and unchanged.

Batch count (Point 12) is driven solely by effect (1); overall wall-clock improvement is (1) plus (2). No correctness concern: the speedup is explained by data size and a lighter (intended) adaptor, not by dropped samples, removed augmentation, or a broken gradient path.

---

## Evidence
- `train_all.py:312-315`, `train.py:181-186` — DataLoader config (bs=2, no drop_last)
- `train_all.py:97-160` — active `SegDataset` (no skip)
- `models/maffnet.py:229,232,274,277` — backbone freeze + no_grad
- `training/trainer.py:269-270,331` — optimizer/clip on trainable params only
- CSV counts: 2016 895/192 · 2017 1600/200 (old 2000) · 2018 1825/377
- `SPLIT_AUDIT.md` (2017 re-split), `RESTORATION_REPORT.md` (28.98M trainable, lighter MMPA)
