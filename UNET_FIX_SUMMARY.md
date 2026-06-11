# U-Net NaN Fix — Summary

Applied recommended fix from `UNET_FAILURE_ANALYSIS.md`. U-Net now emits raw logits, bypasses `_LogitWrapper`, and trainer guards against non-finite loss.

---

## Files Modified

| # | File | Lines | Change |
|---|------|-------|--------|
| 1 | `models/unet.py` | 31 | Drop final `torch.sigmoid` — return raw logits |
| 2 | `train_all.py` | 354 | Remove `_LogitWrapper` wrap |
| 3 | `train.py` | 229 | Remove `_LogitWrapper` wrap |
| 4 | `training/simple_trainer.py` | 132–138 | Add `torch.isfinite(loss)` guard after forward |

Not touched: Attention U-Net, UNet++, TransUNet, MAFFNet, `DiceLoss`, `SimpleLoss`, `MAFFNetLoss`, `MAFFNetTrainer`, `SimpleTrainer` structure (only guard added).

---

## Exact Code Changes

### 1. `models/unet.py:31`

```diff
-        return torch.sigmoid(self.final(d1))
+        return self.final(d1)
```

U-Net `forward` now returns `(B, 1, H, W)` raw logits, matching Attention U-Net, UNet++, TransUNet contract.

### 2. `train_all.py:354`

```diff
                     elif model_name == "unet":
                         print("Building U-Net …")
-                        model   = _LogitWrapper(UNet())
+                        model   = UNet()
                         trainer = SimpleTrainer(
```

`_LogitWrapper` class definition (`train_all.py:49–57`) left in place but unused — no caller; harmless dead code. Can be removed in cleanup pass.

### 3. `train.py:229`

```diff
     elif args.model == "unet":
         print("Building U-Net …")
-        model   = _LogitWrapper(UNet())
+        model   = UNet()
         trainer = SimpleTrainer(
```

Same treatment. `_LogitWrapper` def at `train.py:27–39` left in place, unused.

### 4. `training/simple_trainer.py` — NaN guard in `_train_epoch`

Inserted after the autocast forward, before `pbar.set_postfix`:

```python
            with autocast("cuda"):
                logits = self.model(imgs)
                loss   = self.criterion(logits, masks)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss detected at epoch {epoch}"
                )

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            self.scaler.scale(loss).backward()
```

Fails fast on first NaN/Inf — prevents 20 wasted epochs of silent NaN propagation.

---

## Why Fix Is Numerically Stable

Old path (broken under fp16 autocast):

```
final_conv → sigmoid → clamp(1e-6, 1-1e-6) → logit → BCEWithLogits
                                    ↑
                  fp16: 1-1e-6 rounds to 1.0 → logit(1.0) = +inf → NaN
```

New path (stable):

```
final_conv → BCEWithLogits(logits, target)
                  ↑
       internal log-sum-exp; safe for any finite logit in fp16
```

Three stability guarantees:

1. **`BCEWithLogitsLoss` uses log-sum-exp identity:** `BCE(z, y) = max(z, 0) - z·y + log(1 + exp(-|z|))`. Never evaluates `log(0)` or `log(inf)`. Stable across fp16 dynamic range (~6e-5 to 6e4).
2. **`DiceLoss` receives `sigmoid(logits)` inside `SimpleLoss.forward`** (`simple_trainer.py:42`). `torch.sigmoid` is fp16-safe — saturates to 0 or 1 but never produces inf/NaN. Dice numerator/denominator finite.
3. **No `clamp → logit` round trip.** Source of the fp16 boundary collapse (`1 - 1e-6 == 1.0` in fp16) eliminated. Logits stay in their natural range; no manual probability inversion.

U-Net forward now identical contract to Attention U-Net (already proven stable on all 3 datasets in the `2026-04-28_15-19-03` run — Dice 0.66–0.87).

Bonus: NaN guard catches any future regression on **any** model trained via `SimpleTrainer` (Attention U-Net, UNet++, TransUNet) within one mini-batch instead of one full run.

---

## Verification Commands

### 1. fp16 logit-boundary unit test (confirms original bug existed)

```bash
python -c "
import torch
from torch.amp import autocast
x = torch.tensor([0.999999], device='cuda')
with autocast('cuda'):
    y = x.clamp(1e-6, 1 - 1e-6)
    z = torch.logit(y)
print('dtype:', y.dtype, 'clamped:', y.item(), 'logit:', z.item())
"
# expect: dtype: torch.float16  clamped: 1.0  logit: inf
```

### 2. Smoke test — single epoch on smallest dataset

```bash
python train.py --model unet --dataset ISIC_2016 --epochs 1 --exp unet_fix_smoke
```

Pass criteria:
- No `RuntimeError: Non-finite loss detected ...`
- `train_loss` finite (typical: 0.8–1.2 epoch 1)
- `val_loss` finite, `val_dice > 0` (expect 0.55–0.75)

### 3. Regression — verify Attention U-Net unaffected

```bash
python train.py --model attention_unet --dataset ISIC_2016 --epochs 1 --exp att_unet_regression
```

Pass criteria:
- Trains identically to prior runs (epoch-1 Dice ≈ 0.72, matches prior log `attention_unet_ISIC_2016_2026-04-28_15-19-03/checkpoints/training_log.csv`).

### 4. NaN guard self-test (optional)

Force a NaN by injecting `loss = loss + float('nan')` temporarily; expect `RuntimeError: Non-finite loss detected at epoch 0` on first batch. Revert.

### 5. Full re-run

```bash
python train_all.py
```

Pass criteria (vs. prior `results/2026-04-28_15-19-03/summary.csv`):
- U-Net rows: `Final Val Loss` finite, `Final Val Dice` in [0.65, 0.85] across ISIC 2016/2017/2018.
- Other four models: metrics within ±0.01 Dice of prior run (no regression).

---

## Out of Scope (Not Done, By Instruction)

- `_LogitWrapper` class definitions still present in `train.py:27–39` and `train_all.py:49–57` — unused, safe to delete later.
- No changes to loss functions, trainers, or other architectures.
- No hyperparameter changes.
- No new tests committed beyond the NaN guard.
