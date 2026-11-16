# U-Net Baseline Failure — Root Cause Analysis

**Run:** `2026-04-28_15-19-03`
**Symptom:** U-Net produced `NaN` val loss + `Dice = 0.0000` on ISIC 2016/2017/2018. All four other models trained fine.

---

## TL;DR

`UNet.forward` ends with `torch.sigmoid`. `_LogitWrapper` clamps probs to `[1e-6, 1 - 1e-6]` then calls `torch.logit`. Whole stack run inside `autocast("cuda")` = fp16. In fp16, `1 - 1e-6` rounds to **exactly 1.0** → `torch.logit(1.0) = +inf` → loss = NaN → grad poison → permanent NaN. No other model hits this path.

---

## A. Exact Root Cause

Three things combine, only U-Net hits all three:

1. **U-Net only:** `models/unet.py:31` returns `torch.sigmoid(self.final(d1))` (probabilities, not logits).
2. **U-Net only:** wrapped in `_LogitWrapper` (`train_all.py:354`, `train.py:229`). Wrapper does:
   ```python
   probs = self.model(x).clamp(1e-6, 1 - 1e-6)
   return torch.logit(probs)
   ```
3. **Shared:** trainer forward runs inside `with autocast("cuda")` (`training/simple_trainer.py:132`, `167`). Activations cast to fp16.

**fp16 number facts** (IEEE 754 binary16, eps ≈ 9.77e-4):

- `1 - 1e-6` not representable in fp16. Nearest fp16 below 1.0 = `0.99951...`. `1 - 1e-6 = 0.999999` rounds to **1.0** (round-to-nearest-even).
- `clamp(..., 1.0)` upper bound = 1.0 in fp16.
- Any sigmoid output ≥ 0.99951 (very common after few epochs of training; saturated background pixels) clamps to **1.0** instead of `1 - 1e-6`.
- `torch.logit(1.0) = log(1/0) = +inf`.
- `BCEWithLogitsLoss(inf, target)` → NaN.
- NaN flows backward → all weights NaN → forever NaN. Matches CSV: NaN from epoch 1.

Lower bound `1e-6` also degenerate in fp16 (smallest normal ≈ 6.1e-5; 1e-6 = subnormal ~9.5e-7), but inf at upper bound triggers first.

## B. Files + Line Numbers

| File | Line(s) | Role in bug |
|------|---------|-------------|
| `models/unet.py` | **31** | `return torch.sigmoid(self.final(d1))` — emits probs not logits |
| `train_all.py` | **49–57** | `_LogitWrapper` class def |
| `train_all.py` | **354** | wraps U-Net: `model = _LogitWrapper(UNet())` |
| `train.py` | **27–39** | duplicate `_LogitWrapper` |
| `train.py` | **229** | wraps U-Net same way |
| `training/simple_trainer.py` | **132** | `with autocast("cuda"):` train forward (fp16) |
| `training/simple_trainer.py` | **167** | `with autocast("cuda"):` val forward (fp16) |
| `training/simple_trainer.py` | **40–44** | `SimpleLoss.forward`: `BCEWithLogits(logits, mask)` receives `+inf` |
| `training/trainer.py` | **126** | `compute_metrics` calls `torch.sigmoid(inf)` = NaN → Dice = 0 |

Failure trace:
```
UNet.forward → sigmoid (fp16 probs ∈ [0,1])
  → _LogitWrapper.clamp(1e-6, 1-1e-6)  [upper bound = 1.0 in fp16]
  → torch.logit(1.0) = +inf
  → SimpleLoss → BCEWithLogitsLoss(+inf, mask) = NaN
  → scaler.scale(NaN).backward() → NaN grads
  → params NaN → forever NaN
```

## C. Why Attention U-Net / UNet++ / TransUNet / MAFFNet Survive

| Model | Output | Wrapper? | Result |
|-------|--------|----------|--------|
| Attention U-Net | raw logits (`attention_unet.py:129`, `return self.final(d1)`) | none | direct to `BCEWithLogits` — stable |
| UNet++ | raw logits | none | stable |
| TransUNet | raw logits | none | stable |
| MAFFNet | raw logits (p1–p4) | none, uses `MAFFNetLoss` | stable |
| **U-Net** | **sigmoid(probs)** | **`_LogitWrapper` → logit(fp16 saturated)** | **NaN** |

`BCEWithLogitsLoss` uses log-sum-exp internally → numerically stable for any finite logit. Direct logit models never see inf. U-Net path inserts manual `sigmoid → logit` round trip, which under fp16 cannot survive saturation.

## D. Secondary Concerns (not primary cause but compound)

1. **`UNet.final` uses no BN before sigmoid.** Raw conv output can be large (±10). `sigmoid(large) → 1.0` quickly. Increases saturation rate.
2. **`SimpleLoss.forward` (`simple_trainer.py:42`) calls `torch.sigmoid(logits)` again** for Dice path, then BCE on logits. If logits = inf, dice sigmoid(inf) = 1.0 (ok), but BCE blows up. Both losses summed → NaN propagates.
3. **`GradScaler.update()` does not skip step on NaN reliably** when NaN is in loss before scaling (scaler only checks unscaled grads via `unscale_`). NaN grads should trigger skip, but params already poisoned via running stats / Adam moments if first step bypassed scaler check.
4. **BatchNorm in U-Net** with batch size = 2: running stats can drift, but not the failure cause (other BN-based models pass).
5. **No grad checking / NaN sentinel** in `SimpleTrainer._train_epoch` — bug invisible until val log written.

## E. Minimal Fix (recommended)

**Best fix:** make U-Net emit logits like every other model. One-line change in `models/unet.py:31`:

```python
# before
return torch.sigmoid(self.final(d1))
# after
return self.final(d1)
```

Then drop `_LogitWrapper` wrapping at `train_all.py:354` and `train.py:229`:

```python
# before
model = _LogitWrapper(UNet())
# after
model = UNet()
```

Net effect: U-Net path becomes identical to Attention U-Net path. `BCEWithLogits` handles sigmoid internally with log-sum-exp = numerically stable in fp16. No clamp, no logit, no inf.

**Fallback fix (if `unet.py` must stay sigmoid-output):** force fp32 inside wrapper:

```python
def forward(self, x):
    probs = self.model(x).float().clamp(1e-6, 1 - 1e-6)
    return torch.logit(probs)
```

`.float()` upcasts to fp32 → `1 - 1e-6` representable → `logit` finite. Works but adds compute + breaks autocast intent.

## F. Verification Plan

1. **Unit test the fp16 hypothesis** (no training needed):
   ```python
   import torch
   from torch.amp import autocast
   x = torch.tensor([0.99999], device="cuda")
   with autocast("cuda"):
       y = x.clamp(1e-6, 1 - 1e-6)
       print(y.dtype, y.item(), torch.logit(y).item())
   # expect: torch.float16  1.0  inf
   ```
   If prints `inf` → hypothesis confirmed.

2. **Apply Section E fix.** Run 1 epoch U-Net on ISIC 2018 (smallest val split):
   ```
   python train.py --model unet --dataset ISIC_2018 --epochs 1 --exp unet_fix_test
   ```
   Expect: finite `train_loss`, finite `val_loss`, `dice > 0`.

3. **Compare against Attention U-Net epoch 1** on same dataset (from prior run = Dice 0.7188 @ epoch 1 ISIC 2016). Plain U-Net should land in similar 0.6–0.75 Dice range epoch 1.

4. **Full 20-epoch run** on all 3 datasets via `train_all.py`. Expect U-Net Dice to fall between UNet++ and Attention U-Net (typical literature ordering).

5. **Add NaN sentinel** to `SimpleTrainer._train_epoch` (post-fix, defensive):
   ```python
   if not torch.isfinite(loss):
       raise RuntimeError(f"non-finite loss at epoch {epoch}, step {step}")
   ```
   Fails fast on future regressions.

6. **Regression check** Attention U-Net + UNet++ + TransUNet + MAFFNet unchanged — fix touches only `unet.py` and two wrapping lines; other models untouched.

---

## Evidence Summary

| Claim | Source |
|-------|--------|
| U-Net output is `sigmoid(...)` | `models/unet.py:31` |
| Other models output raw logits | `models/attention_unet.py:129`, docstring lines 1–5 |
| LogitWrapper clamps then calls `torch.logit` | `train_all.py:55–57`, `train.py:37–39` |
| U-Net wrapped, others not | `train_all.py:354` vs. `367, 380, 393` (no wrap) |
| Forward runs under autocast fp16 | `training/simple_trainer.py:132, 167` |
| Loss = Dice + BCEWithLogits | `training/simple_trainer.py:40–44` |
| Result NaN / Dice 0.0 on all 3 datasets | `results/2026-04-28_15-19-03/summary.csv` rows 1–3 |
| Attention U-Net (same trainer, no wrapper) trained fine | same `summary.csv` rows 4–6 |

Conclusion: bug isolated to U-Net output + wrapper + autocast interaction. Section E fix removes the round trip. No other architecture or trainer change needed.
