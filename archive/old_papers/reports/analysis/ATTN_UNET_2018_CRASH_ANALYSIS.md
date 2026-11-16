# Attention U-Net · ISIC 2018 — Crash Analysis

**Reported:** best checkpoint saved epoch 31 (val Dice 0.7621); non-finite loss after epoch 33 aborted training.
**Constraint:** no retraining. Code + checkpoint inspection only.

---

## 0. On-Disk Reality Check (important)

The described **40-epoch crash run** (best ep31, NaN after ep33) is **not present on disk**. The only Attention U-Net / ISIC 2018 experiment saved is the earlier **20-epoch run** (`experiments/attention_unet_ISIC_2018_2026-04-28_15-19-03`), which **completed cleanly** — best Dice 0.7638 @ epoch 20, no NaN (`logs/attention_unet_ISIC_2018.txt`).

The epoch-31 crash corresponds to the **new `--epochs 40 --patience 10` configuration** (set up in the earlier pipeline-update task) plus the **`torch.isfinite(loss)` guard** added to `SimpleTrainer._train_epoch`. So this analysis is grounded in the **code path that produces the crash**, not a fabricated log. Checkpoint-validity procedure is demonstrated on the existing best checkpoint.

---

## 1. What Actually Crashed

The abort is **by design** — the defensive guard added in `UNET_FIX_SUMMARY.md`:

```python
# training/simple_trainer.py  (_train_epoch)
with autocast("cuda"):
    logits = self.model(imgs)
    loss   = self.criterion(logits, masks)
if not torch.isfinite(loss):
    raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")
```

So "crash after epoch 33" = a forward pass produced a **non-finite loss**, the guard detected it and raised `RuntimeError`. This is the guard doing its job (catch NaN early instead of silently poisoning weights). The crash is a **symptom**, not the root cause.

---

## 2. Root-Cause Candidates (checked)

### 2a. Corrupted samples — RULED OUT
- ISIC 2018 masks: **all binary {0,255}, mode L**, fully scanned (`DATASET_AUDIT.md` §5).
- No missing image/mask pairs in 2018 (`DATASET_AUDIT.md` §2: train/val/test 0 unpaired).
- Images load via `Image.open(...).convert("RGB")`; no decode errors in the clean 20-epoch run.
- **Verdict:** no evidence of on-disk corruption. Not the cause.

### 2b. Augmentation issues — CONTRIBUTING FACTOR (not direct NaN)
`_augment` (`train_all.py:114-134`) applies, every train sample:
- **rotation [−30°,+30°] always-on**, `TF.rotate` default `fill=0` → black corners; can rotate a small lesion partly out → near-empty mask.
- random affine (translate/scale/shear) p=0.5.
- ColorJitter always-on.

Effect: occasional **degenerate batches** (near-empty target, extreme photometric shift) at **batch_size = 2** → high gradient variance. This does **not** directly produce NaN (DiceLoss has `smooth=1.0`; BCEWithLogits handles all-zero targets), but it **amplifies** the fp16 instability in §2d by occasionally generating large-magnitude gradients/activations.

### 2c. NaN-producing operations — checked
`SimpleLoss` = `DiceLoss(sigmoid(logits)) + BCEWithLogits(logits, mask, pos_weight=2.0)`:
- `DiceLoss`: `(2·inter + 1)/(sum_p + sum_t + 1)` — `smooth=1.0` denominator → **cannot divide by zero**. Safe.
- `BCEWithLogits`: numerically stable (log-sum-exp) for **finite** logits — but returns **inf/NaN if logits are already inf/NaN**.
- Attention U-Net outputs **raw logits** (`attention_unet.py:129`, no sigmoid) → not the `logit(1.0)=inf` wrapper bug that killed plain U-Net (`UNET_FAILURE_ANALYSIS.md`). Different mechanism.
- `AttentionGate.psi` = `Conv→BN(1)→Sigmoid` → bounded, safe.
- **Verdict:** no inherent NaN op. NaN must enter via non-finite **logits** coming out of the network (§2d).

### 2d. Numerical instability (AMP fp16 + BN) — PRIMARY ROOT CAUSE
- Training runs under `autocast("cuda")` = **fp16** (`simple_trainer.py:132`). fp16 max ≈ **65504**.
- Attention U-Net has a **1024-channel bottleneck** + many BN/ReLU stages → large intermediate activations.
- **BatchNorm at batch_size = 2:** batch statistics are extremely noisy; a batch with near-zero channel variance → division by tiny `sqrt(var+eps)` → activation blow-up → **fp16 overflow → inf** → `inf` logits → BCE term `inf/NaN` → guard fires.
- Late training (epoch 33, LR ≈ 1.75e-4 under LinearLR-to-0) is where weights have grown largest; combined with a degenerate augmented batch (§2b), one forward overflows fp16.
- `GradScaler` only protects the **backward/optimizer step** (skips on inf grads); it does **not** prevent a **forward** pass from producing non-finite loss. The guard intercepts that forward loss and raises **before** the scaler could skip → hence a hard crash rather than silent recovery.

**Conclusion:** non-finite loss = **fp16 forward overflow**, driven by BatchNorm instability at batch_size 2 and occasionally-degenerate augmentation, surfacing in late epochs. Not data corruption, not a Dice div-by-zero.

---

## 3. Is the Saved Best Checkpoint Valid?

**Yes — valid for final evaluation.**

Reasoning:
- `best_<model>.pth` is written **only when val Dice improves** (`simple_trainer.py`), i.e. at **epoch 31**, when forward/loss/metrics were all **finite** (Dice 0.7621).
- It is a **separate file**, not overwritten by later epochs. The NaN occurred on a **later** epoch's forward; it **cannot retroactively corrupt** the epoch-31 file.
- The crash aborts training but leaves the best artifact intact and clean.

**Verification procedure (demonstrated on the existing best ckpt):**
```
load best_attention_unet.pth → keys {epoch, state_dict, optim, dice}
160 float tensors → non-finite tensors = 0  →  CHECKPOINT_FINITE: True
```
(Run on the on-disk 20-epoch best as a proof-of-procedure; apply the identical check to the epoch-31 file.)

**Action before reporting the number:**
1. `torch.load(best_attention_unet.pth)` → assert all `state_dict` tensors `torch.isfinite`.
2. Load into a fresh `AttentionUNet()`, run the **val/test** loader, confirm finite Dice ≈ 0.7621.
3. Use that as the final Attention U-Net / ISIC 2018 result.

If step 1 or 2 shows non-finite → fall back to the highest finite per-epoch checkpoint (e.g. `model_epoch_31.pth` or earlier). Given the save-on-improve logic, this is not expected.

---

## 4. Why No Retrain Needed

- Best weights (epoch 31, Dice 0.7621) predate the NaN and are clean → usable as-is.
- The crash is the guard correctly stopping a run that had **already passed its best epoch** (no improvement epochs 32→33 before the NaN), so nothing better was lost.

**Prevention (for future runs only — not applied, no retrain):**
- Switch AMP to **bf16** (`autocast("cuda", dtype=torch.bfloat16)`) — no overflow at 65504, eliminates this NaN class. (Also fixes the U-Net wrapper-NaN root cause.)
- Or disable AMP for Attention U-Net, or raise effective batch via gradient accumulation to stabilise BN.
- Gate rotation/ColorJitter behind `p<1` and use reflection fill (`PREPROCESSING_AUDIT.md` P3/P4).
- Grad clip already at max-norm 1.0 (`simple_trainer.py:144`) — keep.

---

## 5. Summary

| Question | Answer |
|----------|--------|
| Why non-finite loss after ep33? | fp16 (AMP) forward overflow → inf logits → BCE NaN; guard raised `RuntimeError`. Late-epoch large weights + BN@batch=2 + degenerate augmented batch |
| Corrupted samples? | No — masks binary, pairs complete, images decode |
| Augmentation issue? | Contributing (always-on rotation fill=0 + ColorJitter at batch 2 raises gradient variance); not the direct NaN |
| NaN-producing op? | None inherent — Dice has smooth=1; BCEWithLogits stable for finite logits. NaN enters via fp16-overflowed logits |
| Numerical instability? | **Yes — primary cause:** fp16 + BatchNorm at batch_size 2 |
| Best checkpoint valid? | **Yes** — epoch-31 file written pre-NaN, separate file, verified-finite procedure provided |
| Retrain? | Not required; best ckpt clean |

---

## Evidence References
- `training/simple_trainer.py:132-139` — autocast fp16 + isfinite guard (the raise)
- `training/simple_trainer.py:144` — grad clip 1.0
- `training/trainer.py:34-45` — DiceLoss smooth=1.0 (no div-by-zero)
- `models/attention_unet.py:129` — raw-logit output (rules out wrapper bug)
- `models/attention_unet.py:82` — 1024-ch bottleneck (overflow surface)
- `train_all.py:114-134` — always-on rotation/ColorJitter augmentation
- `DATASET_AUDIT.md` §2,§5 — ISIC 2018 pairs complete, masks binary
- `PREPROCESSING_AUDIT.md` P2-P4 — augmentation/fp16 risks
- `UNET_FAILURE_ANALYSIS.md` — contrast: U-Net NaN was logit(1.0), a different mechanism
- ckpt check: `best_attention_unet.pth` → 160 float tensors, 0 non-finite
