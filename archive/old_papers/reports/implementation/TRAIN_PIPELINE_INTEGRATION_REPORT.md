# Training-Pipeline Integration Report — MAFFNetOriginalReconstructed

**Task:** register and integrate `MAFFNetOriginalReconstructed` into the existing training pipeline. Architecture, layers, tensor flow, and losses **not** modified. No dataset executed; no training run — only `py_compile` and `--help`.

---

## 1. Files modified

| File | Change | Architecture touched? |
|---|---|---|
| [`train.py`](../../train.py) | import, argparse choice, output adapter shim, model-build branch | No |
| `models/maffnet_original_reconstructed.py` | **none** | No (unchanged) |
| `training/trainer.py` | **none** | No (unchanged) |

Only `train.py` was edited. The reconstructed model and the trainer are untouched.

---

## 2. Model registration

1. **Import:**
   ```python
   from models.maffnet_original_reconstructed import MAFFNetOriginalReconstructed
   ```
2. **Argparse choice** added:
   ```text
   maffnet_original_reconstructed
   ```
3. **Build branch** (mirrors the existing maffnet/stabilized blocks, wrapped in the shim):
   ```python
   elif args.model == "maffnet_original_reconstructed":
       base    = MAFFNetOriginalReconstructed(sam_checkpoint=args.checkpoint)
       model   = _MAFFNetOriginalAdapter(base)   # 5-tuple interface shim
       trainer = MAFFNetTrainer(model=model, train_loader=..., val_loader=...,
                                save_dir=..., num_epochs=..., lr=..., weight_decay=...)
   ```

---

## 3. Interface compatibility analysis

**Trainer expectation.** `MAFFNetTrainer` calls `preds = self.model(imgs)` and passes `preds` straight to `MAFFNetLoss`, which unpacks **`p1, p2, p3, p4, rec = preds`** (a 5-tuple), and uses **`preds[0]`** for metrics. This holds in **both** `_train_epoch` and `_val_epoch` (val sets `model.eval()` first). So the trainer requires `model(imgs)` to return the 5-tuple in train **and** eval.

**Reconstructed-model interface (differs):**
| Mode | Reconstructed model returns | Trainer needs |
|---|---|---|
| train | `dict {segmentation_logits:[p1..p4], reconstruction:rec}` | `(p1,p2,p3,p4,rec)` |
| eval | single tensor `p1` | `(p1,p2,p3,p4,rec)` |

**Mismatch in both paths** → a compatibility layer is required.

**Compatibility layer (`_MAFFNetOriginalAdapter`, in `train.py`):** a thin `nn.Module` wrapping the model; the wrapped model is not modified.
- **train:** unpacks the dict → `(p1, p2, p3, p4, rec)`.
- **eval:** the model yields only `p1`; the shim returns `(p1, p1, p1, p1, zeros_rec)` so the val-loss unpack and `preds[0]` metric path both work (metrics use `preds[0] = p1`, which is correct; the duplicated aux heads only affect the auxiliary val-loss value, not checkpoint selection, which is driven by Dice on `p1`).

This is the smallest change that satisfies the existing interface without editing the model.

**Loss source (flagged):** the trainer drives the model with its existing `MAFFNetLoss` (BcDS), exactly as it does for `maffnet`/`stabilized_maffnet`. The reconstructed model's own paper-faithful Dice+MSE loss (`compute_losses`) is **not invoked** in this path — it remains available for standalone use and is unmodified. Using the trainer's loss is consistent with how the pipeline already treats the maffnet family; switching to the model's internal loss would require trainer changes, which were intentionally avoided.

---

## 4. Were trainer changes required?

**No.** `training/trainer.py` was not touched. The single output adapter in `train.py` bridges the interface. Optimizer param-grouping still works (gamma params end with `"gamma"` → still excluded from weight decay even under the `model.` prefix; frozen encoder params `requires_grad=False` → still skipped). Train/eval modes, checkpoint saving (`self.model.state_dict()` of the wrapped module), and experiment-directory creation (`experiments/<exp>`) are all handled by the unchanged trainer/`main()`.

| Capability | Mechanism | Works? |
|---|---|---|
| Train mode | `MAFFNetTrainer._train_epoch` + adapter (dict→tuple) | ✔ |
| Eval mode | `_val_epoch` (model.eval) + adapter (p1→tuple) | ✔ |
| Checkpoint saving | `torch.save(self.model.state_dict())` (adapter-wrapped) | ✔ |
| Experiment dir | `main()` `experiments/<exp>` + `trainer.train(exp_dir=...)` | ✔ |

> Note: checkpoint state-dict keys carry a `model.` prefix from the adapter; loading is via the same adapter, so it is self-consistent. (A bare-model load would need to strip the prefix — documented for future use.)

---

## 5. Compile status

| Check | Command | Result |
|---|---|---|
| Syntax compile | `python -m py_compile train.py` | **OK** |
| CLI help | `python train.py --help` | **OK** — new choice listed |

`--help` output confirms registration:
```
--model {maffnet,stabilized_maffnet,maffnet_original_reconstructed,attention_unet,unetpp,unet,transunet}
```

No dataset, no training executed.

---

## 6. Final supported model list

```
maffnet
stabilized_maffnet
maffnet_original_reconstructed   ← new
attention_unet
unetpp
unet
transunet
```

| Model | Trainer | Output interface |
|---|---|---|
| maffnet | MAFFNetTrainer | 5-tuple (native) |
| stabilized_maffnet | MAFFNetTrainer | 5-tuple (native) |
| **maffnet_original_reconstructed** | MAFFNetTrainer | dict/p1 → 5-tuple via `_MAFFNetOriginalAdapter` |
| attention_unet / unetpp / unet / transunet | SimpleTrainer | single logits |

---

**Integration complete and verified (compile + --help). No training executed; no dataset run.**

*Status: pipeline integration / train.py only / trainer + model unchanged / no training run.*
