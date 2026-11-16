# Integration Report — StabilizedMAFFNet → train.py

**Task:** repository integration only — register `stabilized_maffnet` in `train.py` and add its build branch. No existing model modified. Static compile only — no training, smoke test, or benchmark.

---

## Files Modified
| File | Change |
|------|--------|
| `train.py` | (1) added import; (2) added `stabilized_maffnet` to `--model` choices; (3) added model-construction branch |

No other file modified. `models/stabilized_maffnet.py` already existed (assembly task); `models/maffnet.py` and `training/trainer.py` **untouched** (timestamps unchanged: maffnet.py Jun 22, trainer.py Jun 24). No existing model class altered.

---

## Import Changes (`train.py`)
Added one import beneath the existing MAFFNet import:
```python
from models.maffnet import MAFFNet
from models.stabilized_maffnet import StabilizedMAFFNet   # <-- added
from models.attention_unet import AttentionUNet
```

---

## Model Registry Changes (`train.py`)

**`--model` choices:**
```python
# before
choices=["maffnet", "attention_unet", "unetpp", "unet", "transunet"]
# after
choices=["maffnet", "stabilized_maffnet", "attention_unet", "unetpp", "unet", "transunet"]
```

**Construction branch** (added directly after the `maffnet` branch; mirrors it and reuses `MAFFNetTrainer`):
```python
elif args.model == "stabilized_maffnet":
    print("Building Stabilized MAFFNet (SAM2 + frequency branch + gated fusion) …")
    model   = StabilizedMAFFNet(checkpoint=args.checkpoint)
    trainer = MAFFNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=args.save_dir,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
```
The branch reuses the existing `MAFFNetTrainer` (and therefore the existing boundary supervision and metrics/JSON logging) unchanged, since `StabilizedMAFFNet` returns the same `(p1,p2,p3,p4,rec)` interface.

---

## Verification

| Check | Result |
|-------|--------|
| `py_compile train.py models/stabilized_maffnet.py` | ✅ `COMPILE_OK` |
| `--model` choices include `stabilized_maffnet` (argparse `--help`, no model build) | ✅ shown in usage |
| `from models.stabilized_maffnet import StabilizedMAFFNet` resolves | ✅ `import OK: StabilizedMAFFNet` |
| Existing models untouched | ✅ `maffnet.py` / `trainer.py` timestamps unchanged |
| Training / smoke / benchmark | ⏸ not executed (per instruction) |

---

## Usage (when a run is authorised — not executed here)
```
python train.py --model stabilized_maffnet --dataset ISIC_2016 --epochs 40 --patience 10 \
  --exp stab_maffnet_ISIC_2016 \
  --checkpoint checkpoints/sam2_hiera_large.pt \
  --save_dir experiments/stab_maffnet_ISIC_2016/checkpoints
```
`--checkpoint` is required (same constructor guard as MAFFNet — see `SMOKE_INIT_ANALYSIS.md`).

---

## Notes
- `train_all.py` was **not** modified (its `model_names` loop still lists the original five). If `stabilized_maffnet` should run in the all-models sweep, add it there as a separate follow-up.
- Constraints honoured: only `train.py` edited; no existing model changed; static compile only; nothing trained/benchmarked.

## Evidence References
- `train.py` (import, choices, build branch), `models/stabilized_maffnet.py` (target model)
- `STABILIZED_MAFFNET_ASSEMBLY_REPORT.md`, `SMOKE_INIT_ANALYSIS.md` (checkpoint requirement)
