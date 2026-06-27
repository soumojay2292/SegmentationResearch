# Pre-Training Verification — Official TransUNet (Route B)

**Run before the 40-epoch experiments. No code modified.**
**Verdict: ALL 7 CHECKS PASS. Cleared to train.**

Environment: conda `segmentation` (timm 1.0.26, torch, numpy 2.2.6).

---

## 1. Pretrained Weights Actually Loaded (not silent random fallback)

| Check | Result |
|-------|--------|
| `pretrained_loaded` flag | **True** |
| Stem weight `==` fresh pretrained reference (`torch.equal`) | **True** |

The strong check loads a separate `timm.create_model(..., pretrained=True)` and compares `patch_embed.backbone.stem.conv.weight` byte-for-byte → **identical**. This proves the model holds the real `vit_base_r50_s16_384.orig_in21k_ft_in1k` ImageNet-21k→1k weights, **not** a random init. The offline fallback (random + warning) did **not** trigger. ✅

---

## 2. Total Parameter Count

**101,123,809 (101.1 M)** — ViT-B scale, consistent with R50+ViT-B/16. ✅
(Legacy impl was ~2 M.)

---

## 3. Trainable Parameter Count

**101,123,809 (101.1 M) — frozen: 0.** Entire backbone + decoder fine-tuned (standard full-fine-tune TransUNet; no layers frozen). ✅

---

## 4. ImageNet Normalisation Applied Only for TransUNet

Grep `0.485 | Normalize | norm_mean | norm_std` across `models/` + `train.py` + `train_all.py`:

| Location | Has norm? |
|----------|:---------:|
| `models/transunet.py:70,71,93` | ✅ yes (in-forward) |
| `models/{unet,attention_unet,unet_plus_plus,maffnet}.py` | ❌ none |
| `train.py`, `train_all.py` (shared `SegDataset`) | ❌ none |

Normalisation is isolated inside `TransUNet.forward` (`x = (x - norm_mean)/norm_std`, buffers `[0.485,0.456,0.406]`/`[0.229,0.224,0.225]`). Other models and the shared dataloader are untouched → only the TransUNet pipeline is normalised. ✅

---

## 5. Output Tensor Shape Always (B,1,384,384)

| Batch | Output | OK |
|------:|--------|:--:|
| 1 | (1, 1, 384, 384) | ✅ |
| 2 | (2, 1, 384, 384) | ✅ |
| 4 | (4, 1, 384, 384) | ✅ |

Raw logits, single channel — matches `SimpleLoss` / `compute_metrics` contract. ✅

---

## 6. metrics.json / report.html / Benchmark Still Work

| Artifact | Check | Result |
|----------|-------|--------|
| `metrics.json` | writer present in `SimpleTrainer.train` (`simple_trainer.py:394-412`) + `compute_metrics` finite on TransUNet output (keys acc/dice/mIoU/sensitivity/specificity/f1) | ✅ a real run will emit valid metrics.json |
| `report.html` | `generate_report` on a TransUNet run dir → HTML written, model inferred as **TransUNet** | ✅ (verified in migration smoke) |
| Benchmark | `generate_benchmark()` → `benchmark_2026-06-21_14-36-36.html` (105,234 bytes) | ✅ |
| Aggregate | `aggregate_benchmark.main()` → 11 experiments → CSV+XLSX+MD | ✅ |

All reporting/benchmark utilities run unchanged (none were modified). ✅

---

## 7. No Legacy TransUNet Class Imported Anywhere

Grep `transunet_legacy | TransUNetLegacy` across repo (excluding `src/sam2` and the legacy file itself):
- Only hit: a **docstring comment** in `models/transunet.py:4` ("archived in transunet_legacy.py") — **not an import**.
- `train.py:22` and `train_all.py:26` both `from models.transunet import TransUNet` → resolves to the **new** hybrid class (confirmed: uses timm).

**No code path imports the legacy class.** ✅

---

## Summary

| # | Check | Result |
|---|-------|:------:|
| 1 | Pretrained really loaded (not random) | ✅ |
| 2 | Total params | 101.1 M |
| 3 | Trainable params | 101.1 M (0 frozen) |
| 4 | ImageNet norm only for TransUNet | ✅ |
| 5 | Output (B,1,384,384) | ✅ |
| 6 | metrics.json / report.html / benchmark | ✅ |
| 7 | No legacy class imported | ✅ |

**Cleared for 40-epoch runs:**
```
python train.py --model transunet --dataset ISIC_2016 --epochs 40 --patience 10 --exp transunet_ISIC_2016 --save_dir experiments/transunet_ISIC_2016/checkpoints
python train.py --model transunet --dataset ISIC_2017 --epochs 40 --patience 10 --exp transunet_ISIC_2017 --save_dir experiments/transunet_ISIC_2017/checkpoints
python train.py --model transunet --dataset ISIC_2018 --epochs 40 --patience 10 --exp transunet_ISIC_2018 --save_dir experiments/transunet_ISIC_2018/checkpoints
```

---

## Evidence References
- `models/transunet.py:60-71,93` — pretrained load + norm buffers + in-forward norm
- `models/transunet_legacy.py` — archived, unreferenced
- `train.py:22,259` / `train_all.py:26,393` — import + `TransUNet()` (new class)
- `training/simple_trainer.py:394-412` — metrics.json writer
- verification run: pretrained stem weight == fresh pretrained ref; 101.1M params; shapes (B,1,384,384); benchmark 105,234 bytes; aggregate 11 rows
