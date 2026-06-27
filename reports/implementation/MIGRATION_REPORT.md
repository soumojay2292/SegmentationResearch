# TransUNet Migration Report — Route B (timm R50+ViT-B/16)

**Done:** legacy in-name-only TransUNet replaced with the official-style **ResNet-50 + ViT-B/16 hybrid** (ImageNet-21k→1k pretrained) via timm + CUP decoder. Public API unchanged; trainer / report / benchmark untouched and compatible. All smoke checks pass.

---

## 1. Files Modified / Added

| File | Action | Detail |
|------|--------|--------|
| `models/transunet_legacy.py` | **Added** | Verbatim archive of the old impl; class renamed `TransUNetLegacy` (+`TransUNet` alias). Nothing imports it. |
| `models/transunet.py` | **Replaced** | New R50+ViT-B/16 hybrid + CUP decoder. Same public class `TransUNet(in_ch=3, out_ch=1)`. |
| `MIGRATION_REPORT.md` | **Added** | This file. |

**Not modified (per constraints):**
- `train.py`, `train_all.py` — **no edit needed**. Both still call `TransUNet()` (lines 259 / 393); the new constructor keeps the same defaults, so `--model transunet` works unchanged.
- `training/` (trainer, losses), `utils/report_generator.py`, `utils/benchmark_report.py`, `utils/aggregate_benchmark.py` — untouched.
- `models/maffnet.py` and all other baselines — untouched.
- Dataloaders — untouched (ImageNet normalisation is applied **inside** the TransUNet forward, not in the shared loader).

---

## 2. Architecture Implemented

| Component | Implementation |
|-----------|----------------|
| Backbone | **ResNet-50 stem** (3 stages) — part of timm `vit_base_r50_s16_384` |
| Tokenizer | patch-16 on R50 feature grid → 24×24 = **576 tokens** @384 (+1 cls) |
| Transformer | **ViT-B/16**: 12 layers, **768** dim, **12** heads, **MLP 3072**, LayerNorm |
| Positional emb | learned (native 384 variant — no interpolation needed) |
| Pretrained | **`vit_base_r50_s16_384.orig_in21k_ft_in1k`** (ImageNet-21k→1k), `pretrained=True` default |
| Normalisation | ImageNet mean/std applied inside `forward` (input assumed [0,1]); TransUNet-only |
| Decoder | **CUP** (cascaded upsampler): 4 ×2 stages 768→256→128→64→32, /16 → /1 |
| Seg head | `Conv2d(32, out_ch=1, 1)` → raw logits |
| Output | `(B, 1, H, W)` logits (no sigmoid — matches `SimpleLoss`) |
| Param count | **101.1 M** (vs legacy ~2 M) |
| Fallback | if pretrained download fails (offline) → random init + warning (training still runs) |

**Deviation from paper (documented):** CUP decoder is **skipless** (no R50 intermediate skips fused) for robustness with the timm hybrid wrapper. Encoder + pretraining + ViT-B are faithful; skip fusion can be added later if boundary detail needs it.

---

## 3. Compatibility Preserved

| Subsystem | Status | Why |
|-----------|:------:|-----|
| `--model transunet` (train.py) | ✅ | `TransUNet()` API unchanged |
| `train_all.py` loop | ✅ | same `TransUNet()` call |
| Output shape `(B,1,H,W)` | ✅ | verified (2,1,384,384) |
| Trainer (`SimpleTrainer`) | ✅ | `logits=model(imgs)` → single logit map; early-stop/ckpt/metrics.json unaffected |
| Loss (`SimpleLoss` Dice+BCE) | ✅ | finite on new output (smoke: 1.538) |
| Report generator | ✅ | infers `TransUNet` from folder name; no change |
| Benchmark + aggregate | ✅ | read training_log.csv / metrics.json generically; `transunet` token already mapped |
| ImageNet norm | ✅ | inside model forward → no loader/trainer edit |

---

## 4. Smoke-Test Results (no full training)

| Check | Result |
|-------|--------|
| `py_compile` transunet.py + transunet_legacy.py | ✅ OK |
| `py_compile` train.py + train_all.py (build sites) | ✅ OK |
| Build `TransUNet()` (default API) | ✅ |
| **Pretrained weights loaded** | ✅ `pretrained_loaded = True` (downloaded + cached) |
| Param count | 101.1 M (ViT-B scale) |
| **Output shape** | ✅ `(2, 1, 384, 384)` |
| **Finite loss** (SimpleLoss) | ✅ 1.5379, `isfinite=True` |
| **Report generation** | ✅ `report.html` written, model inferred as `TransUNet` |
| Build sites still `TransUNet()` | ✅ lines 259 / 393 unchanged |
| New class resolves (not legacy) | ✅ timm hybrid confirmed |

---

## 5. Notes / Follow-ups (not done here)

- **Re-train required** to get real metrics: `python train.py --model transunet --dataset ISIC_2016 --epochs 40 --patience 10 --exp transunet_ISIC_2016 --save_dir experiments/transunet_ISIC_2016/checkpoints` (likewise 2017/2018). First run downloads the backbone once (cached).
- Batch size 2 is small for a 101 M ViT-B — consider gradient accumulation (optional; not changed).
- Old transunet results remain on disk and are now architecturally incomparable — re-run for fair comparison.
- Optional future: add R50 skip fusion to the CUP decoder for finer boundaries.
- HF symlink warning on Windows is benign (cache works without symlinks).

---

## Evidence References
- `models/transunet.py` — new hybrid impl (timm `vit_base_r50_s16_384.orig_in21k_ft_in1k`, CUP decoder, in-forward ImageNet norm)
- `models/transunet_legacy.py` — archived old impl
- `train.py:259`, `train_all.py:393` — unchanged `TransUNet()` build sites
- smoke test: pretrained_loaded=True, output (2,1,384,384), SimpleLoss=1.5379 finite, report→TransUNet
- `OFFICIAL_TRANSUNET_MIGRATION_PLAN.md` — the plan this implements (Route B)
- `TRANSUNET_AUDIT.md` — gap analysis motivating the migration
