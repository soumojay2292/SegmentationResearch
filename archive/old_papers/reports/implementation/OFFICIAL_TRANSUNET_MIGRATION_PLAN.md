# Official TransUNet — Migration Plan

**Goal:** replace the in-name-only TransUNet (`models/transunet.py`) with the real architecture (R50+ViT-B_16 hybrid, CUP decoder, ImageNet-21k pretrained).
**Status:** PLAN ONLY — no code modified. See `TRANSUNET_AUDIT.md` for the gap analysis this builds on.

---

## 1. Ours vs. Official (delta to close)

| Component | Ours | Official (Chen et al. 2021 / Beckschen repo) |
|-----------|------|----------------------------------------------|
| Backbone | 2 conv layers, ×2 down | **ResNet-50** (3 stages), ×16 down, ImageNet-pretrained |
| Tokenizer | none (36,864 tokens @384) | patch-16 on R50 feature map → ~576 tokens @384 |
| Transformer | 2 layers / 128 / 4h / MLP256 | **12 / 768 / 12h / MLP3072 (ViT-B)** |
| Pos-emb | fixed sinusoidal | learned, interpolated |
| Decoder | 1 skip, 1×2 upsample | **CUP**: 4 upsample stages + 3 R50 skips |
| Pretrained | none | **R50+ViT-B_16, ImageNet-21k** |
| Output | conv→1 (logits) | conv→n_classes (logits) |

---

## 2. Missing Components (full list)

1. ResNet-50 hybrid stem + 3 skip-emitting stages (`vit_seg_modeling_resnet_skip.py`).
2. Patch embedding on the R50 feature grid.
3. Full ViT-B encoder: 12 layers, 768 dim, 12 heads, MLP 3072, LayerNorm, MSA+MLP blocks.
4. Learned positional embedding + interpolation for non-224 inputs.
5. CUP decoder: 4 `DecoderBlock`s with 3 skip fusions + segmentation head.
6. Config object (`vit_seg_configs.get_r50_b16_config()`), needs **`ml_collections`** (not installed).
7. Pretrained weight loader `VisionTransformer.load_from(np.load(npz))`.
8. The pretrained file **`R50+ViT-B_16.npz`** (ImageNet-21k, ~300–400 MB) → download into `checkpoints/`.
9. ImageNet input normalisation (pretrained backbone needs it; our loader gives raw [0,1]).

---

## 3. Two Migration Routes

### Route A — Vendor the official repo (faithful reproduction)
Copy Beckschen/TransUNet `networks/{vit_seg_modeling.py, vit_seg_configs.py, vit_seg_modeling_resnet_skip.py}` into `models/transunet_official/`; add `ml_collections`; download `R50+ViT-B_16.npz`.
- **Pro:** exact paper architecture + official weights.
- **Con:** +3 vendored files, +`ml_collections` dep, +npz download, pos-emb interpolation wiring for 384.

### Route B — timm hybrid (pragmatic, recommended)
Build encoder from **`timm.create_model("vit_base_r50_s16_224.orig_in21k", pretrained=True, num_classes=0)`** (R50+ViT-B hybrid, ImageNet-21k — **timm 1.0.26 already installed**), add a CUP-style decoder (reuse our `DecoderBlock` pattern from `maffnet.py`).
- **Pro:** no `ml_collections`, no manual npz, modern weights, less vendored code, timm handles pos-emb interpolation.
- **Con:** not byte-identical to the paper repo (still R50+ViT-B + CUP — architecturally equivalent).

**Recommendation: Route B.** Lower risk/effort, uses installed deps, same architectural class. Use Route A only if exact paper-repo reproduction is required for the write-up.

---

## 4. Files To Modify / Add

| File | Action | Route |
|------|--------|-------|
| `models/transunet.py` | **Replace** body with official impl, OR keep + add new `models/transunet_official.py` and re-point imports | A & B |
| `models/transunet_official/` (new, 3 files) | vendor official networks | A only |
| `train.py:259` | build branch: `TransUNet()` → construct official + `load_from(weights)` + pass `img_size`/`n_classes=1` | A & B |
| `train_all.py:393` | same build-branch change | A & B |
| `requirements.txt` | add `ml_collections` (A); timm already present (B) | A (B: none) |
| `checkpoints/R50+ViT-B_16.npz` (new) | download pretrained (A); timm auto-downloads (B) | A only |
| Input normalisation | add ImageNet `Normalize` for TransUNet path (loader is shared — see §5) | A & B |

**Not modified:** `training/`, `utils/report_generator.py`, `utils/benchmark_report.py`, `utils/aggregate_benchmark.py`, datasets split, other models.

---

## 5. Compatibility With Current Pipeline

| Concern | Status | Detail |
|---------|:------:|--------|
| Output shape | ✅ | Official seg head with `n_classes=1` → `(B,1,H,W)` logits. SimpleTrainer does `logits=model(imgs)` then `SimpleLoss=sigmoid+BCE+Dice` — compatible (no sigmoid in head). |
| Loss | ✅ | `SimpleLoss` (Dice+BCEWithLogits) works on 1-channel logits. No change. |
| Image size 384 | ⚠️ | Pretrained ViT is 224 (14×14 grid). Need pos-emb **interpolation** to 384 (24×24). Route A: pass `img_size=384` to `VisionTransformer` (supports interp in `load_from`). Route B: timm `set_input_size`/`dynamic_img_size`. Or simply train at 224 (change loader img_size for this model). |
| **Normalisation** | ❌ | Pretrained backbone expects ImageNet mean/std; loader emits raw [0,1] (`PREPROCESSING_AUDIT.md` P2). Without norm, pretrained weights degrade. **Must add** ImageNet `Normalize` for the TransUNet path. |
| AMP fp16 | ✅ | ViT under autocast fine; token count drops 36,864→576 @384 → far more stable than current. |
| Batch size 2 | ⚠️ | Small for ViT but pretrained tolerates; consider grad-accum (optional). |
| Early stopping / metrics.json | ✅ | SimpleTrainer features apply unchanged. |

---

## 6. Reuse of Existing Subsystems

| Subsystem | Reusable? | Notes |
|-----------|:---------:|-------|
| **Dataset loader** (`SegDataset`) | ✅ with 1 caveat | Returns `(img[3,384,384], mask[1,384,384])` — shape-compatible. Caveat: **add ImageNet normalisation** for this model (pretrained). Otherwise loader unchanged. |
| **Trainer** (`SimpleTrainer`) | ✅ as-is | Expects single `(B,1,H,W)` logit output → official TransUNet(n_classes=1) delivers it. Early-stop, best-ckpt, metrics.json, test eval all work. Only `train.py`/`train_all.py` **build branch** changes (construct + load weights). |
| **Report generator** (`report_generator.py`) | ✅ zero change | Infers model from folder name; `"transunet"`→`"TransUNet"` already mapped. Reads training_log.csv/metrics.json generically. |
| **Benchmark system** (`benchmark_report.py` + `aggregate_benchmark.py`) | ✅ zero change | `"transunet"` already in `MODELS`. Reads training_log.csv / metrics.json generically. Picks up the new runs automatically. |

**Net:** trainer + both report/benchmark systems are **fully reusable**; only the model build sites + (shared) input normalisation need touching.

---

## 7. Effort Estimate

| Task | Route B (recommended) | Route A (faithful) |
|------|----------------------|--------------------|
| Model implementation | ~0.5 day (timm encoder + CUP decoder) | ~1 day (vendor 3 files, wire config) |
| Deps / weights | none (timm installed, auto-download) | +`ml_collections`, +npz download/verify (~0.5 day) |
| Build-site wiring (train.py/train_all.py) | ~1–2 h | ~1–2 h |
| Normalisation fix | ~1 h | ~1 h |
| Smoke test (1 epoch, shape/finite/load check) | ~1 h | ~1 h |
| Full re-train ×3 datasets (40 ep) | compute only (~1–2 GPU-days) | same |
| **Dev total** | **~1 day** | **~2 days** |

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Pretrained pos-emb mismatch at 384 | use interpolation (Route A `img_size=384`; Route B `dynamic_img_size=True`) or train at 224 |
| Missing ImageNet norm silently degrades pretrained features | add `Normalize` for TransUNet path before first run |
| Larger model + batch 2 OOM | gradient accumulation or img 224; ViT-B is ~100M params |
| npz download/version drift (Route A) | pin URL + SHA; prefer Route B to avoid |
| Historical results incomparable | label new runs distinctly; keep old transunet results archived; re-run all baselines under same config for fairness |

---

## 9. Execution Order (when approved)

1. Add model (Route B: timm encoder + CUP decoder) as `models/transunet.py` replacement (or new module).
2. Wire build sites `train.py:259`, `train_all.py:393` (construct, n_classes=1, img_size, pretrained).
3. Add ImageNet normalisation for the TransUNet path.
4. Smoke test: 1 epoch on ISIC 2016 — assert output `(B,1,384,384)`, finite loss, pretrained weights loaded.
5. Full re-train ×3 datasets, 40 ep, patience 10.
6. Confirm report.html + benchmark + aggregate pick up runs (zero change expected).

---

## Evidence References
- `models/transunet.py` — current non-official impl
- `train.py:259`, `train_all.py:393` — `TransUNet()` build sites
- `training/simple_trainer.py:139,182,219` — `logits=model(imgs)` contract (single logit map)
- deps probe: `ml_collections` missing; `timm 1.0.26`, `scipy 1.15.3`, `einops 0.8.2` present
- `TRANSUNET_AUDIT.md` — architecture gap analysis
- `PREPROCESSING_AUDIT.md` P2 — missing ImageNet normalisation
- official: Beckschen/TransUNet `networks/vit_seg_modeling*.py`, `R50+ViT-B_16.npz` (ImageNet-21k)
