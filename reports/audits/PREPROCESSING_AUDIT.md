# Preprocessing Pipeline Audit

**Scope:** preprocessing used by all baseline experiments.
**Active loader:** inline `SegDataset` in `train_all.py:97-160` (identical copy in `train.py:46-111`). Used by **all** models (U-Net, Attention U-Net, UNet++, TransUNet, MAFFNet).
**Inactive:** `datasets.py` `ISICDataset` — legacy, cv2-based, hardcoded `_Segmentation.png` naming, **not imported** by any training entry point. Ignored here.
**Method:** code read only. No code modified.

**Headline:** geometric augmentation is correct (image+mask kept in sync, right interpolation). But three decisions risk hurting segmentation: **(1) anisotropic 384×384 resize distorts lesion shape**, **(2) rotation applied to every sample with black-corner fill**, **(3) no ImageNet normalisation — penalises the SAM2-pretrained MAFFNet**.

---

## 1. Image Normalisation

| Aspect | Finding | Source |
|--------|---------|--------|
| Tensor scaling | `T.ToTensor()` → float [0, 1] | `train_all.py:154` |
| Mean/std normalisation | **NONE** — no `T.Normalize`, no ImageNet stats | grep: no `Normalize` in loader |
| Per-channel standardisation | none | — |

**Assessment:**
- Baselines (U-Net, Attention U-Net, UNet++, TransUNet) train from scratch with BatchNorm → [0,1] input is acceptable; BN absorbs scale.
- **MAFFNet uses a frozen SAM2 Hiera-Large backbone pretrained with ImageNet-style normalisation.** Feeding raw [0,1] (no mean-subtraction, no /std) means the backbone sees an input distribution it was **not** trained on → degraded features. This is a real, silent performance leak for the headline model. (Cross-ref `REPRODUCIBILITY_CHECKLIST.md` F3, `ARCHITECTURE_AUDIT.md`.)

---

## 2. Resize Dimensions & Interpolation

| Aspect | Finding | Source |
|--------|---------|--------|
| Target size | **384 × 384** (square) | `train_all.py:148-149` |
| Image interpolation | `Image.BILINEAR` | `:148` |
| Mask interpolation | `Image.NEAREST` | `:149` |
| Aspect-ratio handling | **NONE — forced square resize** | `:148-149` |

**Assessment:**
- Interpolation choice correct: bilinear for image, nearest for mask (avoids fractional mask values). ✅
- **Anisotropic distortion (HIGH concern):** ISIC dermoscopy images are non-square and vary widely (e.g. 1022×767, 4288×2848, 6748×4499). `resize((384,384))` ignores aspect ratio → horizontal/vertical squash. Lesion shape (a clinically meaningful feature — asymmetry, border) is geometrically distorted, and the distortion factor differs per image. No letterbox-pad, no aspect-preserving resize, no center-crop.

---

## 3. Mask Preprocessing & Binary Enforcement

| Step | Finding | Source |
|------|---------|--------|
| Load mode | `convert("L")` (8-bit grayscale) | `train_all.py:146` |
| Resize | NEAREST → no new gray levels | `:149` |
| Scale | `np.array(mask, float32) / 255.0` → [0,1] | `:155-157` |
| Binary enforce | `(mask_t > 0.5).float()` → {0.0, 1.0} | `:159` |
| Channel | `.unsqueeze(0)` → (1, H, W) | `:157` |

**Assessment:** correct and robust. Source masks are already strictly binary {0,255} (verified in `DATASET_AUDIT.md` §5); NEAREST resize preserves that; `>0.5` threshold guarantees binary even if any interpolation crept in. ✅ No issue.

---

## 4. Data Augmentation — Techniques & Consistency

`_augment` (`train_all.py:114-134`), train split only.

| Technique | Probability | Image interp | Mask interp | Applied to mask? |
|-----------|:-----------:|:------------:|:-----------:|:----------------:|
| Horizontal flip | 0.5 | — | — | ✅ yes |
| Vertical flip | 0.5 | — | — | ✅ yes |
| Rotation [−30°,+30°] | **1.0 (always)** | BILINEAR | NEAREST | ✅ yes |
| Random affine (translate 0.1, scale 0.9–1.1, shear ±5, deg ±15) | 0.5 | BILINEAR | NEAREST | ✅ yes |
| ColorJitter (b0.2,c0.2,s0.2,h0.1) | **1.0 (always)** | — | — | image only (correct) |

**Consistency checks:**
- Geometric transforms applied to **both** image and mask with matching parameters and correct per-modality interpolation. ✅
- Photometric (ColorJitter) applied to image only. ✅
- Same `_augment` for all models and all 3 datasets → **consistent across experiments**. ✅
- Val/test: **no augmentation** (`if self.split == "train"`, `:151`). ✅

**Concerns:**
- **Rotation always-on (MEDIUM-HIGH):** every training image rotated by a random angle each epoch. `TF.rotate` default `fill=0` → **black triangular corners** introduced in the image (and 0=background in the mask). Image black corners are out-of-distribution artifacts the network learns to ignore, wasting capacity; corner regions also dilute the lesion/background statistics. Most pipelines gate rotation behind a probability and/or use reflection fill.
- **ColorJitter always-on (MEDIUM):** strong photometric shift applied to 100% of training samples (no `p<1`). Aggressive but mask-safe.
- **Order:** resize → augment. Rotation/affine run on the already-squashed 384² image, compounding distortion (§2) with rotation fill.
- **Unseeded RNG:** uses Python `random.*` directly (`:115,117,120,124`), no `torch.Generator`, no seed → augmentation is non-deterministic run-to-run (repro impact, cross-ref `REPRODUCIBILITY_CHECKLIST.md` §3).

---

## 5. Cross-Dataset Compatibility (ISIC 2016 / 2017 / 2018)

| Aspect | 2016 | 2017 | 2018 | Consistent? |
|--------|:----:|:----:|:----:|:-----------:|
| Loader | inline SegDataset | same | same | ✅ |
| Resize / interp | 384², bilin/nearest | same | same | ✅ |
| Normalisation | [0,1] none | same | same | ✅ |
| Augmentation | full `_augment` | same | same | ✅ |
| Mask filename in CSV | plain `.png` | plain `.png` | `_Segmentation.png` | ✅ handled — CSV stores explicit mask name; loader reads `row["mask"]` (`:112,143`) |

**Assessment:** the active loader is dataset-agnostic — it reads image/mask filenames from each split's CSV and applies one identical pipeline, so all three datasets are processed compatibly. The differing 2018 mask suffix is absorbed because the CSV carries the literal mask filename. ✅

**Caveat:** the *legacy* `datasets.py` `ISICDataset` hardcodes `_Segmentation.png` (`datasets.py:16,27`) → would silently drop **all** ISIC 2016/2017 masks (which use plain `.png`). It is unused, but is a latent landmine if anyone wires it in. Recommend deleting or fixing.

---

## 6. Preprocessing Decisions That Could Hurt Segmentation

Ranked by expected impact.

| # | Severity | Decision | Why it hurts | Fix direction (not applied) |
|---|:--------:|----------|--------------|------------------------------|
| P1 | 🔴 High | Anisotropic `resize((384,384))` ignoring aspect ratio | Distorts lesion geometry (asymmetry/border are diagnostic); distortion varies per image → inconsistent target shapes | Aspect-preserving resize + pad to square (letterbox), or resize-shorter-side + center/random crop |
| P2 | 🔴 High (MAFFNet) | No ImageNet normalisation before SAM2 backbone | Pretrained backbone sees wrong input distribution → weaker features for the headline model | Apply `T.Normalize(ImageNet mean/std)` for MAFFNet (and document for baselines) |
| P3 | 🟠 Med-High | Rotation applied to 100% of samples, `fill=0` | Black-corner artifacts (OOD) + over-strong geometric aug every epoch | Gate behind `p≈0.5`; use reflection/`fill` matching border; or smaller angle range |
| P4 | 🟠 Med | ColorJitter applied to 100% of samples | Very strong photometric shift; may slow convergence / hurt small datasets (ISIC 2016) | Gate behind `p<1` (e.g. 0.5) |
| P5 | 🟡 Low-Med | Resize-then-rotate ordering | Compounds squash distortion with rotation fill | Augment on native image, resize last; or pad-before-rotate |
| P6 | 🟡 Low | Unseeded augmentation RNG | Non-reproducible epochs; can't bit-match runs | Seed `random`/use `torch.Generator` (see `REPRODUCIBILITY_CHECKLIST.md`) |
| P7 | 🟡 Low | Legacy `datasets.py` 2018-only mask naming | Latent bug if reused — drops 2016/2017 masks | Delete or fix suffix logic |

**Net:** the pipeline is *functionally correct* (no mask corruption, augments synced), but two high-impact choices — aspect-distorting resize and missing normalisation for the pretrained backbone — likely cap achievable Dice, and always-on rotation/ColorJitter add avoidable noise. None affect the *validity* of the cross-model comparison (all models share the pipeline), but they cap the absolute ceiling — and P2 specifically biases against MAFFNet, the model the work is trying to promote.

---

## Evidence References

- `train_all.py:97-160` — active `SegDataset` (load, resize, augment, tensorise, binarise)
- `train_all.py:114-134` — `_augment` (flip/rotate/affine/colorjitter)
- `train_all.py:148-149` — resize dims + interpolation
- `train_all.py:154-159` — ToTensor (no normalize) + mask binarisation
- `train.py:46-111` — identical loader copy
- `datasets.py:5-44` — legacy unused `ISICDataset` (2018-only naming)
- `DATASET_AUDIT.md` §5 — source masks strictly binary {0,255}
- `REPRODUCIBILITY_CHECKLIST.md` §3, F3 — seeds + ImageNet-norm corroboration
- `ARCHITECTURE_AUDIT.md` — SAM2 backbone expectations
