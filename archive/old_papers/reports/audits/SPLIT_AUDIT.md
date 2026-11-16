# Split Audit — ISIC 2017 Re-Split (80/10/10, seed 42)

**Date:** 2026-06-11
**Reason:** original ISIC 2017 test split had 600 images and **zero masks** (see `DATASET_AUDIT.md` F1) — test evaluation impossible. Carved a proper masked train/val/test split from the 2 000-image masked training pool.
**Script:** `scripts/resplit_isic2017.py` (re-runnable, deterministic).
**Scope:** ISIC 2017 only. **ISIC 2016 and ISIC 2018 untouched.**
**Mode:** non-destructive / additive — no file deleted; old CSVs archived; carved files copied (not moved).

**Result: all 7 integrity checks pass.** ✅

---

## 1. Method

| Parameter | Value |
|-----------|-------|
| Source pool | `dataset_split/ISIC_2017/train.csv` — 2 000 masked rows |
| Seed | 42 |
| Shuffler | `random.Random(42).shuffle` (Mersenne-Twister, version-stable) |
| Ratios | 80 / 10 / 10 |
| Slice | rows[0:1600] → train, [1600:1800] → val, [1800:2000] → test |
| Pairing | preserved by splitting whole `(image, mask)` rows, never columns |

---

## 2. Counts

| Split | Rows (CSV) | Disk images | Disk masks | % of 2000 |
|-------|----------:|------------:|-----------:|----------:|
| train | 1 600 | 1 600 referenced (2 000 present¹) | 1 600 referenced | 80.0 % |
| val | 200 | 200 | 200 | 10.0 % |
| test | 200 | 200 | 200 | 10.0 % |
| **Total** | **2 000** | — | — | 100 % |

¹ `train/images` and `train/masks` physically retain all 2 000 source files; only 1 600 are CSV-referenced. The other 400 (now in val+test) remain as harmless orphans in `train/` (loader reads only CSV-listed files). No duplication of *content* across splits — see §4.

---

## 3. Pairing Integrity (image stem ↔ mask stem)

| Split | Mismatched pairs |
|-------|-----------------:|
| train | 0 |
| val | 0 |
| test | 0 |

Every row's image stem (`ISIC_XXXXXXX`) equals its mask stem. Pairing preserved.

---

## 4. Cross-Split Leakage

| Pair | Overlap (stem) |
|------|---------------:|
| train ∩ val | 0 |
| train ∩ test | 0 |
| val ∩ test | 0 |
| **union of all 3** | **2 000** (= full pool, no loss) |

No image appears in more than one split. Disjoint and exhaustive.

---

## 5. File Existence (CSV → disk in own split folder)

Loader convention: `dataset_split/ISIC_2017/<split>/{images,masks}/<file>`.

| Split | Missing images | Missing masks |
|-------|---------------:|--------------:|
| train | 0 | 0 |
| val | 0 | 0 |
| test | 0 | 0 |

Carved val (200) and test (200) image+mask files were copied from `train/` into `val/` and `test/` folders. Every CSV-referenced file resolves on disk. **Loadable.**

---

## 6. Duplicates

| Split | Duplicate image IDs |
|-------|--------------------:|
| train | 0 |
| val | 0 |
| test | 0 |

---

## 7. Mask Binarity (carved val + test, full scan)

| Split | PIL mode | Unique pixel values | Non-binary masks |
|-------|:--------:|:-------------------:|:----------------:|
| val | L | {0, 255} | 0 |
| test | L | {0, 255} | 0 |

All carved masks strictly binary, single-channel 8-bit. Loader `(mask/255) > 0.5` safe.

---

## 8. Archive of Originals

Backup dir: `dataset_split/ISIC_2017/original_csv_backup/`

| File | Note |
|------|------|
| `train.csv.2026-06-11_12-11-22.bak` | 2 000 rows preserved (verified) |
| `val.csv.2026-06-11_12-11-22.bak` | original 150-row val |
| `test.csv.2026-06-11_12-11-22.bak` | original 600-row (mask-less) test |

Originals recoverable. Re-running the script archives again with a fresh timestamp (non-overwriting).

---

## 9. Reproducibility Fingerprint

First 3 rows of each new CSV (seed-42 deterministic; re-run must match exactly):

```
train.csv : ISIC_0010264, ISIC_0013370, ISIC_0002246
val.csv   : ISIC_0011366, ISIC_0000129, ISIC_0000384
test.csv  : ISIC_0010584, ISIC_0009882, ISIC_0015284
```

To reproduce: `python scripts/resplit_isic2017.py` from repo root with the original 2 000-row `train.csv` restored from backup.

---

## 10. Notes & Caveats

- **Old orphan files remain.** `val/images` still holds the 150 original-val images, `test/images` still holds the 600 original mask-less test images. These are **not** referenced by the new CSVs → ignored by the loader. They occupy disk but cause no correctness issue. Optional cleanup (deletion) deferred — not performed to keep this operation reversible.
- **Now usable for test reporting.** The new test split (200 masked images) enables genuine held-out test metrics on ISIC 2017 — closes the gap flagged in `EXPERIMENTAL_GAP_ANALYSIS.md` §1 and `latest_experiment_report.md` §7.
- **Re-train required.** Any prior ISIC 2017 result used the old 2000-train / 150-val split. To use this split, models must be re-trained on the new 1600-train and evaluated on the new 200-test.
- **No stratification.** Split is random, not stratified by lesion class (melanoma / seborrheic keratosis / nevus). If class balance matters, add stratified sampling in a future revision — current ISIC 2017 split CSVs carry no class label to stratify on.

---

## Evidence References

- `scripts/resplit_isic2017.py` — the split implementation
- `dataset_split/ISIC_2017/{train,val,test}.csv` — new splits (1600/200/200)
- `dataset_split/ISIC_2017/original_csv_backup/*.bak` — archived originals
- `dataset_split/ISIC_2017/{val,test}/{images,masks}/` — carved files
- `DATASET_AUDIT.md` F1 — original-test-no-masks finding that motivated this
