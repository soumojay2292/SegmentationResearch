# Dataset Audit — ISIC 2017 & ISIC 2018 Splits

**Scope:** `dataset_split/ISIC_2017/` and `dataset_split/ISIC_2018/`. CSV vs. disk reconciliation, pairing, duplicates, mask binarity, cross-split leakage.
**Method:** pure inspection via `segmentation` conda env (numpy 2.2.6, PIL 12.2.0). No files modified.

**Headline:** **ISIC_2017 test split has ZERO masks** — all 600 mask cells empty, `test/masks/` dir empty. Test-set evaluation on ISIC 2017 is impossible with current data. This explains why all reports use **val**, not test (see `latest_experiment_report.md` §7). ISIC_2018 has **1 100 orphan disk images** not referenced by any CSV. No cross-split leakage anywhere. All present masks binary.

---

## 1. Image / Mask Counts (train / val / test)

### ISIC 2017
| Split | CSV rows | CSV images | CSV masks (non-empty) | Disk images | Disk masks |
|-------|--------:|-----------:|----------------------:|------------:|-----------:|
| train | 2 000 | 2 000 | 2 000 | 2 000 | 2 000 |
| val | 150 | 150 | 150 | 150 | 150 |
| **test** | 600 | 600 | **0** | 600 | **0** |
| **Total** | 2 750 | 2 750 | 2 150 | 2 750 | 2 150 |

### ISIC 2018
| Split | CSV rows | CSV images | CSV masks (non-empty) | Disk images | Disk masks |
|-------|--------:|-----------:|----------------------:|------------:|-----------:|
| train | 1 825 | 1 825 | 1 825 | **2 585** | 1 825 |
| val | 377 | 377 | 377 | **554** | 377 |
| test | 392 | 392 | 392 | **555** | 392 |
| **Total** | 2 594 | 2 594 | 2 594 | 3 694 | 2 594 |

Bold = mismatch between CSV and disk.

---

## 2. Missing Image–Mask Pairs

| Dataset | Split | Rows with image but **no mask** | Verdict |
|---------|-------|-------------------------------:|---------|
| ISIC 2017 | train | 0 | OK |
| ISIC 2017 | val | 0 | OK |
| ISIC 2017 | **test** | **600** | ❌ **all test rows lack mask** |
| ISIC 2018 | train | 0 | OK |
| ISIC 2018 | val | 0 | OK |
| ISIC 2018 | test | 0 | OK |

- **CSV → disk missing:** none. Every CSV-referenced image and mask exists on disk (where the CSV names one).
- **ISIC 2017 test:** CSV mask column entirely empty (`ISIC_0012086.jpg,` …). `dataset_split/ISIC_2017/test/masks/` is an empty directory. Not a path bug — the ground truth simply is not present.

---

## 3. Duplicate Image IDs

| Dataset | Split | CSV image dupes | CSV mask dupes |
|---------|-------|----------------:|---------------:|
| ISIC 2017 | train/val/test | 0 / 0 / 0 | 0 / 0 / 0 |
| ISIC 2018 | train/val/test | 0 / 0 / 0 | 0 / 0 / 0 |

No duplicate IDs within any split CSV. Clean.

---

## 4. Cross-Split Leakage (data contamination)

Stem-level (`ISIC_XXXXXXX`) intersection between splits.

### By CSV reference
| Dataset | train ∩ val | train ∩ test | val ∩ test |
|---------|-----------:|------------:|-----------:|
| ISIC 2017 | 0 | 0 | 0 |
| ISIC 2018 | 0 | 0 | 0 |

### By disk image folders (ISIC 2018, where extra files exist)
| Pair | Overlap |
|------|--------:|
| train ∩ val | 0 |
| train ∩ test | 0 |
| val ∩ test | 0 |

**No leakage.** Splits are disjoint at the patient-image-ID level, both in CSV and on disk. Train/val/test integrity holds.

---

## 5. Mask Binarity

Full scan — **every** mask on disk read, not sampled.

| Dataset | Split | Masks scanned | PIL mode | Global unique values | Non-binary |
|---------|-------|-------------:|:--------:|:--------------------:|:----------:|
| ISIC 2017 | train | 2 000 | L | {0, 255} | 0 |
| ISIC 2017 | val | 150 | L | {0, 255} | 0 |
| ISIC 2017 | test | 0 | — | — | n/a (no masks) |
| ISIC 2018 | train | 1 825 | L | {0, 255} | 0 |
| ISIC 2018 | val | 377 | L | {0, 255} | 0 |
| ISIC 2018 | test | 392 | L | {0, 255} | 0 |

**All 4 744 present masks are strictly binary** — single-channel 8-bit grayscale (`mode L`), pixel values exclusively {0, 255}. No anti-aliased edges, no gray intermediate values, no RGB masks. Loader's `(mask/255) > 0.5` binarisation is safe and lossless.

---

## 6. CSV → File Reference Integrity

| Dataset | Split | All CSV images exist? | All CSV masks exist? | Orphan disk images (not in CSV) | Orphan disk masks |
|---------|-------|:--------------------:|:--------------------:|--------------------------------:|------------------:|
| ISIC 2017 | train | ✅ | ✅ | 0 | 0 |
| ISIC 2017 | val | ✅ | ✅ | 0 | 0 |
| ISIC 2017 | test | ✅ | n/a (none named) | 0 | 0 |
| ISIC 2018 | train | ✅ | ✅ | **760** | 0 |
| ISIC 2018 | val | ✅ | ✅ | **177** | 0 |
| ISIC 2018 | test | ✅ | ✅ | **163** | 0 |

- **Every file the CSV names exists on disk.** No broken references → no `FileNotFoundError` at load time. Loader (`train_all.py` `SegDataset`) reads only CSV-listed files, so orphans are silently ignored.
- **ISIC 2018 orphan images: 1 100 total** (760 train + 177 val + 163 test) `.jpg` files present on disk but not listed in any CSV. They have **no masks** and are never loaded. Dead weight; potential confusion if someone counts disk files instead of CSV rows.
- Naming conventions differ by dataset:
  - ISIC 2017 mask = `ISIC_XXXX.png` (same stem as image)
  - ISIC 2018 mask = `ISIC_XXXX_Segmentation.png` (suffix)
  - Both handled correctly by the CSV (mask filename stored explicitly).

---

## 7. Findings Summary

| # | Severity | Finding |
|---|:--------:|---------|
| F1 | 🔴 **Critical** | **ISIC 2017 test split has no masks** (600 images, 0 GT). Test evaluation impossible; all 2017 metrics in reports are **val**, not test |
| F2 | 🟡 Medium | **ISIC 2018: 1 100 orphan disk images** with no masks, not in any CSV. Unused, waste ~space, risk of miscounting |
| F3 | 🟢 Info | No cross-split leakage in either dataset (CSV and disk both disjoint) |
| F4 | 🟢 Info | All 4 744 present masks strictly binary {0,255}, mode L |
| F5 | 🟢 Info | No duplicate IDs; no broken CSV references; no missing CSV-named files |
| F6 | 🟢 Info | Mask naming differs (2017 plain stem, 2018 `_Segmentation` suffix), handled by CSV |

---

## 8. Impact on Experiments

- **F1 forces val-only reporting for ISIC 2017.** Any "test" claim on ISIC 2017 is unverifiable. For a publication, either (a) obtain ISIC 2017 official test masks from the ISIC Archive, or (b) carve a held-out test split from the current 2000-image train set and re-train. Reporting on val inflates numbers (val tunes implicitly via best-checkpoint selection).
- **F2 has no correctness impact** — loader ignores orphans — but inflates the apparent dataset size if disk files are counted. Document the true CSV-driven counts (train 1825 / val 377 / test 392 for ISIC 2018) in the paper.
- **F3/F4/F5 are clean bills** — split integrity and mask quality are publication-grade.

---

## 9. Recommendations (no files modified)

1. **ISIC 2017 test masks (P0).** Download official 2017 test ground-truth from ISIC Archive (`ISIC-2017_Test_v2_Part1_GroundTruth`) into `dataset_split/ISIC_2017/test/masks/` and populate the empty mask column in `test.csv`. Without this, no honest ISIC 2017 test number can be reported.
2. **Fallback if masks unobtainable:** re-partition the 2 000-image 2017 train set into train/val/test (e.g., 80/10/10) so a true masked test set exists. Re-run all 2017 experiments.
3. **Quarantine ISIC 2018 orphans (P2).** Either remove the 1 100 unreferenced `.jpg` or add them (with masks, if available) to the CSVs. Currently they only cause counting confusion.
4. **Add a `verify_dataset.py` pre-flight** that fails loudly if any split has 0 masks or CSV→disk mismatch, run before every training job. This would have caught F1 immediately.
5. **Record true counts in the paper** from CSV rows, not disk listings: ISIC 2017 = 2000/150/600(no GT); ISIC 2018 = 1825/377/392.

---

## Evidence References

- `dataset_split/ISIC_2017/{train,val,test}.csv` — test.csv mask column empty (600 rows)
- `dataset_split/ISIC_2017/test/masks/` — empty directory (verified via `os.listdir` → `[]`)
- `dataset_split/ISIC_2018/{train,val,test}.csv` — 1825/377/392 rows
- `dataset_split/ISIC_2018/{train,val,test}/images/` — 2585/554/555 disk files (orphans)
- Full mask scan: 4 744 masks, all mode L, values {0,255}
- Loader behaviour: `train_all.py` `SegDataset.__getitem__` reads only CSV-listed files
- Corroborates val-only metric caveat in `latest_experiment_report.md` §7 and `EXPERIMENTAL_GAP_ANALYSIS.md` §1
