# ISIC 2018 Orphan-Image Analysis

**Scope:** the 1 100 disk images in `dataset_split/ISIC_2018/{train,val,test}/images/` that no CSV references (flagged in `DATASET_AUDIT.md` F2).
**Method:** filesystem + CSV reconciliation, repo-wide mask search, split-script source review. No files modified.

**Verdict:** orphans are **valid official ISIC images with no segmentation masks** (none exist anywhere in the repo). They were copied to disk by `split_dataset.py` but excluded from CSVs because the CSV builder pairs image↔mask and these have no mask. **Recommendation: archive to a labeled holdout folder (or ignore in place). Do NOT incorporate into supervised train/eval — unusable without masks.**

---

## 1. Orphan Characterisation

| Metric | Value |
|--------|------:|
| Total orphan files | **1 100** (unique) |
| In train/ | 760 |
| In val/ | 177 |
| In test/ | 163 |
| Cross-folder duplication | 0 (no orphan in >1 split folder) |
| Filename format | `ISIC_XXXXXXX.jpg` (official ISIC naming) |
| Numeric ID range | **12 169 … 36 347** |
| Numeric overlap with the 2 594 masked images | **0** |

Distribution: orphan IDs are **all ≥ 12 169**, extending to 36 347 (258 in the 36 000 bucket). The masked CSV set spans IDs 0 … 16 072. The two sets are numerically disjoint.

---

## 2. Task 1 — Do They Belong to the Official ISIC 2018 Release?

**Almost certainly yes.**

Evidence:
- All 1 100 use the canonical `ISIC_XXXXXXX` archive naming.
- IDs fall in valid ISIC archive ranges (12 169–36 347).
- Zero ID collision with the masked Task 1 segmentation training set.

**High-confidence hypothesis:** these are the official **ISIC 2018 Task 1 validation (100) + test (1000) = 1 100** images, which the challenge released **without public ground-truth masks** (masks withheld for leaderboard scoring). The count `1 100 = 100 + 1 000` is an exact match.

**Alternative (cannot be excluded locally):** some or all may be **ISIC 2018 Task 3 (classification)** images, which carry no segmentation masks and occupy higher ID ranges. The wide spread up to 36 347 is consistent with classification-set IDs.

**Cannot fully confirm provenance from the repo** — `dataset_raw/` contains **only ISIC 2017** (no ISIC 2018 raw source present), so no local cross-check against an official manifest is possible.

---

## 3. Task 2 — Do Corresponding Masks Exist Elsewhere?

**No.** Repo-wide search (`find . -iname "*<id>*"`) for sample orphan IDs (`ISIC_0012236`, `ISIC_0036347`) returns **only the `.jpg`** in `train/images/` — no `.png`, no `_Segmentation.png`, nowhere in the tree.

| Location checked | Mask for orphans? |
|------------------|:-----------------:|
| `ISIC_2018/*/masks/` | ❌ none |
| `dataset_raw/` | ❌ (no 2018 raw at all) |
| `ISIC_2018/train/previews/` | ❌ (20 RGB preview JPGs, not masks) |
| anywhere else in repo | ❌ |

Total ISIC 2018 masks on disk = 2 594, matching exactly the 2 594 CSV-referenced images. Orphans have zero mask coverage.

---

## 4. Task 3 — Why Are They Excluded from the CSV Splits?

Root cause = behaviour of `split_dataset.py`.

```python
# split_dataset.py:25-41 (copy_files)
for f in tqdm(file_list, desc=f"Copying {split}"):
    shutil.copy(os.path.join(img_dir, f), os.path.join(img_out, f))   # ← image copied UNCONDITIONALLY
    base = os.path.splitext(f)[0]
    ...
    if os.path.exists(mask_path_seg):
        shutil.copy(mask_path_seg, ...)
    elif os.path.exists(mask_path_plain):
        shutil.copy(mask_path_plain, ...)
    else:
        print(f"⚠️ No mask found for {f}")          # ← image already on disk, but no mask
```

Mechanism:
1. A flat image pool (Task 1 train **+** the no-mask val/test/classification images) was shuffled and split 70/15/15.
2. `split_dataset.py` **copies every image to disk** (line 26), regardless of mask existence.
3. For mask-less images it prints `⚠️ No mask found` and copies no mask.
4. A **separate CSV builder** (not in `split_dataset.py` — that script writes no CSV) then constructed `{train,val,test}.csv` by enumerating only **image↔mask pairs**.
5. Result: the 1 100 mask-less images sit on disk but never enter a CSV → orphans.

Confirmation by the numbers: CSV-referenced counts 1825/377/392 = 70.4 / 14.5 / 15.1 % of the **2 594 masked** images — exactly the 70/15/15 ratio applied to the masked subset. The orphans are the residue copied alongside.

**No correctness impact on training/eval:** the loader (`train_all.py` `SegDataset`) reads only CSV-listed files, so orphans are silently ignored. They are wasted disk, and a counting hazard if disk listings are mistaken for dataset size.

---

## 5. Task 4 — Recommendation: Archive, Incorporate, or Ignore?

| Option | Feasible? | Verdict |
|--------|:---------:|---------|
| **Incorporate into supervised train/val/test** | ❌ | **No.** Zero masks → no supervision signal. Adding mask-less images to a segmentation split is impossible (loader needs pairs) and would corrupt counts. |
| **Ignore in place** | ✅ | Acceptable short-term. Loader already ignores them; no bug. But ~1 100 files inflate disk and confuse anyone counting files. |
| **Archive to labeled holdout** | ✅ | **Recommended.** Move (or copy) the 1 100 to `dataset_split/ISIC_2018/_unlabeled_holdout/` with a README noting provenance uncertainty. Cleans the split folders, preserves the data, removes the counting hazard. |
| **Use for official leaderboard inference only** | ✅ (conditional) | If confirmed to be the official Task 1 val/test, run inference-only and submit predictions to the ISIC leaderboard. No local metrics possible (no GT). |

**Primary recommendation: ARCHIVE.**
- Relocate the 1 100 orphans into `dataset_split/ISIC_2018/_unlabeled_holdout/{train,val,test}/` (preserving which folder they came from) or a single flat `_unlabeled/`.
- Add a `PROVENANCE.md` stating: official ISIC source suspected (Task 1 val+test or Task 3), masks unavailable, excluded from supervised use.
- This is a **move**, i.e. destructive to current layout → not performed here (instruction: do not modify files). Defer to an explicit follow-up.

**Do NOT delete** until provenance is confirmed — if they are the official val/test set, they have leaderboard value.

---

## 6. Action Items (none executed — analysis only)

| # | Action | Priority |
|---|--------|:--------:|
| 1 | Confirm provenance: diff the 1 100 IDs against the official ISIC 2018 Task 1 val/test manifest (and Task 3 list) from the ISIC Archive | P1 |
| 2 | Archive orphans to `_unlabeled_holdout/` + `PROVENANCE.md` | P2 |
| 3 | If official Task 1 val/test: run inference-only, submit to leaderboard | P3 (optional) |
| 4 | Fix `split_dataset.py`: skip copying images that have no mask (add a `continue` in the `else` branch) so future runs don't strand orphans | P2 |
| 5 | Document true ISIC 2018 size in the paper as **2 594 masked** (1825/377/392), never the 3 694 disk count | P1 |

---

## Evidence References

- `dataset_split/ISIC_2018/{train,val,test}/images/` — 2585/554/555 disk images
- `dataset_split/ISIC_2018/{train,val,test}.csv` — 1825/377/392 masked rows
- `split_dataset.py:25-41` — unconditional image copy + no-mask warning (orphan mechanism)
- `prepare_isic.py` — raw→normalized stage (no CSV, no filtering of mask-less images)
- Repo-wide `find` for orphan IDs → only the `.jpg`, no masks
- `dataset_raw/` — contains ISIC 2017 only (no 2018 source to cross-check)
- `DATASET_AUDIT.md` F2 — original orphan-count finding
