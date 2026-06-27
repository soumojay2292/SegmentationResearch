"""
Reproducible re-split of ISIC 2017.

Reason: original ISIC 2017 test split has 600 images but ZERO masks
(see DATASET_AUDIT.md F1) — test evaluation impossible. This script carves a
proper masked train/val/test split (80/10/10) from the 2000-image masked
training pool, seed=42, preserving image-mask pairing.

Behaviour (non-destructive / additive):
  1. Archive original train/val/test CSVs to original_csv_backup/.
  2. Shuffle the 2000 train rows with random.Random(42).
  3. Slice 1600 / 200 / 200 -> train/val/test.
  4. Write new train.csv, val.csv, test.csv.
  5. COPY (never move) carved val/test image+mask files from train/ into the
     val/ and test/ split folders so the loader path convention works.
     No file is deleted. Old val/test folder contents remain as harmless
     orphans (loader reads only CSV-listed files).

ISIC 2016 and ISIC 2018 are never touched.
"""
import csv
import random
import shutil
from datetime import datetime
from pathlib import Path

SEED = 42
RATIOS = (0.80, 0.10, 0.10)
ROOT = Path("dataset_split/ISIC_2017")
SRC_SPLIT = "train"  # masked pool lives here


def read_rows(csv_path):
    with open(csv_path, newline="") as f:
        return [(r["image"].strip(), r["mask"].strip())
                for r in csv.DictReader(f)]


def write_rows(csv_path, rows):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "mask"])
        w.writerows(rows)


def main():
    src_csv = ROOT / "train.csv"
    rows = read_rows(src_csv)
    assert len(rows) == 2000, f"expected 2000 rows, got {len(rows)}"

    # 1. archive originals
    backup = ROOT / "original_csv_backup"
    backup.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for name in ("train.csv", "val.csv", "test.csv"):
        p = ROOT / name
        if p.exists():
            shutil.copy2(p, backup / f"{name}.{stamp}.bak")

    # 2. deterministic shuffle
    rng = random.Random(SEED)
    rows_shuffled = rows[:]            # copy
    rng.shuffle(rows_shuffled)         # Mersenne-Twister, version-stable

    # 3. slice
    n = len(rows_shuffled)
    n_train = int(n * RATIOS[0])       # 1600
    n_val = int(n * RATIOS[1])         # 200
    train_rows = rows_shuffled[:n_train]
    val_rows = rows_shuffled[n_train:n_train + n_val]
    test_rows = rows_shuffled[n_train + n_val:]
    assert len(train_rows) == 1600
    assert len(val_rows) == 200
    assert len(test_rows) == 200

    # 4. write new CSVs
    write_rows(ROOT / "train.csv", train_rows)
    write_rows(ROOT / "val.csv", val_rows)
    write_rows(ROOT / "test.csv", test_rows)

    # 5. copy carved val/test files into their split folders
    src_img = ROOT / SRC_SPLIT / "images"
    src_msk = ROOT / SRC_SPLIT / "masks"
    copied = {"val": 0, "test": 0}
    for split, srows in (("val", val_rows), ("test", test_rows)):
        dst_img = ROOT / split / "images"
        dst_msk = ROOT / split / "masks"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_msk.mkdir(parents=True, exist_ok=True)
        for img, msk in srows:
            shutil.copy2(src_img / img, dst_img / img)
            shutil.copy2(src_msk / msk, dst_msk / msk)
            copied[split] += 1

    print(f"seed={SEED} ratios={RATIOS}")
    print(f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    print(f"copied val files={copied['val']} test files={copied['test']}")
    print(f"backup -> {backup}")


if __name__ == "__main__":
    main()
