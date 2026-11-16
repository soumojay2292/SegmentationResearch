"""
Prepare ISIC 2017 dataset for segmentation training.

Source layout (dataset_raw/ISIC_2017/):
    ISIC-2017_Training_Data/            ← train images (.jpg)
    ISIC-2017_Training_Part1_GroundTruth/ ← train masks (*_segmentation.png)
    ISIC-2017_Validation_Data/          ← val images (.jpg)
    ISIC-2017_Validation_Part1_GroundTruth/ ← val masks (*_segmentation.png)
    ISIC-2017_Test_v2_Data/             ← test images (.jpg)
    [No Part1 masks for test — ISIC 2017 never released them publicly]

Output layout (dataset_split/ISIC_2017/):
    train/{images,masks}/
    val/{images,masks}/
    test/{images,masks}/   ← masks folder will be empty for test
    train.csv, val.csv, test.csv
"""

import csv
import shutil
from pathlib import Path

RAW_ROOT   = Path("dataset_raw/ISIC_2017")
OUT_ROOT   = Path("dataset_split/ISIC_2017")

SPLITS = {
    "train": {
        "img_dir":  RAW_ROOT / "ISIC-2017_Training_Data",
        "mask_dir": RAW_ROOT / "ISIC-2017_Training_Part1_GroundTruth",
    },
    "val": {
        "img_dir":  RAW_ROOT / "ISIC-2017_Validation_Data",
        "mask_dir": RAW_ROOT / "ISIC-2017_Validation_Part1_GroundTruth",
    },
    "test": {
        "img_dir":  RAW_ROOT / "ISIC-2017_Test_v2_Data",
        "mask_dir": None,  # no binary segmentation masks released
    },
}


def _stem(p: Path) -> str:
    """Return ISIC_XXXXXXX from any ISIC filename."""
    name = p.stem  # e.g. ISIC_0000001_segmentation
    return name.split("_segmentation")[0]


def process_split(split: str, cfg: dict) -> list[dict]:
    img_dir  = cfg["img_dir"]
    mask_dir = cfg["mask_dir"]

    out_img_dir  = OUT_ROOT / split / "images"
    out_mask_dir = OUT_ROOT / split / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    # Collect images (skip superpixel PNGs and metadata CSVs)
    images = sorted(p for p in img_dir.iterdir() if p.suffix == ".jpg")

    # Build mask lookup: stem → Path
    if mask_dir is not None:
        mask_lookup = {
            _stem(p): p
            for p in mask_dir.iterdir()
            if p.suffix == ".png"
        }
    else:
        mask_lookup = {}

    rows      = []
    missing   = []
    copied_i  = 0
    copied_m  = 0

    for img_path in images:
        stem      = img_path.stem          # ISIC_XXXXXXX
        dst_img   = out_img_dir / f"{stem}.jpg"
        dst_mask  = out_mask_dir / f"{stem}.png"

        # Copy image
        shutil.copy2(img_path, dst_img)
        copied_i += 1

        # Copy mask if available
        if stem in mask_lookup:
            shutil.copy2(mask_lookup[stem], dst_mask)
            copied_m += 1
            rows.append({"image": f"{stem}.jpg", "mask": f"{stem}.png"})
        else:
            missing.append(stem)
            if mask_dir is None:
                rows.append({"image": f"{stem}.jpg", "mask": ""})

    if missing and mask_dir is not None:
        print(f"  WARNING: {len(missing)} images have no matching mask in {split}")
        for s in missing[:5]:
            print(f"    {s}")

    return rows, copied_i, copied_m


def write_csv(split: str, rows: list[dict]) -> Path:
    csv_path = OUT_ROOT / f"{split}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "mask"])
        w.writeheader()
        w.writerows(rows)
    return csv_path


def main():
    print(f"Output root: {OUT_ROOT.resolve()}\n")

    for split, cfg in SPLITS.items():
        print(f"Processing '{split}' split …")
        rows, n_img, n_mask = process_split(split, cfg)
        csv_path = write_csv(split, rows)

        if cfg["mask_dir"] is None:
            print(f"  Images copied : {n_img}")
            print(f"  Masks         : 0  (no Part1 ground-truth released for ISIC 2017 test)")
        else:
            print(f"  Images copied : {n_img}")
            print(f"  Masks copied  : {n_mask}")
            assert n_img == n_mask, f"Image/mask count mismatch in {split}!"
        print(f"  CSV written   : {csv_path}\n")

    print("Done.")
    print()
    print("NOTE: The test split has images but NO binary segmentation masks.")
    print("      ISIC 2017 test Part1 (segmentation) ground truth was never publicly")
    print("      released. Use train + val for training/evaluation, or treat test as")
    print("      an unlabelled inference set.")


if __name__ == "__main__":
    main()
