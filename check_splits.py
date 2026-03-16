import os

TRAIN_IMG = "dataset_split/train/images"
TRAIN_MASK = "dataset_split/train/masks"
VAL_IMG = "dataset_split/val/images"
VAL_MASK = "dataset_split/val/masks"

def check_split(img_dir, mask_dir, name):
    missing = []
    for f in os.listdir(img_dir):
        base = os.path.splitext(f)[0]
        mask_file = base + ".png"  # enforce .png masks
        if not os.path.exists(os.path.join(mask_dir, mask_file)):
            missing.append(f)
    print(f"{name} set: {len(os.listdir(img_dir))} images, {len(os.listdir(mask_dir))} masks")
    print(f"Missing masks: {len(missing)}")
    if missing:
        print(missing[:10])  # show first 10 missing

check_split(TRAIN_IMG, TRAIN_MASK, "Train")
check_split(VAL_IMG, VAL_MASK, "Val")
