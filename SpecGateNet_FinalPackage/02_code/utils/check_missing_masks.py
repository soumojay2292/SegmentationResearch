import os

def check_missing_masks(dataset="ISIC_2016", split="train"):
    img_dir = f"dataset_split/{dataset}/{split}/images"
    mask_dir = f"dataset_split/{dataset}/{split}/masks"

    missing = []
    for fname in os.listdir(img_dir):
        mask_name = fname.replace(".jpg", "_Segmentation.png")
        mask_path = os.path.join(mask_dir, mask_name)
        if not os.path.exists(mask_path):
            missing.append(mask_name)

    print(f"Dataset: {dataset} | Split: {split}")
    print(f"Total images: {len(os.listdir(img_dir))}")
    print(f"Missing masks: {len(missing)}")
    if missing:
        print("Examples:", missing[:10])

if __name__ == "__main__":
    for ds in ["ISIC_2016", "ISIC_2017", "ISIC_2018"]:
        for split in ["train", "val", "test"]:
            check_missing_masks(ds, split)
