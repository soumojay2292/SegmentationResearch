import os

def filter_split(dataset="ISIC_2016", split="train"):
    img_dir = f"dataset_split/{dataset}/{split}/images"
    mask_dir = f"dataset_split/{dataset}/{split}/masks"

    valid_images = []
    for fname in os.listdir(img_dir):
        mask_name = fname.replace(".jpg", "_Segmentation.png")
        mask_path = os.path.join(mask_dir, mask_name)
        if os.path.exists(mask_path):
            valid_images.append(fname)

    print(f"✅ {dataset}/{split}: {len(valid_images)} valid samples out of {len(os.listdir(img_dir))}")

if __name__ == "__main__":
    for ds in ["ISIC_2016", "ISIC_2017", "ISIC_2018"]:
        for split in ["train", "val", "test"]:
            filter_split(ds, split)
