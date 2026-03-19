import os
import pandas as pd

def generate_csv(dataset="ISIC_2016", split="train"):
    img_dir = f"dataset_split/{dataset}/{split}/images"
    mask_dir = f"dataset_split/{dataset}/{split}/masks"

    rows = []
    for fname in os.listdir(img_dir):
        mask_name = fname.replace(".jpg", "_Segmentation.png")
        mask_path = os.path.join(mask_dir, mask_name)
        if os.path.exists(mask_path):
            rows.append({"image": fname, "mask": mask_name})

    df = pd.DataFrame(rows)
    csv_path = f"dataset_split/{dataset}/{split}.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Generated {csv_path} with {len(rows)} valid samples")

if __name__ == "__main__":
    for ds in ["ISIC_2016", "ISIC_2017", "ISIC_2018"]:
        for split in ["train", "val", "test"]:
            generate_csv(ds, split)
