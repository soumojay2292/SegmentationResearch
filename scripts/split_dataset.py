import os, shutil, random
from tqdm import tqdm
import argparse

def split_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    img_dir = os.path.join(input_dir, "images")
    mask_dir = os.path.join(input_dir, "masks")

    files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg",".jpeg",".png"))]
    random.shuffle(files)

    n = len(files)
    train_split = int(train_ratio * n)
    val_split   = int((train_ratio + val_ratio) * n)

    train_files = files[:train_split]
    val_files   = files[train_split:val_split]
    test_files  = files[val_split:]

    def copy_files(file_list, split):
        img_out = os.path.join(output_dir, split, "images")
        mask_out = os.path.join(output_dir, split, "masks")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(mask_out, exist_ok=True)
        for f in tqdm(file_list, desc=f"Copying {split}"):
            shutil.copy(os.path.join(img_dir, f), os.path.join(img_out, f))
            base = os.path.splitext(f)[0]

            # Handle both mask naming conventions
            mask_file_seg = base + "_Segmentation.png"
            mask_path_seg = os.path.join(mask_dir, mask_file_seg)

            mask_file_plain = base + ".png"
            mask_path_plain = os.path.join(mask_dir, mask_file_plain)

            if os.path.exists(mask_path_seg):
                shutil.copy(mask_path_seg, os.path.join(mask_out, mask_file_seg))
            elif os.path.exists(mask_path_plain):
                shutil.copy(mask_path_plain, os.path.join(mask_out, mask_file_plain))
            else:
                print(f"⚠️ No mask found for {f}")


    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to dataset with images/ and masks/")
    parser.add_argument("--output", required=True, help="Output folder for dataset_split_YEAR")
    args = parser.parse_args()
    split_dataset(args.input, args.output)
