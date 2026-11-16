import os
import pandas as pd
import numpy as np
import cv2

# Paths
csv_path = "dataset_raw/ISIC_2019/ISIC_2019_Training_GroundTruth.csv"
out_dir = "dataset_raw/ISIC_2019/masks"
os.makedirs(out_dir, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path)

# ISIC 2019 images are 256x256
HEIGHT, WIDTH = 256, 256

def rle_decode(mask_rle, shape=(HEIGHT, WIDTH)):
    """Decode RLE string into numpy mask."""
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 255
    return img.reshape(shape)

print("Converting CSV masks to PNG...")
for idx, row in df.iterrows():
    image_id = row["image"]  # column name may vary, check your CSV
    mask_rle = row["segmentation"]
    if pd.isna(mask_rle):
        continue
    mask = rle_decode(mask_rle)
    cv2.imwrite(os.path.join(out_dir, image_id + ".png"), mask)

print(f"✅ Done! Masks saved in {out_dir}")
