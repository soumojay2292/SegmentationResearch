import os, sys, cv2, numpy as np

year = sys.argv[1]  # e.g. "2016", "2017", "2018"
img_dir = f"dataset_split/ISIC_{year}/train/images"
mask_dir = f"dataset_split/ISIC_{year}/train/masks"
out_dir = f"dataset_split/ISIC_{year}/train/previews"
os.makedirs(out_dir, exist_ok=True)

# Check pairing
img_files = set([f.replace(".jpg", "") for f in os.listdir(img_dir) if f.endswith(".jpg")])
mask_files = set([f.replace("_Segmentation.png", "") for f in os.listdir(mask_dir) if f.endswith(".png")])

missing_masks = img_files - mask_files
missing_images = mask_files - img_files

print("Images without masks:", len(missing_masks))
print("Masks without images:", len(missing_images))

# Generate previews
for f in list(os.listdir(img_dir))[:20]:  # first 20 samples
    img_path = os.path.join(img_dir, f)
    mask_path = os.path.join(mask_dir, f.replace(".jpg", "_Segmentation.png"))

    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Could not read image: {img_path}")
        continue

    if not os.path.exists(mask_path):
        print(f"⚠️ No mask found for: {f}")
        continue

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"⚠️ Could not read mask: {mask_path}")
        continue

    # ✅ Resize mask to match image dimensions
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # ✅ Normalize mask values to binary (0 or 255)
    mask = (mask > 0).astype(np.uint8) * 255

    overlay = img.copy()
    overlay[mask > 0] = (0, 255, 0)  # green overlay
    cv2.imwrite(os.path.join(out_dir, f), overlay)

print(f"✅ Previews saved in {out_dir}")
