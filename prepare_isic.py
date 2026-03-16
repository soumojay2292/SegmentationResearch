import os, shutil, cv2, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--raw_images", required=True, help="Path to raw training images")
parser.add_argument("--raw_masks", required=True, help="Path to raw training masks")
parser.add_argument("--output", required=True, help="Output folder for normalized dataset")
args = parser.parse_args()

IMG_DIR = os.path.join(args.output, "images")
MASK_DIR = os.path.join(args.output, "masks")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

raw_img = args.raw_images
raw_mask = args.raw_masks

print("Copying images...")
for f in os.listdir(raw_img):
    # Skip non-image files
    if f.lower() in ["attribution", "license"] or f.lower().endswith(".csv"):
        continue
    if "superpixels" in f.lower() or "metadata" in f.lower():
        continue
    shutil.copy(os.path.join(raw_img, f), os.path.join(IMG_DIR, f))

print("Copying masks and normalizing to .png...")
for f in os.listdir(raw_mask):
    # Skip non-image files
    if f.lower() in ["attribution", "license"] or f.lower().endswith(".csv"):
        continue
    if "superpixels" in f.lower() or "metadata" in f.lower():
        continue

    # Normalize mask filename by removing "_segmentation" or "_Segmentation"
    base = f.replace("_segmentation", "").replace("_Segmentation", "")
    mask = cv2.imread(os.path.join(raw_mask, f), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"⚠️ Could not read mask {f}")
        continue
    cv2.imwrite(os.path.join(MASK_DIR, os.path.splitext(base)[0] + ".png"), mask)

print(f"✅ Done! Images in {IMG_DIR}, masks in {MASK_DIR} (all .png)")
