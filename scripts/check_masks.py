import os
import random
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

def check_dataset(dataset_path: str, visualize: bool = False, num_samples: int = 5, save_preview: bool = True):
    img_dir = Path(dataset_path) / "images"
    mask_dir = Path(dataset_path) / "masks"

    img_files = sorted([f.stem for f in img_dir.glob("*")])
    mask_files = sorted([f.stem for f in mask_dir.glob("*")])

    missing_masks = [f for f in img_files if f not in mask_files]
    missing_images = [f for f in mask_files if f not in img_files]

    print(f"Dataset: {dataset_path}")
    print(f"  Images: {len(img_files)}")
    print(f"  Masks:  {len(mask_files)}")
    print(f"  Missing masks: {len(missing_masks)}")
    print(f"  Missing images: {len(missing_images)}")
    print("-" * 50)

    if missing_masks:
        print("Missing mask files for:", missing_masks) 
    if missing_images: 
        print("Missing image files for:", missing_images)

    # Visualize and/or save a few random samples
    common = list(set(img_files).intersection(set(mask_files)))
    samples = random.sample(common, min(num_samples, len(common)))

    # Always define out_dir so Pylance knows it's bound
    out_dir: Path = Path(dataset_path) / "preview"
    if save_preview:
        out_dir.mkdir(parents=True, exist_ok=True)


    for s in samples:
        img_path = img_dir / (s + ".jpg")
        if not img_path.exists():
            img_path = img_dir / (s + ".png")
        mask_path = mask_dir / (s + ".png")

        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue

        # Overlay mask in red
        overlay = img.copy()
        overlay[mask > 0] = [255, 0, 0]

        if visualize:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Image")
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.title("Image + Mask Overlay")
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()

        if save_preview:
            cv2.imwrite(str(out_dir / f"{s}_overlay.png"), overlay)


if __name__ == "__main__":
    for year in [
        "dataset_2016", "dataset_2017", "dataset_2018",
        "dataset_split_2016", "dataset_split_2017", "dataset_split_2018"
    ]:
        check_dataset(year, visualize=False, save_preview=False)


