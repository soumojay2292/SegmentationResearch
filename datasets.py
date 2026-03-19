import os
import cv2
from torch.utils.data import Dataset

class ISICDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Only keep files that have both image and mask
        self.files = []
        for f in os.listdir(img_dir):
            if not f.endswith(".jpg"):
                continue
            mask_name = f.replace(".jpg", "_Segmentation.png")
            mask_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path):
                self.files.append(f)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_name = img_name.replace(".jpg", "_Segmentation.png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        # ✅ Check if mask exists
        if not os.path.exists(mask_path):
            print(f"⚠️ Skipping missing mask: {mask_path}")
            # fallback: move to next sample safely
            return self.__getitem__((idx + 1) % len(self.files))

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        return img, mask

