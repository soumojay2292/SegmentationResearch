import os, cv2, torch
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A

class SegDataset(Dataset):
    def __init__(self, csv_file, img_dir, mask_dir, image_size=256):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = os.listdir(img_dir)
        self.transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(),
        ])

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["image"]
        mask_name = self.df.iloc[idx]["mask"]

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Load with OpenCV (NumPy arrays)
        img = cv2.imread(img_path)              # shape (H, W, 3)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # shape (H, W)

        # Albumentations expects NumPy arrays
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        # ✅ Convert to PyTorch tensors AFTER augmentation
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        return img, mask

    def __len__(self):
        return len(self.df)



