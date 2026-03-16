import os, cv2, torch
from torch.utils.data import Dataset
import albumentations as A

class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, image_size=256):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = os.listdir(img_dir)
        self.transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Always load mask as .png
        mask_path = os.path.join(self.mask_dir, os.path.splitext(self.files[idx])[0] + ".png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        augmented = self.transform(image=image, mask=mask)
        image, mask = augmented["image"], augmented["mask"]

        image = torch.tensor(image, dtype=torch.float32).permute(2,0,1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0

        return image, mask


