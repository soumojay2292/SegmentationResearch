"""
train.py – MAFFNet training entry point.

Usage:
    python train.py --dataset ISIC_2016 --epochs 100 --batch_size 8
    python train.py --dataset ISIC_2018 --checkpoint path/to/sam2.pth
"""
import sys
import os

sys.path.insert(0, os.path.abspath("."))         # project first ✅
sys.path.insert(1, os.path.abspath("src/sam2"))  # sam2 second
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import random
from PIL import Image
import numpy as np

from models.maffnet import MAFFNet
from training.trainer import MAFFNetTrainer  # keep this


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SegDataset(torch.utils.data.Dataset):
    """
    Reads image/mask pairs from dataset_split/{dataset}/{split}/
    CSV columns: image_path, mask_path  (absolute or relative to project root)
    """
    def __init__(self, csv_path: str, split: str = "train", img_size: int = 384):
        import csv as _csv
        self.split    = split
        self.img_size = img_size
        self.samples  = []

        with open(csv_path) as f:
            reader = _csv.DictReader(f)
            print(reader.fieldnames)
            for row in reader:
                self.samples.append((row["image"], row["mask"]))

    # --- Augmentation ---
    def _augment(self, img: Image.Image, mask: Image.Image):
        # Horizontal flip
        if random.random() > 0.5:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)
        # Vertical flip
        if random.random() > 0.5:
            img  = TF.vflip(img)
            mask = TF.vflip(mask)
        # Random rotation ±30°
        angle = random.uniform(-30, 30)
        img   = TF.rotate(img,  angle, interpolation=InterpolationMode.BILINEAR)
        mask  = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
        # Random affine
        if random.random() > 0.5:
            params = T.RandomAffine.get_params(
                degrees=(-15, 15), translate=(0.1, 0.1),
                scale_ranges=(0.9, 1.1), shears=(-5, 5),
                img_size=[self.img_size, self.img_size]
            )
            img  = TF.affine(img,  *params, interpolation=InterpolationMode.BILINEAR)
            mask = TF.affine(mask, *params, interpolation=InterpolationMode.NEAREST)
        # Color jitter (image only)
        img = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)(img)
        return img, mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img  = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Resize
        img  = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # Augment train only
        if self.split == "train":
            img, mask = self._augment(img, mask)

        # To tensor
        img_t  = T.ToTensor()(img)                # [0,1], (3,H,W)
        mask_t = torch.from_numpy(
            np.array(mask, dtype=np.float32) / 255.0
        ).unsqueeze(0)                             # (1,H,W)
        mask_t = (mask_t > 0.5).float()

        return img_t, mask_t


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train MAFFNet")
    p.add_argument("--dataset",    default="ISIC_2018",
                   choices=["ISIC_2016", "ISIC_2018"])
    p.add_argument("--data_root",  default="dataset_split")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--rec_lambda", type=float, default=0.1)
    p.add_argument("--img_size",   type=int,   default=384)
    p.add_argument("--workers",    type=int,   default=4)
    p.add_argument("--checkpoint", default=None,
                   help="Path to SAM2 Hiera-Large checkpoint (.pth)")
    p.add_argument("--resume",     default=None,
                   help="Resume from MAFFNet checkpoint")
    p.add_argument("--save_dir",   default="checkpoints")
    p.add_argument("--test_only",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    root = Path(args.data_root) / args.dataset

    def make_loader(split, shuffle):
        csv_path = root / f"{split}.csv"
        if not csv_path.exists():
            print(f"[WARN] {csv_path} not found, skipping {split} split.")
            return None
        ds = SegDataset(str(csv_path), split=split, img_size=args.img_size)
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )

    train_loader = make_loader("train", shuffle=True)
    val_loader   = make_loader("val",   shuffle=False)
    test_loader  = make_loader("test",  shuffle=False)

    if train_loader is None or val_loader is None:
        print("ERROR: train/val CSVs missing. Aborting.")
        sys.exit(1)

    # Build model
    print("Building MAFFNet with SAM2 Hiera-Large backbone …")
    model = MAFFNet(checkpoint=args.checkpoint)

    trainer = MAFFNetTrainer(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        save_dir     = args.save_dir,
        num_epochs   = args.epochs,
        lr           = args.lr,
        weight_decay = args.weight_decay,
        rec_lambda   = args.rec_lambda,
    )

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        print(f"Resumed from {args.resume}  (epoch {ckpt.get('epoch', '?')})")

    if args.test_only:
        best_ckpt = str(Path(args.save_dir) / "best_maffnet.pth")
        trainer.evaluate(test_loader, checkpoint=best_ckpt)
    else:
        trainer.train()
        if test_loader:
            print("\n--- Final test evaluation ---")
            best_ckpt = str(Path(args.save_dir) / "best_maffnet.pth")
            trainer.evaluate(test_loader, checkpoint=best_ckpt)


if __name__ == "__main__":
    main()