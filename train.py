import sys
import os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(1, os.path.abspath("src/sam2"))

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import random
from PIL import Image
import numpy as np
from tqdm import tqdm   # ✅ NEW

from models.maffnet import MAFFNet
from training.trainer import MAFFNetTrainer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SegDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, split: str = "train", img_size: int = 384):
        import csv as _csv

        self.split    = split
        self.img_size = img_size
        self.samples  = []

        # 🔥 base dirs (IMPORTANT FIX)
        dataset_root = Path(csv_path).parent  
        self.img_dir  = dataset_root / self.split / "images"
        self.mask_dir = dataset_root / self.split / "masks"

        with open(csv_path) as f:
            reader = _csv.DictReader(f)
            for row in reader:
                self.samples.append((row["image"], row["mask"]))

    def _augment(self, img, mask):
        if random.random() > 0.5:
            img, mask = TF.hflip(img), TF.hflip(mask)
        if random.random() > 0.5:
            img, mask = TF.vflip(img), TF.vflip(mask)

        angle = random.uniform(-30, 30)
        img   = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
        mask  = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

        if random.random() > 0.5:
            params = T.RandomAffine.get_params(
                degrees=(-15, 15), translate=(0.1, 0.1),
                scale_ranges=(0.9, 1.1), shears=(-5, 5),
                img_size=[self.img_size, self.img_size]
            )
            img  = TF.affine(img,  *params, interpolation=InterpolationMode.BILINEAR)
            mask = TF.affine(mask, *params, interpolation=InterpolationMode.NEAREST)

        img = T.ColorJitter(0.2, 0.2, 0.2, 0.1)(img)
        return img, mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, mask_name = self.samples[idx]

        # 🔥 FIX: correct full path
        img_path  = self.img_dir / img_name
        mask_path = self.mask_dir / mask_name

        img  = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img  = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        if self.split == "train":
            img, mask = self._augment(img, mask)

        img_t  = T.ToTensor()(img)
        mask_t = torch.from_numpy(
            np.array(mask, dtype=np.float32) / 255.0
        ).unsqueeze(0)

        mask_t = (mask_t > 0.5).float()
        return img_t, mask_t


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ISIC_2018")
    p.add_argument("--data_root", default="dataset_split")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--rec_lambda", type=float, default=0.1)
    p.add_argument("--img_size", type=int, default=384)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--exp", default="experiment",
                   help="Experiment name; outputs go to experiments/<exp>")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Experiment helpers
# ---------------------------------------------------------------------------

def _write_config(exp_dir: Path, args) -> None:
    with open(exp_dir / "config.txt", "w") as f:
        f.write(f"model:      MAFFNet\n")
        f.write(f"backbone:   SAM2 Hiera-Large\n")
        f.write(f"dataset:    {args.dataset}\n")
        f.write(f"epochs:     {args.epochs}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"img_size:   {args.img_size}\n")
        f.write(f"lr:         {args.lr}\n")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.fast:
        args.epochs = 3
        args.batch_size = 2
        args.img_size = 224

    exp_dir = Path("experiments") / args.exp
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment → {exp_dir}")
    _write_config(exp_dir, args)

    root = Path(args.data_root) / args.dataset

    def make_loader(split, shuffle):
        csv_path = root / f"{split}.csv"
        ds = SegDataset(str(csv_path), split=split, img_size=args.img_size)
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=True,
        )

    train_loader = make_loader("train", True)
    val_loader   = make_loader("val", False)

    print("Building MAFFNet with SAM2 Hiera-Large backbone …")
    model = MAFFNet(checkpoint=args.checkpoint)

    trainer = MAFFNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=args.save_dir,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        rec_lambda=args.rec_lambda,
    )

    trainer.train(config=vars(args), exp_dir=exp_dir)
    print("\nTraining finished.")


if __name__ == "__main__":
    main()