import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import os
import subprocess
from datasets.seg_dataset import SegDataset
from models.unet import UNet
from models.maffnet import MAFFNet
from models.transunet import TransUNet
from training.trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
import datetime
import csv
from torch.cuda.amp import GradScaler
# from segment_anything import sam_model_registry
# from maffnet import MAFFNet  


# Models to train
# sam_encoder = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
MODELS = {
    "unet": lambda: UNet(),
    # "maffnet": lambda: MAFFNet(encoder=sam_encoder),
    "transunet": lambda: TransUNet(in_ch=3, out_ch=1, embed_dim=64, num_heads=2),
}

# Datasets to train on
DATASETS = ["ISIC_2016", "ISIC_2017", "ISIC_2018"]

def run_all():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    summary_path = os.path.join(results_dir, "summary.csv")
    with open(summary_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Dataset", "Final Train Loss", "Final Val Loss", "Final Val Dice", "Final Val IoU", "Checkpoint Path"])




    for dataset_name in DATASETS:
        print(f"\n📂 Starting dataset: {dataset_name}\n")
        base_path = f"dataset_split/{dataset_name}"
        train_csv = f"{base_path}/train.csv"
        val_csv   = f"{base_path}/val.csv"

        train_ds = SegDataset(
            train_csv,
            f"{base_path}/train/images",
            f"{base_path}/train/masks",
            image_size=64
        )

        val_ds = SegDataset(
            val_csv,
            f"{base_path}/val/images",
            f"{base_path}/val/masks",
            image_size=64
        )


        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

        for model_name, model_fn in MODELS.items():
            print(f"🚀 Training {model_name} on {dataset_name}...\n")
            model = model_fn().to(device)
            optimizer = Adam(model.parameters(), lr=1e-4)

            # Each run gets its own trainer + log directory
            trainer = Trainer(model, optimizer, device)
            trainer.scaler = GradScaler()   # ✅ add this
            trainer.writer = SummaryWriter(
                log_dir=f"runs/{model_name}_{dataset_name}"
            )

            # Train for fixed epochs
            train_loss = trainer.fit(train_loader, val_loader, num_epochs=100)

            # Save checkpoint
            ckpt_path = os.path.join(results_dir, f"{model_name}_{dataset_name}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"✅ Saved checkpoint: {ckpt_path}\n")

            val_loss, val_dice, val_iou = trainer.evaluate_epoch(val_loader)
            with open(summary_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([model_name, dataset_name, train_loss, val_loss, val_dice, val_iou, ckpt_path])

    print("\n🎉 All models trained on all datasets! Results saved in 'results/' folder.\n")

    # ✅ Launch TensorBoard automatically
    print("📊 Launching TensorBoard...")
    subprocess.Popen(["tensorboard", "--logdir", "runs"])


if __name__ == "__main__":
    run_all()
