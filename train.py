from xml.parsers.expat import model

from networkx import union
import argparse, torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from datasets.seg_dataset import SegDataset
from models.unet import UNet
from models.maffnet import MAFFNet
from training.trainer import Trainer
from utils.metrics import dice_coeff, iou_coeff
from utils.loss import DiceLoss
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from segment_anything import sam_model_registry
from torch.amp import autocast, GradScaler
import os
import torch.nn as nn 

from models.transunet import TransUNet


class ComboLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))

        dice = 1 - ((2 * intersection + 1e-8) / (union + 1e-8))
        dice = dice.mean()
        return bce + dice

criterion = ComboLoss()

# Fix OpenMP duplicate runtime error
torch.backends.cudnn.benchmark = True
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
print("OpenMP duplicate runtime fix applied (KMP_DUPLICATE_LIB_OK=TRUE)")

def log_segmentation_images(writer, inputs, masks, preds, step: int):
    """Log input images, ground truth masks, and predicted masks to TensorBoard."""
    input_grid = vutils.make_grid(inputs, normalize=True, scale_each=True)
    mask_grid = vutils.make_grid(masks.float(), normalize=True, scale_each=True)
    pred_grid = vutils.make_grid(preds.float(), normalize=True, scale_each=True)

    writer.add_image("Inputs", input_grid, global_step=step)
    writer.add_image("GroundTruth", mask_grid, global_step=step)
    writer.add_image("Predictions", pred_grid, global_step=step)

def main(model_name, dataset_name):
    print(f"Starting training with {model_name} on dataset {dataset_name}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset paths
    base_path = f"dataset_split/{dataset_name}"
    train_csv = f"{base_path}/train.csv"
    val_csv   = f"{base_path}/val.csv"
    test_csv  = f"{base_path}/test.csv"

    train_ds = SegDataset(
        train_csv,
        f"{base_path}/train/images",
        f"{base_path}/train/masks",
        image_size=1024
    )

    val_ds = SegDataset(
        val_csv,
        f"{base_path}/val/images",
        f"{base_path}/val/masks",
        image_size=1024
    )

    test_ds = SegDataset(
        test_csv,
        f"{base_path}/test/images",
        f"{base_path}/test/masks",
        image_size=1024
    )


    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)


    # TensorBoard writer with model+dataset name
    writer = SummaryWriter(log_dir=f"runs/{model_name}_{dataset_name}")

# Model selection
    if model_name == "unet":
        model = UNet()
    elif model_name == "maffnet":
        sam_encoder = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
        model = MAFFNet(encoder=sam_encoder)
    elif model_name == "transunet":
        model = TransUNet(in_ch=3, out_ch=1, embed_dim=64, num_heads=2)
    else:
        raise ValueError("Unknown model")


    optimizer = Adam(model.parameters(), lr=1e-4)
    trainer = Trainer(model, optimizer, device)
    model = model.to(device)

    # Print device info
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(next(model.parameters()).device)

    model.eval()
    with torch.no_grad():
        test_out = model(torch.randn(1, 3, 224, 224).to(device))
        print("Model output shape:", test_out.shape)

    scaler = GradScaler("cuda")
    # Training loop
    num_epochs = 50
    global_step = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} started...")   
        model.train()
        for i, (images, labels) in enumerate(train_loader): 
            images, labels = images.to(device), labels.to(device) 
            
            optimizer.zero_grad() 
            with autocast("cuda"):
                outputs = model(images)
                if labels.shape[-1] != outputs.shape[-1]:
                    labels = torch.nn.functional.interpolate(
                        labels, 
                        size=outputs.shape[-2:], 
                        mode='nearest'
                        )
                if labels.shape[-1] != outputs.shape[-1]:
                    print("Mismatch detected!")
                if i % 50 == 0:
                    print("Output:", outputs.shape, "Label:", labels.shape)
                loss = criterion(outputs, labels)

            
            # Backprop with AMP 
            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            preds = (torch.sigmoid(outputs) > 0.5).float()  
                
                # Logging 
            writer.add_scalar("Loss/train", loss.item(), global_step) 
            if i % 100 == 0: 
                log_segmentation_images(writer, images.cpu(), labels.cpu(), preds.cpu(), step=global_step) 
            if i % 10 == 0: 
                print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}") 
                
            global_step += 1

        # Validation after each epoch
        val_dice, val_iou = evaluate(model, val_loader, device) 
        writer.add_scalar("Dice/val", val_dice, epoch) 
        writer.add_scalar("IoU/val", val_iou, epoch) 
        print(f"Epoch {epoch+1}: Val Dice={val_dice:.4f}, Val IoU={val_iou:.4f}")

    # Final test evaluation
    test_dice, test_iou = evaluate(model, test_loader, device)
    print(f"Final Test Dice={test_dice:.4f}, Test IoU={test_iou:.4f}")
    writer.add_scalar("Dice/test", test_dice, num_epochs)
    writer.add_scalar("IoU/test", test_iou, num_epochs)

    # Save final model checkpoint
    torch.save(model.state_dict(), f"final_{model_name}_{dataset_name}.pth")

    writer.close()

def evaluate(model, loader, device):
    model.eval()
    dices, ious = [], []
    with torch.no_grad():
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)

            pred = torch.sigmoid(model(img))

    # 🔥 FIX: resize mask to match prediction
        if mask.shape[-1] != pred.shape[-1]:
            mask = torch.nn.functional.interpolate(
                mask,
                size=pred.shape[-2:],
                mode='nearest'
            )

        dices.append(dice_coeff(pred, mask).item())
        ious.append(iou_coeff(pred, mask).item())
    return sum(dices)/len(dices), sum(ious)/len(ious)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)  # NEW
    args = parser.parse_args()
    main(args.model, args.dataset)
