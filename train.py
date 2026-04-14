import argparse, torch, os
from torch.utils.data import DataLoader
from torch.optim import Adam
from datasets.seg_dataset import SegDataset
from models.unet import UNet
from models.maffnet import MAFFNet
from models.transunet import TransUNet
from utils.metrics import dice_coeff, iou_coeff
from torch.utils.tensorboard import SummaryWriter
from segment_anything import sam_model_registry
from torch.amp import autocast, GradScaler
import torch.nn as nn
import torchvision.utils as vutils
from tqdm import tqdm

# -------------------------
# Loss
# -------------------------
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
        return bce + dice.mean()

criterion = ComboLoss()

# Fix OpenMP
torch.backends.cudnn.benchmark = True
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------------------------
# TensorBoard Images
# -------------------------
def log_segmentation_images(writer, inputs, masks, preds, step):
    writer.add_image("Inputs", vutils.make_grid(inputs, normalize=True), step)
    writer.add_image("GT", vutils.make_grid(masks.float(), normalize=True), step)
    writer.add_image("Pred", vutils.make_grid(preds.float(), normalize=True), step)

# -------------------------
# Evaluation
# -------------------------
def evaluate(model, loader, device):
    model.eval()
    dices, ious = [], []

    sample_images, sample_masks, sample_preds = None, None, None

    with torch.no_grad():
        for i, (img, mask) in enumerate(loader):

            img, mask = img.to(device), mask.to(device)
            pred = torch.sigmoid(model(img))

            if mask.shape[-1] != pred.shape[-1]:
                mask = torch.nn.functional.interpolate(mask, size=pred.shape[-2:], mode='nearest')

            dice = dice_coeff(pred, mask).item()
            iou = iou_coeff(pred, mask).item()

            dices.append(dice)
            ious.append(iou)

            if i == 0:
                sample_images = img.cpu()
                sample_masks = mask.cpu()
                sample_preds = (pred > 0.5).float().cpu()

    return (
        sum(dices)/len(dices),
        sum(ious)/len(ious),
        sample_images,
        sample_masks,
        sample_preds
    )

# -------------------------
# Main
# -------------------------
def main(model_name, dataset_name):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base = f"dataset_split/{dataset_name}"

    train_ds = SegDataset(f"{base}/train.csv", f"{base}/train/images", f"{base}/train/masks", 224)
    val_ds   = SegDataset(f"{base}/val.csv", f"{base}/val/images", f"{base}/val/masks", 224)
    test_ds  = SegDataset(f"{base}/test.csv", f"{base}/test/images", f"{base}/test/masks", 224)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=2)
    test_loader  = DataLoader(test_ds, batch_size=2)

    writer = SummaryWriter(f"runs/{model_name}_{dataset_name}")

    # Model
    if model_name == "unet":
        model = UNet()
    elif model_name == "maffnet":
        sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
        model = MAFFNet(encoder=sam)
    elif model_name == "transunet":
        model = TransUNet(in_ch=3, out_ch=1, embed_dim=64, num_heads=2)

    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    num_epochs = 20
    step = 0

    # -------------------------
    # Training
    # -------------------------
    for epoch in range(num_epochs):

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()

        loop = tqdm(train_loader)

        for i, (images, labels) in enumerate(loop):

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast("cuda"):
                outputs = model(images)

                if labels.shape[-1] != outputs.shape[-1]:
                    labels = torch.nn.functional.interpolate(labels, size=outputs.shape[-2:], mode='nearest')

                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = (torch.sigmoid(outputs) > 0.5).float()

            dice = dice_coeff(preds, labels).item()
            iou = iou_coeff(preds, labels).item()

            loop.set_postfix(loss=loss.item(), dice=dice, iou=iou)

            writer.add_scalar("Loss/train", loss.item(), step)

            if i % 100 == 0:
                log_segmentation_images(writer, images.cpu(), labels.cpu(), preds.cpu(), step)

            step += 1

        val_dice, val_iou, _, _, _ = evaluate(model, val_loader, device)

        writer.add_scalar("Dice/val", val_dice, epoch)
        writer.add_scalar("IoU/val", val_iou, epoch)

        print(f"Val Dice={val_dice:.4f}")

    # -------------------------
    # Test
    # -------------------------
    test_dice, test_iou, imgs, gts, preds = evaluate(model, test_loader, device)

    print(f"\nTest Dice={test_dice:.4f}, IoU={test_iou:.4f}")

    # -------------------------
    # Save dashboard images
    # -------------------------
    os.makedirs("results/dashboard", exist_ok=True)

    vutils.save_image(imgs[0], "results/dashboard/input.png")
    vutils.save_image(gts[0], "results/dashboard/gt.png")
    vutils.save_image(preds[0], "results/dashboard/pred.png")

    # -------------------------
    # Params
    # -------------------------
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # -------------------------
    # Dashboard
    # -------------------------
    from utils.report_generator import generate_report

    generate_report(
        metrics={"val_dice": val_dice, "test_dice": test_dice, "iou": test_iou},
        config={
            "dataset": dataset_name,
            "epochs": num_epochs,
            "batch_size": 2,
            "lr": 1e-4,
            "total_params": total_params,
            "trainable_params": trainable_params
        },
        image_paths={
            "input": "input.png",
            "gt": "gt.png",
            "pred": "pred.png"
        }
    )

    torch.save(model.state_dict(), f"final_{model_name}_{dataset_name}.pth")

    writer.close()

# -------------------------
# Entry
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    main(args.model, args.dataset)