import os
import torch
import torchvision
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


class Trainer:
    def __init__(self, model, optimizer, device, scheduler=None, checkpoint_path=None, loss_fn=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

        # ✅ Default loss: BCE + Dice
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.loss_fn = loss_fn if loss_fn is not None else self._combined_loss

        # ✅ AMP scaler
        self.scaler = GradScaler()

        # ✅ TensorBoard writer
        self.writer = SummaryWriter(log_dir="runs/segmentation_experiment")

        self.global_step = 0
        self.epoch = 0

        # ✅ Resume training if checkpoint exists
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # ---------------- Metrics ----------------
    def dice_coeff(self, pred, target, eps=1e-6):
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return (2. * intersection + eps) / (union + eps)

    def iou_coeff(self, pred, target, eps=1e-6):
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return (intersection + eps) / (union + eps)

    # ---------------- Loss ----------------
    def _combined_loss(self, pred, target):
        dice = self.dice_coeff(pred, target)
        return self.bce(pred, target) + (1 - dice)

    # ---------------- Training ----------------
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        

        for img, mask in loader:
            img, mask = img.to(self.device), mask.to(self.device)
            self.optimizer.zero_grad()

            # ✅ Mixed precision forward
            with autocast():
                output = self.model(img)
                if output.shape[-2:] != mask.shape[-2:]:
                    output = F.interpolate(output, size=mask.shape[-2:], mode="bilinear", align_corners=False)

                loss = self.loss_fn(output, mask)

            # ✅ Scaled backward
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            # Logging
            self.writer.add_scalar("Loss/train", loss.item(), self.global_step)
            self.global_step += 1

        avg_loss = total_loss / len(loader)
        self.writer.add_scalar("Loss/train_epoch", avg_loss, self.epoch)
        return avg_loss

    # ---------------- Evaluation ----------------
    def evaluate_epoch(self, loader):
        self.model.eval()
        running_loss, dice_score, iou_score = 0.0, 0.0, 0.0

        with torch.no_grad():
            for img, mask in loader:
                img, mask = img.to(self.device), mask.to(self.device)
                output = self.model(img)

                # ✅ Ensure prediction and mask sizes match
                if output.shape[-2:] != mask.shape[-2:]:
                    output = F.interpolate(output, size=mask.shape[-2:], mode="bilinear", align_corners=False)

                loss = self.loss_fn(output, mask)

                running_loss += loss.item()
                dice_score += self.dice_coeff(output, mask).item()
                iou_score += self.iou_coeff(output, mask).item()

            # Log predictions from last batch
            self.log_predictions(img.detach().cpu(), mask.detach().cpu(), output.detach().cpu(), self.epoch)

        avg_loss = running_loss / len(loader)
        avg_dice = dice_score / len(loader)
        avg_iou = iou_score / len(loader)

        self.writer.add_scalar("Loss/val", avg_loss, self.epoch)
        self.writer.add_scalar("Dice/val", avg_dice, self.epoch)
        self.writer.add_scalar("IoU/val", avg_iou, self.epoch)

        return avg_loss, avg_dice, avg_iou

    # ---------------- Utilities ----------------
    def step_epoch(self):
        self.epoch += 1
        if self.scheduler:
            self.scheduler.step()

    def log_predictions(self, images, masks, preds, epoch):
        images = images[:4]
        masks = masks[:4]
        preds = (torch.sigmoid(preds[:4]) > 0.5).float()

        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        elif masks.ndim == 2:
            masks = masks.unsqueeze(0).unsqueeze(0)

        if preds.ndim == 3:
            preds = preds.unsqueeze(1)

        img_grid = torchvision.utils.make_grid(images, normalize=True, scale_each=True)
        mask_grid = torchvision.utils.make_grid(masks, normalize=True, scale_each=True)
        pred_grid = torchvision.utils.make_grid(preds, normalize=True, scale_each=True)

        self.writer.add_image("Input Images", img_grid, epoch)
        self.writer.add_image("Ground Truth Masks", mask_grid, epoch)
        self.writer.add_image("Predicted Masks", pred_grid, epoch)

        overlay = images.clone()
        overlay[:, 0, :, :] = torch.where(preds.squeeze(1) > 0.5, 1.0, overlay[:, 0, :, :])
        overlay[:, 1, :, :] = torch.where(masks.squeeze(1) > 0.5, 1.0, overlay[:, 1, :, :])
        overlay_grid = torchvision.utils.make_grid(overlay, normalize=True, scale_each=True)
        self.writer.add_image("Overlay Predictions", overlay_grid, epoch)

    def fit(self, train_loader, val_loader, num_epochs):
        for _ in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_dice, val_iou = self.evaluate_epoch(val_loader)
            self.step_epoch()

            print(f"Epoch {self.epoch}: Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Dice={val_dice:.4f}, Val IoU={val_iou:.4f}")

            os.makedirs("checkpoints", exist_ok=True)
            torch.save(self.model.state_dict(), f"checkpoints/model_epoch{self.epoch}.pth")
