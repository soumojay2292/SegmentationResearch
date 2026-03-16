import torch
import torchvision
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard.writer import SummaryWriter

class Trainer:
    def __init__(self, model, optimizer, device, scheduler=None):
        self.model = model.to(device)
        self.optimizer = optimizer 
        self.device = device 
        self.scheduler = scheduler

        self.bce = torch.nn.BCEWithLogitsLoss() 
        self.scaler = torch.amp.grad_scaler.GradScaler(device="cuda") 
        self.writer = SummaryWriter(log_dir="runs/segmentation_experiment")

        self.global_step = 0
        self.epoch = 0

    def dice_coeff(self, pred, target, eps=1e-6):
        """Compute Dice coefficient."""
        pred = torch.sigmoid(pred)        # convert logits to probabilities
        pred = (pred > 0.5).float()       # threshold to binary mask

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + eps) / (union + eps)
        return dice

    def loss_fn(self, pred, target):
        dice = self.dice_coeff(pred, target)
        return self.bce(pred, target) + (1 - dice)


    def train_epoch(self, loader):
        self.model.train()
        running_loss = 0.0

        for img, mask in loader:
            img, mask = img.to(self.device), mask.to(self.device)

        with torch.amp.autocast_mode.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.model(img)
            loss = self.loss_fn(output, mask)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            running_loss += loss.item()
            self.writer.add_scalar("Loss/train", loss.item(), self.global_step)
            self.global_step += 1

        avg_loss = running_loss / len(loader)
        self.writer.add_scalar("Loss/train_epoch", avg_loss, self.epoch)
        return avg_loss

    def evaluate_epoch(self, loader):
        self.model.eval()
        running_loss, dice_score = 0.0, 0.0

        with torch.no_grad():
            for img, mask in loader:
                img, mask = img.to(self.device), mask.to(self.device)
                output = self.model(img)
                loss = self.loss_fn(output, mask)

                running_loss += loss.item()
                dice_score += self.dice_coeff(output, mask).item()

            # Take the last batch for visualization
            self.log_predictions(
                images=img.detach().cpu(),
                masks=mask.detach().cpu(),
                preds=output.detach().cpu(),
                epoch=self.epoch
            )

        avg_loss = running_loss / len(loader)
        avg_dice = dice_score / len(loader)

        # Log scalars
        self.writer.add_scalar("Loss/val", avg_loss, self.epoch)
        self.writer.add_scalar("Dice/val", avg_dice, self.epoch)

        return avg_loss, avg_dice


    def step_epoch(self):
        self.epoch += 1
        if self.scheduler:
            self.scheduler.step()

    def log_predictions(self, images, masks, preds, epoch): 
        """Log input images, ground truth masks, predicted masks, and overlays to TensorBoard.""" 
        # Convert tensors to CPU and detach
        images = images[:4].detach().cpu()
        masks = masks[:4].detach().cpu()
        preds = (torch.sigmoid(preds[:4].detach().cpu()) > 0.5).float()

        # Ensure masks are 4D: [B, 1, H, W]
        if masks.ndim == 3:   # [B, H, W]
            masks = masks.unsqueeze(1)
        elif masks.ndim == 2: # [H, W] single mask
            masks = masks.unsqueeze(0).unsqueeze(0)

        # Ensure preds are 4D: [B, 1, H, W]
        if preds.ndim == 3:
            preds = preds.unsqueeze(1)

        # Grids for raw visualization
        img_grid = torchvision.utils.make_grid(images, normalize=True, scale_each=True) 
        mask_grid = torchvision.utils.make_grid(masks, normalize=True, scale_each=True) 
        pred_grid = torchvision.utils.make_grid(preds, normalize=True, scale_each=True) 

        self.writer.add_image("Input Images", img_grid, epoch) 
        self.writer.add_image("Ground Truth Masks", mask_grid, epoch) 
        self.writer.add_image("Predicted Masks", pred_grid, epoch)

        # Overlay visualization: red = prediction, green = ground truth
        overlay = images.clone()
        overlay[:, 0, :, :] = torch.where(preds.squeeze(1) > 0.5, 1.0, overlay[:, 0, :, :])  # red channel
        overlay[:, 1, :, :] = torch.where(masks.squeeze(1) > 0.5, 1.0, overlay[:, 1, :, :])  # green channel
        overlay_grid = torchvision.utils.make_grid(overlay, normalize=True, scale_each=True)
        self.writer.add_image("Overlay Predictions", overlay_grid, epoch)



    def fit(self, train_loader, val_loader, num_epochs):
        """Run full training loop with logging."""
        for _ in range(num_epochs):
            # Train for one epoch
            train_loss = self.train_epoch(train_loader)

            # Evaluate after training
            val_loss, val_dice = self.evaluate_epoch(val_loader)

            # Step scheduler if provided
            self.step_epoch()

            print(f"Epoch {self.epoch}: "
                f"Train Loss={train_loss:.4f}, "
                f"Val Loss={val_loss:.4f}, "
                f"Val Dice={val_dice:.4f}")

