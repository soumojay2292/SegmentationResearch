"""
Trainer for MAFFNet skin lesion segmentation.

Loss  : Dice + BCEWithLogits on p1–p4  +  λ·MSE(rec, input)
Optim : AdamW, lr=1e-3, wd=0.01
Sched : Linear LR decay over 100 epochs
AMP   : enabled (GradScaler)

Metrics: acc, dice, mIoU, sensitivity, specificity, F1
"""

import os
import time
import csv
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        flat_p = probs.flatten(1)
        flat_t = targets.flatten(1)
        inter  = (flat_p * flat_t).sum(1)
        return 1.0 - (2.0 * inter + self.smooth) / (
            flat_p.sum(1) + flat_t.sum(1) + self.smooth
        ).mean()


class MAFFNetLoss(nn.Module):
    """
    total = mean(dice + bce) over p1–p4   +   λ · mse(rec, img)
    λ = 0.1 by default (ablation in paper uses λ ∈ {0.05, 0.1, 0.2})
    """
    def __init__(self, rec_lambda: float = 0.1):
        super().__init__()
        self.dice = DiceLoss()
        self.bce  = nn.BCEWithLogitsLoss()
        self.mse  = nn.MSELoss()
        self.lam  = rec_lambda

    def forward(
        self,
        preds: Tuple[torch.Tensor, ...],   # (p1,p2,p3,p4,rec)
        mask:  torch.Tensor,               # (B,1,H,W) float in [0,1]
        img:   torch.Tensor,               # (B,3,H,W)  original image
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        p1, p2, p3, p4, rec = preds

        seg_loss = 0.0
        for p in (p1, p2, p3, p4):
            seg_loss = seg_loss + self.dice(p, mask) + self.bce(p, mask)
        seg_loss = seg_loss / 4.0

        # Normalise input image to [-1,1] to match RDB Tanh output
        img_norm = img * 2.0 - 1.0
        rec_loss = self.mse(rec, img_norm)

        total = seg_loss + self.lam * rec_loss
        return total, {
            "seg_loss": seg_loss.item(),
            "rec_loss": rec_loss.item(),
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(logits: torch.Tensor, masks: torch.Tensor, thresh: float = 0.5) -> Dict[str, float]:
    """
    logits: (B,1,H,W)  masks: (B,1,H,W) float in [0,1]
    Returns: acc, dice, mIoU, sensitivity, specificity, f1
    """
    probs = torch.sigmoid(logits)
    pred  = (probs > thresh).float()
    tgt   = masks

    TP = (pred * tgt).sum().item()
    FP = (pred * (1 - tgt)).sum().item()
    TN = ((1 - pred) * (1 - tgt)).sum().item()
    FN = ((1 - pred) * tgt).sum().item()

    eps = 1e-8
    acc         = (TP + TN) / (TP + FP + TN + FN + eps)
    dice        = 2 * TP / (2 * TP + FP + FN + eps)
    iou         = TP / (TP + FP + FN + eps)
    sensitivity = TP / (TP + FN + eps)
    specificity = TN / (TN + FP + eps)
    f1          = 2 * TP / (2 * TP + FP + FN + eps)   # same as dice for binary

    return {
        "acc":         acc,
        "dice":        dice,
        "mIoU":        iou,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1":          f1,
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class MAFFNetTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        save_dir:     str  = "checkpoints",
        num_epochs:   int  = 100,
        lr:           float = 1e-3,
        weight_decay: float = 0.01,
        rec_lambda:   float = 0.1,
        device:       Optional[torch.device] = None,
    ):
        self.model       = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.num_epochs   = num_epochs
        self.save_dir     = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = self.model.to(self.device)

        # Trainable params only (backbone is frozen)
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optim = AdamW(trainable, lr=lr, weight_decay=weight_decay)

        # Linear decay: lr → 0 over num_epochs
        self.scheduler = LinearLR(
            self.optim,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=num_epochs,
        )

        self.criterion = MAFFNetLoss(rec_lambda=rec_lambda).to(self.device)
        self.scaler    = GradScaler()

        self.best_dice = 0.0
        self._csv_path = self.save_dir / "training_log.csv"
        self._init_csv()

    # ------------------------------------------------------------------
    # CSV logging
    # ------------------------------------------------------------------
    def _init_csv(self):
        with open(self._csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "epoch", "lr",
                "train_loss", "train_seg_loss", "train_rec_loss",
                "val_loss", "val_dice", "val_miou",
                "val_acc", "val_sensitivity", "val_specificity", "val_f1",
            ])

    def _log_csv(self, row: list):
        with open(self._csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    # ------------------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = seg_loss_sum = rec_loss_sum = 0.0
        n = 0

        for imgs, masks in self.train_loader:
            imgs  = imgs.to(self.device)
            masks = masks.to(self.device).float()
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            self.optim.zero_grad(set_to_none=True)

            with autocast():
                preds = self.model(imgs)
                loss, sub = self.criterion(preds, masks, imgs)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optim)
            nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0
            )
            self.scaler.step(self.optim)
            self.scaler.update()

            bs = imgs.size(0)
            total_loss   += loss.item() * bs
            seg_loss_sum += sub["seg_loss"] * bs
            rec_loss_sum += sub["rec_loss"] * bs
            n += bs

        return {
            "loss":     total_loss   / n,
            "seg_loss": seg_loss_sum / n,
            "rec_loss": rec_loss_sum / n,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _val_epoch(self) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        metric_sums: Dict[str, float] = {
            k: 0.0 for k in ("acc", "dice", "mIoU", "sensitivity", "specificity", "f1")
        }
        n = 0

        for imgs, masks in self.val_loader:
            imgs  = imgs.to(self.device)
            masks = masks.to(self.device).float()
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            with autocast():
                preds = self.model(imgs)
                loss, _ = self.criterion(preds, masks, imgs)

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            n += bs

            # Use p1 (finest prediction) for metrics
            m = compute_metrics(preds[0], masks)
            for k, v in m.items():
                metric_sums[k] += v * bs

        avg_metrics = {k: v / n for k, v in metric_sums.items()}
        return total_loss / n, avg_metrics

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------
    def train(self):
        print(f"Training on {self.device}  |  {self.num_epochs} epochs")
        print(f"Trainable params: "
              f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6:.1f}M")

        for epoch in range(1, self.num_epochs + 1):
            t0 = time.time()

            train_stats = self._train_epoch(epoch)
            val_loss, val_metrics = self._val_epoch()

            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()

            elapsed = time.time() - t0
            dice    = val_metrics["dice"]

            print(
                f"Ep {epoch:03d}/{self.num_epochs}  "
                f"lr={current_lr:.2e}  "
                f"train_loss={train_stats['loss']:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"dice={dice:.4f}  "
                f"mIoU={val_metrics['mIoU']:.4f}  "
                f"[{elapsed:.0f}s]"
            )

            self._log_csv([
                epoch, f"{current_lr:.6f}",
                f"{train_stats['loss']:.4f}",
                f"{train_stats['seg_loss']:.4f}",
                f"{train_stats['rec_loss']:.4f}",
                f"{val_loss:.4f}",
                f"{dice:.4f}",
                f"{val_metrics['mIoU']:.4f}",
                f"{val_metrics['acc']:.4f}",
                f"{val_metrics['sensitivity']:.4f}",
                f"{val_metrics['specificity']:.4f}",
                f"{val_metrics['f1']:.4f}",
            ])

            # Save best checkpoint
            if dice > self.best_dice:
                self.best_dice = dice
                ckpt_path = self.save_dir / "best_maffnet.pth"
                torch.save({
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "optim": self.optim.state_dict(),
                    "dice": dice,
                }, ckpt_path)
                print(f"  ↑ New best dice={dice:.4f}, saved → {ckpt_path}")

            # Save latest checkpoint every 10 epochs
            if epoch % 10 == 0:
                ckpt_path = self.save_dir / f"maffnet_ep{epoch:03d}.pth"
                torch.save({
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "optim": self.optim.state_dict(),
                    "dice": dice,
                }, ckpt_path)

        print(f"\nTraining complete. Best val dice: {self.best_dice:.4f}")
        return self.best_dice

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader, checkpoint: str = None) -> Dict[str, float]:
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location=self.device)
            self.model.load_state_dict(ckpt["state_dict"])
            print(f"Loaded checkpoint: {checkpoint}  (epoch {ckpt.get('epoch','?')})")

        self.model.eval()
        metric_sums: Dict[str, float] = {
            k: 0.0 for k in ("acc", "dice", "mIoU", "sensitivity", "specificity", "f1")
        }
        n = 0

        for imgs, masks in test_loader:
            imgs  = imgs.to(self.device)
            masks = masks.to(self.device).float()
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            with autocast():
                preds = self.model(imgs)

            bs = imgs.size(0)
            n += bs
            m = compute_metrics(preds[0], masks)
            for k, v in m.items():
                metric_sums[k] += v * bs

        results = {k: v / n for k, v in metric_sums.items()}
        print("\n=== Test Results ===")
        for k, v in results.items():
            print(f"  {k:<14}: {v:.4f}")
        return results