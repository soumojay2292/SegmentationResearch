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
from datetime import datetime
from typing import Dict, Tuple, Optional

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
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

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        flat_p = probs.flatten(1)
        flat_t = targets.flatten(1)
        inter  = (flat_p * flat_t).sum(1)
        return 1.0 - (2.0 * inter + self.smooth) / (
            flat_p.sum(1) + flat_t.sum(1) + self.smooth
        ).mean()


class BoundaryLoss(nn.Module):
    """
    Edge-aware L1 loss: extracts boundaries from GT mask and predicted
    probabilities using Sobel filters, then penalises edge mismatches.
    Loss = L1(sobel(sigmoid(logits)), sobel(targets))
    """
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def _edges(self, x: torch.Tensor) -> torch.Tensor:
        kx = self.kx.to(x.dtype)
        ky = self.ky.to(x.dtype)
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        return (gx ** 2 + gy ** 2).sqrt().clamp(0, 1)

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred_edges = self._edges(probs)
        gt_edges   = self._edges(targets)
        return F.l1_loss(pred_edges, gt_edges)


class MAFFNetLoss(nn.Module):
    """
    total = mean(dice + bce) over p1–p4
          + λ_bnd · boundary_l1  (optional, default λ_bnd=0.1)
    boundary_l1 = L1(sobel(pred), sobel(gt))
    """
    def __init__(self, boundary_lambda: float = 0.1):
        super().__init__()
        self.dice     = DiceLoss()
        self.bce      = nn.BCEWithLogitsLoss()
        self.boundary = BoundaryLoss()
        self.blam     = boundary_lambda

    def forward(
        self,
        preds: Tuple[torch.Tensor, ...],   # (p1,p2,p3,p4,rec)
        mask:  torch.Tensor,               # (B,1,H,W) float in [0,1]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        p1, p2, p3, p4, _rec = preds
        mask = mask.clamp(0, 1)

        seg_loss = bnd_loss = 0.0
        for p in (p1, p2, p3, p4):
            prob = torch.sigmoid(p)
            seg_loss = seg_loss + self.dice(prob, mask) + self.bce(p, mask)
            if self.blam > 0.0:
                bnd_loss = bnd_loss + self.boundary(prob, mask)

        seg_loss = (seg_loss / 4.0).mean()

        if self.blam > 0.0:
            bnd_loss = (bnd_loss / 4.0).mean()
            total = seg_loss + self.blam * bnd_loss
        else:
            total = seg_loss

        return total, {
            "seg_loss":      seg_loss.item(),
            "boundary_loss": bnd_loss.item() if self.blam > 0.0 else 0.0,
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
        save_dir:        str  = "checkpoints",
        num_epochs:      int  = 100,
        lr:              float = 1e-3,
        weight_decay:    float = 0.01,
        boundary_lambda: float = 0.1,
        device:          Optional[torch.device] = None,
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

        self.criterion = MAFFNetLoss(
            boundary_lambda=boundary_lambda
        ).to(self.device)
        self.scaler    = GradScaler('cuda')

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
                "train_loss", "train_seg_loss", "train_boundary_loss",
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
        total_loss = seg_loss_sum = bnd_loss_sum = 0.0
        n = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False)

        for imgs, masks in pbar:
            imgs  = imgs.to(self.device)
            masks = masks.to(self.device).float()
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            self.optim.zero_grad(set_to_none=True)

            with autocast('cuda'):
                preds = self.model(imgs)
                loss, sub = self.criterion(preds, masks)

            pbar.set_postfix({"loss": float(loss.item())})
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
            bnd_loss_sum += sub["boundary_loss"] * bs
            n += bs

        return {
            "loss":          total_loss   / n,
            "seg_loss":      seg_loss_sum / n,
            "boundary_loss": bnd_loss_sum / n,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _val_epoch(self, epoch: int = 0) -> Tuple[float, Dict[str, float]]:
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

            with autocast('cuda'):
                preds = self.model(imgs)
                loss, _ = self.criterion(preds, masks)

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
    # Save prediction samples to run_dir/samples/
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _save_samples(self, run_dir: Path, n: int = 5) -> Path:
        import numpy as np
        from PIL import Image

        sample_dir = run_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        saved = 0

        for imgs, masks in self.val_loader:
            if saved >= n:
                break
            imgs  = imgs.to(self.device)
            masks = masks.to(self.device).float()
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            with autocast('cuda'):
                preds = self.model(imgs)

            probs   = torch.sigmoid(preds[0])
            pred_bin = (probs > 0.5).float()

            for j in range(imgs.size(0)):
                if saved >= n:
                    break
                img_np  = (imgs[j].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                gt_np   = (masks[j, 0].cpu().numpy() * 255).astype(np.uint8)
                pred_np = (pred_bin[j, 0].cpu().numpy() * 255).astype(np.uint8)

                Image.fromarray(img_np).save(sample_dir / f"input_{saved}.png")
                Image.fromarray(gt_np,   mode="L").save(sample_dir / f"gt_{saved}.png")
                Image.fromarray(pred_np, mode="L").save(sample_dir / f"pred_{saved}.png")
                saved += 1

        print(f"[trainer] Saved {saved} prediction samples → {sample_dir}")
        return sample_dir

    # ------------------------------------------------------------------
    # Save run summary to run_dir/summary.csv
    # ------------------------------------------------------------------
    def _save_summary(self, run_dir: Path, config: dict,
                      val_metrics: dict, val_loss: float) -> Path:
        summary_path = run_dir / "summary.csv"
        with open(summary_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerow(["val_loss",    f"{val_loss:.4f}"])
            for k, v in val_metrics.items():
                w.writerow([k, f"{v:.4f}"])
            w.writerow(["best_dice", f"{self.best_dice:.4f}"])
            for k, v in config.items():
                w.writerow([k, v])
        return summary_path

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------
    def train(self, config: dict = None, exp_dir=None):
        config = config or {}

        total_p   = sum(p.numel() for p in self.model.parameters()) / 1e6
        train_p   = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
        print(f"Training on {self.device}  |  {self.num_epochs} epochs")
        print(f"Trainable params: {train_p:.1f}M  /  Total: {total_p:.1f}M")

        run_dir = Path(exp_dir) if exp_dir is not None else Path("results") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        last_val_metrics: Dict[str, float] = {}
        last_val_loss = 0.0

        for epoch in tqdm(range(self.num_epochs), desc="Epochs"):
            t0 = time.time()

            train_stats = self._train_epoch(epoch)
            val_loss, val_metrics = self._val_epoch(epoch)

            last_val_metrics = val_metrics
            last_val_loss    = val_loss

            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()

            elapsed = time.time() - t0
            dice    = val_metrics["dice"]

            tqdm.write(
                f"  Ep {epoch+1:03d}/{self.num_epochs}  "
                f"lr={current_lr:.2e}  "
                f"train={train_stats['loss']:.4f}  "
                f"val={val_loss:.4f}  "
                f"dice={dice:.4f}  "
                f"mIoU={val_metrics['mIoU']:.4f}  "
                f"[{elapsed:.0f}s]"
            )

            self._log_csv([
                epoch + 1, f"{current_lr:.6f}",
                f"{train_stats['loss']:.4f}",
                f"{train_stats['seg_loss']:.4f}",
                f"{train_stats['boundary_loss']:.4f}",
                f"{val_loss:.4f}",
                f"{dice:.4f}",
                f"{val_metrics['mIoU']:.4f}",
                f"{val_metrics['acc']:.4f}",
                f"{val_metrics['sensitivity']:.4f}",
                f"{val_metrics['specificity']:.4f}",
                f"{val_metrics['f1']:.4f}",
            ])

            if dice > self.best_dice:
                self.best_dice = dice
                ckpt_path = self.save_dir / "best_maffnet.pth"
                torch.save({
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "optim": self.optim.state_dict(),
                    "dice": dice,
                }, ckpt_path)
                tqdm.write(f"  ↑ New best dice={dice:.4f} → {ckpt_path}")

            if (epoch + 1) % 10 == 0:
                ckpt_path = self.save_dir / f"maffnet_ep{epoch+1:03d}.pth"
                torch.save({
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "optim": self.optim.state_dict(),
                    "dice": dice,
                }, ckpt_path)

        print(f"\nTraining complete. Best val dice: {self.best_dice:.4f}")

        # Post-training: samples, summary, report
        image_dir    = self._save_samples(run_dir)
        summary_path = self._save_summary(run_dir, config, last_val_metrics, last_val_loss)

        report_config = {
            "dataset":          config.get("dataset",    "—"),
            "epochs":           self.num_epochs,
            "batch_size":       config.get("batch_size", "—"),
            "lr":               config.get("lr",         "—"),
            "img_size":         config.get("img_size",   "—"),
            "total_params":     f"{total_p:.1f}M",
            "trainable_params": f"{train_p:.1f}M",
        }

        from utils.report_generator import generate_report
        generate_report(
            str(self._csv_path),
            str(summary_path),
            str(image_dir),
            report_config,
        )

        if exp_dir is not None:
            import shutil
            best_ckpt = self.save_dir / "best_maffnet.pth"
            if best_ckpt.exists():
                shutil.copy2(best_ckpt, run_dir / "model.pth")
            for curve in ("loss_curve.png", "iou_curve.png"):
                src = Path("results") / "dashboard" / curve
                if src.exists():
                    shutil.copy2(src, run_dir / curve)
            print(f"[exp] Experiment saved → {run_dir}")

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

            with autocast('cuda'):
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