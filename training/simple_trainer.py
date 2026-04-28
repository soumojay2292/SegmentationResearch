"""
Trainer for single-output segmentation models (Attention U-Net, UNet++, …).

Loss   : Dice + BCEWithLogits (pos_weight=2.0)
Optim  : AdamW, linear LR decay
AMP    : GradScaler + autocast (mirrors MAFFNetTrainer)
Logging: same CSV columns as MAFFNetTrainer (boundary_loss always 0.0)
"""

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.trainer import DiceLoss, compute_metrics


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class SimpleLoss(nn.Module):
    """Dice + BCEWithLogits for models that output a single (B,1,H,W) logit map."""

    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.register_buffer("pos_weight", torch.tensor([2.0]))

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.clamp(0, 1)
        prob = torch.sigmoid(logits)
        bce  = F.binary_cross_entropy_with_logits(logits, mask, pos_weight=self.pos_weight)
        return self.dice(prob, mask) + bce


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SimpleTrainer:
    """Generic trainer for segmentation models with a single (B,1,H,W) output.

    Produces the same CSV log format, checkpoint structure, and post-training
    artifacts (samples, summary, report) as MAFFNetTrainer.
    """

    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        model_name:   str   = "model",
        save_dir:     str   = "checkpoints",
        num_epochs:   int   = 100,
        lr:           float = 1e-3,
        weight_decay: float = 0.01,
        device:       Optional[torch.device] = None,
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.model_name   = model_name
        self.num_epochs   = num_epochs
        self.save_dir     = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = self.model.to(self.device)

        self.optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = LinearLR(
            self.optim,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=num_epochs,
        )
        self.criterion = SimpleLoss().to(self.device)
        self.scaler    = GradScaler("cuda")

        self.best_dice = 0.0
        self._csv_path = self.save_dir / "training_log.csv"
        self._init_csv()

    # ------------------------------------------------------------------
    # CSV logging — same columns as MAFFNetTrainer for report compatibility
    # ------------------------------------------------------------------

    def _init_csv(self):
        with open(self._csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
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
        total_loss = 0.0
        n = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}", leave=False)
        for imgs, masks in pbar:
            imgs  = imgs.to(self.device)
            masks = masks.to(self.device).float()
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            self.optim.zero_grad(set_to_none=True)

            with autocast("cuda"):
                logits = self.model(imgs)
                loss   = self.criterion(logits, masks)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optim)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optim)
            self.scaler.update()

            bs          = imgs.size(0)
            total_loss += loss.item() * bs
            n          += bs

        avg = total_loss / n
        return {"loss": avg, "seg_loss": avg, "boundary_loss": 0.0}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _val_epoch(self) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss  = 0.0
        metric_sums = {k: 0.0 for k in ("acc", "dice", "mIoU", "sensitivity", "specificity", "f1")}
        n = 0

        for imgs, masks in self.val_loader:
            imgs  = imgs.to(self.device)
            masks = masks.to(self.device).float()
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            with autocast("cuda"):
                logits = self.model(imgs)
                loss   = self.criterion(logits, masks)

            bs          = imgs.size(0)
            total_loss += loss.item() * bs
            n          += bs

            m = compute_metrics(logits, masks)
            for k, v in m.items():
                metric_sums[k] += v * bs

        avg_metrics = {k: v / n for k, v in metric_sums.items()}
        return total_loss / n, avg_metrics

    # ------------------------------------------------------------------
    # Save prediction samples
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

            with autocast("cuda"):
                logits = self.model(imgs)

            probs    = torch.sigmoid(logits)
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
    # Save run summary
    # ------------------------------------------------------------------

    def _save_summary(self, run_dir: Path, config: dict,
                      val_metrics: dict, val_loss: float) -> Path:
        summary_path = run_dir / "summary.csv"
        with open(summary_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerow(["val_loss",  f"{val_loss:.4f}"])
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

        total_p = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"Training on {self.device}  |  {self.num_epochs} epochs")
        print(f"Total params: {total_p:.1f}M")

        run_dir = (
            Path(exp_dir) if exp_dir is not None
            else Path("results") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        last_val_metrics: Dict[str, float] = {}
        last_val_loss = 0.0

        for epoch in tqdm(range(self.num_epochs), desc="Epochs"):
            t0 = time.time()

            train_stats              = self._train_epoch(epoch)
            val_loss, val_metrics    = self._val_epoch()
            last_val_metrics         = val_metrics
            last_val_loss            = val_loss

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

            # Save every epoch
            ckpt = {
                "epoch":      epoch + 1,
                "state_dict": self.model.state_dict(),
                "optim":      self.optim.state_dict(),
                "dice":       dice,
            }
            torch.save(ckpt, self.save_dir / f"model_epoch_{epoch + 1}.pth")

            if dice > self.best_dice:
                self.best_dice = dice
                torch.save(ckpt, self.save_dir / f"best_{self.model_name}.pth")
                tqdm.write(f"  ↑ New best dice={dice:.4f} → best_{self.model_name}.pth")

        print(f"\nTraining complete. Best val dice: {self.best_dice:.4f}")

        image_dir    = self._save_samples(run_dir)
        summary_path = self._save_summary(run_dir, config, last_val_metrics, last_val_loss)

        report_config = {
            "dataset":      config.get("dataset",    "—"),
            "epochs":       self.num_epochs,
            "batch_size":   config.get("batch_size", "—"),
            "lr":           config.get("lr",         "—"),
            "img_size":     config.get("img_size",   "—"),
            "total_params": f"{total_p:.1f}M",
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
            best_ckpt = self.save_dir / f"best_{self.model_name}.pth"
            if best_ckpt.exists():
                shutil.copy2(best_ckpt, run_dir / "model.pth")
            print(f"[exp] Experiment saved → {run_dir}")

        return self.best_dice
