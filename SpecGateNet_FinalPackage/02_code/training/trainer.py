"""
Trainer for MAFFNet skin lesion segmentation.

Loss  : Dice + BCEWithLogits on p1–p4  +  0.01·BoundaryL1 (after epoch 10 warmup)
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
        return (1.0 - (2.0 * inter + self.smooth) / (
            flat_p.sum(1) + flat_t.sum(1) + self.smooth
        )).mean()


class BoundaryLoss(nn.Module):
    """Sobel edge L1: L1(edges(sigmoid(logits)), edges(gt))."""
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
        # +1e-6 keeps the sqrt gradient finite at flat regions (gx=gy=0);
        # without it, d/du sqrt(u) = 1/(2*sqrt(u)) -> inf at u=0 (see FGFF_NUMERICAL_ANALYSIS_REPORT.md)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6).clamp(0, 1)

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._edges(probs), self._edges(targets))


class MAFFNetLoss(nn.Module):
    """
    Boundary-centric Dual Supervision (BcDS) — see CONTRIBUTION_DESIGN.md.

        L_total = L_seg
                  + lambda_bds(t) * L_BDS      # N1: boundary-aware deep supervision
                  + lambda_ewr     * L_EWR      # N2: edge-weighted reconstruction

    L_seg = mean_i [ Dice(sigmoid(p_i), y) (+ BCE if use_bce) ]            (4 heads)
    L_BDS = (1/Σw) Σ_i w_i · L1( Sobel(sigmoid(p_i)), Sobel(y) )
    B(y)  = boundary map from GT  (sobel-thin | morphological; optional band_r)
    W     = 1 + alpha · B_r(src)   ;  src ∈ {GT boundary, input-image edges}
    L_EWR = Σ( W ⊙ (rec01 - x)^2 ) / ( ΣW · C )         (normalised weighted MSE)
    lambda_bds(t) = lambda_bds * min(1, t / t_warm)     (t_warm=0 ⇒ always on)

    All terms are switchable (use_bds / use_ewr / use_bce) for ablation.
    Defaults = full BcDS.
    """
    def __init__(
        self,
        lambda_bds: float = 0.05,
        lambda_ewr: float = 0.05,
        alpha:      float = 3.0,
        t_warm:     int   = 5,
        band_r:     int   = 1,
        ds_weights        = (1.0, 1.0, 1.0, 0.01),
        use_bds:    bool  = True,
        use_ewr:    bool  = True,
        use_bce:    bool  = True,
        smooth_pred: bool = True,
        bmap:       str   = "sobel",      # "sobel" | "morph"
        ewr_edge_source: str = "gt",      # "gt" | "image"
        pos_weight: float = 2.0,
    ):
        super().__init__()
        self.dice     = DiceLoss()
        self.boundary = BoundaryLoss()
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        self.lambda_bds = lambda_bds
        self.lambda_ewr = lambda_ewr
        self.alpha      = alpha
        self.t_warm     = t_warm
        self.band_r     = band_r
        self.ds_weights = tuple(ds_weights)
        self.use_bds    = use_bds
        self.use_ewr    = use_ewr
        self.use_bce    = use_bce
        self.smooth_pred = smooth_pred
        self.bmap       = bmap
        self.ewr_edge_source = ewr_edge_source

    def _edge_mag(self, x: torch.Tensor) -> torch.Tensor:
        return self.boundary._edges(x)             # Sobel magnitude, clamp[0,1]

    def _boundary_map(self, y: torch.Tensor) -> torch.Tensor:
        """B(y): GT lesion-boundary map (sobel-thin or morphological), optional band."""
        if self.bmap == "morph":
            k = 3
            dil = F.max_pool2d(y, k, 1, k // 2)
            ero = -F.max_pool2d(-y, k, 1, k // 2)
            b = (dil - ero).clamp(0, 1)
        else:
            b = self._edge_mag(y)
        if self.band_r > 0:
            ksz = 2 * self.band_r + 1
            b = F.max_pool2d(b, ksz, 1, self.band_r)   # widen edge to a band
        return b.clamp(0, 1)

    def forward(
        self,
        preds: Tuple[torch.Tensor, ...],   # (p1,p2,p3,p4,rec)
        mask:  torch.Tensor,               # (B,1,H,W) float in [0,1]
        epoch: int = 0,
        image: torch.Tensor = None,        # (B,3,H,W) in [0,1] — reconstruction target
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        p1, p2, p3, p4, rec = preds
        y = mask.clamp(0, 1)
        heads = (p1, p2, p3, p4)
        w = self.ds_weights

        # --- L_seg: deep-supervised Dice [+ BCE] ---
        seg_loss = 0.0
        for p in heads:
            prob = torch.sigmoid(p)
            l = self.dice(prob, y)
            if self.use_bce:
                l = l + F.binary_cross_entropy_with_logits(p, y, pos_weight=self.pos_weight)
            seg_loss = seg_loss + l
        seg_loss = seg_loss / len(heads)
        total = seg_loss

        # --- N1: Boundary-aware Deep Supervision ---
        # Compute the warmup weight first; when it is exactly zero, skip the
        # entire BDS computation so no autograd graph is built. This prevents a
        # zeroed term from injecting 0 * inf = NaN through the edge sqrt.
        bds_loss = torch.zeros((), device=y.device)
        ramp = 1.0 if self.t_warm <= 0 else min(1.0, epoch / float(self.t_warm))
        bds_weight = self.lambda_bds * ramp
        if self.use_bds and bds_weight > 0:
            sy = self._edge_mag(y)
            acc = 0.0
            for wi, p in zip(w, heads):
                prob = torch.sigmoid(p)
                if self.smooth_pred:
                    prob = F.avg_pool2d(prob, kernel_size=3, stride=1, padding=1)
                acc = acc + wi * F.l1_loss(self._edge_mag(prob), sy)
            bds_loss = acc / float(sum(w))
            total = total + bds_weight * bds_loss

        # --- N2: Edge-weighted Reconstruction ---
        ewr_loss = torch.zeros((), device=y.device)
        if self.use_ewr and image is not None and rec is not None and self.lambda_ewr > 0:
            rec01 = (rec + 1.0) / 2.0
            if rec01.shape[-2:] != image.shape[-2:]:
                rec01 = F.interpolate(rec01, size=image.shape[-2:],
                                      mode='bilinear', align_corners=False)
            if self.ewr_edge_source == "image":
                src = image.mean(dim=1, keepdim=True)
                B = self._edge_mag(src)
                if self.band_r > 0:
                    B = F.max_pool2d(B, 2 * self.band_r + 1, 1, self.band_r)
                B = B.clamp(0, 1)
            else:
                B = self._boundary_map(y)             # GT-boundary (coherent with N1)
            Wt = 1.0 + self.alpha * B                 # (B,1,H,W), base weight 1 everywhere
            sq = (rec01 - image) ** 2                 # (B,C,H,W)
            ewr_loss = (Wt * sq).sum() / (Wt.sum() * sq.shape[1] + 1e-8)
            total = total + self.lambda_ewr * ewr_loss

        return total, {
            "seg_loss":      float(seg_loss.item()),
            "boundary_loss": float(bds_loss.item()),   # = BDS (kept name for CSV compat)
            "bds_loss":      float(bds_loss.item()),
            "ewr_loss":      float(ewr_loss.item()),
            "rec_loss":      float(ewr_loss.item()),   # alias (back-compat)
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
        device:       Optional[torch.device] = None,
        loss_kwargs:  Optional[dict] = None,
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

        # Trainable params only (backbone is frozen); MMPA gamma excluded from weight decay
        decay, no_decay = [], []
        for pname, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if pname.endswith("gamma") else decay).append(p)
        self.optim = AdamW(
            [{"params": decay,    "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=lr,
        )

        # Linear decay: lr → 0 over num_epochs
        self.scheduler = LinearLR(
            self.optim,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=num_epochs,
        )

        self.criterion = MAFFNetLoss(**(loss_kwargs or {})).to(self.device)
        # bf16 autocast: GradScaler not required (bf16 has fp32-range exponent)

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
                "train_ewr_loss",
            ])

    def _log_csv(self, row: list):
        with open(self._csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    # ------------------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = seg_loss_sum = bnd_loss_sum = ewr_loss_sum = 0.0
        n = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False)

        for imgs, masks in pbar:
            imgs  = imgs.to(self.device)
            masks = masks.to(self.device).float()
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            self.optim.zero_grad(set_to_none=True)

            with autocast('cuda', dtype=torch.bfloat16):
                preds = self.model(imgs)
                loss, sub = self.criterion(preds, masks, epoch=epoch, image=imgs)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")

            pbar.set_postfix({"loss": float(loss.item())})
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0
            )
            self.optim.step()

            bs = imgs.size(0)
            total_loss   += loss.item()                 * bs
            seg_loss_sum += sub["seg_loss"]             * bs
            bnd_loss_sum += sub["boundary_loss"]        * bs
            ewr_loss_sum += sub.get("ewr_loss", 0.0)    * bs
            n += bs

        return {
            "loss":          total_loss   / n,
            "seg_loss":      seg_loss_sum / n,
            "boundary_loss": bnd_loss_sum / n,
            "ewr_loss":      ewr_loss_sum / n,
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

            with autocast('cuda', dtype=torch.bfloat16):
                preds = self.model(imgs)
                loss, _ = self.criterion(preds, masks, epoch=epoch, image=imgs)

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

            with autocast('cuda', dtype=torch.bfloat16):
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
                f"{train_stats.get('ewr_loss', 0.0):.4f}",
            ])

            # Save every epoch
            ckpt = {
                "epoch":      epoch + 1,
                "state_dict": self.model.state_dict(),
                "optim":      self.optim.state_dict(),
                "dice":       dice,
            }
            torch.save(ckpt, self.save_dir / f"model_epoch_{epoch+1}.pth")

            # Keep best model separately
            if dice > self.best_dice:
                self.best_dice = dice
                torch.save(ckpt, self.save_dir / "best_maffnet.pth")
                tqdm.write(f"  ↑ New best dice={dice:.4f} → best_maffnet.pth")

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

            with autocast('cuda', dtype=torch.bfloat16):
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


# ---------------------------------------------------------------------------
# General-purpose Trainer  (used by train_all.py for all models)
# ---------------------------------------------------------------------------

class Trainer:
    """
    Lightweight trainer compatible with all segmentation models.

    Handles both single-tensor outputs (logits) and MAFFNet's 5-tuple
    (p1, p2, p3, p4, rec).  Reuses DiceLoss / MAFFNetLoss / compute_metrics
    defined above so there is no duplicated logic.

    Interface expected by train_all.py:
        trainer = Trainer(model, optimizer, device)
        trainer.scaler = GradScaler()          # optional override
        trainer.writer = SummaryWriter(...)    # optional TensorBoard
        train_loss = trainer.fit(train_loader, val_loader, num_epochs=N)
        val_loss, val_dice, val_iou = trainer.evaluate_epoch(val_loader)
    """

    def __init__(self, model: nn.Module, optimizer, device):
        self.model        = model.to(device)
        self.optimizer    = optimizer
        self.device       = device
        self._device_type = "cuda" if "cuda" in str(device) else "cpu"

        # Defaults — callers may replace these after construction
        self.scaler = GradScaler(self._device_type)
        self.writer = None  # optional SummaryWriter

        self._dice_loss    = DiceLoss()
        self._maffnet_loss = MAFFNetLoss().to(device)
        self._bce          = nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------------
    def _forward(
        self, imgs: torch.Tensor, masks: torch.Tensor, epoch: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass + loss. Returns (loss, primary_logits)."""
        with autocast(self._device_type):
            output = self.model(imgs)

            if isinstance(output, tuple):
                # MAFFNet: (p1, p2, p3, p4, rec)
                loss, _ = self._maffnet_loss(output, masks, epoch=epoch)
                logits  = output[0]
            else:
                logits  = output
                prob    = torch.sigmoid(logits)
                loss    = self._bce(logits, masks) + self._dice_loss(prob, masks)

        return loss, logits

    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        num_epochs:   int,
    ) -> float:
        """Train for num_epochs. Returns the final epoch's average train loss."""
        final_train_loss = 0.0

        for epoch in range(num_epochs):
            self.model.train()
            total_loss, n = 0.0, 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
            for imgs, masks in pbar:
                imgs  = imgs.to(self.device)
                masks = masks.to(self.device).float()
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)

                self.optimizer.zero_grad(set_to_none=True)
                loss, _ = self._forward(imgs, masks, epoch=epoch)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                bs          = imgs.size(0)
                total_loss += loss.item() * bs
                n          += bs
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            epoch_loss       = total_loss / max(n, 1)
            final_train_loss = epoch_loss

            val_loss, val_dice, val_iou = self.evaluate_epoch(val_loader)
            print(
                f"  Ep {epoch+1:>3}/{num_epochs}  "
                f"train={epoch_loss:.4f}  val={val_loss:.4f}  "
                f"dice={val_dice:.4f}  iou={val_iou:.4f}"
            )

            if self.writer is not None:
                self.writer.add_scalar("Loss/train", epoch_loss, epoch)
                self.writer.add_scalar("Loss/val",   val_loss,   epoch)
                self.writer.add_scalar("Dice/val",   val_dice,   epoch)
                self.writer.add_scalar("IoU/val",    val_iou,    epoch)

        return final_train_loss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_epoch(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """Evaluate one pass over val_loader. Returns (val_loss, val_dice, val_iou)."""
        self.model.eval()
        total_loss = dice_sum = iou_sum = 0.0
        n = 0

        for imgs, masks in val_loader:
            imgs  = imgs.to(self.device)
            masks = masks.to(self.device).float()
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            loss, logits = self._forward(imgs, masks)

            bs          = imgs.size(0)
            total_loss += loss.item() * bs
            m           = compute_metrics(logits, masks)
            dice_sum   += m["dice"] * bs
            iou_sum    += m["mIoU"] * bs
            n          += bs

        denom = max(n, 1)
        return total_loss / denom, dice_sum / denom, iou_sum / denom