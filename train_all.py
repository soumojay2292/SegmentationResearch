import sys
import os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(1, os.path.abspath("src/sam2"))

import csv
import json
import datetime
import subprocess
import webbrowser
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image

from models.maffnet import MAFFNet
from models.attention_unet import AttentionUNet
from models.unet_plus_plus import UNetPlusPlus
from models.unet import UNet
from models.transunet import TransUNet
from training.trainer import MAFFNetTrainer
from training.simple_trainer import SimpleTrainer


EPOCHS = 20
MAFFNET_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"
DATASETS = ["ISIC_2016", "ISIC_2017", "ISIC_2018"]


def safe_scalar(x) -> float:
    """Convert any scalar-like value (tensor, list, tuple, float) to a Python float."""
    if isinstance(x, (list, tuple)):
        x = x[0]
    if torch.is_tensor(x):
        x = x.mean().item()
    return float(x)


# ---------------------------------------------------------------------------
# LogitWrapper (same as train.py)
# ---------------------------------------------------------------------------

class _LogitWrapper(torch.nn.Module):
    """Converts probability outputs (sigmoid already applied) to logits."""
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.model(x).clamp(1e-6, 1 - 1e-6)
        return torch.logit(probs)


# ---------------------------------------------------------------------------
# Logging tee
# ---------------------------------------------------------------------------

class _Tee:
    """Mirror stdout to both the console and a per-run log file."""

    def __init__(self, log_path: str):
        self._log_path = log_path
        self._file = None
        self._orig = None

    def __enter__(self):
        os.makedirs("logs", exist_ok=True)
        self._file = open(self._log_path, "w", encoding="utf-8")
        self._orig = sys.stdout
        sys.stdout = self
        return self

    def write(self, data: str):
        self._orig.write(data)
        self._file.write(data)

    def flush(self):
        self._orig.flush()
        self._file.flush()

    def __exit__(self, *_):
        sys.stdout = self._orig
        if self._file:
            self._file.close()


# ---------------------------------------------------------------------------
# Dataset (same as train.py — uses csv_path + split convention)
# ---------------------------------------------------------------------------

class SegDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, split: str = "train", img_size: int = 384):
        import csv as _csv

        self.split    = split
        self.img_size = img_size
        self.samples  = []

        dataset_root  = Path(csv_path).parent
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
# Dashboard generator
# ---------------------------------------------------------------------------

def _generate_dashboard(summary_csv: str, out_html: str) -> None:
    """Read summary.csv and write a self-contained HTML results dashboard."""

    def _safe_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    rows = []
    try:
        with open(summary_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        return
    if not rows:
        return

    os.makedirs(os.path.dirname(out_html), exist_ok=True)

    table_rows_html = ""
    for r in rows:
        dice = _safe_float(r.get("Final Val Dice"))
        bg   = f"hsl({int(dice * 120)}, 55%, 93%)"
        table_rows_html += (
            f"<tr style='background:{bg}'>"
            f"<td>{r.get('Model','')}</td>"
            f"<td>{r.get('Dataset','')}</td>"
            f"<td>{r.get('Final Train Loss','')}</td>"
            f"<td>{r.get('Final Val Loss','')}</td>"
            f"<td><b>{r.get('Final Val Dice','')}</b></td>"
            f"<td>{r.get('Final Val IoU','')}</td>"
            f"<td style='font-size:.75em'>{r.get('Checkpoint Path','')}</td>"
            "</tr>\n"
        )

    chart_labels = json.dumps([f"{r['Model']} / {r['Dataset']}" for r in rows])
    chart_dice   = json.dumps([_safe_float(r.get("Final Val Dice")) for r in rows])
    chart_iou    = json.dumps([_safe_float(r.get("Final Val IoU"))  for r in rows])
    generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<title>Segmentation Research Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body  {{ font-family: Arial, sans-serif; margin: 24px; background: #f4f6f8; color: #222; }}
  h1    {{ margin-bottom: 4px; }}
  .meta {{ color: #666; font-size: .88em; margin-bottom: 28px; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff;
           box-shadow: 0 1px 6px rgba(0,0,0,.12); border-radius: 6px;
           overflow: hidden; margin-bottom: 36px; }}
  th    {{ background: #2c3e50; color: #fff; padding: 10px 14px; text-align: left; }}
  td    {{ padding: 8px 14px; border-bottom: 1px solid #e8e8e8; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ filter: brightness(.96); }}
  .chart-box {{ background: #fff; border-radius: 8px;
                box-shadow: 0 1px 6px rgba(0,0,0,.12);
                padding: 24px; max-width: 960px; }}
</style>
</head>
<body>
<h1>Segmentation Research Dashboard</h1>
<p class="meta">Generated: {generated_at} &nbsp;|&nbsp; Source: {summary_csv}</p>

<table>
<thead>
  <tr>
    <th>Model</th><th>Dataset</th>
    <th>Train Loss</th><th>Val Loss</th>
    <th>Dice ↑</th><th>IoU ↑</th>
    <th>Checkpoint</th>
  </tr>
</thead><tbody>
{table_rows_html}</tbody>
</table>

<div class="chart-box">
  <canvas id="metricsChart"></canvas>
</div>

<script>
new Chart(document.getElementById("metricsChart"), {{
  type: "bar",
  data: {{
    labels: {chart_labels},
    datasets: [
      {{
        label: "Val Dice",
        data: {chart_dice},
        backgroundColor: "rgba(52,152,219,.75)",
        borderColor:     "rgba(52,152,219,1)",
        borderWidth: 1
      }},
      {{
        label: "Val IoU",
        data: {chart_iou},
        backgroundColor: "rgba(46,204,113,.75)",
        borderColor:     "rgba(46,204,113,1)",
        borderWidth: 1
      }}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ position: "top" }},
      title:  {{ display: true, text: "Validation Metrics — All Models & Datasets" }}
    }},
    scales: {{ y: {{ beginAtZero: true, max: 1.0,
                     title: {{ display: true, text: "Score" }} }} }}
  }}
}});
</script>
</body></html>"""

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all():
    timestamp   = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    summary_path = os.path.join(results_dir, "summary.csv")
    with open(summary_path, mode="w", newline="") as f:
        csv.writer(f).writerow([
            "Model", "Dataset", "Final Train Loss",
            "Final Val Loss", "Final Val Dice", "Final Val IoU",
            "Checkpoint Path",
        ])

    for dataset_name in DATASETS:
        print(f"\n📂 Starting dataset: {dataset_name}\n")
        base_path = f"dataset_split/{dataset_name}"

        train_ds = SegDataset(f"{base_path}/train.csv", split="train", img_size=384)
        val_ds   = SegDataset(f"{base_path}/val.csv",   split="val",   img_size=384)

        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,
                                  num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=2, shuffle=False,
                                  num_workers=0, pin_memory=True)

        model_names = ["unet", "attention_unet", "unetpp", "transunet", "maffnet"]

        for model_name in model_names:
            print(f"\n===== {model_name} | {dataset_name} =====")
            print(f"🚀 Training {model_name} on {dataset_name}...\n")

            log_path = os.path.join("logs", f"{model_name}_{dataset_name}.txt")
            exp_dir  = Path("experiments") / f"{model_name}_{dataset_name}_{timestamp}"
            exp_dir.mkdir(parents=True, exist_ok=True)

            save_dir = str(exp_dir / "checkpoints")

            config = {
                "model":      model_name,
                "dataset":    dataset_name,
                "epochs":     EPOCHS,
                "batch_size": 2,
                "lr":         1e-3,
                "img_size":   384,
            }

            try:
                with _Tee(log_path):
                    if model_name == "maffnet":
                        print("Building MAFFNet with SAM2 Hiera-Large backbone …")
                        model   = MAFFNet(checkpoint=MAFFNET_CHECKPOINT)
                        trainer = MAFFNetTrainer(
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            save_dir=save_dir,
                            num_epochs=EPOCHS,
                            lr=1e-3,
                            weight_decay=0.01,
                        )
                    elif model_name == "unet":
                        print("Building U-Net …")
                        model   = _LogitWrapper(UNet())
                        trainer = SimpleTrainer(
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            model_name="unet",
                            save_dir=save_dir,
                            num_epochs=EPOCHS,
                            lr=1e-3,
                            weight_decay=0.01,
                        )
                    elif model_name == "attention_unet":
                        print("Building Attention U-Net …")
                        model   = AttentionUNet()
                        trainer = SimpleTrainer(
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            model_name="attention_unet",
                            save_dir=save_dir,
                            num_epochs=EPOCHS,
                            lr=1e-3,
                            weight_decay=0.01,
                        )
                    elif model_name == "unetpp":
                        print("Building UNet++ …")
                        model   = UNetPlusPlus()
                        trainer = SimpleTrainer(
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            model_name="unetpp",
                            save_dir=save_dir,
                            num_epochs=EPOCHS,
                            lr=1e-3,
                            weight_decay=0.01,
                        )
                    elif model_name == "transunet":
                        print("Building TransUNet …")
                        model   = TransUNet()
                        trainer = SimpleTrainer(
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            model_name="transunet",
                            save_dir=save_dir,
                            num_epochs=EPOCHS,
                            lr=1e-3,
                            weight_decay=0.01,
                        )
                    else:
                        raise ValueError(f"Unknown model: {model_name}")

                    trainer.train(config=config, exp_dir=exp_dir)

                    # Final validation pass to collect summary metrics
                    if model_name == "maffnet":
                        val_loss, val_metrics = trainer._val_epoch(epoch=0)
                    else:
                        val_loss, val_metrics = trainer._val_epoch()

                    val_loss = safe_scalar(val_loss)
                    val_dice = safe_scalar(val_metrics["dice"])
                    val_iou  = safe_scalar(val_metrics["mIoU"])

                    ckpt_path = str(exp_dir / "model.pth")

                    print(f"✅ Experiment saved: {exp_dir}\n")

                    with open(summary_path, mode="a", newline="") as f:
                        csv.writer(f).writerow([
                            model_name, dataset_name,
                            "N/A",
                            f"{val_loss:.4f}",
                            f"{val_dice:.4f}",
                            f"{val_iou:.4f}",
                            ckpt_path,
                        ])

                    metrics = {
                        "model":    model_name,
                        "dataset":  dataset_name,
                        "val_loss": val_loss,
                        "dice":     val_dice,
                        "iou":      val_iou,
                    }
                    json_path = os.path.join(
                        results_dir, f"{model_name}_{dataset_name}_metrics.json"
                    )
                    with open(json_path, "w", encoding="utf-8") as jf:
                        json.dump(metrics, jf, indent=2)

            except Exception as e:
                print(f"[ERROR] {model_name} on {dataset_name} failed: {e}")
                continue
            finally:
                torch.cuda.empty_cache()

    print(f"\n🎉 All models trained on all datasets!")
    print(f"📁 Results saved at: {results_dir}\n")

    # --- Dashboard generation ---
    dashboard_dir  = os.path.join("results", "dashboard")
    dashboard_html = os.path.join(dashboard_dir, "segmentation_research_dashboard.html")
    gen_script     = os.path.join(dashboard_dir, "generate_dashboard.py")

    try:
        if os.path.isfile(gen_script):
            subprocess.run([sys.executable, gen_script, summary_path], check=True)
        else:
            _generate_dashboard(summary_path, dashboard_html)
        print("✅ Dashboard generated successfully.")
        print(f"   → {os.path.abspath(dashboard_html)}")
        webbrowser.open(os.path.abspath(dashboard_html))
    except Exception as _dash_err:
        print(f"[WARN] Dashboard generation failed: {_dash_err}")

    print("\n📊 Launching TensorBoard...")
    subprocess.Popen(["tensorboard", "--logdir", "runs"])


if __name__ == "__main__":
    run_all()
