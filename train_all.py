import sys
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
from segment_anything import sam_model_registry
from models.maffnet import MAFFNet
from models.attention_unet import AttentionUNet
from models.unet_plus_plus import UNetPlusPlus as UNetPP
import json
import webbrowser


EPOCHS = 20


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


# Models to train
def create_maffnet():
    sam_encoder = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    return MAFFNet(encoder=sam_encoder)
print("Encoder loaded successfully!")
MODELS = {
    "unet":           lambda: UNet(),
    "maffnet":        create_maffnet,
    "transunet":      lambda: TransUNet(in_ch=3, out_ch=1, embed_dim=64, num_heads=2),
    "attention_unet": lambda: AttentionUNet(),
    "unetpp":         lambda: UNetPP(),
}

# Datasets to train on
DATASETS = ["ISIC_2016", "ISIC_2017", "ISIC_2018"]


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

    # Table rows — background tinted green→red by Dice score
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
            image_size=224
        )

        val_ds = SegDataset(
            val_csv,
            f"{base_path}/val/images",
            f"{base_path}/val/masks",
            image_size=224
        )


        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)

        for model_name, model_fn in MODELS.items():
            print(f"\n===== {model_name} | {dataset_name} =====")
            print(f"🚀 Training {model_name} on {dataset_name}...\n")

            log_path = os.path.join("logs", f"{model_name}_{dataset_name}.txt")
            try:
                with _Tee(log_path):
                    model = model_fn().to(device)
                    optimizer = Adam(model.parameters(), lr=1e-4)

                    # Each run gets its own trainer + log directory
                    trainer = Trainer(model, optimizer, device)
                    trainer.scaler = GradScaler()
                    trainer.writer = SummaryWriter(
                        log_dir=f"runs/{model_name}_{dataset_name}"
                    )

                    train_loss = trainer.fit(train_loader, val_loader, num_epochs=EPOCHS)

                    # Save checkpoint
                    ckpt_path = os.path.join(results_dir, f"{model_name}_{dataset_name}.pth")
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"✅ Saved checkpoint: {ckpt_path}\n")

                    val_loss, val_dice, val_iou = trainer.evaluate_epoch(val_loader)
                    with open(summary_path, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([model_name, dataset_name, train_loss, val_loss, val_dice, val_iou, ckpt_path])

                    # Per-run structured metrics
                    metrics = {
                        "model":      model_name,
                        "dataset":    dataset_name,
                        "train_loss": float(train_loss) if train_loss is not None else None,
                        "val_loss":   float(val_loss)   if val_loss   is not None else None,
                        "dice":       float(val_dice)   if val_dice   is not None else None,
                        "iou":        float(val_iou)    if val_iou    is not None else None,
                    }
                    json_path = os.path.join(results_dir, f"{model_name}_{dataset_name}_metrics.json")
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

    # ✅ Launch TensorBoard automatically
    print("\n📊 Launching TensorBoard...")
    subprocess.Popen(["tensorboard", "--logdir", "runs"])


if __name__ == "__main__":
    run_all()
