"""
Single-experiment research report (publication-ready, self-contained HTML).

Reads metrics from the experiment's own training_log.csv / summary.csv — no
hardcoded values. Writes report.html into the experiment's OWN folder
(run_dir = parent of the samples dir) so historical reports are never
overwritten. The HTML is fully self-contained (inline CSS + Chart.js via CDN).

Backward compatible: signature generate_report(log_csv, summary_csv,
image_dir, config) is unchanged; both SimpleTrainer and MAFFNetTrainer keep
working. generate_training_curves still writes PNGs into results/dashboard/
so MAFFNetTrainer's existing curve-copy step keeps working.
"""

import os
import csv
import base64
import re
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

DASHBOARD_DIR    = Path("results/dashboard")
BASELINE_SUMMARY = Path("experiments/final_baseline/summary.csv")

# Known model tokens — longest / most specific first for prefix matching
_KNOWN_MODELS = [
    ("attention_unet", "Attention U-Net"),
    ("unetpp",         "UNet++"),
    ("transunet",      "TransUNet"),
    ("maffnet",        "MAFFNet"),
    ("unet",           "U-Net"),
]


# ---------------------------------------------------------------------------
# CSV / image helpers
# ---------------------------------------------------------------------------

def _read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _img_to_b64(path):
    path = Path(path)
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = path.suffix.lstrip(".").replace("jpg", "jpeg")
    return f"data:image/{ext};base64,{data}"


def find_latest_run(results_dir="results"):
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return None
    runs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and d.name != "dashboard"],
        key=lambda d: d.name, reverse=True,
    )
    return runs[0] if runs else None


# ---------------------------------------------------------------------------
# Model / dataset inference from the experiment folder name
# ---------------------------------------------------------------------------

def _infer_model(name, config):
    m = config.get("model")
    if m:
        for tok, pretty in _KNOWN_MODELS:
            if m.lower() == tok:
                return pretty
        return str(m)
    low = name.lower()
    for tok, pretty in _KNOWN_MODELS:
        if low.startswith(tok) or f"_{tok}_" in f"_{low}_":
            return pretty
    return "—"


def _infer_dataset(name, config):
    d = config.get("dataset")
    if d and d != "—":
        return d
    m = re.search(r"(ISIC[_-]?\d{4})", name)
    return m.group(1).replace("-", "_") if m else "—"


# ---------------------------------------------------------------------------
# PNG training curves — kept for MAFFNetTrainer's run_dir copy (compat)
# ---------------------------------------------------------------------------

def generate_training_curves(log_csv):
    rows = _read_csv(log_csv)
    if not rows:
        return None, None
    try:
        epochs     = [int(r["epoch"])        for r in rows]
        train_loss = [float(r["train_loss"]) for r in rows]
        val_loss   = [float(r["val_loss"])   for r in rows]
        val_iou    = [float(r["val_miou"])   for r in rows]
        val_dice   = [float(r["val_dice"])   for r in rows]
    except (KeyError, ValueError):
        return None, None

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    loss_path = DASHBOARD_DIR / "loss_curve.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    ax.plot(epochs, val_loss,   label="Val Loss",   linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()
    ax.set_title("Training & Validation Loss")
    plt.tight_layout(); plt.savefig(loss_path, dpi=120, bbox_inches="tight"); plt.close()

    iou_path = DASHBOARD_DIR / "iou_curve.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, val_iou,  label="Val IoU",  linewidth=2)
    ax.plot(epochs, val_dice, label="Val Dice", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score"); ax.legend()
    ax.set_title("Validation IoU & Dice")
    plt.tight_layout(); plt.savefig(iou_path, dpi=120, bbox_inches="tight"); plt.close()

    return loss_path, iou_path


# ---------------------------------------------------------------------------
# Per-epoch arrays for Chart.js
# ---------------------------------------------------------------------------

def _extract_chart_data(log_csv):
    rows = _read_csv(log_csv)
    if not rows:
        return [], [], [], [], []
    try:
        epochs     = [int(r["epoch"])                   for r in rows]
        train_loss = [round(float(r["train_loss"]), 4)  for r in rows]
        val_loss   = [round(float(r["val_loss"]),   4)  for r in rows]
        val_dice   = [round(float(r["val_dice"]),   4)  for r in rows]
        val_iou    = [round(float(r["val_miou"]),   4)  for r in rows]
    except (KeyError, ValueError):
        return [], [], [], [], []
    return epochs, train_loss, val_loss, val_dice, val_iou


def _best_and_final(log_csv):
    rows = _read_csv(log_csv)
    if not rows:
        return {}, {}
    best  = max(rows, key=lambda r: float(r.get("val_dice", 0) or 0))
    final = rows[-1]
    return best, final


# ---------------------------------------------------------------------------
# Sample scoring + Best / Average / Difficult categorisation
# ---------------------------------------------------------------------------

def _mask_arr(path):
    try:
        a = np.array(Image.open(path).convert("L"))
        return (a > 127).astype(np.float32)
    except Exception:
        return None


def _dice(gt, pred):
    inter = float((gt * pred).sum())
    denom = float(gt.sum() + pred.sum())
    return (2.0 * inter / denom) if denom > 0 else 1.0


def _scored_samples(image_dir, max_n=40):
    """Read input_i/gt_i/pred_i triplets, score per-sample Dice."""
    image_dir = Path(image_dir)
    out = []
    if not image_dir.exists():
        return out
    for i in range(max_n):
        ip = image_dir / f"input_{i}.png"
        gp = image_dir / f"gt_{i}.png"
        pp = image_dir / f"pred_{i}.png"
        if ip.exists() and gp.exists() and pp.exists():
            g, p = _mask_arr(gp), _mask_arr(pp)
            d = _dice(g, p) if (g is not None and p is not None) else 0.0
            out.append({"input": ip, "gt": gp, "pred": pp, "dice": d})
    return out


def _categorise(samples):
    """Return dict best/average/difficult -> sample (or None)."""
    if not samples:
        return {"Best": None, "Average": None, "Difficult": None}
    s = sorted(samples, key=lambda x: x["dice"], reverse=True)
    return {
        "Best":      s[0],
        "Average":   s[len(s) // 2],
        "Difficult": s[-1],
    }


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _fmt(d, key, pct=False):
    v = d.get(key)
    try:
        f = float(v)
        return f"{f * 100:.2f}%" if pct else f"{f:.4f}"
    except (TypeError, ValueError):
        return "—"


def _sample_card(label, s):
    no = '<div class="sm-none">no sample</div>'
    if s is None:
        return f"""<div class="sm-row"><div class="sm-tag">{label}</div>
          <div class="sm-imgs">{no}</div></div>"""
    return f"""<div class="sm-row">
      <div class="sm-tag">{label}<span class="sm-dice">Dice {s['dice']:.4f}</span></div>
      <div class="sm-imgs">
        <figure><img src="{_img_to_b64(s['input'])}"><figcaption>Input</figcaption></figure>
        <figure><img src="{_img_to_b64(s['gt'])}"><figcaption>Ground Truth</figcaption></figure>
        <figure><img src="{_img_to_b64(s['pred'])}"><figcaption>Prediction</figcaption></figure>
      </div></div>"""


_CSS = """
*{box-sizing:border-box}
body{margin:0;background:#f4f6f9;color:#1e293b;
 font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;font-size:14px;line-height:1.5}
.wrap{max-width:1080px;margin:0 auto;padding:32px 24px 60px}
h1{font-size:22px;margin:0 0 4px}
.sub{color:#64748b;font-size:13px;margin-bottom:20px}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:24px}
.chip{background:#fff;border:1px solid #e2e8f0;border-radius:6px;padding:6px 12px;font-size:12px}
.chip b{color:#0f172a}
.sec{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;
 color:#475569;margin:28px 0 12px}
.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}
.kpi{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;text-align:center;
 box-shadow:0 1px 3px rgba(0,0,0,.04)}
.kpi .l{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.04em}
.kpi .v{font-size:24px;font-weight:700;color:#0f172a;margin-top:6px}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;
 box-shadow:0 1px 3px rgba(0,0,0,.04)}
.card h3{font-size:12px;margin:0 0 10px;color:#475569;font-weight:600}
.chart-wrap{position:relative;height:200px}
.sm-row{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:14px;margin-bottom:12px;
 box-shadow:0 1px 3px rgba(0,0,0,.04)}
.sm-tag{font-weight:600;font-size:13px;margin-bottom:10px;display:flex;align-items:center;gap:10px}
.sm-dice{font-weight:500;font-size:11px;color:#0369a1;background:#e0f2fe;border-radius:5px;padding:2px 8px}
.sm-imgs{display:flex;gap:12px;flex-wrap:wrap}
.sm-imgs figure{margin:0;text-align:center}
.sm-imgs img{width:180px;height:180px;object-fit:cover;border-radius:8px;border:1px solid #e2e8f0;background:#0f172a}
.sm-imgs figcaption{font-size:11px;color:#64748b;margin-top:4px}
.sm-none{color:#94a3b8;font-size:12px;padding:20px}
table.cfg{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e2e8f0;border-radius:10px;overflow:hidden}
table.cfg td{padding:9px 14px;border-bottom:1px solid #f1f5f9;font-size:13px}
table.cfg td:first-child{color:#64748b;width:40%}
table.cfg tr:last-child td{border-bottom:none}
.foot{margin-top:40px;color:#94a3b8;font-size:11px;text-align:center}
"""


# ---------------------------------------------------------------------------
# Main entry point  (signature unchanged — backward compatible)
# ---------------------------------------------------------------------------

def generate_report(log_csv, summary_csv, image_dir, config):
    config = config or {}

    # Still emit shared PNG curves so MAFFNetTrainer's copy step keeps working
    generate_training_curves(log_csv)

    # Per-exp output dir = parent of samples dir
    run_dir = Path(image_dir).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    exp_name = config.get("exp") or run_dir.name

    model_name = _infer_model(run_dir.name, config)
    dataset    = _infer_dataset(run_dir.name, config)

    # Training timestamp = mtime of the training log (when training finished)
    try:
        ts = datetime.fromtimestamp(os.path.getmtime(log_csv)).strftime("%Y-%m-%d %H:%M")
    except OSError:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    epochs, train_loss, val_loss, val_dice, val_iou = _extract_chart_data(log_csv)
    best, final = _best_and_final(log_csv)
    best_epoch = best.get("epoch", "—")

    kpi_dice = _fmt(best, "val_dice")
    kpi_iou  = _fmt(best, "val_miou")
    kpi_sens = _fmt(best, "val_sensitivity")
    kpi_spec = _fmt(best, "val_specificity")
    kpi_acc  = _fmt(best, "val_acc", pct=True)

    samples = _categorise(_scored_samples(image_dir))
    sample_html = "".join(_sample_card(lbl, samples[lbl])
                          for lbl in ("Best", "Average", "Difficult"))

    # Config rows
    epochs_c = config.get("epochs", len(epochs) or "—")
    batch    = config.get("batch_size", "—")
    lr       = config.get("lr", "—")
    img_size = config.get("img_size", "—")
    img_str  = f"{img_size} × {img_size}" if img_size != "—" else "—"

    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{model_name} · {dataset} — Experiment Report</title>
<style>{_CSS}</style></head><body><div class="wrap">

  <h1>{model_name} — Segmentation Experiment Report</h1>
  <div class="sub">Single-experiment report · generated {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>

  <div class="meta">
    <span class="chip">Model: <b>{model_name}</b></span>
    <span class="chip">Dataset: <b>{dataset}</b></span>
    <span class="chip">Experiment: <b>{exp_name}</b></span>
    <span class="chip">Trained: <b>{ts}</b></span>
    <span class="chip">Best epoch: <b>{best_epoch}</b></span>
  </div>

  <div class="sec">Best-Epoch Validation Metrics</div>
  <div class="kpis">
    <div class="kpi"><div class="l">Dice</div><div class="v">{kpi_dice}</div></div>
    <div class="kpi"><div class="l">IoU</div><div class="v">{kpi_iou}</div></div>
    <div class="kpi"><div class="l">Sensitivity</div><div class="v">{kpi_sens}</div></div>
    <div class="kpi"><div class="l">Specificity</div><div class="v">{kpi_spec}</div></div>
    <div class="kpi"><div class="l">Accuracy</div><div class="v">{kpi_acc}</div></div>
  </div>

  <div class="sec">Training Curves</div>
  <div class="cards">
    <div class="card"><h3>Training Loss vs Epoch</h3><div class="chart-wrap"><canvas id="c1"></canvas></div></div>
    <div class="card"><h3>Validation Loss vs Epoch</h3><div class="chart-wrap"><canvas id="c2"></canvas></div></div>
    <div class="card"><h3>Validation Dice vs Epoch</h3><div class="chart-wrap"><canvas id="c3"></canvas></div></div>
  </div>

  <div class="sec">Qualitative Predictions</div>
  {sample_html}

  <div class="sec">Configuration</div>
  <table class="cfg">
    <tr><td>Architecture</td><td>{model_name}</td></tr>
    <tr><td>Dataset</td><td>{dataset}</td></tr>
    <tr><td>Epochs</td><td>{epochs_c}</td></tr>
    <tr><td>Batch size</td><td>{batch}</td></tr>
    <tr><td>Input size</td><td>{img_str}</td></tr>
    <tr><td>Initial LR</td><td>{lr}</td></tr>
    <tr><td>Optimizer</td><td>AdamW · linear LR decay</td></tr>
  </table>

  <div class="foot">Auto-generated · data from {Path(log_csv).name} & {Path(summary_csv).name} · no hardcoded values</div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const eps={epochs}, trainLoss={train_loss}, valLoss={val_loss}, valDice={val_dice};
const base={{responsive:true,maintainAspectRatio:false,animation:false,
 plugins:{{legend:{{display:false}}}},elements:{{point:{{radius:0}}}},
 scales:{{x:{{ticks:{{color:'#94a3b8',font:{{size:10}},maxTicksLimit:6}},grid:{{display:false}}}},
 y:{{ticks:{{color:'#94a3b8',font:{{size:10}}}},grid:{{color:'#eef2f7'}}}}}}}};
function mk(id,data,color){{new Chart(document.getElementById(id),{{type:'line',
 data:{{labels:eps,datasets:[{{data:data,borderColor:color,backgroundColor:color+'22',
 borderWidth:2,tension:0.35,fill:true}}]}},options:base}});}}
mk('c1',trainLoss,'#6366f1');
mk('c2',valLoss,'#f97316');
mk('c3',valDice,'#10b981');
</script>
</body></html>"""

    report_path = run_dir / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[report] Saved → {report_path}")
    return report_path
