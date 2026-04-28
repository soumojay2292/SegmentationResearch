"""
Research-grade HTML dashboard for MAFFNet segmentation.
Reads all metrics from real CSV files — no hardcoded values.
Generates report.html using the new dashboard UI (style.css + Chart.js).
"""

import csv
import base64
import webbrowser
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DASHBOARD_DIR    = Path("results/dashboard")
BASELINE_SUMMARY = Path("experiments/final_baseline/summary.csv")


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


# ---------------------------------------------------------------------------
# Auto-detect latest run folder
# ---------------------------------------------------------------------------

def find_latest_run(results_dir="results"):
    results_dir = Path(results_dir)
    runs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and d.name != "dashboard"],
        key=lambda d: d.name,
        reverse=True,
    )
    return runs[0] if runs else None


# ---------------------------------------------------------------------------
# PNG training curves — kept for experiment folder copy
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

    def _dark_axes(ax, fig, title):
        ax.set_facecolor("#1e293b")
        fig.patch.set_facecolor("#0f172a")
        ax.tick_params(colors="#94a3b8")
        ax.xaxis.label.set_color("#94a3b8")
        ax.yaxis.label.set_color("#94a3b8")
        ax.title.set_color("#38bdf8")
        for sp in ax.spines.values():
            sp.set_edgecolor("#334155")
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
        ax.set_title(title)

    loss_path = DASHBOARD_DIR / "loss_curve.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_loss, label="Train Loss", color="#38bdf8", linewidth=2)
    ax.plot(epochs, val_loss,   label="Val Loss",   color="#f97316", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend()
    _dark_axes(ax, fig, "Training & Validation Loss")
    plt.tight_layout()
    plt.savefig(loss_path, dpi=120, bbox_inches="tight")
    plt.close()

    iou_path = DASHBOARD_DIR / "iou_curve.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, val_iou,  label="Val IoU",  color="#22c55e", linewidth=2)
    ax.plot(epochs, val_dice, label="Val Dice", color="#a78bfa", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.legend()
    _dark_axes(ax, fig, "Validation IoU & Dice")
    plt.tight_layout()
    plt.savefig(iou_path, dpi=120, bbox_inches="tight")
    plt.close()

    return loss_path, iou_path


# ---------------------------------------------------------------------------
# Per-epoch arrays for Chart.js injection
# ---------------------------------------------------------------------------

def _extract_chart_data(log_csv):
    rows = _read_csv(log_csv)
    if not rows:
        return [], [], [], [], []
    try:
        epochs     = [int(r["epoch"])              for r in rows]
        train_loss = [round(float(r["train_loss"]), 4) for r in rows]
        val_loss   = [round(float(r["val_loss"]),   4) for r in rows]
        val_dice   = [round(float(r["val_dice"]),   4) for r in rows]
        val_iou    = [round(float(r["val_miou"]),   4) for r in rows]
    except (KeyError, ValueError):
        return [], [], [], [], []
    return epochs, train_loss, val_loss, val_dice, val_iou


# ---------------------------------------------------------------------------
# Best / final epoch rows from training log
# ---------------------------------------------------------------------------

def _best_and_final(log_csv):
    rows = _read_csv(log_csv)
    if not rows:
        return {}, {}
    best  = max(rows, key=lambda r: float(r.get("val_dice", 0)))
    final = rows[-1]
    return best, final


# ---------------------------------------------------------------------------
# Prediction samples
# ---------------------------------------------------------------------------

def _collect_samples(image_dir, n=5):
    image_dir = Path(image_dir)
    samples = []
    if not image_dir.exists():
        return samples
    for i in range(n):
        ip = image_dir / f"input_{i}.png"
        gp = image_dir / f"gt_{i}.png"
        pp = image_dir / f"pred_{i}.png"
        if ip.exists() and gp.exists() and pp.exists():
            samples.append((ip, gp, pp))
    if not samples:
        ip = image_dir / "input.png"
        gp = image_dir / "gt.png"
        pp = image_dir / "pred.png"
        if ip.exists() and gp.exists() and pp.exists():
            samples.append((ip, gp, pp))
    return samples


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def _read_baseline_summary(path=BASELINE_SUMMARY):
    path = Path(path)
    if not path.exists():
        return {}
    rows = _read_csv(path)
    if not rows:
        return {}
    row0 = rows[0]
    if "metric" in row0:
        return {r["metric"]: r["value"] for r in rows}
    return dict(row0)


# ---------------------------------------------------------------------------
# HTML fragment helpers
# ---------------------------------------------------------------------------

def _safe(d, k, fmt=".4f"):
    v = d.get(k)
    return f"{float(v):{fmt}}" if v is not None else "—"


def _delta_html(curr_val, base_val, label="vs baseline"):
    """KPI delta line: green if improved, red if regressed."""
    try:
        delta = float(curr_val) - float(base_val)
        sign  = "↑" if delta >= 0 else "↓"
        cls   = "up" if delta >= 0 else "dn"
        return f'<div class="kpi-delta {cls}">{sign} {delta:+.4f} {label}</div>'
    except (TypeError, ValueError):
        return '<div class="kpi-delta up">↑ Best validation epoch</div>'


def _cmp_row(label, base_val, curr_val):
    """One comparison bar row — baseline (dark) vs current (accent)."""
    try:
        bv = float(base_val)
        bw = f"{min(bv * 100, 100):.2f}"
        bs = f"{bv:.4f}"
    except (TypeError, ValueError):
        bw, bs = "0", "—"

    try:
        cv = float(curr_val)
        cw = f"{min(cv * 100, 100):.2f}"
        cs = f"{cv:.4f}"
    except (TypeError, ValueError):
        cw, cs = "0", "—"

    return f"""
      <div class="cmp-row">
        <span class="cmp-metric">{label}</span>
        <div class="cmp-bar-wrap">
          <div class="cmp-bar bar-base" style="width:{bw}%"></div>
          <div class="cmp-bar bar-imp"  style="width:{cw}%"></div>
        </div>
        <div class="cmp-vals"><span class="v-base">{bs}</span><span class="v-imp">{cs}</span></div>
      </div>"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_report(log_csv, summary_csv, image_dir, config):
    """
    log_csv     : path to checkpoints/training_log.csv
    summary_csv : path to results/<run>/summary.csv
    image_dir   : directory containing input_N / gt_N / pred_N PNGs
    config      : dict — dataset, epochs, batch_size, lr, img_size,
                         total_params, trainable_params
    """
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    # PNG curves (copied to experiment folder by trainer)
    generate_training_curves(log_csv)

    # Per-epoch arrays → injected into Chart.js
    epochs, train_loss, val_loss, val_dice, val_iou = _extract_chart_data(log_csv)
    num_epochs = len(epochs)

    # Best / final rows
    best, final = _best_and_final(log_csv)

    # Fallback: populate from summary.csv if training log is empty
    if not best and summary_csv:
        summary_rows = _read_csv(summary_csv)
        if summary_rows:
            row0 = summary_rows[0]
            sd = ({r["metric"]: r["value"] for r in summary_rows}
                  if "metric" in row0 else dict(row0))
            key_map = {
                "val_dice": "val_dice", "val_miou": "val_miou",
                "val_loss": "val_loss", "train_loss": "train_loss",
                "acc": "val_acc", "sensitivity": "val_sensitivity",
                "specificity": "val_specificity",
            }
            best = final = {key_map[k]: v for k, v in sd.items() if k in key_map}

    best_epoch  = best.get("epoch",  "—")
    final_epoch = final.get("epoch", num_epochs if num_epochs else "—")

    # KPI values
    kpi_dice = _safe(best,  "val_dice",  ".4f")
    kpi_iou  = _safe(best,  "val_miou",  ".4f")
    kpi_loss = f"{min(val_loss):.4f}" if val_loss else _safe(final, "val_loss", ".4f")
    try:
        kpi_acc = f"{float(best.get('val_acc', 0)) * 100:.2f}%"
    except (TypeError, ValueError):
        kpi_acc = "—"

    # Baseline for comparison bars + KPI deltas
    baseline     = _read_baseline_summary()
    base_label   = "final_baseline" if baseline else "—"
    curr_label   = f"Current (ep {best_epoch})"

    delta_dice = _delta_html(best.get("val_dice"), baseline.get("dice"), "vs baseline")
    delta_iou  = _delta_html(best.get("val_miou"), baseline.get("mIoU"), "vs baseline")

    cmp_dice = _cmp_row("Dice",        baseline.get("dice"),        best.get("val_dice"))
    cmp_iou  = _cmp_row("IoU",         baseline.get("mIoU"),        best.get("val_miou"))
    cmp_acc  = _cmp_row("Accuracy",    baseline.get("acc"),         best.get("val_acc"))
    cmp_sens = _cmp_row("Sensitivity", baseline.get("sensitivity"), best.get("val_sensitivity"))
    cmp_spec = _cmp_row("Specificity", baseline.get("specificity"), best.get("val_specificity"))

    # Chart y-axis mins (a little below the data min)
    try:
        dice_min = round(max(0.0, min(val_dice) - 0.05), 2)
        iou_min  = round(max(0.0, min(val_iou)  - 0.05), 2)
    except (ValueError, TypeError):
        dice_min, iou_min = 0.70, 0.60

    # Visual results — first sample, base64 for file:// compatibility
    samples = _collect_samples(image_dir, n=1)
    _no_img = '<p style="color:#475569;padding:20px;text-align:center;font-size:12px">No sample</p>'
    if samples:
        ip, gp, pp = samples[0]
        vis_input = f'<img src="{_img_to_b64(ip)}" alt="Input skin lesion image">'
        vis_gt    = f'<img src="{_img_to_b64(gp)}" alt="Ground truth mask">'
        vis_pred  = f'<img src="{_img_to_b64(pp)}" alt="MAFFNet prediction">'
    else:
        vis_input = vis_gt = vis_pred = _no_img

    # Config
    dataset      = config.get("dataset",          "—")
    epochs_c     = config.get("epochs",           num_epochs or "—")
    batch        = config.get("batch_size",       "—")
    lr           = config.get("lr",               "—")
    img_size     = config.get("img_size",         "—")
    img_size_str = f"{img_size} × {img_size}" if img_size != "—" else "—"
    ts           = datetime.now().strftime("%b %d %Y")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MAFFNet — Segmentation Research Dashboard</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
<div class="dash">

  <!-- ── Header ── -->
  <div class="hdr">
    <div class="hdr-left">
      <h1>MAFFNet — Segmentation Research Dashboard</h1>
      <p>Dataset: {dataset} &nbsp;·&nbsp; SAM2 Hiera-Large backbone &nbsp;·&nbsp; {epochs_c} epochs · {ts}</p>
    </div>
    <div style="display:flex;gap:8px;align-items:center">
      <span class="badge">Epoch {final_epoch} / {epochs_c}</span>
      <span class="badge" style="color:#34d399;border-color:#14532d;background:#0c2818">&#9679; Complete</span>
    </div>
  </div>

  <!-- ── KPI cards ── -->
  <div class="sec-label">Key Metrics — Validation Set (Best: Epoch {best_epoch})</div>
  <div class="kpi-grid">
    <div class="kpi">
      <div class="kpi-label">Dice Score</div>
      <div class="kpi-val">{kpi_dice}</div>
      {delta_dice}
    </div>
    <div class="kpi">
      <div class="kpi-label">Mean IoU</div>
      <div class="kpi-val">{kpi_iou}</div>
      {delta_iou}
    </div>
    <div class="kpi">
      <div class="kpi-label">Pixel Accuracy</div>
      <div class="kpi-val">{kpi_acc}</div>
      <div class="kpi-delta up">&#8593; Best validation epoch</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Val Loss</div>
      <div class="kpi-val">{kpi_loss}</div>
      <div class="kpi-delta up">&#8595; Lowest across {epochs_c} epochs</div>
    </div>
  </div>

  <!-- ── Comparison + Loss curve ── -->
  <div class="two-col">
    <div class="card">
      <div class="sec-label" style="margin-bottom:12px">Comparison: {base_label} vs {curr_label}</div>
      <div class="legend">
        <span><span class="leg-dot" style="background:#3b4275"></span>{base_label}</span>
        <span><span class="leg-dot" style="background:#6366f1"></span>{curr_label}</span>
      </div>
      {cmp_dice}
      {cmp_iou}
      {cmp_acc}
      {cmp_sens}
      {cmp_spec}
    </div>

    <div class="card">
      <div class="sec-label" style="margin-bottom:8px">Training Curves — Loss</div>
      <div class="legend">
        <span><span class="leg-dot" style="background:#6366f1"></span>Train</span>
        <span><span class="leg-dot" style="background:#818cf8"></span>Val</span>
      </div>
      <div class="chart-wrap"><canvas id="lossChart"></canvas></div>
    </div>
  </div>

  <!-- ── Dice + IoU curves ── -->
  <div class="curves-grid">
    <div class="card">
      <div class="sec-label" style="margin-bottom:8px">Training Curves — Dice Score</div>
      <div class="legend">
        <span><span class="leg-dot" style="background:#34d399"></span>Val Dice</span>
      </div>
      <div class="chart-wrap"><canvas id="diceChart"></canvas></div>
    </div>
    <div class="card">
      <div class="sec-label" style="margin-bottom:8px">Training Curves — IoU</div>
      <div class="legend">
        <span><span class="leg-dot" style="background:#f59e0b"></span>Val IoU</span>
      </div>
      <div class="chart-wrap"><canvas id="iouChart"></canvas></div>
    </div>
  </div>

  <!-- ── Visual results ── -->
  <div class="sec-label">Visual Results — Sample Predictions</div>
  <div class="visual-grid">
    <div class="vis-card">
      <div class="vis-label">Input Image</div>
      <div class="vis-img">{vis_input}</div>
    </div>
    <div class="vis-card">
      <div class="vis-label">Ground Truth</div>
      <div class="vis-img">{vis_gt}</div>
    </div>
    <div class="vis-card">
      <div class="vis-label">Prediction</div>
      <div class="vis-img">{vis_pred}</div>
    </div>
  </div>

  <!-- ── Config ── -->
  <div class="sec-label" style="margin-top:4px">Model Configuration</div>
  <div class="cfg-grid">
    <div class="cfg-item"><span class="cfg-key">Architecture</span><span class="cfg-val">MAFFNet</span></div>
    <div class="cfg-item"><span class="cfg-key">Backbone</span><span class="cfg-val">SAM2 Hiera-Large</span></div>
    <div class="cfg-item"><span class="cfg-key">Input Size</span><span class="cfg-val">{img_size_str}</span></div>
    <div class="cfg-item"><span class="cfg-key">Epochs</span><span class="cfg-val">{epochs_c}</span></div>
    <div class="cfg-item"><span class="cfg-key">Batch Size</span><span class="cfg-val">{batch}</span></div>
    <div class="cfg-item"><span class="cfg-key">Optimizer</span><span class="cfg-val">AdamW</span></div>
    <div class="cfg-item"><span class="cfg-key">LR Schedule</span><span class="cfg-val">Linear decay</span></div>
    <div class="cfg-item"><span class="cfg-key">Loss Function</span><span class="cfg-val">Dice + BCE + Boundary</span></div>
    <div class="cfg-item"><span class="cfg-key">Init LR</span><span class="cfg-val">{lr}</span></div>
  </div>

</div><!-- end .dash -->

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const eps       = {epochs};
const trainLoss = {train_loss};
const valLoss   = {val_loss};
const valDice   = {val_dice};
const valIoU    = {val_iou};

const baseOpts = {{
  responsive: true,
  maintainAspectRatio: false,
  animation: false,
  plugins: {{ legend: {{ display: false }}, tooltip: {{ enabled: false }} }},
  elements: {{ point: {{ radius: 0 }} }},
  scales: {{
    x: {{ display: false, grid: {{ display: false }} }},
    y: {{
      display: true,
      grid: {{ color: '#1e2235' }},
      ticks: {{ color: '#475569', font: {{ size: 10 }}, maxTicksLimit: 4 }},
      border: {{ color: 'transparent' }}
    }}
  }}
}};

new Chart(document.getElementById('lossChart'), {{
  type: 'line',
  data: {{
    labels: eps,
    datasets: [
      {{ data: trainLoss, borderColor: '#6366f1', borderWidth: 1.5, tension: 0.4, fill: false }},
      {{ data: valLoss,   borderColor: '#818cf8', borderWidth: 1.5, borderDash: [4, 3], tension: 0.4, fill: false }}
    ]
  }},
  options: baseOpts
}});

new Chart(document.getElementById('diceChart'), {{
  type: 'line',
  data: {{
    labels: eps,
    datasets: [
      {{ data: valDice, borderColor: '#34d399', borderWidth: 1.5, tension: 0.4, fill: false }}
    ]
  }},
  options: {{ ...baseOpts, scales: {{ ...baseOpts.scales, y: {{ ...baseOpts.scales.y, min: {dice_min} }} }} }}
}});

new Chart(document.getElementById('iouChart'), {{
  type: 'line',
  data: {{
    labels: eps,
    datasets: [
      {{ data: valIoU, borderColor: '#f59e0b', borderWidth: 1.5, tension: 0.4, fill: false }}
    ]
  }},
  options: {{ ...baseOpts, scales: {{ ...baseOpts.scales, y: {{ ...baseOpts.scales.y, min: {iou_min} }} }} }}
}});
</script>
</body>
</html>"""

    report_path = DASHBOARD_DIR / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[report] Saved → {report_path}")
    webbrowser.open(f"file://{report_path.resolve()}")
    return report_path
