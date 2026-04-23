"""
Research-grade HTML dashboard for MAFFNet segmentation.
Reads all metrics from real CSV files — no hardcoded baselines.
"""

import csv
import base64
import webbrowser
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DASHBOARD_DIR = Path("results/dashboard")


# ---------------------------------------------------------------------------
# Internal helpers
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
# Training curves
# ---------------------------------------------------------------------------

def generate_training_curves(log_csv):
    rows = _read_csv(log_csv)
    if not rows:
        return None, None

    try:
        epochs     = [int(r["epoch"])      for r in rows]
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

    # Loss curve
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

    # IoU / Dice curve
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
# Best / final epoch from training log
# ---------------------------------------------------------------------------

def _best_and_final(log_csv):
    rows = _read_csv(log_csv)
    if not rows:
        return {}, {}
    best  = max(rows, key=lambda r: float(r.get("val_dice", 0)))
    final = rows[-1]
    return best, final


# ---------------------------------------------------------------------------
# Prediction sample grid
# ---------------------------------------------------------------------------

def _collect_samples(image_dir, n=5):
    image_dir = Path(image_dir)
    samples = []
    if not image_dir.exists():
        return samples

    # Flat layout: input_0.png / gt_0.png / pred_0.png
    for i in range(n):
        ip = image_dir / f"input_{i}.png"
        gp = image_dir / f"gt_{i}.png"
        pp = image_dir / f"pred_{i}.png"
        if ip.exists() and gp.exists() and pp.exists():
            samples.append((ip, gp, pp))

    # Fallback: single triplet in root
    if not samples:
        ip = image_dir / "input.png"
        gp = image_dir / "gt.png"
        pp = image_dir / "pred.png"
        if ip.exists() and gp.exists() and pp.exists():
            samples.append((ip, gp, pp))

    return samples


def _sample_grid_html(samples):
    if not samples:
        return "<p style='color:#64748b;padding:16px'>No prediction samples found in image directory.</p>"

    rows = []
    for ip, gp, pp in samples:
        ib = _img_to_b64(ip)
        gb = _img_to_b64(gp)
        pb = _img_to_b64(pp)
        rows.append(f"""
<div class="sample-row">
  <div class="sample-cell"><span class="label">Input</span><img src="{ib}" loading="lazy"></div>
  <div class="sample-cell"><span class="label">Ground Truth</span><img src="{gb}" loading="lazy"></div>
  <div class="sample-cell"><span class="label">Prediction</span><img src="{pb}" loading="lazy"></div>
</div>""")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# HTML fragments
# ---------------------------------------------------------------------------

_COLORS = {
    "val_dice":        "#a78bfa",
    "val_miou":        "#22c55e",
    "val_acc":         "#38bdf8",
    "val_sensitivity": "#f97316",
    "val_specificity": "#fb7185",
    "val_f1":          "#fbbf24",
}
_LABELS = {
    "val_dice":        "Dice",
    "val_miou":        "IoU",
    "val_acc":         "Accuracy",
    "val_sensitivity": "Sensitivity",
    "val_specificity": "Specificity",
    "val_f1":          "F1",
}


def _metric_rows_html(row_dict):
    html = ""
    for key, label in _LABELS.items():
        if key not in row_dict:
            continue
        val   = float(row_dict[key])
        color = _COLORS[key]
        pct   = min(val * 100, 100)
        html += f"""
<tr>
  <td class="metric-label">{label}</td>
  <td class="metric-value" style="color:{color}">{val:.4f}</td>
  <td class="bar-cell">
    <div class="bar"><div class="fill" style="width:{pct:.1f}%;background:{color}"></div></div>
  </td>
</tr>"""
    return html


def _config_item(label, value):
    return f'<div class="ci"><span class="ck">{label}</span><span class="cv">{value}</span></div>'


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_report(log_csv, summary_csv, image_dir, config):
    """
    log_csv     : path to checkpoints/training_log.csv
    summary_csv : path to results/<run>/summary.csv  (may be None)
    image_dir   : directory containing input_N / gt_N / pred_N PNGs
    config      : dict with keys: dataset, epochs, batch_size, lr, img_size,
                  total_params, trainable_params
    """
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    # Training curves
    loss_path, iou_path = generate_training_curves(log_csv)
    loss_b64 = _img_to_b64(loss_path) if loss_path else ""
    iou_b64  = _img_to_b64(iou_path)  if iou_path  else ""

    no_curve = "<p style='color:#64748b;padding:16px'>No training log data yet.</p>"
    loss_img = f'<img src="{loss_b64}">' if loss_b64 else no_curve
    iou_img  = f'<img src="{iou_b64}">'  if iou_b64  else no_curve

    # Metrics — primary source: training log; fallback: summary.csv
    best, final = _best_and_final(log_csv)

    # If log is empty, try to populate from summary.csv
    if not best and summary_csv:
        summary_rows = _read_csv(summary_csv)
        if summary_rows:
            # summary.csv rows are key/value pairs written by _save_summary
            row0 = summary_rows[0]
            if "metric" in row0:
                summary_dict = {r["metric"]: r["value"] for r in summary_rows}
            else:
                summary_dict = dict(row0)
            # Map summary keys → log keys where possible
            _key_map = {
                "Final Val Dice": "val_dice", "Final Val IoU": "val_miou",
                "Final Val Loss": "val_loss", "Final Train Loss": "train_loss",
                "val_dice": "val_dice", "val_miou": "val_miou",
                "val_loss": "val_loss", "train_loss": "train_loss",
            }
            merged = {_key_map[k]: v for k, v in summary_dict.items() if k in _key_map}
            best = final = merged

    best_epoch  = best.get("epoch",  "—")
    final_epoch = final.get("epoch", "—")

    best_rows  = _metric_rows_html(best)
    final_rows = _metric_rows_html(final)

    def _safe(d, k, fmt=".4f"):
        v = d.get(k)
        return f"{float(v):{fmt}}" if v is not None else "—"

    kpi_dice  = _safe(best,  "val_dice",  ".3f")
    kpi_iou   = _safe(best,  "val_miou",  ".3f")
    kpi_tloss = _safe(final, "train_loss", ".4f")
    kpi_vloss = _safe(final, "val_loss",   ".4f")
    kpi_dice_pct  = f"{float(best.get('val_dice',  0)) * 100:.1f}" if best  else "0"
    kpi_iou_pct   = f"{float(best.get('val_miou',  0)) * 100:.1f}" if best  else "0"

    # Samples
    samples    = _collect_samples(image_dir)
    sample_html = _sample_grid_html(samples)

    # Config
    dataset  = config.get("dataset",          "—")
    epochs   = config.get("epochs",           "—")
    batch    = config.get("batch_size",       "—")
    lr       = config.get("lr",               "—")
    img_size = config.get("img_size",         "—")
    total_p  = config.get("total_params",     "—")
    train_p  = config.get("trainable_params", "—")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MAFFNet Research Dashboard</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f172a;color:#e2e8f0;
     padding:28px 40px;min-height:100vh;}}
h1{{color:#38bdf8;font-size:1.85rem;margin-bottom:4px}}
.sub{{color:#64748b;font-size:.85rem;margin-bottom:32px}}
h2{{color:#94a3b8;font-size:.8rem;text-transform:uppercase;letter-spacing:.1em;
    margin:36px 0 14px;border-bottom:1px solid #1e293b;padding-bottom:8px}}
.card{{background:#1e293b;border:1px solid #334155;border-radius:12px;
       padding:20px 24px;transition:border-color .2s}}
.card:hover{{border-color:#38bdf888}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
.four{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}
.kpi{{text-align:center;padding:22px 16px}}
.kpi-lbl{{font-size:.72rem;color:#64748b;text-transform:uppercase;
           letter-spacing:.07em;margin-bottom:10px}}
.kpi-val{{font-size:2.1rem;font-weight:700}}
.kpi-ep{{color:#475569;font-size:.72rem;margin-top:8px}}
.bar{{height:8px;border-radius:4px;background:#334155;margin-top:10px;overflow:hidden}}
.fill{{height:100%;border-radius:4px}}
table{{width:100%;border-collapse:collapse;font-size:.88rem}}
thead tr{{border-bottom:1px solid #334155}}
th{{color:#475569;font-weight:500;padding:8px 12px;text-align:left;
    font-size:.72rem;text-transform:uppercase;letter-spacing:.06em}}
td{{padding:10px 12px}}
tbody tr:hover{{background:#0f172a55}}
.metric-label{{color:#94a3b8}}
.metric-value{{font-weight:600;font-size:1rem}}
.bar-cell{{width:38%}}
.tbl-hdr{{color:#475569;font-size:.72rem;margin-bottom:12px;
           text-transform:uppercase;letter-spacing:.06em}}
.sample-row{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;
              margin-bottom:20px}}
.sample-cell{{text-align:center}}
.label{{display:block;font-size:.68rem;color:#64748b;text-transform:uppercase;
         letter-spacing:.07em;margin-bottom:6px}}
img{{width:100%;border-radius:8px;display:block}}
.sample-cell img{{transition:opacity .2s;cursor:zoom-in}}
.sample-cell img:hover{{opacity:.85}}
.cg{{display:grid;grid-template-columns:1fr 1fr;gap:6px 24px;margin-top:4px}}
.ci{{display:flex;justify-content:space-between;padding:7px 0;
     border-bottom:1px solid #0f172a55;font-size:.84rem}}
.ck{{color:#64748b}}.cv{{color:#e2e8f0;font-weight:500}}
.tags{{margin-top:14px}}
.tag{{display:inline-block;padding:3px 10px;border-radius:999px;
       font-size:.7rem;font-weight:500;margin:3px}}
.tb{{background:#1e3a5f;color:#38bdf8}}
.tg{{background:#14532d;color:#4ade80}}
.ta{{background:#78350f;color:#fbbf24}}
.foot{{color:#334155;font-size:.72rem;margin-top:44px;text-align:center}}
</style>
</head>
<body>

<h1>MAFFNet — Research Dashboard</h1>
<div class="sub">Skin lesion segmentation · ISIC · MAFFNet + SAM2 Hiera-Large · {ts}</div>

<!-- KPIs -->
<div class="four">
  <div class="card kpi">
    <div class="kpi-lbl">Best Val Dice</div>
    <div class="kpi-val" style="color:#a78bfa">{kpi_dice}</div>
    <div class="bar"><div class="fill" style="width:{kpi_dice_pct}%;background:#a78bfa"></div></div>
    <div class="kpi-ep">epoch {best_epoch}</div>
  </div>
  <div class="card kpi">
    <div class="kpi-lbl">Best Val IoU</div>
    <div class="kpi-val" style="color:#22c55e">{kpi_iou}</div>
    <div class="bar"><div class="fill" style="width:{kpi_iou_pct}%;background:#22c55e"></div></div>
    <div class="kpi-ep">epoch {best_epoch}</div>
  </div>
  <div class="card kpi">
    <div class="kpi-lbl">Final Train Loss</div>
    <div class="kpi-val" style="color:#38bdf8">{kpi_tloss}</div>
    <div class="kpi-ep">epoch {final_epoch}</div>
  </div>
  <div class="card kpi">
    <div class="kpi-lbl">Final Val Loss</div>
    <div class="kpi-val" style="color:#f97316">{kpi_vloss}</div>
    <div class="kpi-ep">epoch {final_epoch}</div>
  </div>
</div>

<!-- Training Curves -->
<h2>Training Curves</h2>
<div class="two">
  <div class="card">{loss_img}</div>
  <div class="card">{iou_img}</div>
</div>

<!-- Metrics Table -->
<h2>Metrics Table</h2>
<div class="two">
  <div class="card">
    <div class="tbl-hdr">Best epoch &nbsp;·&nbsp; ep {best_epoch}</div>
    <table>
      <thead><tr><th>Metric</th><th>Value</th><th>Score</th></tr></thead>
      <tbody>{best_rows}</tbody>
    </table>
  </div>
  <div class="card">
    <div class="tbl-hdr">Final epoch &nbsp;·&nbsp; ep {final_epoch}</div>
    <table>
      <thead><tr><th>Metric</th><th>Value</th><th>Score</th></tr></thead>
      <tbody>{final_rows}</tbody>
    </table>
  </div>
</div>

<!-- Visual Results -->
<h2>Visual Results</h2>
<div class="card">
{sample_html}
</div>

<!-- Configuration -->
<h2>Configuration</h2>
<div class="two">
  <div class="card">
    <div class="tbl-hdr">Training</div>
    <div class="cg">
      {_config_item("Dataset",    dataset)}
      {_config_item("Epochs",     epochs)}
      {_config_item("Batch size", batch)}
      {_config_item("LR",         lr)}
      {_config_item("Image size", f"{img_size}×{img_size}" if img_size != "—" else "—")}
      {_config_item("Loss",       "Dice + BCE + MSE")}
    </div>
  </div>
  <div class="card">
    <div class="tbl-hdr">Model</div>
    <div class="cg">
      {_config_item("Total params",  total_p)}
      {_config_item("Trainable",     train_p)}
      {_config_item("Backbone",      "SAM2 Hiera-Large")}
      {_config_item("Backbone",      "Frozen")}
      {_config_item("Optimiser",     "AdamW")}
      {_config_item("Scheduler",     "Linear decay")}
    </div>
    <div class="tags">
      <span class="tag tb">SAM2</span>
      <span class="tag tg">MMPA</span>
      <span class="tag tg">FFT Fusion</span>
      <span class="tag tg">FGFF</span>
      <span class="tag ta">AMP</span>
    </div>
  </div>
</div>

<div class="foot">MAFFNet · {ts} · results/dashboard/report.html</div>

</body>
</html>"""

    report_path = DASHBOARD_DIR / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[report] Saved → {report_path}")
    webbrowser.open(f"file://{report_path.resolve()}")
    return report_path
