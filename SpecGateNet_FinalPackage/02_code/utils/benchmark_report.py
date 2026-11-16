"""
Benchmark dashboard — aggregates completed experiments into one comparison.

Scans experiments/<model>_<dataset>_<timestamp>/checkpoints/training_log.csv
(works on ALL historical experiments — every run has this file), takes the
best-Dice epoch row, and compares the models across the three ISIC datasets.

Outputs a self-contained, publication-ready HTML to
results/benchmark/benchmark_<timestamp>.html (timestamped → never overwrites
history) plus a convenience copy benchmark_latest.html.

Comparison: U-Net, Attention U-Net, UNet++, TransUNet, MAFFNet
Across:     ISIC 2016, ISIC 2017, ISIC 2018
Includes:   metric table, separate Dice/IoU/Sensitivity/Specificity plots,
            ranking by average Dice, % improvement vs the U-Net baseline.
"""

import csv
import re
import base64
from io import BytesIO
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# canonical order + display names
MODELS = [
    ("unet",           "U-Net"),
    ("attention_unet", "Attention U-Net"),
    ("unetpp",         "UNet++"),
    ("transunet",      "TransUNet"),
    ("maffnet",        "MAFFNet"),
]
MODEL_TOKENS = [m[0] for m in sorted(MODELS, key=lambda x: -len(x[0]))]  # longest first
DATASETS = ["ISIC_2016", "ISIC_2017", "ISIC_2018"]
BASELINE = "unet"

METRIC_KEYS = {           # display -> training_log.csv column
    "Dice":        "val_dice",
    "IoU":         "val_miou",
    "Sensitivity": "val_sensitivity",
    "Specificity": "val_specificity",
    "Accuracy":    "val_acc",
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_dirname(name):
    low = name.lower()
    model = next((t for t in MODEL_TOKENS if low.startswith(t)), None)
    md = re.search(r"(ISIC[_-]?\d{4})", name)
    dataset = md.group(1).replace("-", "_") if md else None
    mt = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", name)
    stamp = mt.group(1) if mt else ""
    return model, dataset, stamp


def _best_row(log_csv):
    p = Path(log_csv)
    if not p.exists():
        return None
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    try:
        return max(rows, key=lambda r: float(r.get("val_dice", 0) or 0))
    except ValueError:
        return None


def collect(experiments_dir="experiments"):
    """Return results[model][dataset] = {metric: float, ...} (latest run wins)."""
    exp_dir = Path(experiments_dir)
    results = {}
    chosen_stamp = {}
    if not exp_dir.exists():
        return results
    for d in exp_dir.iterdir():
        if not d.is_dir():
            continue
        model, dataset, stamp = _parse_dirname(d.name)
        if model is None or dataset not in DATASETS:
            continue
        log_csv = d / "checkpoints" / "training_log.csv"
        row = _best_row(log_csv)
        if row is None:
            continue
        key = (model, dataset)
        # keep the most recent run (by timestamp string, then folder mtime)
        rank = (stamp, d.stat().st_mtime)
        if key in chosen_stamp and rank <= chosen_stamp[key]:
            continue
        chosen_stamp[key] = rank
        vals = {}
        for disp, col in METRIC_KEYS.items():
            try:
                vals[disp] = float(row.get(col))
            except (TypeError, ValueError):
                vals[disp] = None
        vals["best_epoch"] = row.get("epoch", "—")
        results.setdefault(model, {})[dataset] = vals
    return results


# ---------------------------------------------------------------------------
# Plots (grouped bar, base64-embedded)
# ---------------------------------------------------------------------------

def _plot_metric(results, metric):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    n_ds = len(DATASETS)
    width = 0.15
    xs = range(n_ds)
    for i, (tok, name) in enumerate(MODELS):
        ys = []
        for ds in DATASETS:
            v = results.get(tok, {}).get(ds, {}).get(metric)
            ys.append(v if v is not None else 0.0)
        offs = [x + (i - 2) * width for x in xs]
        ax.bar(offs, ys, width, label=name)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([d.replace("_", " ") for d in DATASETS])
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{metric} by model & dataset", fontsize=11)
    ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close()
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Derived stats
# ---------------------------------------------------------------------------

def _mean_dice(results, tok):
    vs = [results.get(tok, {}).get(ds, {}).get("Dice") for ds in DATASETS]
    vs = [v for v in vs if v is not None]
    return sum(vs) / len(vs) if vs else None


def _ranking(results):
    rows = []
    for tok, name in MODELS:
        md = _mean_dice(results, tok)
        if md is not None:
            rows.append((name, md))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def _improvement(results):
    """% improvement of each model vs U-Net baseline, per dataset."""
    out = {}
    for tok, name in MODELS:
        if tok == BASELINE:
            continue
        row = {}
        for ds in DATASETS:
            base = results.get(BASELINE, {}).get(ds, {}).get("Dice")
            cur  = results.get(tok, {}).get(ds, {}).get("Dice")
            if base and cur is not None and base > 0:
                row[ds] = (cur - base) / base * 100.0
            else:
                row[ds] = None
        out[name] = row
    return out


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_CSS = """
*{box-sizing:border-box}
body{margin:0;background:#f4f6f9;color:#1e293b;
 font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;font-size:14px}
.wrap{max-width:1120px;margin:0 auto;padding:32px 24px 60px}
h1{font-size:22px;margin:0 0 4px}.sub{color:#64748b;font-size:13px;margin-bottom:24px}
.sec{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:#475569;margin:30px 0 12px}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e2e8f0;border-radius:10px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.04)}
th,td{padding:9px 12px;text-align:center;font-size:13px;border-bottom:1px solid #f1f5f9}
th{background:#0f172a;color:#fff;font-weight:600}
td:first-child,th:first-child{text-align:left}
tr:last-child td{border-bottom:none}
tbody tr:hover td{background:#f8fafc}
.best{font-weight:700;color:#047857}
.plots{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.card img{width:100%;display:block}
.rank{display:flex;gap:12px;flex-wrap:wrap}
.rank .r{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px 18px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.rank .r .p{font-size:11px;color:#64748b}.rank .r .m{font-size:18px;font-weight:700}
.pos{color:#047857}.neg{color:#b91c1c}
.foot{margin-top:40px;color:#94a3b8;font-size:11px;text-align:center}
"""


def _metric_table(results, metric):
    head = "".join(f"<th>{d.replace('_',' ')}</th>" for d in DATASETS)
    body = ""
    # per-dataset best for bolding
    best_per_ds = {}
    for ds in DATASETS:
        vals = [(tok, results.get(tok, {}).get(ds, {}).get(metric)) for tok, _ in MODELS]
        vals = [(t, v) for t, v in vals if v is not None]
        best_per_ds[ds] = max(vals, key=lambda x: x[1])[0] if vals else None
    for tok, name in MODELS:
        cells = ""
        for ds in DATASETS:
            v = results.get(tok, {}).get(ds, {}).get(metric)
            if v is None:
                cells += "<td>—</td>"
            else:
                cls = ' class="best"' if best_per_ds[ds] == tok else ""
                cells += f"<td{cls}>{v:.4f}</td>"
        body += f"<tr><td>{name}</td>{cells}</tr>"
    return f"<table><thead><tr><th>Model</th>{head}</tr></thead><tbody>{body}</tbody></table>"


def generate_benchmark(experiments_dir="experiments", out_dir="results/benchmark"):
    results = collect(experiments_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    stamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    if not results:
        html = ("<html><body style='font-family:sans-serif;padding:40px'>"
                "<h2>Benchmark dashboard</h2><p>No experiments found under "
                f"<code>{experiments_dir}</code>.</p></body></html>")
        path = out_dir / f"benchmark_{stamp}.html"
        path.write_text(html, encoding="utf-8")
        (out_dir / "benchmark_latest.html").write_text(html, encoding="utf-8")
        print(f"[benchmark] no data — wrote placeholder → {path}")
        return path

    # tables
    tables = "".join(
        f'<div class="sec">{m} — comparative table</div>{_metric_table(results, m)}'
        for m in ("Dice", "IoU", "Sensitivity", "Specificity")
    )
    # plots
    plots = "".join(
        f'<div class="card"><img src="{_plot_metric(results, m)}" alt="{m}"></div>'
        for m in ("Dice", "IoU", "Sensitivity", "Specificity")
    )
    # ranking
    rank_html = ""
    for i, (name, md) in enumerate(_ranking(results), 1):
        rank_html += (f'<div class="r"><div class="p">#{i} · avg Dice</div>'
                      f'<div class="m">{name}</div><div class="p">{md:.4f}</div></div>')
    # improvement table
    imp = _improvement(results)
    imp_head = "".join(f"<th>{d.replace('_',' ')}</th>" for d in DATASETS)
    imp_body = ""
    for name, row in imp.items():
        cells = ""
        for ds in DATASETS:
            v = row[ds]
            if v is None:
                cells += "<td>—</td>"
            else:
                cls = "pos" if v >= 0 else "neg"
                sign = "+" if v >= 0 else ""
                cells += f'<td class="{cls}">{sign}{v:.1f}%</td>'
        imp_body += f"<tr><td>{name}</td>{cells}</tr>"
    imp_table = (f"<table><thead><tr><th>Model (vs U-Net)</th>{imp_head}</tr></thead>"
                 f"<tbody>{imp_body}</tbody></table>")

    n_exp = sum(len(v) for v in results.values())
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Segmentation Benchmark Dashboard</title><style>{_CSS}</style></head>
<body><div class="wrap">
  <h1>Segmentation Benchmark Dashboard</h1>
  <div class="sub">Aggregated from {n_exp} experiments · generated {now.strftime("%Y-%m-%d %H:%M")} ·
   models: U-Net, Attention U-Net, UNet++, TransUNet, MAFFNet · datasets: ISIC 2016/2017/2018</div>

  <div class="sec">Model Ranking — by Average Dice</div>
  <div class="rank">{rank_html}</div>

  <div class="sec">% Improvement vs U-Net Baseline (Dice)</div>
  {imp_table}

  <div class="sec">Metric Plots</div>
  <div class="plots">{plots}</div>

  {tables}

  <div class="foot">Auto-generated from experiments/*/checkpoints/training_log.csv (best-Dice epoch per run) · no hardcoded values</div>
</div></body></html>"""

    path = out_dir / f"benchmark_{stamp}.html"
    path.write_text(html, encoding="utf-8")
    (out_dir / "benchmark_latest.html").write_text(html, encoding="utf-8")
    print(f"[benchmark] → {path}")
    return path


if __name__ == "__main__":
    generate_benchmark()
