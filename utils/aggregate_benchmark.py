"""
Benchmark aggregation utility — standalone, read-only.

Scans experiments/*/metrics.json (written by the trainers at end of a completed
run), aggregates every completed experiment, and emits:
    results/benchmark/master_results.csv
    results/benchmark/master_results.xlsx
    results/benchmark/benchmark_summary.md

Schema-driven: model and dataset are read from each metrics.json, so ANY future
model (MAFFNet included) that writes a metrics.json with the same shape is picked
up automatically — no code change. Does not import or modify any training code.

Run:  python -m utils.aggregate_benchmark      (from repo root)
  or: python utils/aggregate_benchmark.py
"""

import csv
import json
import zipfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

EXPERIMENTS_DIR = Path("experiments")
OUT_DIR         = Path("results/benchmark")

# Exact required columns (+ trailing Experiment for traceability)
COLUMNS = [
    "Model", "Dataset", "Best Epoch",
    "Validation Dice", "Test Dice",
    "Validation IoU", "Test IoU",
    "Accuracy", "Sensitivity", "Specificity",
    "Training Time", "Early Stopped (Yes/No)",
    "Experiment",
]

# Pretty model names (display only; unknown models pass through unchanged)
_PRETTY = {
    "unet": "U-Net", "attention_unet": "Attention U-Net", "attn_unet": "Attention U-Net",
    "unetpp": "UNet++", "unet_plus_plus": "UNet++",
    "transunet": "TransUNet", "maffnet": "MAFFNet",
}

# metrics.json keys that might hold a training duration (future-proof)
_TIME_KEYS = ("training_time", "train_time_sec", "duration_sec", "elapsed_sec", "train_seconds")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _num(v, nd=4):
    try:
        if v is None:
            return ""
        return round(float(v), nd)
    except (TypeError, ValueError):
        return ""


def _pretty_model(m):
    if not m:
        return "—"
    return _PRETTY.get(str(m).lower(), str(m))


def _fmt_seconds(sec):
    try:
        sec = int(round(float(sec)))
    except (TypeError, ValueError):
        return ""
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _training_time(exp_dir: Path, data: dict):
    """Prefer an explicit duration key in metrics.json; else estimate from the
    span of checkpoint mtimes (marked '~'); else ''."""
    for k in _TIME_KEYS:
        if k in data:
            t = _fmt_seconds(data[k])
            if t:
                return t
    ck = exp_dir / "checkpoints"
    if ck.is_dir():
        mtimes = [p.stat().st_mtime for p in ck.glob("model_epoch_*.pth")]
        if len(mtimes) >= 2:
            return "~" + _fmt_seconds(max(mtimes) - min(mtimes))
    return ""


def _early_stopped(data):
    es = data.get("early_stopped")
    if isinstance(es, bool):
        return "Yes" if es else "No"
    # infer: best epoch strictly before the configured budget
    be = data.get("best_epoch")
    budget = data.get("epochs_run") or data.get("config", {}).get("epochs")
    try:
        if be is not None and budget is not None and int(be) < int(budget):
            return "Yes"
        if be is not None and budget is not None:
            return "No"
    except (TypeError, ValueError):
        pass
    return "—"


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def collect(experiments_dir=EXPERIMENTS_DIR):
    rows = []
    skipped = []
    if not Path(experiments_dir).exists():
        return rows, skipped
    for exp_dir in sorted(Path(experiments_dir).iterdir()):
        if not exp_dir.is_dir():
            continue
        mj = exp_dir / "metrics.json"
        if not mj.is_file():
            skipped.append(exp_dir.name)
            continue
        try:
            data = json.loads(mj.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            skipped.append(exp_dir.name + " (unreadable)")
            continue

        val  = data.get("val", {})  or {}
        test = data.get("test", {}) or {}
        rows.append({
            "Model":                 _pretty_model(data.get("model")),
            "Dataset":               data.get("dataset", "—"),
            "Best Epoch":            data.get("best_epoch", ""),
            "Validation Dice":       _num(val.get("dice")),
            "Test Dice":             _num(test.get("dice")),
            "Validation IoU":        _num(val.get("mIoU")),
            "Test IoU":              _num(test.get("mIoU")),
            "Accuracy":              _num(val.get("acc")),
            "Sensitivity":           _num(val.get("sensitivity")),
            "Specificity":           _num(val.get("specificity")),
            "Training Time":         _training_time(exp_dir, data),
            "Early Stopped (Yes/No)": _early_stopped(data),
            "Experiment":            exp_dir.name,
        })
    # sort Dataset -> Model -> Experiment
    rows.sort(key=lambda r: (str(r["Dataset"]), str(r["Model"]), str(r["Experiment"])))
    return rows, skipped


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _col_ref(idx):  # 0 -> A, 26 -> AA
    s = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        s = chr(65 + rem) + s
    return s


def _xlsx_stdlib(rows, path):
    """Minimal dependency-free .xlsx writer (single sheet, inline strings)."""
    def cell(c, r, val):
        ref = f"{_col_ref(c)}{r}"
        if isinstance(val, (int, float)) and not isinstance(val, bool) and val != "":
            return f'<c r="{ref}"><v>{val}</v></c>'
        if val == "" or val is None:
            return f'<c r="{ref}" t="inlineStr"><is><t></t></is></c>'
        return f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(val))}</t></is></c>'

    sheet_rows = []
    sheet_rows.append("<row r=\"1\">" + "".join(cell(c, 1, h) for c, h in enumerate(COLUMNS)) + "</row>")
    for ri, r in enumerate(rows, start=2):
        sheet_rows.append("<row r=\"%d\">" % ri +
                          "".join(cell(c, ri, r[h]) for c, h in enumerate(COLUMNS)) + "</row>")
    sheet = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
             '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
             '<sheetData>' + "".join(sheet_rows) + '</sheetData></worksheet>')

    content_types = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '</Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>')
    workbook = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="master_results" sheetId="1" r:id="rId1"/></sheets></workbook>')
    wb_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '</Relationships>')

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("xl/workbook.xml", workbook)
        z.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        z.writestr("xl/worksheets/sheet1.xml", sheet)


def write_xlsx(rows, path):
    """Try pandas/openpyxl; fall back to the stdlib writer (no dependency)."""
    try:
        import pandas as pd
        pd.DataFrame(rows, columns=COLUMNS).to_excel(path, index=False, engine="openpyxl")
        return "pandas+openpyxl"
    except Exception:
        _xlsx_stdlib(rows, path)
        return "stdlib"


def write_summary_md(rows, skipped, path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    datasets = sorted({r["Dataset"] for r in rows})
    lines = []
    lines.append("# Benchmark Summary\n")
    lines.append(f"Aggregated **{len(rows)}** completed experiments · generated {now}\n")
    lines.append("Source: `experiments/*/metrics.json` · sorted Dataset → Model\n")

    # Master table
    lines.append("## Master Results\n")
    lines.append("| " + " | ".join(COLUMNS) + " |")
    lines.append("|" + "|".join("---" for _ in COLUMNS) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in COLUMNS) + " |")
    lines.append("")

    # Best model per dataset (by Validation Dice)
    lines.append("## Best Model per Dataset (Validation Dice)\n")
    lines.append("| Dataset | Best Model | Val Dice | Test Dice |")
    lines.append("|---|---|---|---|")
    for ds in datasets:
        sub = [r for r in rows if r["Dataset"] == ds and r["Validation Dice"] != ""]
        if not sub:
            continue
        b = max(sub, key=lambda r: float(r["Validation Dice"]))
        lines.append(f"| {ds} | {b['Model']} | {b['Validation Dice']} | {b['Test Dice'] or '—'} |")
    lines.append("")

    # Ranking by mean Validation Dice across datasets
    lines.append("## Ranking — Mean Validation Dice\n")
    by_model = {}
    for r in rows:
        if r["Validation Dice"] != "":
            by_model.setdefault(r["Model"], []).append(float(r["Validation Dice"]))
    ranking = sorted(((m, sum(v) / len(v)) for m, v in by_model.items()),
                     key=lambda x: x[1], reverse=True)
    lines.append("| Rank | Model | Mean Val Dice | # datasets |")
    lines.append("|---|---|---|---|")
    for i, (m, mean) in enumerate(ranking, 1):
        lines.append(f"| {i} | {m} | {mean:.4f} | {len(by_model[m])} |")
    lines.append("")

    es = sum(1 for r in rows if r["Early Stopped (Yes/No)"] == "Yes")
    lines.append(f"_Early-stopped runs: {es}/{len(rows)}._\n")
    if skipped:
        lines.append(f"_Skipped (no metrics.json) — {len(skipped)} folder(s): "
                     + ", ".join(f"`{s}`" for s in skipped) + "._\n")

    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(experiments_dir=EXPERIMENTS_DIR, out_dir=OUT_DIR):
    rows, skipped = collect(experiments_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "master_results.csv"
    xlsx_path = out_dir / "master_results.xlsx"
    md_path  = out_dir / "benchmark_summary.md"

    write_csv(rows, csv_path)
    engine = write_xlsx(rows, xlsx_path)
    write_summary_md(rows, skipped, md_path)

    print(f"[aggregate] {len(rows)} experiments aggregated "
          f"({len(skipped)} skipped, no metrics.json)")
    print(f"  CSV  → {csv_path}")
    print(f"  XLSX → {xlsx_path}  (engine: {engine})")
    print(f"  MD   → {md_path}")
    return rows


if __name__ == "__main__":
    main()
