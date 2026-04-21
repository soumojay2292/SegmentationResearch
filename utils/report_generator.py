import os
import webbrowser
import matplotlib.pyplot as plt

def generate_report(metrics, config, image_paths):

    os.makedirs("results/dashboard", exist_ok=True)

    # -------------------------
    # SAVE FIGURE (BAR CHART)
    # -------------------------
    labels = ["U-Net", "TransUNet", "MAFFNet (Best)", "MAFFNet (Final)"]
    values = [0.80, 0.75, metrics['val_dice'], metrics['test_dice']]

    plt.figure()
    plt.bar(labels, values)
    plt.ylabel("Dice Score")
    plt.title("Model Comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("results/dashboard/metrics.png")
    plt.close()

    # -------------------------
    # HTML DASHBOARD
    # -------------------------
    html = f"""
<html>
<head>
<title>MAFFNet Research Dashboard</title>

<style>
body {{
    font-family: 'Segoe UI', sans-serif;
    background: #0f172a;
    color: white;
    margin: 20px;
}}

h1 {{
    color: #38bdf8;
    margin-bottom: 5px;
}}

.subtitle {{
    color: #94a3b8;
    margin-bottom: 20px;
}}

.card {{
    background: #1e293b;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 15px;
}}

.grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 15px;
}}

.row {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 15px;
}}

.metric {{
    font-size: 24px;
    font-weight: bold;
}}

.bar {{
    height: 12px;
    border-radius: 6px;
    background: #334155;
    margin-top: 5px;
}}

.fill {{
    height: 100%;
    border-radius: 6px;
}}

.green {{ background: #22c55e; }}
.orange {{ background: #f97316; }}
.red {{ background: #ef4444; }}
.blue {{ background: #38bdf8; }}

img {{
    width: 100%;
    border-radius: 10px;
}}

ul {{
    padding-left: 20px;
}}

.tag {{
    display: inline-block;
    padding: 4px 8px;
    border-radius: 6px;
    margin: 3px;
    font-size: 12px;
}}

.good {{ background: #14532d; }}
.mid {{ background: #78350f; }}
.bad {{ background: #7f1d1d; }}

</style>
</head>

<body>

<h1>MAFFNet — Research Dashboard</h1>
<div class="subtitle">Skin lesion segmentation | ISIC | Frequency Fusion + SAM</div>

<!-- METRICS -->
<div class="grid">

<div class="card">
<h3>Best Dice</h3>
<div class="metric blue">{metrics['val_dice']:.3f}</div>
<div class="bar"><div class="fill blue" style="width:{metrics['val_dice']*100}%"></div></div>
<p>mid-training peak</p>
</div>

<div class="card">
<h3>Final Dice</h3>
<div class="metric orange">{metrics['test_dice']:.3f}</div>
<div class="bar"><div class="fill orange" style="width:{metrics['test_dice']*100}%"></div></div>
<p>generalization drop</p>
</div>

<div class="card">
<h3>IoU</h3>
<div class="metric green">{metrics['iou']:.3f}</div>
<div class="bar"><div class="fill green" style="width:{metrics['iou']*100}%"></div></div>
<p>boundary overlap</p>
</div>

</div>

<!-- DATA + PIPELINE -->
<div class="row">

<div class="card">
<h3>Dataset</h3>
<p>{config['dataset']}</p>
<p>Resolution: 224 × 224</p>
<p>Task: Binary segmentation</p>
<p>Format: CSV pipeline</p>
</div>

<div class="card">
<h3>Model Pipeline</h3>
<p>
<span class="tag good">Input</span>
<span class="tag good">SAM</span>
<span class="tag mid">MMPA</span>
<span class="tag mid">FFT</span>
<span class="tag mid">FGFF</span>
<span class="tag good">Decoder</span>
</p>
</div>

</div>

<!-- PARAMETERS -->
<div class="row">

<div class="card">
<h3>Parameters</h3>
<p>Total: {config['total_params']}</p>
<p>Trainable: {config['trainable_params']}</p>
<p>SAM: Frozen encoder</p>
</div>

<div class="card">
<h3>Training Config</h3>
<p>Epochs: {config['epochs']}</p>
<p>Batch Size: {config['batch_size']}</p>
<p>LR: {config['lr']}</p>
<p>Loss: BCE + Dice</p>
</div>

</div>

<!-- MODEL COMPARISON -->
<div class="card">
<h3>Metric Comparison</h3>

<p>U-Net (baseline)</p>
<div class="bar"><div class="fill green" style="width:80%"></div></div>

<p>TransUNet</p>
<div class="bar"><div class="fill green" style="width:75%"></div></div>

<p>MAFFNet (best)</p>
<div class="bar"><div class="fill blue" style="width:{metrics['val_dice']*100}%"></div></div>

<p>MAFFNet (final)</p>
<div class="bar"><div class="fill orange" style="width:{metrics['test_dice']*100}%"></div></div>

</div>

<!-- VISUALS -->
<div class="card">
<h3>Sample Prediction</h3>
<div class="grid">
    <img src="{image_paths['input']}">
    <img src="{image_paths['gt']}">
    <img src="{image_paths['pred']}">
</div>
</div>

<!-- INSIGHTS -->
<div class="card">
<h3>Key Observations</h3>
<ul>
<li>✔ Lesion structure captured correctly</li>
<li>✔ Frequency fusion improves boundary sensitivity</li>
<li>⚠ Training instability observed</li>
<li>⚠ Validation > Test gap (overfitting)</li>
<li>⚠ SAM frozen limits domain adaptation</li>
</ul>
</div>

<!-- STATUS -->
<div class="card">
<h3>Status</h3>
<span class="tag good">Model Learning</span>
<span class="tag mid">Unstable Training</span>
<span class="tag bad">Generalization Gap</span>
</div>

</body>
</html>
"""

    path = "results/dashboard/report.html"
    print(f"[INFO] Report saved at: {path}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    webbrowser.open(f"file://{os.path.abspath(path)}")