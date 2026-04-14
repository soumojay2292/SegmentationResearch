import os
import webbrowser

def generate_report(metrics, config, image_paths):

    html = f"""
<html>
<head>
<title>MAFFNet Dashboard</title>

<style>
body {{
    font-family: 'Segoe UI', sans-serif;
    background: #0f172a;
    color: white;
    margin: 20px;
}}

h1 {{
    color: #38bdf8;
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

.metric {{
    font-size: 22px;
    font-weight: bold;
}}

.good {{ color: #22c55e; }}
.mid {{ color: #facc15; }}
.bad {{ color: #ef4444; }}

.bar {{
    height: 10px;
    border-radius: 5px;
    background: #334155;
    margin-top: 5px;
}}

.fill {{
    height: 100%;
    border-radius: 5px;
}}

.green {{ background: #22c55e; }}
.orange {{ background: #f97316; }}
.red {{ background: #ef4444; }}

img {{
    width: 100%;
    border-radius: 10px;
}}

</style>
</head>

<body>

<h1>MAFFNet Result Dashboard</h1>

<div class="grid">

<div class="card">
<h3>Validation Dice</h3>
<div class="metric mid">{metrics['val_dice']:.3f}</div>
<div class="bar"><div class="fill green" style="width:{metrics['val_dice']*100}%"></div></div>
</div>

<div class="card">
<h3>Test Dice</h3>
<div class="metric bad">{metrics['test_dice']:.3f}</div>
<div class="bar"><div class="fill orange" style="width:{metrics['test_dice']*100}%"></div></div>
</div>

<div class="card">
<h3>IoU</h3>
<div class="metric mid">{metrics['iou']:.3f}</div>
<div class="bar"><div class="fill green" style="width:{metrics['iou']*100}%"></div></div>
</div>

</div>

<div class="card">
<h3>Pipeline</h3>
<p>Input → SAM → MMPA → FFT → FGFF → Decoder</p>
</div>

<div class="grid">

<div class="card">
<h3>Model Info</h3>
<p>Total Params: {config['total_params']}</p>
<p>Trainable: {config['trainable_params']}</p>
</div>

<div class="card">
<h3>Training</h3>
<p>Epochs: {config['epochs']}</p>
<p>Batch: {config['batch_size']}</p>
<p>LR: {config['lr']}</p>
</div>

<div class="card">
<h3>Status</h3>
<p class="good">✔ Model learning</p>
<p class="mid">⚠ Unstable training</p>
<p class="bad">✖ Generalization gap</p>
</div>

</div>

<div class="card">
<h3>Visual Results</h3>
<div class="grid">
    <img src="{image_paths['input']}">
    <img src="{image_paths['gt']}">
    <img src="{image_paths['pred']}">
</div>
</div>

<div class="card">
<h3>Key Insights</h3>
<ul>
<li>✔ Model captures lesion structure</li>
<li>✔ Frequency fusion improves edges</li>
<li>⚠ Training instability observed</li>
<li>⚠ Val > Test gap indicates overfitting</li>
</ul>
</div>

</body>
</html>
"""

    os.makedirs("results/dashboard", exist_ok=True)

    path = "results/dashboard/report.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    webbrowser.open(f"file://{os.path.abspath(path)}")