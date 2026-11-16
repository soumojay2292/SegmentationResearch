# Feature Alignment — Implementation Report

**Task:** create the reusable Feature Alignment module from the synopsis (§1.4). Module only — not integrated. `maffnet.py` and `train.py` untouched. Nothing executed.

**File created:** `modules/feature_alignment.py` → class `FeatureAlignment`.

---

## Design

Purpose: align the activation statistics of two feature streams (SAM encoder features vs frequency-branch features) before fusion, so a downstream gate is not skewed by mismatched scales/means. Pipeline:

```
x  ->  InstanceNorm2d  ->  Conv2d(1x1)  ->  ReLU  ->  out
```

- **InstanceNorm2d** — per-sample, per-channel normalisation over the spatial dimensions. Removes per-channel mean/variance differences between streams (the actual "alignment"). `affine=False` by default (pure normalisation); `affine=True` optional for a learnable per-channel scale/shift.
- **1×1 Conv** — learnable channel projection. Supports **arbitrary** `in_channels → out_channels`; lets the aligned features be re-mapped to a common width for fusion.
- **ReLU** — non-linearity; bounds output to non-negative, matching the synopsis block.

API: `FeatureAlignment(in_channels, out_channels=None, affine=False, eps=1e-5)`. `out_channels` defaults to `in_channels`. Arbitrary positive channel dims supported.

---

## Equations

Input `x ∈ ℝ^{B×C_in×H×W}`. For sample `b`, channel `c`:

**Instance normalisation** (statistics over the `H×W` plane of each sample/channel):

$$\mu_{b,c} = \frac{1}{HW}\sum_{h,w} x_{b,c,h,w}, \qquad \sigma^2_{b,c} = \frac{1}{HW}\sum_{h,w}\big(x_{b,c,h,w}-\mu_{b,c}\big)^2,$$

$$\hat{x}_{b,c,h,w} = \frac{x_{b,c,h,w}-\mu_{b,c}}{\sqrt{\sigma^2_{b,c}+\varepsilon}}\,,\qquad \varepsilon = 10^{-5}.$$

If `affine=True`: $\hat{x}_{b,c} \leftarrow \gamma_c\,\hat{x}_{b,c} + \beta_c$ with learnable $\gamma,\beta \in \mathbb{R}^{C_{in}}$.

**1×1 channel projection** (pointwise linear map across channels):

$$z_{b,o,h,w} = b_o + \sum_{c=1}^{C_{in}} W_{o,c}\,\hat{x}_{b,c,h,w}, \qquad W \in \mathbb{R}^{C_{out}\times C_{in}},\; b \in \mathbb{R}^{C_{out}}.$$

**ReLU:**

$$\text{out}_{b,o,h,w} = \max\big(0,\; z_{b,o,h,w}\big).$$

---

## Tensor Shapes

| Stage | Shape |
|-------|-------|
| input `x` | `(B, C_in, H, W)` |
| after InstanceNorm2d | `(B, C_in, H, W)` |
| after 1×1 Conv | `(B, C_out, H, W)` |
| after ReLU (output) | `(B, C_out, H, W)` |

Spatial dimensions `H, W` are preserved; only the channel dimension changes (`C_in → C_out`).

---

## Parameters

| Component | Parameters |
|-----------|-----------|
| InstanceNorm2d (`affine=False`, default) | 0 (no learnable params, no running stats) |
| InstanceNorm2d (`affine=True`, optional) | `2·C_in` (γ, β) |
| 1×1 Conv (with bias) | `C_in·C_out + C_out` |
| ReLU | 0 |
| **Total (default)** | **`C_in·C_out + C_out`** |

Example: `C_in = C_out = 256` → `256·256 + 256 = 65,792` params (~0.066 M). Lightweight.

---

## Expected Computational Complexity

Per forward pass, batch `B`, spatial size `H×W`:

| Component | Cost |
|-----------|------|
| InstanceNorm2d | `O(B · C_in · H · W)` (mean/var reduction + normalise) |
| 1×1 Conv | `O(B · C_in · C_out · H · W)` MACs |
| ReLU | `O(B · C_out · H · W)` |

**Dominant term:** the 1×1 convolution, `O(B · C_in · C_out · H · W)` — **linear in spatial resolution** (no quadratic/attention cost), consistent with the synopsis's lightweight-fusion objective.

Worked example (`B=2, C_in=C_out=256, H=W=96`): 1×1 conv ≈ `2·256·256·96·96 ≈ 1.21 G` MACs; InstanceNorm/ReLU negligible by comparison.

---

## Constraints Honoured
- New reusable module only: `modules/feature_alignment.py`.
- `models/maffnet.py` and `train.py` **not modified**.
- Module **not integrated** into any model or pipeline.
- Nothing executed (no training, no smoke test, no compile run).

## Notes for Future Integration (not done here)
- Instantiate one `FeatureAlignment` per stream prior to gated fusion in the future `models/stabilized_maffnet.py`; choose a common `out_channels` for both streams so the gate operates on matched widths.
- `affine=True` may be enabled if a learnable per-channel rescale proves helpful during ablation.
