# Lightweight Gated Fusion — Implementation Report

**Task:** create the reusable Gated Fusion module from the synopsis (§1.4). Module only — not integrated. `maffnet.py` and `trainer.py` untouched. Nothing executed.

**File created:** `modules/gated_fusion.py` → class `GatedFusion`.

---

## 1. Architecture

```
F_spatial ─┐
           ├─ Concatenate (channel dim) ─> 1x1 Conv (2C->C) ─> Sigmoid ─> g
F_freq    ─┘
                                                                       │
out = g * F_spatial + (1 - g) * F_freq  <───────────────────────────────┘
```

- Concatenate the two streams along channels → `(B, 2C, H, W)`.
- A single **1×1 convolution** (`2C → C`) produces a gate logit per channel per pixel.
- **Sigmoid** maps logits to `g ∈ (0,1)`.
- **Convex weighted fusion**: `out = g·F_spatial + (1−g)·F_freq`.
- No attention, no MultiHeadAttention, no Transformer — only one 1×1 conv, a sigmoid, and two element-wise products.

API: `GatedFusion(channels)`. Both inputs and output share width `C` and spatial size `(H, W)`. Arbitrary `C`. If streams differ in width, align to a common `C` first (FeatureAlignment, Task 1).

---

## 2. Equations

Inputs `F_s, F_f ∈ ℝ^{B×C×H×W}`. Let `W_g ∈ ℝ^{C×2C}`, `b_g ∈ ℝ^{C}` be the 1×1 conv weights.

$$g = \sigma\!\Big( W_g * \big[\,F_s \,\|\, F_f\,\big] + b_g \Big), \qquad g \in (0,1)^{B\times C\times H\times W},$$

$$\text{out} = g \odot F_s + (1 - g) \odot F_f,$$

where `[·‖·]` is channel concatenation, `*` is the 1×1 convolution (a per-pixel linear map across channels), `σ` the logistic sigmoid, and `⊙` element-wise multiplication. The gate is **spatially varying and channel-wise**: a distinct weight at every `(b, c, h, w)`.

---

## 3. Tensor Shapes

| Stage | Shape |
|-------|-------|
| `F_spatial` | `(B, C, H, W)` |
| `F_freq` | `(B, C, H, W)` |
| concat | `(B, 2C, H, W)` |
| 1×1 conv | `(B, C, H, W)` |
| sigmoid `g` | `(B, C, H, W)` |
| `g·F_s + (1−g)·F_f` (output) | `(B, C, H, W)` |

Spatial resolution `(H, W)` and channel count `C` are both preserved.

---

## 4. Parameter Count

Single 1×1 conv `2C → C` with bias:

$$\#\text{params} = (2C \cdot C \cdot 1 \cdot 1) + C = 2C^2 + C.$$

Example `C = 256`: `2·256² + 256 = 131,328` (~0.13 M). No other learnable parameters (sigmoid and the fusion are parameter-free).

---

## 5. Computational Complexity

Per forward, batch `B`, spatial `H×W`:

| Step | Cost |
|------|------|
| 1×1 conv (`2C→C`) | `O(B · 2C · C · H · W) = O(B·C²·H·W)` MACs |
| sigmoid | `O(B·C·H·W)` |
| `g·F_s + (1−g)·F_f` | `O(B·C·H·W)` |

**Dominant:** the 1×1 conv, `O(B·C²·H·W)` — **linear in the number of pixels** `H·W`. Worked example (`B=2, C=256, H=W=96`): `2·2·256²·96·96 ≈ 2.4×10⁹` MACs. Activation memory: `O(B·C·H·W)` for the concat/gate.

---

## 6. Numerical Stability Analysis

- **Bounded gate:** `g = σ(·) ∈ (0,1)`, so `g` and `(1−g)` are both bounded; the output is a **convex combination** of `F_s` and `F_f`. Each output element lies between the two inputs — fusion **cannot amplify** activations, preventing blow-up.
- **No singular operations:** no division, no `sqrt`, no complex `abs`. The only nonlinearity is sigmoid. Gradients are finite everywhere: `dσ/dx = σ(1−σ) ∈ (0, 0.25]` (bounded), and the conv backward is linear.
- **Backward of the fusion:** `∂out/∂F_s = g`, `∂out/∂F_f = (1−g)`, both in `(0,1)` — well-scaled, no exploding gradient. `∂out/∂g = F_s − F_f` (finite for finite inputs).
- **Init behaviour:** with default conv init and zero bias, `g ≈ 0.5` initially → a balanced average of the two streams, a stable starting point that the gate refines during training.
- **AMP-safe:** all ops (conv, sigmoid, multiply-add) are autocast-stable in both fp16 and bf16; no FFT/magnitude here. No special precision handling required.

---

## 7. Why This Is Lighter Than Attention-Based Fusion

| Aspect | Gated Fusion (this module) | Cross/Multi-Head Attention fusion |
|--------|----------------------------|-----------------------------------|
| Token interaction | none (pixel-local 1×1 conv) | all-pairs over `N = H·W` tokens |
| Compute | `O(B·C²·H·W)` — **linear in pixels** | `O(B·N²·C) = O(B·(H·W)²·C)` — **quadratic in pixels** |
| Activation memory | `O(B·C·H·W)` | `O(B·heads·N²)` attention matrix |
| Learnable params | `2C² + C` (one 1×1 conv) | ~`4C²` (Q,K,V,O projections) + norms |
| Operations | 1 conv + sigmoid + 2 muls | QKV projections, scaled dot-product, softmax, output proj |

At dermoscopic resolution the attention matrix is the bottleneck: for `H=W=96`, `N=9216` and `N² ≈ 8.5×10⁷` entries **per head**, which is exactly the quadratic cost the synopsis avoids. The gated fusion replaces this with a per-pixel 1×1 convolution — no `N²` term, no multi-head projections, no softmax — so it scales **linearly** with resolution and uses a fraction of the parameters and memory, while still learning a spatially adaptive blend of the two streams.

---

## Constraints Honoured
- New reusable module only: `modules/gated_fusion.py`.
- Arbitrary channel width; spatial resolution and channel dimension preserved.
- No attention / MultiHeadAttention / Transformer; lightweight (one 1×1 conv).
- `models/maffnet.py` and `training/trainer.py` **not modified**.
- Module **not integrated**; nothing executed.

## Notes for Future Integration (not done here)
- Intended use: `out = GatedFusion(C)(F_spatial_aligned, F_freq_aligned)` after both streams pass through `FeatureAlignment` (Task 1), with `F_freq` produced by `FrequencyBranch` (Task 2). Assembly belongs in a future `models/stabilized_maffnet.py`.
