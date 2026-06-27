# Frequency Branch — Implementation Report

**Task:** create the reusable Frequency Branch from the synopsis (§1.4). Module only — not integrated. `maffnet.py` and `trainer.py` untouched. Nothing executed.

**File created:** `modules/frequency_branch.py` → class `FrequencyBranch`.

---

## 1. Design

Pipeline (exact to spec):

```
x  ->  rfft2 (fp32)  ->  radial high-pass mask  ->  irfft2 (fp32)  ->  out
```

- **fp32 FFT** — `torch.fft.rfft2(x.float(), norm="ortho")`. Forced single precision because half-precision FFT overflows under AMP.
- **Radial high-pass mask** — a **fixed, non-learnable** binary mask `M(u,v) ∈ {0,1}` that keeps spectral coefficients whose radial frequency `r ≥ cutoff` and zeros those below. Cutoff is configurable.
- **Inverse FFT** — `torch.fft.irfft2(·, s=(H,W), norm="ortho")`, reconstructing a **real** spatial map at the **original resolution**.
- Output cast back to the input dtype. The branch isolates boundary-sensitive (edge) content by discarding the low-frequency bulk.

API: `FrequencyBranch(cutoff: float = 0.1)`. Works for arbitrary `(B, C, H, W)`. The mask is cached per `(H, W, device)`.

---

## 2. Mathematical Formulation

Input `x ∈ ℝ^{B×C×H×W}`. For each sample/channel, the 2-D real FFT (orthonormal):

$$X(u,v) = \frac{1}{\sqrt{HW}} \sum_{h=0}^{H-1}\sum_{w=0}^{W-1} x(h,w)\, e^{-j 2\pi\left(\frac{uh}{H} + \frac{vw}{W}\right)}, \quad u\in[0,H),\ v\in[0,\lfloor W/2\rfloor].$$

High-pass filtering is an element-wise mask in the frequency domain:

$$X_{\text{hp}}(u,v) = M(u,v)\,\cdot\,X(u,v).$$

Reconstruction by the orthonormal inverse real FFT at the original size `(H,W)`:

$$\text{out}(h,w) = \frac{1}{\sqrt{HW}}\sum_{u,v} X_{\text{hp}}(u,v)\, e^{+j 2\pi\left(\frac{uh}{H}+\frac{vw}{W}\right)} \;\in\; \mathbb{R}.$$

Because the mask zeros the DC term (`r=0 < cutoff`), the output is **zero-mean** per channel — it carries only edge/texture (high-frequency) structure.

---

## 3. High-pass Mask Construction

Frequency coordinates follow the `rfft2` layout (DC at index `[0,0]`, **not** fft-shifted):

- vertical (full spectrum): `fy = torch.fft.fftfreq(H) ∈ [-0.5, 0.5)` cycles/pixel,
- horizontal (half spectrum): `fx = torch.fft.rfftfreq(W) ∈ [0, 0.5]` cycles/pixel.

Radial frequency and binary mask:

$$r(u,v) = \sqrt{f_y(u)^2 + f_x(v)^2}, \qquad M(u,v) = \begin{cases} 1 & r(u,v) \ge \text{cutoff} \\ 0 & r(u,v) < \text{cutoff} \end{cases}.$$

- `cutoff ∈ [0, ~0.707]` (max radius `√(0.5²+0.5²)≈0.707`). Default `0.1`.
- `cutoff → 0`: pass-through (only exact DC removed). Larger cutoff → more aggressive high-pass (only sharper edges survive).
- Mask shape `(H, W//2+1)`, broadcast over batch and channels.
- Cached per `(H, W, device)`; recomputed only when a new spatial size/device appears.

*(Ideal binary high-pass can introduce spatial ringing (Gibbs). A smooth radial mask — e.g. Butterworth/Gaussian — would mitigate this; left as a future option since the spec asks for a fixed mask.)*

---

## 4. Tensor Shapes

| Stage | Shape | dtype |
|-------|-------|-------|
| input `x` | `(B, C, H, W)` | input dtype (e.g. bf16/fp32) |
| `x.float()` | `(B, C, H, W)` | fp32 |
| `rfft2(...)` | `(B, C, H, W//2+1)` | complex64 |
| mask `M` | `(H, W//2+1)` | fp32 (broadcast) |
| `X·M` | `(B, C, H, W//2+1)` | complex64 |
| `irfft2(..., s=(H,W))` | `(B, C, H, W)` | fp32 |
| output | `(B, C, H, W)` | input dtype |

Spatial resolution `(H, W)` and channel count `C` are preserved exactly (the `s=(H,W)` argument guarantees correct even/odd reconstruction).

---

## 5. Parameter Count

**0 learnable parameters.** The mask is fixed (binary, derived from `cutoff`); `cutoff` is a stored Python float, not a `Parameter`. No buffers tracked in `state_dict` (mask is cached in a plain dict, recomputed on load). Memory: one cached mask tensor of `H·(W//2+1)` fp32 values per spatial size (e.g. 384×193 ≈ 74 k floats ≈ 0.3 MB).

---

## 6. Computational Complexity

Per forward, batch `B`, channels `C`, spatial `H×W`:

| Step | Cost |
|------|------|
| `rfft2` | `O(B · C · H · W · log(H·W))` |
| mask multiply | `O(B · C · H · (W//2+1))` |
| `irfft2` | `O(B · C · H · W · log(H·W))` |

**Dominant:** the two FFTs, `O(B·C·H·W·log(HW))` — quasi-linear in pixels, no quadratic/attention cost. Worked example (`B=2, C=256, H=W=96`): ~`2·256·96·96·log₂(9216) ≈ 2·256·9216·13.2 ≈ 6.2×10⁷` butterfly ops per transform; negligible relative to the surrounding convolutions.

---

## 7. Numerical Stability Considerations

- **fp32 FFT** (`x.float()`): avoids the half-precision FFT overflow that produces NaN under AMP (the original MAFFNet failure class).
- **No `abs()` / `sqrt()` on the spectrum.** The pipeline is `rfft2 → (× real mask) → irfft2`, all linear operations. There is **no complex-magnitude or Euclidean-norm step**, so the backward has **no singularity** — this avoids the `SqrtBackward0` / complex-`abs` gradient blow-ups documented in `FGFF_NUMERICAL_ANALYSIS_REPORT.md`. The mask multiply is a constant `{0,1}` scaling: its backward is the same mask, always finite.
- **Orthonormal norm (`norm="ortho"`)**: keeps the energy scale of forward/inverse transforms matched, preventing amplitude growth across the round trip.
- **Exact reconstruction size** (`s=(H,W)`): correct for both even and odd `H, W`; no off-by-one resolution drift.
- **DC removal** → zero-mean output, so the branch cannot inject a large constant offset downstream.
- **Backward through `rfft2`/`irfft2`** is well-defined and finite for finite inputs; combined with the points above, the branch is gradient-stable end to end.

---

## Constraints Honoured
- New reusable module only: `modules/frequency_branch.py`.
- fp32 FFT; original spatial resolution preserved; configurable cutoff; fixed (non-learnable) mask.
- `models/maffnet.py` and `training/trainer.py` **not modified**.
- Module **not integrated** into any model or pipeline.
- Nothing executed (no training, smoke test, or compile run).

## Notes for Future Integration (not done here)
- Pair the output of `FrequencyBranch` with the spatial stream via `FeatureAlignment` (Task 1) and a gated fusion (future task) inside `models/stabilized_maffnet.py`.
- If ringing from the ideal mask is observed, switch to a smooth radial mask and/or make `cutoff` learnable in a later iteration.
