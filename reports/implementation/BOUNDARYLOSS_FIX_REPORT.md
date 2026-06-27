# BoundaryLoss Fix Report

**Scope:** the two changes requested, in `training/trainer.py` only. No training, smoke test, or model run was executed (per instruction). FGFF, MMPA, SAM, the optimizer, and the model architecture are untouched.

---

## Files Modified

| File | Change |
|------|--------|
| `training/trainer.py` | (1) epsilon-stabilized Sobel magnitude in `BoundaryLoss._edges`; (2) BDS computation skipped when its warmup weight is exactly zero |

No other file was modified. `models/maffnet.py` (FGFF, MMPA, SAM wrapper), and the optimizer construction were not changed.

---

## Exact Lines Changed

### Change 1 — `BoundaryLoss._edges` (≈ line 62)

**Before:**
```python
gx = F.conv2d(x, kx, padding=1)
gy = F.conv2d(x, ky, padding=1)
return (gx ** 2 + gy ** 2).sqrt().clamp(0, 1)
```
**After:**
```python
gx = F.conv2d(x, kx, padding=1)
gy = F.conv2d(x, ky, padding=1)
# +1e-6 keeps the sqrt gradient finite at flat regions (gx=gy=0);
# without it, d/du sqrt(u) = 1/(2*sqrt(u)) -> inf at u=0
return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6).clamp(0, 1)
```

### Change 2 — BDS block in `MAFFNetLoss.forward` (≈ lines 159–177)

**Before:**
```python
# --- N1: Boundary-aware Deep Supervision ---
bds_loss = torch.zeros((), device=y.device)
if self.use_bds and self.lambda_bds > 0:
    sy = self._edge_mag(y)
    acc = 0.0
    for wi, p in zip(w, heads):
        prob = torch.sigmoid(p)
        if self.smooth_pred:
            prob = F.avg_pool2d(prob, kernel_size=3, stride=1, padding=1)
        acc = acc + wi * F.l1_loss(self._edge_mag(prob), sy)
    bds_loss = acc / float(sum(w))
    ramp = 1.0 if self.t_warm <= 0 else min(1.0, epoch / float(self.t_warm))
    total = total + (self.lambda_bds * ramp) * bds_loss
```
**After:**
```python
# --- N1: Boundary-aware Deep Supervision ---
# Compute the warmup weight first; when it is exactly zero, skip the
# entire BDS computation so no autograd graph is built. This prevents a
# zeroed term from injecting 0 * inf = NaN through the edge sqrt.
bds_loss = torch.zeros((), device=y.device)
ramp = 1.0 if self.t_warm <= 0 else min(1.0, epoch / float(self.t_warm))
bds_weight = self.lambda_bds * ramp
if self.use_bds and bds_weight > 0:
    sy = self._edge_mag(y)
    acc = 0.0
    for wi, p in zip(w, heads):
        prob = torch.sigmoid(p)
        if self.smooth_pred:
            prob = F.avg_pool2d(prob, kernel_size=3, stride=1, padding=1)
        acc = acc + wi * F.l1_loss(self._edge_mag(prob), sy)
    bds_loss = acc / float(sum(w))
    total = total + bds_weight * bds_loss
```

Static check: `python -m py_compile training/trainer.py` → **OK** (no model executed).

---

## Why the Fix Is Mathematically Correct

### Change 1 — epsilon-stabilized magnitude
Let `u = gx² + gy² ≥ 0`. The original edge map is `sqrt(u)` with derivative

```
d/du sqrt(u) = 1 / (2*sqrt(u))  -> +inf  as u -> 0.
```

At flat regions (gx = gy = 0, e.g., the interior/background of a near-binary prediction), `u = 0`, so the backward of `SqrtBackward0` returns a non-finite value. The replacement uses `sqrt(u + ε)` with `ε = 1e-6`:

```
d/du sqrt(u+ε) = 1 / (2*sqrt(u+ε))  <=  1 / (2*sqrt(ε)) = 1 / (2e-3) = 500.
```

The gradient is now **bounded by 500** for every pixel, removing the singularity. The forward value is essentially unchanged: for `u >> ε`, `sqrt(u+ε) ≈ sqrt(u)`; at `u = 0`, `sqrt(ε) = 1e-3`, which is negligible and still inside the existing `clamp(0, 1)`. So the edge maps used by BDS and by the EWR boundary prior are numerically the same, while the gradient is finite everywhere.

### Change 2 — skip BDS when weight is zero
Mathematically, `total = seg + (λ_b·ramp)·BDS`. When the warmup weight `λ_b·ramp = 0`, the BDS term contributes nothing to the value of `total`. But if the BDS subgraph is still built, autograd evaluates the local Jacobian of the (singular) edge `sqrt` and multiplies it by the zero upstream gradient: `0 · inf = NaN`. Gating on `bds_weight > 0` and not building the subgraph when it is zero yields a result that is **identical in value** (BDS weight 0 ⇔ BDS absent) but **safe in the backward**, because the singular operation is never entered. This is a redundant safeguard given Change 1, and it also avoids the wasted computation of a term that does not affect the loss.

---

## Expected Effect on Backward Stability

- The first non-finite backward operation identified by anomaly detection (`SqrtBackward0` in `BoundaryLoss._edges`) is **eliminated**: the edge gradient is bounded (≤ 500) at every pixel, including flat regions.
- During warmup (`ramp = 0`, e.g. epoch 0 with `t_warm = 5`), BDS is no longer built, so the `0 · inf = NaN` path cannot occur.
- After warmup (`ramp > 0`), BDS is active and the ε-stabilized `sqrt` keeps its gradient finite, so the active phase is also stable.
- The forward pass and all loss terms were already finite; with a finite first backward, the optimizer step no longer writes non-finite values into parameters, so iteration 2 (and beyond) is expected to remain finite and training should progress.

**Runtime verification (one-batch overfit, 5-epoch smoke) was NOT performed here, as instructed.** Acceptance — iteration 2 finite, gradients finite, optimizer updating, Dice → > 0.95 — remains to be confirmed in a separate run.

---

## Constraints Honoured
- Only the two requested changes, both in `training/trainer.py`.
- FGFF, MMPA, SAM, optimizer, and model architecture unchanged.
- No training, no smoke test, no model execution (only a static `py_compile`).

## Evidence References
- `FGFF_NUMERICAL_ANALYSIS_REPORT.md` (anomaly: `SqrtBackward0` @ `trainer.py:62`, via BDS `_edge_mag(prob)`)
- `training/trainer.py` `BoundaryLoss._edges` (Change 1); `MAFFNetLoss.forward` BDS block (Change 2)
