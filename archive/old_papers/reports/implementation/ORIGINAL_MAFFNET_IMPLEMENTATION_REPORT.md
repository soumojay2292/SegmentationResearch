# Original MAFF-Net — Faithful Reproduction Report

**New file:** [`models/maffnet_original.py`](../../models/maffnet_original.py)
**Source paper:** Y. Zhou, H. Gong, R. Qiu, S. Wei, Z. Li, W. Zhang, *"MAFF-Net: SAM-powered Mixed Multi-scale Perception Adaptation and frequency-guided feature fusion for robust skin lesion segmentation,"* Biomedical Signal Processing and Control 113 (2026) 109222.
**Goal:** reproduce the published architecture exactly as described in Section 3 (Eqs. 1–14, Figs. 1–6), with no improvements from our later research models and no edits to any existing file.

The module is **self-contained**: it does not import or reuse any block from `models/maffnet.py` or `models/stabilized_maffnet.py`. Where the existing repository code diverges from the paper, the **paper** definition was used (see the comparison table in §6).

---

## 1. Paper-module → code mapping

| Paper element | Paper ref | Code (class / function in `maffnet_original.py`) |
|---|---|---|
| Frozen SAM (Hiera, `sam2_hiera_large`) encoder, 4 stages, channels 144/288/576/1152 | Eq. 1; §3 | `FrozenSAMEncoder` (frozen; loads real SAM2 trunk, else shape-compatible `_DummyHieraTrunk`) |
| MMPA projection `Z = GELU(W_p·LN(X)+b_p)`, C→32 | Eq. 2 | `MMPA.ln`, `MMPA.proj`, `MMPA.proj_act` |
| Three perceptionNets, dilations {3,5,7}, depthwise k3 (d=1 then d=r) | Eq. 3; Fig. 3 | `PerceptionNet`; `MMPA.experts` |
| WetGeneration weight generator: GAP → Linear → GELU → Linear → Softmax (3 experts) | Eq. 4 | `WeightGenerator`; `MMPA.weight_gen` |
| Fused experts `E_fused = Σ α_j·E_j`, scaling Γ (init 1e-7), residual, reconstruct →C | Eq. 5 | `MMPA.forward` (fusion, `self.gamma`, residual, `self.recon`) |
| Frequency transform: gray = mean(RGB)·255 → FFT → fft-shift → magnitude → min-max → pseudo-RGB | Eq. 6 | `FFTMagnitude` |
| Resolution alignment `f_in = Interpolate(f_freq, 0.25)` | Eq. 7 | `FrequencyEncoder.forward` (`F.interpolate(..., scale_factor=0.25)`) |
| Lightweight frequency Transformer: depthwise Q/K/V, learnable temperature T, residual + LN + FFN, 4 stages | Eq. 8; §3.2.2 | `FreqTransformerBlock`; `FrequencyEncoder` |
| FGFF: branches `b_i=DilatedConv(Conv1x1(f_spatial))` {1,3,5,7}; `Conv3x3(Cat)`; freq attn `σ(Conv3x3([Avg;Max]))`; `f_fused = β·(attn·spatial)` | Eq. 9; Fig. 4 | `FGFF` |
| EDB: upsample → pad-to-match → concat → DoubleConv | Eq. 10; Fig. 5 | `EDB` (`F.interpolate`, `torch.cat`, `DoubleConv`) |
| EDB channel-attention residual `attn=σ(Conv([avg;max]))`, `x_out=x+γ·(attn·x)` | Eq. 11 | `EDB.attn_conv`, `EDB.gamma` |
| Deep-supervision seg heads p1–p4 | §3.3; Fig. 1 | `SegHead`; `head1`…`head4` |
| RDB: grouped ConvTranspose2d (k4,s2,p1), hierarchical concat + Conv+ReLU, →3ch, interpolate; train-only MSE | Eq. 12; Fig. 6 | `RDB` |
| Loss `Σ λ_i·Dice(Up(p_i),y) + λ_rec·MSE(rec,x)`, λ={1,1,1,0.01} | Eq. 13 | `MAFFNetOriginalLoss` |
| Linear LR schedule (start 1.0 → end 0.001), AdamW lr 1e-3 wd 0.01, bs 12, 100 ep, 384² | Eq. 14; §4.2 | *training recipe — documented only; not implemented here (no trainer added)* |

---

## 2. Assumptions (paper under-specifies these)

1. **WetGeneration widths.** The weight generator is applied to the 32-d projected feature `Z`; the paper's "C→C/2→3" is realised as `Linear(32→16) → GELU → Linear(16→3) → Softmax`.
2. **Eq. 3 Permute.** The explicit `Permute(Z,(0,3,1,2))` is read as an implementation artefact that places the 32 reduced channels on the convolution channel axis. Depthwise convolutions with `groups=32` over `(B,32,H,W)` give the equivalent depthwise behaviour and are used directly.
3. **Eq. 5 second line.** The paper writes `P_enhanced = P + Permute(α ⊙ (E_fused ⊗ Γ))`; since `α` is already folded into `E_fused` (Eq. 5, line 1), the re-appearance of `α` is treated as redundant notation. Implemented as `P_enhanced = Z + (E_fused · Γ)` with reverse permute implied.
4. **EDB attention shape.** The stated output `[0,1]^{1×H×W}` indicates a **spatial** attention map; avg/max are taken over the **channel** axis (CBAM-style), concatenated, and convolved to one channel.
5. **Frequency-Transformer hyperparameters** (embedding width, head count, depth) are not given. Faithful but reasonable values were chosen so the four frequency stages pair channel-for-channel with the spatial stages `(144,288,576,1152)` at strides 4/8/16/32. `num_heads=4`, `ffn_ratio=2`, one block per stage.
6. **RDB feature source.** The paper feeds RDB "four feature maps … generated during the encoding stage"; the four **fused** (post-FGFF) stage features `g1…g4` are used as `s1…s4`.
7. **Loss head count.** Eq. 13 lists weights `{1,1,1,0.01}` (four heads) while the summation index shows three Dice terms plus the reconstruction term; the four deep-supervision heads `p1…p4` are each supervised with the four listed weights, the coarsest (`p4`) receiving `0.01`.
8. **Encoder fallback.** When the real SAM2 Hiera-large checkpoint is unavailable, a shape-compatible frozen placeholder emits the paper channel widths so the network is constructible and shape-testable. It is **not** part of the architecture and is replaced by the real frozen trunk when `sam2` + the checkpoint are present.
9. **min-max normalisation (Eq. 6)** is implemented exactly as written, with no added epsilon, to stay faithful (no extra numerical stabilisation).

---

## 3. Tensor flow

Input `I ∈ R^{B×3×384×384}`.

```
                          ┌──────────────── Spatial stream ────────────────┐
I ──► FrozenSAMEncoder ──► X1..X4  (144,288,576,1152 @ H/4,H/8,H/16,H/32)
                              │ per stage
                              ▼
                          MMPA(X_i) ──► spatial_i  (same C_i)
                                                   │
                          ┌──────────── Frequency stream ───────────┐      │
I ──► FFTMagnitude ──► f_freq(3ch) ──► Interp×0.25 ──► FreqEncoder ──► freq_i (C_i)
                                                   │                  │
                                                   ▼                  ▼
                          FGFF(spatial_i, freq_i) ──► g_i   (i = 1..4)
                                                   │
   ┌── deep supervision ──┐                        ▼
   p4 ◄── head4(g4)                         g4 ──► EDB3(g4,g3) ──► d3 ──► head3 ──► p3
                                            d3 ──► EDB2(d3,g2) ──► d2 ──► head2 ──► p2
                                            d2 ──► EDB1(d2,g1) ──► d1 ──► head1 ──► p1
                                                   │
                          RDB([g1,g2,g3,g4]) ──► rec (3ch, train-only)
```

**Outputs** — train: `(p1, p2, p3, p4, rec)`, all seg maps upsampled to `384×384`, `rec ∈ R^{B×3×384×384}`. Eval: `p1` only (RDB skipped).

Verified shapes (`python models/maffnet_original.py`, batch 2):
```
p1: (2,1,384,384)  p2: (2,1,384,384)  p3: (2,1,384,384)  p4: (2,1,384,384)
rec: (2,3,384,384)  loss: 2.4914  eval: (2,1,384,384)
```

---

## 4. Parameter summary

Measured with the **placeholder** encoder (real SAM2 weights absent in this environment):

| Quantity | Value |
|---|---|
| Total parameters | ≈ 193.3 M |
| Trainable parameters | ≈ 185.4 M |
| Encoder (frozen) | all `requires_grad = False` |

**Paper-reported budget:** frozen SAM-ViT-H ≈ 227 M (876 MB); trainable MMPA ≈ 1.7 M, FGFF ≈ 2.2 M, EDB ≈ 0.4 M, RDB ≈ 7.8 M; **total trainable ≈ 14.8 M (6.5 %)**.

**Why the measured trainable count is higher than 14.8 M — and how to close it.** The paper's per-module budgets (EDB 0.4 M, FGFF 2.2 M) imply that fusion and decoding run at **substantially reduced channel widths that the text never specifies**. This reproduction instead pairs every fusion/decoder stage **channel-for-channel** with the encoder stages (up to 1152 wide), as the per-stage notation in Eqs. 1/9/10 suggests, which inflates the count. The module **structure and equations are faithful**; only the unstated widths differ. Matching the 14.8 M figure exactly would require the authors' unpublished reduced widths for the frequency encoder, FGFF, and EDB (e.g. a common narrow decoder dimension), which are not derivable from the paper. This is recorded as a known, paper-driven limitation rather than a deviation from any stated design. Substituting the real frozen SAM-ViT-H trunk additionally moves the frozen count toward the reported ≈ 227 M.

---

## 5. Confirmation: existing models were NOT modified

Only **one new file** was created — `models/maffnet_original.py` — plus this report. None of the protected files were edited:

| File | Status |
|---|---|
| `models/maffnet.py` | **unchanged** (current stable model) |
| `models/stabilized_maffnet.py` | **unchanged** (our proposed model) |
| `train.py` | **unchanged** (no new model registered) |
| `training/trainer.py` | **unchanged** |

No imports were altered, no source file was renamed, no training or experiment was run. The shape check uses only random tensors via `python models/maffnet_original.py`.

Resulting `models/` (three independent models):
```
models/
├── maffnet.py            # current stable implementation (unchanged)
├── stabilized_maffnet.py # our proposed architecture (unchanged)
└── maffnet_original.py   # faithful reproduction of the published MAFF-Net  (NEW)
```

> **Repository note.** Two checkouts exist on disk: `Desktop/New_Sm_P/SegmentationResearch` (this one — contains `maffnet.py` **and** `stabilized_maffnet.py`, matching the requested three-model tree) and `Desktop/SegmentationResearch` (older; lacks `stabilized_maffnet.py`). The new file was placed in the former because it is the only checkout matching the requested final structure. It is self-contained, so it can be copied verbatim into the other checkout if intended.

---

## 6. Paper vs. existing `maffnet.py` (key faithfulness corrections)

| Aspect | Paper (used here) | Existing `maffnet.py` (not reused) |
|---|---|---|
| Encoder stage channels | 144 / 288 / 576 / 1152 | 96 / 192 / 384 / 768 |
| MMPA dilations | {3, 5, 7} | {3, 5, 17} |
| MMPA projection width | C → 32 | bottleneck variant |
| MMPA scaling Γ | fixed vector, init 1e-7 | n/a |
| Weight generator | GAP → Lin(32→16) → GELU → Lin(16→3) → Softmax | SE-style gating |
| Frequency stream | grayscale FFT magnitude + 4-stage lightweight Transformer | absent (FGFF only) |
| FGFF dilations | {1, 3, 5, 7} + freq attn + learnable β | different |
| Decoder | EDB (concat + spatial-attn residual, γ) | generic decoder block |
| Loss | Dice deep-sup {1,1,1,0.01} + reconstruction MSE | Dice + BCE + boundary + edge-weighted recon |

No feature-alignment module, lightweight gated fusion, boundary-aware supervision, bf16-specific code, or extra numerical stabilisation is present — those belong to the later research models.

---

*Status: Draft v1 / Dependencies: real SAM2 checkpoint for end-to-end run, supervisor confirmation of target checkout, authors' unstated fusion/decoder widths for exact param match.*
