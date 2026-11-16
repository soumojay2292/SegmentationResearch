# TransUNet Implementation Audit

**Scope:** `models/transunet.py` (88 lines) vs. official TransUNet (Chen et al., 2021, *"TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation"*) + reference impl (Beckschen/TransUNet).
**Constraint:** no code modified. Inspection only.

---

## 0. Performance Premise — Correction (read first)

The task states TransUNet was "strong on ISIC 2016, poor on ISIC 2018 (Dice ≈ 0.53)." **On-disk data contradicts this.** From `results/2026-04-28_15-19-03/summary.csv` + per-run `training_log.csv` (20-epoch runs):

| Dataset | Best Dice | **Final-epoch Dice** (what summary.csv reports) | Final Val Loss |
|---------|----------:|------------------------------------------------:|---------------:|
| ISIC 2016 | 0.8686 | 0.7714 | 0.8337 |
| ISIC 2017 | 0.5215 | **0.2675** | 2.2276 |
| ISIC 2018 | **0.8716** | **0.8716** | 0.3446 |

**Reality:** ISIC 2018 is the **strongest** (0.8716), ISIC 2017 is the **weakest** (final 0.2675). The "2018 poor" premise is inverted. No `transunet` `metrics.json` exists (those runs predate the metrics-export change), so the task's "0.87 / 0.78 / 0.53" set is **not on disk**.

The audit therefore explains the **real** behaviour: **extreme instability + dataset-sensitivity**, with large **best→final collapses** (2016: 0.8686→0.7714; 2017: 0.5215→0.2675). The summary reports the **final** epoch, so a late-training collapse (2017) reads as catastrophic. This collapse pattern is the true phenomenon, and the architecture below explains it.

---

## 1. Our Implementation vs. Official TransUNet

| Component | Official TransUNet | **Ours (`transunet.py`)** | Match? |
|-----------|--------------------|---------------------------|:------:|
| CNN backbone | **ResNet-50** (R50) hybrid, 3 stages → 1/16 feature map, ImageNet-pretrained | 2 plain conv blocks (`enc1` 3→64 s1, `enc2` 64→128 s2) — `:40-49` | ❌ |
| Downsample before ViT | ×16 | **×2** (only `enc2` stride 2) — `:46` | ❌ |
| Patch embedding | 16×16 patch proj on CNN map (≈196 tokens @224) | **None** — every feature cell is a token (`flatten(2)`) — `:79` | ❌ |
| Token count | ~196 (@224) | **H/2 × W/2 = 36,864** (@384) — measured | ❌ |
| ViT depth | **12** layers | **2** (`num_layers=2`) — `:58` | ❌ |
| Hidden dim | **768** | **128** (`embed_dim=128`) — `:36` | ❌ |
| Attention heads | **12** | **4** (`num_heads=4`) — `:36` | ❌ |
| MLP / feedforward dim | **3072** (4×768) | **256** (`embed_dim*2`) — `:56` | ❌ |
| Positional embedding | **Learned**, interpolated | Fixed **sinusoidal 2D** (`PositionalEncoding2D`) — `:6-32` | ❌ |
| Skip connections | **3** (CNN stages) | **1** (`x1` only) — `:85` | ❌ |
| Decoder | **CUP** — 4 cascaded upsample+conv stages | **1** stage (`up1` ×2 + `dec1`) — `:61-66` | ❌ |
| Decoder upsampling | ×16 back to input | **×2** (`ConvTranspose2d` stride 2) — `:61` | ❌ |
| Output head | 1×1 conv → classes | 1×1 conv → 1 (`out_conv`) — `:67` | ✅ |
| Pretrained weights | **R50+ViT-B_16, ImageNet-21k** | **None** — random init from scratch | ❌ |

**Verdict:** this is **not** TransUNet. It is a small 2-layer transformer wedged between a 2-conv encoder and a 1-stage decoder. It shares only the name and the "conv → transformer → conv-decoder" silhouette. Every capacity/pretraining/multi-scale property that makes official TransUNet strong is absent.

---

## 2. Architecture Verification (task item 2)

- **ViT architecture:** `nn.TransformerEncoder` of 2 `TransformerEncoderLayer`. Standard post-norm encoder. Not ViT-B. `:55-58`
- **Patch embedding:** absent — no patch projection; tokens = raw feature-map cells. `:79`
- **Patch size:** effectively **1** (per feature cell). Not 16.
- **Positional embeddings:** sinusoidal, recomputed each forward, added to feature map. Note bug-risk: `pe[0::2]` fills C/2 channels from `div_term[:C//2]` — works for even C=128 but is non-standard and **not learned** (can't adapt). `:18-32`
- **Skip connections:** one (`x1`, the 64-ch pre-downsample map). `:85`
- **Decoder:** single `ConvTranspose2d(128→64)` + one conv block. `:61-66`
- **Encoder-decoder fusion:** `torch.cat([x_up, x1])` once. `:85`
- **# transformer layers:** 2. **Hidden dim:** 128. **Heads:** 4. **MLP dim:** 256. `:36,56,58`
- **Output head:** `Conv2d(64, 1, 1)` → raw logits (no sigmoid — correct for BCEWithLogits). `:67`

---

## 3. Training Configuration (task item 3)

Source: `train_all.py:391-403`, `train.py:240-252`, `config.py`.

| Setting | Ours | Official | Assessment |
|---------|------|----------|------------|
| Pretrained weights | **disabled / none** | R50+ViT-B_16 ImageNet-21k | ❌ critical |
| Optimizer | AdamW | SGD (mom 0.9) | acceptable, but see LR |
| Learning rate | 1e-3 | 0.01 (SGD) | ❌ high for from-scratch transformer @ batch 2 |
| Scheduler | LinearLR 1.0→0.0 | poly (0.9) | acceptable shape; combined with instability → late collapse |
| Weight decay | 0.01 | 1e-4 | high for a transformer trained from scratch |
| Batch size | **2** | 24 | ❌ far too small for transformer / BN stability |
| Image size | 384 | 224 | larger → 36,864 tokens (pathological without patchify) |
| AMP | fp16 autocast | — | adds fp16 instability on top |

---

## 4. Are Pretrained ViT Weights Actually Loaded? (task item 4)

**No.** Searched `transunet.py`, `train.py`, `train_all.py`:
- No `timm`, no `load_state_dict`, no `.npz`/`R50+ViT-B_16` checkpoint, no `from_pretrained`.
- `TransUNet()` constructed with default random init only (`train_all.py:393`, `train.py:259`).
- `nn.TransformerEncoderLayer` weights are randomly initialised.

**The transformer trains from scratch on ≤2000 dermoscopy images.** Transformers are data-hungry and have weak inductive bias; without ImageNet-21k pretraining and with only 2 layers, attention cannot learn useful structure reliably → **high-variance, unstable training**. This is the single biggest deviation from official TransUNet and the root of the dataset-sensitivity.

---

## 5. What Explains the Cross-Dataset Variance (task item 5)

The variance is **not** a property of any single dataset — it is the **broken-from-scratch architecture amplifying noise**, surfaced through **final-epoch reporting**:

1. **No pretraining + 2-layer/128-dim transformer + batch 2** → the model is under-capacity and under-stabilised. It can reach decent **best** Dice when a run happens to converge (2016 0.8686, 2018 0.8716) but **collapses in late epochs** on others (2017 0.5215→0.2675; val loss 2.2276). High run/dataset variance is the expected outcome.
2. **Token explosion (36,864 tokens, no patchify)** → attention is applied at near-full resolution with a tiny model: O(N²) cost, weak signal, easily destabilised by fp16 + small batch.
3. **Single skip + ×2 decoder** → the decoder cannot reconstruct fine boundaries; output quality hinges on the transformer alone, which is fragile.
4. **Summary reports the FINAL epoch, not best** → a late collapse (2017) is recorded as 0.2675 even though best was 0.5215. This exaggerates the apparent "poor dataset."
5. **Dataset difficulty interacts with fragility:** ISIC 2017 (more class diversity / harder boundaries) is where the fragile model collapses; ISIC 2016/2018 happen to stabilise. Swap a seed and the "poor" dataset could change — the instability is the cause, not an intrinsic 2018/2017 property.

**Bottom line:** the model is too weak and too unstable to give consistent per-dataset results; the spread (≈0.87 down to ≈0.27 final) is the signature of an unpretrained, undersized, token-exploded transformer reported at its final (sometimes collapsed) epoch.

---

## 6. Findings — Classified

### 🔴 Critical Issues
1. **No pretrained ViT/ResNet weights** — trains from scratch; defeats TransUNet's core mechanism. (`transunet.py`, no loader anywhere)
2. **No ResNet-50 hybrid backbone** — 2 conv layers only; ×2 downsample vs ×16. (`:40-49`)
3. **Transformer drastically undersized** — 2 layers / 128 dim / 4 heads / MLP 256 vs 12 / 768 / 12 / 3072. (`:36,56,58`)
4. **No patch embedding → 36,864 tokens @384** — transformer at near-full-res; pathological O(N²), wrong scale. (`:79`)
5. **Single skip + single ×2 decoder stage** (no CUP) — under-decoded; cannot recover boundaries. (`:61-66,85`)
6. **Trains under fp16 + batch 2** — instability stacked on a fragile model → late-epoch collapse (cross-ref `ATTN_UNET_2018_CRASH_ANALYSIS.md` for the same fp16/BN-at-batch-2 failure mode).

### 🟡 Recommended Improvements
1. **Report best-epoch, not final-epoch** Dice — final-epoch reporting turns a 0.52 best into a 0.27 headline (2017). (reporting, not arch)
2. **Lower LR + warmup** (e.g. 1e-4 with linear warmup) or SGD+poly per paper — current AdamW 1e-3 from scratch is unstable.
3. **Increase effective batch** (gradient accumulation or smaller image) — batch 2 is far below transformer needs.
4. **Add real patch embedding** (patch 16 on a downsampled feature map) to cut tokens ~256× and restore ViT scale.
5. **Learned positional embeddings** instead of fixed sinusoidal.
6. **Reduce weight decay** to ~1e-4 for the transformer.

### 🟢 Optional Improvements
1. **Adopt the official R50+ViT-B_16 backbone via `timm`** with ImageNet-21k weights — the proper fix; would likely lift and stabilise all three datasets.
2. **CUP-style multi-skip decoder** (3 skips, 4 upsample stages).
3. **ImageNet normalisation** of inputs (cross-ref `PREPROCESSING_AUDIT.md` P2) — required if pretrained weights are adopted.
4. **bf16 autocast** instead of fp16 to remove overflow instability.
5. **Multi-seed runs + report mean±std** — to characterise (and tame) the variance.

---

## 7. Summary

| Question | Answer |
|----------|--------|
| Is this official TransUNet? | No — name only. No backbone, no pretraining, ~16× smaller transformer, no patchify, 1-stage decoder |
| Pretrained ViT loaded? | No — random init from scratch |
| LR/optimizer suitable? | AdamW 1e-3 + batch 2 + fp16 from scratch → unstable |
| Why the cross-dataset spread? | Under-capacity, unpretrained, token-exploded transformer → high instability; **final-epoch reporting** captures late collapses (worst on ISIC 2017, not 2018) |
| On-disk match to premise? | No — 2018 is strongest (0.8716), 2017 weakest (0.2675 final / 0.5215 best) |

---

## Evidence References
- `models/transunet.py:36,56,58` — embed_dim 128, heads 4, MLP 256, 2 layers
- `models/transunet.py:46` — only one stride-2 (×2 downsample)
- `models/transunet.py:79` — flatten → 36,864 tokens @384 (no patch embedding)
- `models/transunet.py:61-66,85` — single skip, single ×2 decoder stage
- `train_all.py:393`, `train.py:259` — `TransUNet()` random init, no weight load
- `results/2026-04-28_15-19-03/summary.csv` — final-epoch Dice 0.7714 / 0.2675 / 0.8716
- `experiments/transunet_ISIC_*_2026-04-28_15-19-03/checkpoints/training_log.csv` — best Dice 0.8686 / 0.5215 / 0.8716 (best→final collapse)
- official: Chen et al. 2021 — R50+ViT-B_16, 12 layers/768/12 heads/3072 MLP, patch 16, CUP decoder, ImageNet-21k pretrained, SGD lr 0.01
