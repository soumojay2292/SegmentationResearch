# Reproducibility Checklist — MAFFNet / ISIC Segmentation

**Scope:** what an external researcher needs to bit-reproduce results in `results/2026-04-28_15-19-03/summary.csv`. Audits the current repo state, lists gaps, gives concrete fixes.

Status legend: ✅ in repo, ⚠️ partial / scattered, ❌ missing.

---

## 1. Hyperparameters

### 1.1 Training
| Hyperparameter | Value | Source | Status |
|----------------|------:|--------|:------:|
| Epochs | 20 | `train_all.py:31`, `EPOCHS = 20` | ✅ |
| Batch size | 2 | `train_all.py:312, 314` | ✅ |
| Image size | 384 × 384 | `train_all.py:309, 310` | ✅ |
| Learning rate | 1e-3 | `train_all.py:334, 349, 362, …` | ✅ |
| Weight decay | 0.01 | same call sites | ✅ |
| Optimiser | AdamW (default β₁=0.9, β₂=0.999, ε=1e-8) | `training/simple_trainer.py:83`, `training/trainer.py` (analogous) | ⚠️ defaults not documented |
| LR schedule | LinearLR(start=1.0, end=0.0, iters=epochs) | `training/simple_trainer.py:84–89` | ✅ |
| Grad clip | max-norm 1.0 | `simple_trainer.py:139` | ✅ |
| AMP | autocast("cuda") + GradScaler | `simple_trainer.py:91, 132, 167` | ⚠️ dtype not pinned (fp16 vs. bf16 ambiguous) |
| Pos-weight (BCE) | 2.0 | `simple_trainer.py:38`, `training/trainer.py:81` | ✅ |
| Dice smooth | 1.0 | `training/trainer.py:35` | ✅ |
| Boundary λ | 0.01 | `training/trainer.py:105` | ✅ |
| Boundary warmup epoch | ≥ 10 | `training/trainer.py:97, 103` | ✅ |
| Threshold for metrics | 0.5 | `training/trainer.py:121` | ✅ |
| num_workers | 0 | `train_all.py:313, 315` | ✅ |
| pin_memory | True | same | ✅ |
| Device | "cuda" if available | `simple_trainer.py:78–80` | ⚠️ specific GPU not logged |

### 1.2 Architecture
| Field | Value | Status |
|-------|------:|:------:|
| FPN channel width C | 256 | `models/maffnet.py:198, 293` | ✅ |
| MMPA dilations | (3, 5, 7) + 1×1 context | `maffnet.py:46–49` | ✅ |
| MMPA SE reduction | C / 4 | `maffnet.py:59` | ✅ |
| FGFF FFT norm | "ortho" | `maffnet.py:101` | ✅ |
| FGFF normalisation | min-max per channel, ε=1e-8 | `maffnet.py:106–108` | ✅ |
| Decoder upsample | bilinear, align_corners=False | `maffnet.py:122` | ✅ |
| dec1 skip | zero tensor (no real skip) | `maffnet.py:350–351` | ⚠️ undocumented quirk |
| Number of seg heads | 4 (p1–p4) | `maffnet.py:314–317` | ✅ |
| Reported head | p1 (inferred) | not stated anywhere | ❌ |
| RDB output | computed, **discarded by loss** | `maffnet.py:363`, `training/trainer.py:89` | ❌ undocumented |

### 1.3 Loss
| Field | Value | Status |
|-------|------:|:------:|
| Segmentation loss | Dice + BCEWithLogits (pos_weight=2.0), averaged over 4 heads | `training/trainer.py:93–101` | ✅ |
| Boundary kernel | Sobel 3×3 (kx, ky) | `training/trainer.py:52–55` | ✅ |
| Sobel input | `sigmoid(p)` after `avg_pool2d(k=3, s=1, p=1)` | `training/trainer.py:98` | ✅ |
| Edge norm | √(gx² + gy²), clamp(0,1) | `training/trainer.py:62` | ✅ |
| Boundary loss | L1 | `training/trainer.py:65` | ✅ |
| Total loss formula | `seg + 0.01 · bnd` (epoch ≥ 10) else `seg` | `training/trainer.py:103–108` | ✅ |

---

## 2. Preprocessing & Augmentation

| Step | Value | Source | Status |
|------|-------|--------|:------:|
| Image format | JPG → RGB | `train_all.py:145`, `train.py:96` | ✅ |
| Mask format | PNG → L (grayscale) | same | ✅ |
| Resize | 384 × 384, bilinear (img), nearest (mask) | `train_all.py:148–149` | ✅ |
| Mask binarisation | `(m / 255) > 0.5 → {0, 1}` | `train_all.py:159` | ✅ |
| Pixel range (img) | `T.ToTensor()` → [0, 1] | `train_all.py:154` | ✅ |
| ImageNet mean/std normalisation | **none applied** | grep shows no `Normalize` call | ❌ — SAM2 expects ImageNet stats |
| Augment: hflip / vflip | p = 0.5 each | `train_all.py:115–118` | ✅ |
| Augment: rotation | uniform [−30°, +30°], bilinear (img) / nearest (mask) | `train_all.py:120–122` | ✅ |
| Augment: random affine | p = 0.5, degrees=±15, translate=0.1, scale=[0.9, 1.1], shear=±5 | `train_all.py:124–131` | ✅ |
| Augment: ColorJitter | brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, always applied | `train_all.py:133` | ⚠️ no probability gate, paper usually says "p=0.5" |
| Val/Test augmentation | none | `train_all.py:151` (`if split == "train"`) | ✅ |
| Dataset split | static CSVs in `dataset_split/ISIC_{2016,2017,2018}/{train,val,test}.csv` | repo files | ✅ |

---

## 3. Random Seeds

| Seed source | Set? | Where | Status |
|-------------|:----:|-------|:------:|
| Python `random.seed` | ❌ | nowhere | ❌ |
| `numpy.random.seed` | ❌ | nowhere | ❌ |
| `torch.manual_seed` | ❌ | nowhere | ❌ |
| `torch.cuda.manual_seed_all` | ❌ | nowhere | ❌ |
| `DataLoader` worker seed | n/a (num_workers=0) | — | ✅ by accident |
| cuDNN determinism | ❌ | no `torch.backends.cudnn.deterministic = True` | ❌ |
| cuDNN benchmark | ❌ default (True) | implicit | ❌ |
| `torch.use_deterministic_algorithms` | ❌ | not set | ❌ |
| Augmentation RNG | uses `random.*` directly | `train_all.py:115, 120, 124` | ⚠️ unseeded |

**Verdict: zero seed control.** Every run draws fresh state. No two runs are bit-identical. Reported metrics in `summary.csv` cannot be reproduced exactly — only in distribution.

---

## 4. Dependencies

### 4.1 Declared (`requirements.txt`)
| Package | Version pin | Status |
|---------|------------|:------:|
| PyTorch / torchvision / torchaudio | CUDA-enabled, **no version pin** | ⚠️ |
| NumPy | 2.2.6 | ✅ |
| Pandas | (no pin) | ⚠️ |
| SciPy | 1.15.3 | ✅ |
| scikit-learn | (no pin) | ⚠️ |
| OpenCV | 4.13.0.92 | ✅ |
| Pillow | 12.2.0 | ✅ |
| Albumentations | 2.0.8 | ✅ (but **unused** — augment uses torchvision, not albumentations) |
| Matplotlib | 3.10.8 | ✅ |
| TensorBoard | 2.20.0 | ✅ |
| Hydra-core | 1.3.2 | ✅ (not used in `train_all.py`) |
| OmegaConf | 2.3.0 | ✅ |
| HuggingFace `datasets`, `huggingface_hub` | (no pin) | ⚠️ |
| SAM2 | `git+https://github.com/facebookresearch/segment-anything-2.git` | ⚠️ no commit SHA pinned |
| timm, einops, tensordict | (no pin) | ⚠️ |

### 4.2 Undeclared / system-level
| Item | Value | Status |
|------|-------|:------:|
| Python version | not stated | ❌ |
| CUDA version | not stated | ❌ |
| cuDNN version | not stated | ❌ |
| GPU model | not stated (run logs would show) | ❌ |
| Driver version | not stated | ❌ |
| OS | not stated | ❌ |

### 4.3 External assets
| Asset | Path / source | Status |
|-------|--------------|:------:|
| SAM2 Hiera-L checkpoint | `checkpoints/sam2_hiera_large.pt` (~0.5 GB) | ⚠️ download instructions missing |
| SAM2 config YAML | `src/sam2/sam2/configs/sam2/sam2_hiera_l.yaml` | ✅ in submodule |
| ISIC 2016/2017/2018 raw images | external download, see `dataset_raw/` | ⚠️ download URL + dataset version not pinned |

---

## 5. Missing Implementation Details (External Repro Blockers)

Ranked by impact. Each blocks bit-reproducibility or correct re-implementation.

### Severity HIGH

1. **No seed control.** Cannot reproduce any single number. Different runs of the same config will vary by ±0.5 to ±2 pp Dice.
2. **No GPU / CUDA / cuDNN / Python version recorded.** AMP behaviour, conv kernels, and cuFFT can differ between CUDA 11.x / 12.x. cuDNN benchmark = True (default) → non-determinism even with seeds.
3. **No ImageNet normalisation applied** before SAM2 backbone, but SAM2 was pretrained with ImageNet mean/std (or its derivative). Whether MAFFNet was trained with the wrong normalisation, or whether the omission is intentional, **must be stated**. Reproducer who normalises will get different features than the repo.
4. **SAM2 commit SHA unpinned.** `git+https://github.com/facebookresearch/segment-anything-2.git` → HEAD at install time. Future API change → silent break or behavioural drift.
5. **PyTorch / torchvision version unpinned.** `torch.fft.rfft2` and `F.interpolate` semantics have changed across releases; AMP behaviour differs.
6. **Which segmentation head is reported.** Code returns `(p1, p2, p3, p4, rec)`. Tables show one number. Paper must say "we report p1 (full-resolution head)" — currently unstated.
7. **dec1 zero-skip** behaviour (`maffnet.py:350–351`) is undocumented. A faithful re-implementer following the textual architecture spec will likely use a real skip (e.g., stem features) and **diverge from the repo**.
8. **RDB defined but unused.** Reader of `forward` sees `rec` returned. Reader of `MAFFNetLoss.forward` sees `_rec` discarded. Inconsistency must be resolved (drop RDB or wire reconstruction loss).
9. **High-pass filter and inverse FFT missing** vs. published spec (see `ARCHITECTURE_AUDIT.md`). Reproducer following the paper will write the spec'd FGFF and get different numbers than the checkpoint.

### Severity MEDIUM

10. **AMP dtype not pinned.** `autocast("cuda")` defaults to fp16 on most GPUs but bf16 on Ampere+H100 in some configs. Numerical results differ subtly.
11. **Augmentation RNG seed.** Even with a master seed, `random.uniform(-30, 30)` is not under `torch.Generator` control and won't be deterministic in DataLoader with `num_workers > 0`.
12. **ColorJitter always-on at p=1.** Whether other ISIC papers used p=0.5 is a baseline-comparison concern.
13. **Validation metric averaging.** `simple_trainer.py:171–179` weights per-image metrics by batch size, **not** by image count via pixel sum → standard but not paper-stated.
14. **Best model selection criterion.** `simple_trainer.py:310–313` keeps best by val Dice; `summary.csv` reports **final epoch**, not best. Subtle mismatch needs clarifying.
15. **Test set not used.** `train_all.py` only constructs train/val loaders. Reported metrics are on val split. External reviewer expects test split → numbers won't match.
16. **No model architecture YAML / config dump per run.** `experiments/.../config.txt` has only top-level args. Internal arch hyperparameters (dilation rates, SE ratio, λ_bnd, etc.) hard-coded in Python.
17. **No checksum / SHA of dataset splits.** CSV files can drift; need a hash to confirm reproducer has the same split.
18. **No requirements.txt for the *exact* tested env.** `requirements.txt` is install-time hint, not a freeze. Need `pip freeze > requirements-lock.txt`.

### Severity LOW

19. Logging dir hard-coded (`logs/`, `results/<ts>/`).
20. Hard-coded `pos_weight = 2.0`. Justified by class imbalance but not measured / stated.
21. tqdm output goes to stdout via `_Tee` → reproducer parsing logs may mismatch.
22. Augmentation `ColorJitter` is `T.ColorJitter(0.2, 0.2, 0.2, 0.1)`. The exact `torchvision` version determines hue-handling — bug fixed in 0.16 changes hue distribution slightly.

---

## 6. Recommendations (Concrete, Functionality-Preserving)

Priority order. Each is small, isolated, and does not change training behaviour beyond making it reproducible.

### Tier 1 — Must do before sharing code publicly

1. **Add `set_seed(seed)` helper** and call once in `train_all.py:run_all()` and `train.py:main()`:
   ```python
   def set_seed(s):
       import random, numpy as np, torch
       random.seed(s); np.random.seed(s)
       torch.manual_seed(s); torch.cuda.manual_seed_all(s)
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False
   ```
   Expose `--seed` CLI flag. Log to `config.txt`.
2. **Pin PyTorch + torchvision + CUDA**. Replace unversioned lines in `requirements.txt` with `torch==2.3.1+cu121` (or whatever the actual tested combination is).
3. **Pin SAM2 commit SHA**: `git+https://...@<sha>` in requirements.
4. **Generate `requirements-lock.txt`** via `pip freeze` in the tested env. Commit it.
5. **Add `environment.md`** with: Python version, CUDA version, cuDNN version, GPU model, driver. Auto-emit at run start via `python -V`, `torch.version.cuda`, `torch.backends.cudnn.version()`, `nvidia-smi -q`.
6. **Document the reported head.** Add a single-line comment in `MAFFNetTrainer._val_epoch` indicating `metrics = compute_metrics(p1, mask)`. State it in paper.
7. **Document dec1 zero-skip.** Either keep with a comment ("we intentionally drop the stem skip because…") or replace with a real skip; pick one.
8. **Decide RDB fate.** Drop it or add reconstruction loss. Don't leave dead in `forward`.
9. **Add download script** for SAM2 checkpoint and the 3 ISIC datasets, with SHA-256 verification, in `scripts/download_assets.sh`.

### Tier 2 — Should do for journal acceptance

10. **Switch reporting to test split.** Construct `test_loader` from `dataset_split/*/test.csv`. Report test metrics in tables.
11. **Multi-seed runs.** Standard is 3 seeds → mean ± std. Add `--seeds 0,1,2` orchestration.
12. **Hash all CSV splits.** Add `scripts/verify_splits.py` that prints SHA-256 of each split file and fails if mismatch.
13. **Save full config per run.** Dump every hyperparameter (architecture + training + loss + augment) as `experiments/.../config.yaml`. One file = one run, full fidelity.
14. **Resolve ImageNet normalisation.** Either apply `T.Normalize(mean=ImageNet, std=ImageNet)` to match SAM2 pretraining and re-train, or document explicitly that you do not normalise and why.
15. **Pin AMP dtype.** Use `with autocast("cuda", dtype=torch.float16):` explicitly. Or migrate to `bf16` on supported HW (also kills the U-Net NaN bug class root cause — see `UNET_FAILURE_ANALYSIS.md`).
16. **Move hard-coded constants to a single `configs/maffnet.yaml`.** Dilation rates, SE reduction, λ_bnd, warmup epoch, smoothing kernel, head count. Drop torchscript-hostile literals.
17. **Record `pip freeze` per experiment** at the top of each `experiments/<exp>/env.txt`.

### Tier 3 — Nice to have

18. Switch random ops in augmentation to `torch.Generator` for full determinism with workers.
19. Migrate augment to `albumentations` (already in `requirements.txt` but unused) for reproducible transform versioning.
20. Add a CI smoke test: 1-epoch run on a 10-image subset that must hit fixed Dice ± 1e-4. Catches drift.
21. Publish a Docker image with the exact env. `Dockerfile` pinned by CUDA + PyTorch + commit SHA.
22. Release pre-trained final checkpoints alongside code. Include MD5 / SHA for each.

---

## 7. Quick Self-Audit Score

| Dimension | Score (0–5) |
|-----------|------------:|
| Hyperparameter completeness | 4 |
| Preprocessing completeness | 3 (ImageNet-norm question open) |
| Seed control | 0 |
| Dependency pinning | 2 |
| Environment recording | 0 |
| External-asset distribution | 1 |
| Architecture documentation vs. code | 2 (RDB, dec1, head, FGFF gaps) |
| Test-split reporting | 0 |
| Multi-seed reporting | 0 |
| **Overall reproducibility** | **1.4 / 5** |

Current state: an internal team member with access to the same GPU and Python env can run `python train_all.py` and get **statistically similar** numbers. An external reviewer **cannot reproduce any single reported number bit-for-bit**, and following the paper text alone (without code) would produce a **different architecture** than the checkpoint.

Tier 1 actions alone raise the score to ~3/5 and unlock Q3 venue acceptance. Tier 1+2 reach ~4/5 → Q2 acceptable.

---

## Evidence References

- `train_all.py` — full training driver (no seed call anywhere)
- `train.py` — CLI runner (no seed call anywhere)
- `training/simple_trainer.py:78–95` — optimiser, scaler setup
- `training/trainer.py:34–113` — losses and λ values
- `models/maffnet.py:97–113` — FFT path (no high-pass, no iFFT)
- `models/maffnet.py:319–320, 363` — RDB construction + dead output
- `requirements.txt` — partial pins
- `dataset_split/ISIC_{2016,2017,2018}/` — split CSVs (no hash)
- `results/2026-04-28_15-19-03/summary.csv` — metrics that need to be reproducible
- `ARCHITECTURE_AUDIT.md`, `NOVELTY_AUDIT.md`, `COMPUTATIONAL_ANALYSIS.md` — companion audits
