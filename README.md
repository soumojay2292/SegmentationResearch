# SpecGate-Net

**Spatial–Spectral Gated Fusion for Boundary-Faithful Skin Lesion Segmentation**

SpecGate-Net is a dual-stream fusion network for dermoscopic skin-lesion
segmentation. It keeps a **frozen SAM2 (Hiera-Large) encoder**, the convolutional
decoder, and the auxiliary reconstruction branch, and redesigns only the fusion
pathway: multi-scale encoder features are processed along a **spatial stream** and a
parameter-free **frequency stream**, reconciled by per-stream **feature alignment**,
and merged by a learned, spatially varying **gated fusion**. Training adds
boundary-focused supervision on top of a region objective.

This repository reproduces the study reported in the accompanying manuscript
(`paper2/`), evaluated on **ISIC 2016 / 2017 / 2018**.

## Models in this repository

| Role | Model | `--model` id | Source file / class | Needs SAM2 backbone |
|---|---|---|---|---|
| **Proposed** | **SpecGate-Net** | `specgate_net` | `models/specgate_net.py` / `SpecGateNet` | yes |
| **Prior work / reference control** | **MAFF-Net [15]** | `maffnet` | `models/maffnet.py` / `MAFFNet` | yes |
| Baseline | U-Net | `unet` | `models/unet.py` / `UNet` | no |
| Baseline | Attention U-Net | `attention_unet` | `models/attention_unet.py` / `AttentionUNet` | no |
| Baseline | UNet++ | `unetpp` | `models/unet_plus_plus.py` / `UNetPlusPlus` | no |
| Baseline | TransUNet | `transunet` | `models/transunet.py` / `TransUNet` | no |

**MAFF-Net [15]** is the published prior architecture (Zhou et al., 2026),
reproduced here under this study's protocol as a reference control — it is **not**
a proposed model. **SpecGate-Net** is the sole proposed model.

## Repository structure

```
.
├── train.py                     Single training entry point (defines its own SegDataset)
├── requirements.txt
├── models/
│   ├── specgate_net.py          SpecGateNet (proposed)
│   ├── specgate_net_ablation.py SpecGateNetAblation (leave-one-out ablation)
│   ├── maffnet.py               MAFF-Net [15] + shared blocks reused by SpecGate-Net
│   ├── unet.py, attention_unet.py, unet_plus_plus.py, transunet.py   baselines
│   ├── transunet_legacy.py      helper used by transunet.py
│   └── sam2_loader.py           frozen SAM2 encoder loader
├── modules/                     frequency_branch.py, feature_alignment.py, gated_fusion.py
├── training/                    trainer.py (MAFFNetTrainer), simple_trainer.py (SimpleTrainer)
├── utils/                       report_generator.py
├── src/sam2/                    local SAM 2 library (added to sys.path by train.py)
├── dataset_split/               ISIC_2016 / ISIC_2017 / ISIC_2018 (see Dataset preparation)
├── checkpoints/                 place the SAM2 backbone weights here
├── experiments/                 per-run outputs (checkpoints, reports, metrics)
├── paper2/                      manuscript (LaTeX), figures, and result tables/CSVs
└── archive/                     superseded / exploratory material (not part of the active workflow)
```

## Environment setup

Python 3.10 is recommended. Install PyTorch matching your CUDA/CPU setup, then the
remaining dependencies:

```bash
pip install -r requirements.txt
```

The SAM 2 library is used from the local clone in `src/sam2/`; `train.py` adds it to
`sys.path` automatically, so no separate SAM 2 install is required.

## SAM2 checkpoint (required for SpecGate-Net and MAFF-Net)

The proposed model and the MAFF-Net reference control use a **frozen SAM 2.1
Hiera-Large** image encoder. Obtain the SAM 2.1 Hiera-Large checkpoint from the
official Meta SAM 2 release and place it at:

```
checkpoints/sam2.1_hiera_large.pt
```

Its model config is bundled at `src/sam2/sam2/configs/sam2/sam2_hiera_l.yaml` and is
loaded internally — you only need to pass the checkpoint via `--checkpoint`. The four
convolutional/transformer baselines do **not** require this checkpoint.

## Dataset preparation

Place each ISIC dataset under `dataset_split/<name>/` with this layout (already
present in this repository):

```
dataset_split/ISIC_2018/
├── train/  { images/ , masks/ }
├── val/    { images/ , masks/ }
├── test/   { images/ , masks/ }
├── train.csv
├── val.csv
└── test.csv
```

Each CSV has two columns, `image,mask`, listing the file names inside the
corresponding `images/` and `masks/` folders, for example:

```
image,mask
ISIC_0000000.jpg,ISIC_0000000_Segmentation.png
```

The same layout exists for `ISIC_2016`, `ISIC_2017`, and `ISIC_2018`. Images are
read as RGB, masks as grayscale; both are resized to 384×384 and masks binarised at
0.5 inside `train.py`.

## Training and evaluation

The entry point is `train.py`. Outputs for a run go to `experiments/<exp>/`.

**SpecGate-Net (proposed) — ISIC 2018:**

```bash
python train.py --model specgate_net --dataset ISIC_2018 --data_root dataset_split \
  --epochs 40 --batch_size 2 --img_size 384 --lr 1e-3 --weight_decay 0.01 \
  --checkpoint checkpoints/sam2.1_hiera_large.pt --exp specgate_ISIC2018
```

**MAFF-Net [15] (prior-work reference control):**

```bash
python train.py --model maffnet --dataset ISIC_2018 --data_root dataset_split \
  --epochs 40 --batch_size 2 --img_size 384 --lr 1e-3 --weight_decay 0.01 \
  --checkpoint checkpoints/sam2.1_hiera_large.pt --exp maffnet_ISIC2018
```

**Baselines (no SAM2 checkpoint needed):**

```bash
python train.py --model unet           --dataset ISIC_2018 --data_root dataset_split --epochs 40 --exp unet_ISIC2018
python train.py --model attention_unet --dataset ISIC_2018 --data_root dataset_split --epochs 40 --exp attn_unet_ISIC2018
python train.py --model unetpp         --dataset ISIC_2018 --data_root dataset_split --epochs 40 --exp unetpp_ISIC2018
python train.py --model transunet      --dataset ISIC_2018 --data_root dataset_split --epochs 40 --exp transunet_ISIC2018
```

Set `--dataset ISIC_2016` or `--dataset ISIC_2017` to run the other benchmarks.

**Key arguments** (see `python train.py --help` for the full list): `--model`,
`--dataset`, `--data_root` (default `dataset_split`), `--epochs`, `--batch_size`
(default 2), `--lr` (default 1e-3), `--weight_decay` (default 0.01), `--img_size`
(default 384), `--patience` (early-stopping patience on validation Dice for the
baseline models; `<=0` disables), `--checkpoint`, `--exp`, `--fast` (quick smoke
run).

**Outputs.** Each run writes to `experiments/<exp>/`: `config.txt`, the best model
(`model.pth`), a training `summary.csv`, an HTML `report.html`, and sample
predictions. For the four convolutional/transformer baselines, `train.py` also
evaluates the held-out **test** split and writes `experiments/<exp>/metrics.json`.
The per-image metrics and aggregate tables that back the manuscript are provided as
CSVs under `paper2/results/` (`perimage/`, `stats6/`, `final_benchmark_table.*`,
`boundary_metrics.*`).

## Component ablation (three components)

The leave-one-out ablation is implemented in
`models/specgate_net_ablation.py` as `SpecGateNetAblation`, a subclass of
`SpecGateNet` that disables **exactly one** component while leaving the frozen
encoder, decoder, reconstruction branch, and training pipeline unchanged. It is
constructed with `ablate=` set to one of:

- `"freq"`   — disables the **Frequency Branch** (`modules/frequency_branch.py`)
- `"align"`  — disables the **Feature Alignment** module (`modules/feature_alignment.py`)
- `"fusion"` — replaces the **Gated Fusion** module (`modules/gated_fusion.py`) with a fixed unweighted mean

The ablation model is not exposed through `train.py --model`; it is instantiated
directly from `SpecGateNetAblation`. The ablation was run on **ISIC 2018**, and its
run outputs are stored in:

```
experiments/ablation_no_frequency/
experiments/ablation_no_alignment/
experiments/ablation_no_gated_fusion/
```

The corresponding results table is in `paper2/results/ablation_results.*`.

## Reproducing the paper's experiments

The manuscript reports six models across ISIC 2016/2017/2018. The run directories
used for the reported results are (dataset-specific names are historical and are the
authoritative source paths referenced by `paper2/results/final_benchmark_table.csv`):

| Model | Experiment directories |
|---|---|
| SpecGate-Net | `experiments/stabilized_maffnet_ISIC{2016,2017,2018}/` |
| MAFF-Net [15] | `experiments/maffnet_ISIC{2016,2017,2018}_final/` |
| U-Net | `experiments/unet_ISIC_2016/`, `experiments/unet_final_2017/`, `experiments/unet_ISIC_2018/` |
| Attention U-Net | `experiments/attn_unet_ISIC_2016/`, `experiments/attn_unet_ISIC_2017/`, `experiments/attention_unet_ISIC2018_final/` |
| UNet++ | `experiments/unetpp_ISIC_{2016,2017,2018}/` |
| TransUNet | `experiments/transunet_ISIC_{2016,2017,2018}/` |

## Citation

If you use this code, please cite the SpecGate-Net manuscript:

```bibtex
@misc{karmakar_specgatenet,
  author = {Soumojay Karmakar and Lokesh Singh},
  title  = {{SpecGate-Net: Spatial--Spectral Gated Fusion for Boundary-Faithful Skin Lesion Segmentation}},
  note   = {Manipal Institute of Technology Bengaluru, Manipal Academy of Higher Education, Manipal, India}
}
```

This work builds on and uses as a reference control the published MAFF-Net:

```bibtex
@article{maffnet,
  author  = {Y. Zhou and H. Gong and R. Qiu and S. Wei and Z. Li and W. Zhang},
  title   = {{MAFF-Net: SAM-powered mixed multi-scale perception adaptation and frequency-guided feature fusion for robust skin lesion segmentation}},
  journal = {Biomedical Signal Processing and Control},
  volume  = {113},
  pages   = {109222},
  year    = {2026}
}
```

The full reference list (33 entries) is in `paper2/latex/references.bib`.

## License

See [LICENSE](LICENSE).
