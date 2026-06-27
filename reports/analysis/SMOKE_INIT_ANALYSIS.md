# MAFFNet Smoke-Init Failure — Analysis

**Error:** `ValueError: Provide either 'encoder' (a SAM model) or 'checkpoint' (path to .pth).`
**When:** at model construction, before training starts.
**Mode:** analysis only. No code changed.

**One-line cause:** `train.py` was launched for `--model maffnet` **without `--checkpoint`**, so `args.checkpoint` defaulted to `None`, and `MAFFNet(checkpoint=None)` (with `encoder=None`) hits the constructor guard and raises.

---

## 1. Why train.py Is Not Passing the Checkpoint

The failure is a **CLI usage gap**, not a code bug:

| Evidence | Location |
|----------|----------|
| `--checkpoint` defaults to `None` | `train.py:135` → `p.add_argument("--checkpoint", default=None)` |
| MAFFNet built with that raw default | `train.py:202` → `model = MAFFNet(checkpoint=args.checkpoint)` |
| Constructor requires one of the two | `models/maffnet.py:302-315` → `if encoder is not None: … elif checkpoint is not None: … else: raise ValueError(...)` |

So if the launch command omits `--checkpoint`, `args.checkpoint = None` → `MAFFNet(checkpoint=None)` → `encoder` also `None` → the `else` branch raises.

**Contrast — why `train_all.py` works:** it hard-codes the path and never relies on a CLI default:
- `train_all.py:32` → `MAFFNET_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"`
- `train_all.py:342` → `MAFFNet(checkpoint=MAFFNET_CHECKPOINT)`

`train.py` has **no such default** for the MAFFNet path; the checkpoint must be supplied explicitly on the command line.

---

## 2. Which Pretrained SAM Checkpoint Should Be Used

**`checkpoints/sam2_hiera_large.pt`** — SAM2 Hiera-Large.

- Present on disk: **897,952,850 bytes (~898 MB)**, dated 2026-04-17 (verified).
- It is the exact file `train_all.py` already uses (`MAFFNET_CHECKPOINT`).
- It matches the encoder MAFFNet expects: `models/maffnet.py:305-289` loads it via `load_sam2_encoder(config="src/sam2/sam2/configs/sam2/sam2_hiera_l.yaml", checkpoint=...)` → frozen SAM2 Hiera-Large image encoder (the same backbone the original paper uses, `sam2_hiera_large`).
- Confirmed loadable in the earlier restoration smoke test (`[SAM2] … keys`, encoder built, 212.7M frozen params).

No other checkpoint is needed; do **not** pass a SAM ViT checkpoint here (that path is the separate `encoder=` route for `SAMEncoder`, not used by `train.py`).

---

## 3. Exact Command to Launch MAFFNet Successfully

```
python train.py --model maffnet --dataset ISIC_2016 --epochs 1 \
  --exp maffnet_smoke \
  --checkpoint checkpoints/sam2_hiera_large.pt \
  --save_dir experiments/maffnet_smoke/checkpoints
```

Notes:
- The only mandatory addition vs. the failing run is **`--checkpoint checkpoints/sam2_hiera_large.pt`**.
- `--save_dir` is optional but recommended (keeps this run's `training_log.csv` / checkpoints isolated; default is the shared `checkpoints/`).
- For the BcDS ablation runs, the same `--checkpoint` is required on every MAFFNet invocation.
- Run from the repo root (`SegmentationResearch/`) so the relative paths resolve. First build loads ~898 MB SAM2 (slower) — expected.

---

## 4. Is This Caused by the Recent Restoration Changes?

**No — pre-existing.** This is independent of the restoration / contribution work.

| Component | Touched by restoration? | Relevant to this error? |
|-----------|:-----------------------:|:-----------------------:|
| `MAFFNet.__init__` encoder/checkpoint guard (`maffnet.py:302-315`) | No (restoration changed MMPA, FGFF, `dec1`, norm buffers, forward — not the constructor's checkpoint logic) | **Yes — this is the raise site, and it is original** |
| `train.py:135` `--checkpoint` default `None` | No (train.py was not modified in restoration) | **Yes — original** |
| `train.py:202` `MAFFNet(checkpoint=args.checkpoint)` | No | Yes — original |

The restoration edits were confined to `models/maffnet.py` (MMPA/FGFF/dec1/normalization) and `training/trainer.py` (loss). None alter the constructor's `encoder`/`checkpoint` requirement or `train.py`'s argument handling. The guard and the `None` default both predate all restoration/contribution changes, so a `train.py --model maffnet` run without `--checkpoint` would have failed identically before this session's work.

**Conclusion:** the smoke command simply omitted `--checkpoint`. Add it (Section 3) and construction succeeds. No fix to source is required for this error (though a convenience default in `train.py` could be added later if desired — out of scope here).

---

## Evidence References
- `train.py:135` (`--checkpoint` default None), `:202` (`MAFFNet(checkpoint=args.checkpoint)`)
- `train_all.py:32,342` (hard-coded `MAFFNET_CHECKPOINT`, works)
- `models/maffnet.py:302-315` (constructor guard / ValueError), `:305-289` (SAM2 load path)
- `checkpoints/sam2_hiera_large.pt` — present, ~898 MB
- `RESTORATION_REPORT.md` — restoration scope (no train.py / constructor-guard changes)
