# Research Alignment Report

**Sources (only these):** (1) Approved Synopsis — *"Stabilised MAFFNet with SAM-based Encoder for Skin Lesion Segmentation"* (Karmakar, guide Dr. Lokesh Singh, Jan 2026); (2) Original MAFF-Net paper (Zhou et al., BSPC 2026); (3) current source code.
**Mode:** analysis only. No code modified, no paper rewritten.

> **Bottom line.** The approved synopsis and the current code describe **two different methods**. The synopsis commits to a *stabilised frequency-fusion redesign* (stabilised FFT → high-pass → inverse FFT → feature alignment → gated fusion). The code instead contains a *faithful restoration of the original MAFF-Net* plus a *new boundary-supervision contribution* (BDS + EWR). They overlap only on the SAM encoder, FFT float32 stabilisation, and the ISIC evaluation protocol. This gap should be resolved with the guide before the paper is finalised.

---

## 1. Features Present in the Original MAFF-Net (paper)

| # | Component | Description |
|---|-----------|-------------|
| O1 | Frozen SAM2 (Hiera-Large) encoder | Pretrained backbone, frozen |
| O2 | Mixed Multi-scale Perception Adaptor (MMPA) | Lightweight dynamic adaptor (bottleneck, dilated experts, softmax weighting) |
| O3 | Frequency-Guided Feature Fusion (FGFF) | FFT **magnitude** spectrum (phase discarded) + cross-modal fusion |
| O4 | Reconstruction Decoder Block (RDB) | Auxiliary reconstruction, MSE self-supervision (train only) |
| O5 | Decoder with multi-scale **deep supervision** | Several prediction heads |
| O6 | Loss = Dice (deep-supervised) + reconstruction MSE | No high-pass, no inverse-FFT step, no feature-alignment module |
| O7 | Evaluation on ISIC 2016/2017/2018 | Dice, IoU, sensitivity, specificity |

The original FGFF uses the FFT **magnitude** only. It does **not** contain an explicit high-pass filter, an inverse-FFT reconstruction stage, a dedicated feature-alignment module, or a gated-fusion replacement for cross-attention.

---

## 2. Features Proposed in the Approved Synopsis

From the synopsis objectives and proposed methodology (Fig. 1.1, §1.3.1, §1.4):

| # | Proposed component | Synopsis status |
|---|--------------------|-----------------|
| S1 | SAM-based encoder backbone | Core |
| S2 | Two parallel paths: spatial + frequency | Core |
| S3 | **Stabilised FFT (float32 isolation under AMP)** to remove NaN | Objective 2 |
| S4 | **High-pass filtering** of spectral coefficients | Core methodology |
| S5 | **Inverse FFT** → boundary-sensitive spatial map | Core methodology |
| S6 | **Feature-alignment module** (instance norm + channel projection), SAM↔spectral | Objective 3 |
| S7 | **Lightweight gated fusion** replacing cross-attention (linear complexity) | Objective 4 |
| S8 | Projected skip connections to decoder | Core methodology |
| S9 | "Boundary-aware learning as the primary design objective" | Objective 1 (goal-level) |
| S10 | Improve Dice/IoU over MAFF-Net baseline; gains on diffuse/irregular margins | Objective 5 (expected) |
| S11 | Validate on ISIC 2016/2017/2018 vs U-Net, Attention U-Net, UNet++, TransUNet, **SGT-Net**, MAFF-Net (Dice, mIoU, sensitivity, specificity) | Objective 6 |

The synopsis does **not** mention MMPA, RDB, deep supervision, boundary-aware deep supervision, or edge-weighted reconstruction. Its methodology is the *stabilised frequency-fusion* redesign.

---

## 3. Features Currently Implemented (source code)

| # | Implemented component | Notes |
|---|-----------------------|-------|
| C1 | Frozen SAM2 Hiera-Large encoder | matches O1 / S1 |
| C2 | MMPA dynamic adaptor (restored) | from original MAFF-Net (O2); not in synopsis |
| C3 | FGFF cross-modal fusion, FFT **magnitude** + dilated spatial + β | from original (O3); FFT in **float32** (matches S3) |
| C4 | RDB reconstruction branch (restored) | from original (O4); not in synopsis |
| C5 | Decoder + 4-head deep supervision | from original (O5) |
| C6 | **Boundary-aware Deep Supervision (BDS)** | new contribution; not in synopsis or original |
| C7 | **Edge-weighted Reconstruction (EWR)** | new contribution; not in synopsis or original |
| C8 | Loss = Dice + BCE + boundary (BDS) + edge-weighted recon (EWR) | |
| C9 | Evaluation on ISIC 2016/2017/2018; baselines U-Net, Attention U-Net, UNet++, TransUNet, restored MAFF-Net | **SGT-Net not implemented** |

**Absent from the code (though promised by the synopsis):** high-pass filter (S4), inverse FFT (S5), feature-alignment module (S6), gated-fusion replacement (S7), SGT-Net baseline (part of S11).

---

## 4. Differences Among the Three

| Feature | Original MAFF-Net | Approved Synopsis | Current Code | Verdict |
|---------|:-----------------:|:-----------------:|:------------:|---------|
| SAM/SAM2 frozen encoder | ✅ | ✅ | ✅ | aligned |
| FFT float32 stabilisation | (implicit) | ✅ (S3) | ✅ | aligned |
| FFT magnitude fusion | ✅ | ✅ (path) | ✅ (FGFF) | aligned |
| High-pass filter | ❌ | ✅ (S4) | ❌ | **synopsis-only, not built** |
| Inverse FFT stage | ❌ | ✅ (S5) | ❌ | **synopsis-only, not built** |
| Feature-alignment module | ❌ | ✅ (S6) | ❌ | **synopsis-only, not built** |
| Gated-fusion replacement | ❌ | ✅ (S7) | ❌ | **synopsis-only, not built** |
| MMPA dynamic adaptor | ✅ | ❌ | ✅ (restored) | code+original, not in synopsis |
| RDB reconstruction | ✅ | ❌ | ✅ (restored) | code+original, not in synopsis |
| Deep supervision | ✅ | ❌ | ✅ | code+original, not in synopsis |
| Boundary-aware Deep Supervision (BDS) | ❌ | ❌ (goal only, S9) | ✅ | **code-only contribution** |
| Edge-weighted Reconstruction (EWR) | ❌ | ❌ | ✅ | **code-only contribution** |
| Boundary-aware *objective* | indirect | ✅ primary (S9) | ✅ (BDS+EWR) | **goal aligned, mechanism differs** |
| SGT-Net baseline | n/a | ✅ (S11) | ❌ | **synopsis commitment unmet** |
| ISIC 2016/17/18 + Dice/IoU/Sens/Spec | ✅ | ✅ | ✅ | aligned |

**Summary of the gap.** The synopsis's four signature mechanisms (high-pass, inverse FFT, feature alignment, gated fusion) are **not implemented**. The code instead restores the original MAFF-Net modules (MMPA, FGFF, RDB, deep supervision) and adds a **different** contribution (BDS + EWR) that the synopsis does not name. The one point of conceptual continuity is Objective 1: the synopsis declares *boundary-aware learning* the primary goal, and BDS + EWR are boundary-aware — so the **goal** is preserved even though the **mechanism** changed.

---

## 5. Components That MUST Remain to Stay Consistent with the Approved Synopsis

These are commitments the synopsis makes explicitly. To remain consistent, the paper must either retain them or have the synopsis formally amended by the guide.

1. **SAM-based encoder backbone** (S1) — present in code; keep.
2. **Frequency-guided processing with float32-stabilised FFT** (S2, S3) — present in code (FGFF + fp32 FFT); keep and foreground.
3. **Boundary-aware learning as the primary objective** (S9) — satisfied in spirit by BDS + EWR; keep this framing.
4. **Evaluation on ISIC 2016/2017/2018 with Dice, mIoU, sensitivity, specificity** (S11) — present; keep.
5. **Baseline set including SGT-Net** (S11) — **currently missing SGT-Net**; either add it or get the baseline list amended.
6. **Improvement over the MAFF-Net baseline** (S10) — stated as *expected*; keep as an expectation pending results.

**Conflict to resolve with the guide:** the synopsis also commits to high-pass filtering (S4), inverse FFT (S5), feature alignment (S6), and gated fusion (S7) as the *core methodology*. The current code does none of these. Strict consistency requires either (a) implementing them, or (b) an approved amendment recording that the methodology changed to *restored MAFF-Net + Boundary-centric Dual Supervision*. This is a supervisor-approval matter, not one to settle silently in the paper.

---

## 6. Components to Describe as Optional Implementation Improvements (not Core Contributions)

- **Restoration of the original MAFF-Net modules** (MMPA, FGFF, RDB, deep supervision) — engineering/reproduction, the experimental base. Describe as restored base, not contribution (consistent with the methodology sections already drafted).
- **FFT float32 stabilisation** — although the synopsis lists it as an objective, in practice it is a one-line numerical fix. Best described as an implementation/restoration detail that enables stable training, not a standalone headline novelty.
- **BCE term in the loss** — an implementation switch on top of the original Dice; optional, ablatable, not core.
- **TransUNet upgraded to the official R50+ViT-B/16 baseline** — a baseline-fidelity improvement, reported as such.

**What should be presented as the actual core contribution (if the guide approves the changed direction):** Boundary-centric Dual Supervision (BDS + EWR). It is the only genuinely new, non-restoration element in the code, and it aligns with synopsis Objective 1 (boundary-aware learning). It is, however, **not the mechanism named in the synopsis**, so it can be presented as the core contribution **only after** the synopsis direction is reconciled per §5.

---

## 7. Alignment Verdict and Decision Required

| Aspect | State |
|--------|------|
| Goal (boundary-aware dermoscopic segmentation on ISIC, SAM encoder, frequency processing) | **Aligned** |
| Named methodology (high-pass, iFFT, feature alignment, gated fusion) | **Not implemented** |
| Actual contribution in code (BDS + EWR) | **Not in synopsis** |
| Baseline set (SGT-Net) | **Incomplete** |

Two consistent paths forward (decision for the author and guide, not made here):

- **Path A — implement the synopsis as approved:** add high-pass filtering, inverse FFT, the feature-alignment module, and gated fusion; add the SGT-Net baseline. The paper then matches the approved synopsis.
- **Path B — amend the synopsis:** record, with the guide's approval, that the methodology became *faithful MAFF-Net restoration + Boundary-centric Dual Supervision*, and update the objectives and baseline list accordingly. The paper then matches the implemented work.

Either path keeps the SAM encoder, fp32-stabilised frequency processing, the boundary-aware objective, and the ISIC evaluation, all of which are already aligned.

---

## Evidence References
- Synopsis: Abstract, §1.3.1 Objectives, §1.4 Proposed Methodology, Fig. 1.1, Table 1.1 (stabilised FFT, high-pass, inverse FFT, feature alignment, gated fusion, SGT-Net baseline)
- Original paper: §3.1 MMPA, §3.2 FGFF (magnitude), §3.3 RDB + deep supervision, §3.4 loss (Dice + recon MSE)
- Code: `models/maffnet.py` (MMPA, FGFF magnitude fusion, RDB, fp32 FFT, no high-pass / no iFFT / no alignment / no gated-fusion module), `training/trainer.py` (BDS + EWR loss), `train_all.py` baselines (no SGT-Net)
