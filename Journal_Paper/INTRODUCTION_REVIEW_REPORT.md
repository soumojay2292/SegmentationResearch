# Introduction Review Report (Section 1)

**Scope:** Section 1 only. Verified against current code (`models/`, `training/`, `train.py`, `train_all.py`), the approved synopsis, the original MAFF-Net paper, and the project reports (`RESEARCH_ALIGNMENT_REPORT`, `RESTORATION_REPORT`, `CONTRIBUTION_IMPLEMENTATION_REPORT`, `REVIEW_REPORT`). Research direction unchanged; no experimental results invented; no other section modified.

---

## Implementation Verification (basis for the review)

| Claim in Introduction | Code evidence | Verdict |
|----------------------|---------------|:-------:|
| Frozen SAM2 (Hiera) encoder | `maffnet.py:229,274` `requires_grad_(False)`; `sam2_loader.py` `sam2_hiera_l` | ✅ |
| Restored MMPA, FGFF, RDB | `models/maffnet.py` (MMPA adaptor, FGFF cross-modal, RDB) | ✅ |
| FFT isolated to single precision | `maffnet.py:128` `rfft2(x.float())` | ✅ |
| BcDS = BDS + EWR, single boundary prior, loss-only | `training/trainer.py` (`MAFFNetLoss`: BDS, EWR, shared `B(y)`) | ✅ |
| No high-pass filter, no inverse FFT | repo grep → none | ✅ (correctly omitted) |
| Baselines U-Net, Attention U-Net, UNet++, TransUNet | `train_all.py:317` | ✅ |
| ISIC 2016/2017/2018 | `dataset_split/` | ✅ |

Every technical statement in the revised Introduction matches the current implementation.

---

## Changes Made

1. **Added SAM2 citation [16].** Earlier draft used SAM2 in the method but cited only SAM [14]. SAM2 now cited where the encoder is introduced and in the restoration paragraph.
2. **Added a citation marker [20] for the clinical statistics.** Opening melanoma incidence/survival figures were uncited; now flagged to a primary source (entry [20] in `References/`, pending verification). Specific unverifiable percentages were softened to a sourced qualitative statement to avoid unsupported precision.
3. **Sharpened the three-way contribution labels.** The contributions list now explicitly tags item 1 as *Restored base, not claimed as a contribution* and items 2–4 as *Proposed contribution*, with a lead-in sentence separating the original design, the reproduction, and the extension.
4. **Reduced repetition.** The pooling / boundary-smoothing explanation, previously stated across multiple paragraphs, is now concentrated in the U-Net paragraph and referenced rather than repeated.
5. **Neutralized methodology-direction wording.** The Introduction describes only what the implemented framework does and frames BcDS as the proposed contribution under evaluation; it makes no claim of being the approved or sole research direction, and asserts no performance outcome.
6. **Added the Draft v1 Status/Dependencies footer** per the standing drafting rule.

---

## Technical Inconsistencies Fixed

- **SAM vs SAM2 citation gap** — resolved (added [16]).
- **Uncited clinical statistics** — resolved (added [20] marker; precision softened until the source is confirmed).
- No factual statement in the prior draft contradicted the code, so no technical claim required removal; the FFT-stability, frozen-encoder, restored-module, and supervision-only statements all hold.

---

## Remaining Issues Requiring Supervisor Confirmation

These are recorded here rather than in the paper body, since "pending approval" wording is not appropriate in an IEEE Introduction.

1. **Methodology direction vs approved synopsis (primary).** The synopsis commits to stabilised FFT + high-pass filtering + inverse FFT + feature alignment + gated fusion and an SGT-Net baseline; the implemented framework (and hence this Introduction) instead presents restored MAFF-Net + Boundary-Centric Dual Supervision. The Introduction is consistent with the **implementation** but diverges from the **synopsis**. Resolution (Path A implement synopsis modules, or Path B amend the synopsis) is a supervisor decision; the contribution wording can be finalized only afterward (see `RESEARCH_ALIGNMENT_REPORT.md`).
2. **Baseline set.** Synopsis lists SGT-Net as a baseline; it is neither implemented nor claimed in the Introduction. Confirm whether SGT-Net is added (Path A) or formally dropped (Path B).
3. **Reference [20] (clinical statistics)** requires a primary epidemiological source before submission.
4. **Reference [6] (SGT-Net)** metadata is unverified (see `References/References.md`).

---

## Scores

| Metric | Score | Basis |
|--------|:-----:|-------|
| **Consistency (Implementation vs Introduction)** | **9.5 / 10** | Every technical statement matches the code; the −0.5 is the unverified clinical-statistics source [20], not a code mismatch. |
| **IEEE readiness** | **8 / 10** | Standard IEEE flow (motivation → methods → gap → objective → contributions), formal tone, explicit contribution separation. Held below 9 by the pending synopsis-direction decision and the two unverified citations ([6], [20]). |

---

## Status

Status:
Draft v1

Dependencies:
- Experimental Results
- Supervisor Feedback
- Final Figures
- Final Tables
