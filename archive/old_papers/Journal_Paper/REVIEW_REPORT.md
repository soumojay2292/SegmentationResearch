# Review Report — Senior Reviewer / Technical Editor

**Manuscript:** SAM2-based skin-lesion segmentation; restored MAFF-Net + Boundary-centric Dual Supervision.
**Material reviewed:** drafted Sections 1, 2, 3.1–3.6, 4; Section 5 (skeleton, no numbers); Section 6; Section 7. Front matter (title/abstract/keywords) and References not yet written.
**Recommendation: Major Revision** — strong writing and honest framing, but not submittable: no experimental results, no abstract/reference list/figures, one loss inconsistency vs code, and an unresolved mismatch with the approved synopsis.

---

## 1. Technical Correctness

| Item | Finding | Severity |
|------|---------|:--------:|
| §3.2 MMPA | Matches restored code (bottleneck-32, depthwise dilated experts d=3/5/7, softmax weighting, γ≈1e-7 residual). | OK |
| §3.3 FGFF | Matches code: FFT **magnitude** only, fp32, cross-modal fusion with learnable β. Correctly omits high-pass/iFFT (not in code). | OK |
| §3.4 BDS / §3.5 EWR | Match code: Sobel-edge L1 at the deep-supervision heads; edge-weighted MSE with W=1+αB; warmup ramp. | OK |
| §3.6 loss | **Inconsistency.** Paper states L = Dice + BDS + EWR. Code also adds **BCEWithLogits (pos_weight=2)** by default (`use_bce=True`). Paper omits the BCE term. | **High** |
| §4 config | Matches verified config (batch 2, ≤40 epochs / patience 10, AdamW 1e-3, wd 0.01, linear decay, AMP, grad-clip 1.0, fp32 FFT, frozen encoder). | OK |
| §4 hardware | GPU model is a placeholder. Acceptable interim, but must be filled before submission. | Medium |
| Unsupported claims | None overclaimed: Results are placeholders; Intro/Conclusion avoid asserting improvement or SOTA. Good discipline. | OK |
| §4.5 metrics | Lists the six implemented metrics (Dice/IoU/Acc/Sens/Spec/F1). No HD95/Boundary-IoU — consistent with code, but a **boundary** paper should report a boundary metric (see §6 below). | Medium |

**Action:** reconcile §3.6 with code — either set `use_bce=False` for the reported runs (paper-faithful Dice) or add the BCE term to the equation. Fill hardware. Add a boundary-specific metric to support the boundary claim.

---

## 2. Research Consistency (synopsis vs original vs implementation vs draft)

This is the most serious issue. See `RESEARCH_ALIGNMENT_REPORT.md` for the full matrix.

| Comparison | Status |
|-----------|--------|
| Draft ↔ Implementation | **Aligned** (draft was written to match the code). |
| Draft ↔ Original MAFF-Net | **Aligned** (draft distinguishes original vs restored correctly). |
| Draft ↔ **Approved Synopsis** | **Mismatch (major).** Synopsis commits to stabilised-FFT + **high-pass filter + inverse FFT + feature-alignment module + gated fusion** and an **SGT-Net** baseline. The draft (and code) contain none of these; instead they present restored MAFF-Net + BcDS. |
| Synopsis objective 1 ("boundary-aware learning as primary objective") | Goal preserved by BDS+EWR; mechanism differs. |

**Mismatches to highlight:**
1. Synopsis methodology (high-pass, iFFT, feature alignment, gated fusion) absent from draft/code.
2. Draft's contribution (BDS+EWR) not named in synopsis.
3. SGT-Net baseline promised by synopsis, absent from draft (§4) and code.

**Action:** resolve with the supervisor before submission — either implement the synopsis methodology (Path A) or obtain an approved amendment to the synopsis (Path B). The paper cannot be finalised while it silently contradicts the approved synopsis.

---

## 3. IEEE Structure

| Element | Present? | Note |
|---------|:--------:|------|
| Title | ❌ | front matter empty |
| Abstract | ❌ | required |
| Keywords / Index Terms | ❌ | required |
| Introduction (I) | ✅ | strong |
| Related Work (II) | ✅ | strong |
| Proposed Method (III) | ✅ | 3.1–3.6 complete in prose |
| Experimental Setup (IV) | ✅ | complete |
| Results (V) | ⚠️ | skeleton only, all cells placeholders |
| Discussion (VI) | ✅ | framework only (generic) |
| Conclusion (VII) | ✅ | results placeholder |
| References | ❌ | citation numbers used, no bibliography |
| Author/affiliation, Acknowledgements | ❌ | pending |

Structure is sound and broadly IEEE-conformant. Gaps are front matter, References, and the empty Results. Consider whether Results and Discussion should merge (common in IEEE); current split is acceptable.

---

## 4. Writing Quality

- **Tone:** formal, consistent, appropriate for a Q1/IEEE venue. Good.
- **Grammar:** clean; no notable errors found.
- **Flow / transitions:** logical progression; each method subsection states provenance up front (effective).
- **Repetition:** the motif "pooling attenuates high-frequency edge content / boundaries smoothed" recurs across §1, §2.1, §2.2, and §3.3. Tighten — state once in §1, reference thereafter.
- **Clarity:** strong. Equations are introduced with adequate prose.
- Minor: a few long compound sentences in §1 and §2.3 could be split. The word "boundary/margin" is very dense; acceptable for the topic but watch for monotony.

**Action:** reduce the repeated pooling/boundary-smoothing explanation to one canonical statement; lightly split the longest sentences.

---

## 5. Citations

| Issue | Detail |
|-------|--------|
| **No reference list** | Citations [1]–[15] are used but the bibliography does not exist. Build it. |
| **SAM2 not cited** | The method uses SAM2 (Hiera) but only SAM [14] is cited. Add the SAM2 reference (Ravi et al., 2024). |
| **Uncited factual claims (§1)** | Melanoma mortality/survival statistics ("≈75% of deaths", "98%/30% survival") need primary sources. |
| **Boundary-loss prior art (§3.4)** | BDS resembles established edge/boundary losses; cite representative prior work (e.g., boundary/active-contour losses) to position the contribution honestly and pre-empt a novelty objection. |
| **Backbone components** | TransUNet baseline now uses R50+ViT-B/16; cite ViT and ResNet where introduced. |
| **Citation placement** | Generally correct; verify [4]/[11]/[12] map to ISIC 2018/2016/2017 consistently in the final list. |

**Action:** assemble the full `.bib`, add SAM2 + melanoma-statistics + boundary-loss citations, verify number↔reference mapping.

---

## 6. Figures, Tables, Equations

**Figures (none produced yet):**
- F1 Overall architecture (code-accurate). **Required.**
- F2 MMPA block, F3 FGFF block. Recommended.
- F4 Boundary-centric Dual Supervision pipeline (BDS + EWR sharing B(y)). **Required** — this is the contribution figure.
- F5/F6 Qualitative comparison + best/average/difficult panels. **Required.**
- F7 Ablation / training-curve charts. Recommended.

**Tables:** Table 1 (datasets) filled. Tables 2–8 (comparison + ablation) are empty skeletons — must be populated from experiments.

**Equations:** good coverage (FGFF fusion, BDS, EWR, total loss, metrics). **Missing:** a formal equation for MMPA (currently prose only) — add for completeness, since §3.2 is the most heavily mathematical original module. Optionally add the FFT-magnitude definition in §3.3.

---

## 7. Contribution Clarity

**Strength of the manuscript.** The draft consistently separates the three layers:
- **Original MAFF-Net** (cited, described as published),
- **Restored implementation** (explicitly "the experimental base, not a contribution"),
- **Our contribution** (BcDS = BDS + EWR).

This distinction appears in §1 (contributions list), every §3 subsection (provenance sentence), §6, and §7. This is handled well and should be preserved.

**Caveat:** because the contribution is supervision-level and edge losses are well known, the novelty is modest. The paper must (a) cite prior boundary losses, and (b) demonstrate via ablation that the *dual* coupling beats either term alone, or the contribution will read as incremental.

---

## 8. Publication Readiness Scores (/10)

| Criterion | Score | Justification |
|-----------|:-----:|---------------|
| Novelty | 4 | Supervision-level extension; edge losses are established; novelty rests on the dual coupling, unproven until ablation. |
| Technical quality | 4 | Method sound and consistent with code, but **no results**, no boundary metric, one loss inconsistency. |
| Organization | 7 | Clear IEEE-style structure; excellent original/restored/contribution separation; front matter + refs missing. |
| Readability | 8 | Formal, clean, well-sequenced prose. |
| Publication readiness | 3 | Blocked by empty Results, missing abstract/references/figures, synopsis mismatch, and loss inconsistency. |

**Overall: Major Revision.** Not submittable in current state; the writing foundation is strong but the empirical and consistency gaps are decisive.

---

## Prioritised Actions

**P0 (blocking):**
1. Resolve the **approved-synopsis mismatch** with the supervisor (Path A implement, or Path B amend). 
2. Run experiments and populate Section 5 (comparison + ablation), including the **A/B/C/D** test of the dual claim.
3. Fix the **§3.6 ↔ code loss inconsistency** (BCE term).

**P1 (required for submission):**
4. Write Title, Abstract, Keywords; build the **reference list** (incl. SAM2, melanoma stats, boundary-loss prior art).
5. Add a **boundary-specific metric** (e.g., HD95 / Boundary-IoU) to support the boundary claim.
6. Add the **SGT-Net** baseline (synopsis commitment) or record its removal in the amendment.
7. Produce required figures (F1, F4, qualitative).

**P2 (polish):**
8. Fill hardware (GPU model). 9. Reduce repeated pooling/boundary explanation. 10. Add MMPA equation. 11. Note batch/epoch deviation from the original MAFF-Net budget (already flagged in §4 — keep).

---

## Evidence References
- Drafts: `01_Introduction/`, `02_Related_Work/`, `03_Methodology/3.1–3.6`, `04_Experimental_Setup/`, `05_Results/` (skeleton), `06_Discussion/`, `07_Conclusion/`
- Code: `models/maffnet.py`, `training/trainer.py` (loss incl. BCE switch)
- `RESEARCH_ALIGNMENT_REPORT.md`, `RESTORATION_REPORT.md`, `CONTRIBUTION_IMPLEMENTATION_REPORT.md`
- Synopsis §1.3.1 objectives; original paper §3.1–3.4
