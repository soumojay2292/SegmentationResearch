# MAFF-Net Reproduction — Implementation-Fidelity Verification

**Subject:** [`models/maffnet_original.py`](../../models/maffnet_original.py)
**Reference:** Zhou et al., *MAFF-Net*, Biomedical Signal Processing and Control 113 (2026) 109222 (15-page article; full text re-extracted and searched).
**Scope:** verification only. No code was modified; no training was run. Parameter figures come from instantiating the model and counting `numel()` per submodule.

---

## 0. Method

1. Re-extracted the full paper text and keyword-searched it for every hyperparameter the report assumed (`heads`, `depth`, `blocks`, `dim`, `FFN`, `hidden`, `width`, `in_channel`, `groups`, `scale_factor`, numeric channel widths).
2. Checked for any appendix / supplementary / code release.
3. Instantiated `MAFFNetOriginal` and measured parameters per top-level module to attribute the 185 M trainable count precisely.
4. Re-verified each of the report's nine assumptions against the paper text.

**Appendix / supplementary / code release:** none. The article has no appendix; "Data will be made available on request"; no GitHub or code-availability statement. Therefore **no external source can supply the missing dimensions** — they must be inferred.

---

## 1. Verification of the report's assumptions

| # | Assumption (report §2) | Paper evidence | Verdict |
|---|---|---|---|
| 1 | WetGeneration widths 32→16→3 | Eq. 4 operates on `Z'=GAP(Z)` where `Z` is the 32-d projected feature; text "reduce … from C to C/2 … to 3". | **Defensible** — best reading of an ambiguous `C`. |
| 2 | Eq. 3 Permute treated as artefact; depthwise `groups=32` over `(B,32,H,W)` | Eq. 3 "DWConv … number of groups being 32"; permute only reorders axes. | **Verified equivalent.** |
| 3 | Eq. 5 second `α⊙` dropped (already folded into `E_fused`) | `α∈R^3` cannot broadcast element-wise onto `E_fused∈R^{B×32×H×W}`; the re-use is a notation error. | **Verified** (literal eq. is ill-typed; interpretation sound). |
| 4 | EDB skip attention = spatial (pool over channel) → `1×H×W` | Eq. 11 text says per-channel avg/max **but** states output `[0,1]^{1×H×W}`. The two are mutually inconsistent **in the paper**; the stated shape implies channel-axis pooling. | **Defensible** — resolves a paper-internal contradiction toward the stated output shape. |
| 5 | Freq-Transformer `heads=4`, `ffn_ratio=2`, 1 block/stage | Paper: "incorporating **multiple levels** of Transformer blocks"; `d_k = num_heads/dim` symbolic only. No numeric value anywhere. | **Unverifiable / free choice.** "Multiple levels" weakly suggests >1 block per stage, so even the *depth* may be under-counted. |
| 6 | RDB inputs = fused `g1..g4` | Eq. 12 / §3.3.2: "four feature maps … generated during the **encoding** stage `s1..s4`." | **Ambiguous** — could equally be the post-MMPA spatial features. Chosen reading is reasonable but not the only one. |
| 7 | Loss applies `{1,1,1,0.01}` to four heads | Eq. 13 sums `i=0..2` (3 Dice) yet lists 4 weights and `s_i={16,8,4}` (3 factors). Paper-internal mismatch. | **Defensible** — applies all four listed weights to the four heads. |
| 8 | Placeholder encoder when SAM2 absent | N/A (engineering scaffold, clearly flagged). | **Verified non-architectural.** |
| 9 | Min-max (Eq. 6) with no epsilon | Eq. 6 has no epsilon. | **Verified faithful.** |

**Net:** assumptions 1–4, 7–9 are verified or defensible (several resolve genuine paper-internal inconsistencies). Assumptions **5 and 6 are unverifiable free choices** the paper does not constrain.

---

## 2. Omitted implementation details (searched, confirmed absent)

The paper specifies concretely **only**: MMPA projection `C→32`; depthwise `groups=32`; MMPA dilations `{3,5,7}`; Γ init `1e-7`; FGFF dilations `{1,3,5,7}`; RDB `ConvTranspose(k=4,s=2,p=1)` with `C_out=C_in/2`; loss weights `{1,1,1,0.01}`; FFT/min-max formulae; training recipe (AdamW, lr 1e-3, wd 0.01, bs 12, 100 ep, 384², linear schedule).

### Requested hyperparameter checklist

| Detail | Defined in paper? | Evidence |
|---|---|---|
| Frequency-Transformer **depth** | **No** | "multiple levels of Transformer blocks" — unquantified (line ~598). |
| Number of **attention heads** | **No** | only symbolic `num_heads`, `d_k=num_heads/dim` (lines ~626–630). |
| **FFN ratio** | **No** | `FFN(LayerNorm(Z))` named, no dimension (Eq. 8). |
| **Hidden / embedding dim** (`dim`) | **No** | `P(l)∈R^{B×dim×H/4×W/4}` — `dim` never assigned a value. |
| **Decoder channel widths** (EDB) | **No** | Eq. 10 uses symbolic `C_in`, `C_low`; no numbers. |
| **FGFF channel widths** | **No** | Eq. 9 uses `f_spatial`/`f_freq`; no channel counts. |

Every quantity the verification was asked to find is **absent from the paper and from any supplementary material**. The reproduction therefore had to assume them, which is the root of the size discrepancy.

---

## 3. Why ≈185 M trainable vs the paper's ≈14.8 M

Measured per-module (placeholder encoder; `numel` of trainable params):

| Module | Reproduction | Paper budget | Ratio | Driver |
|---|---:|---:|---:|---|
| **FGFF ×4** | **133.98 M** | 2.2 M | **≈61×** | 4 dense 3×3 dilated convs + wide concat-fuse at full stage widths up to **1152** |
| Frequency encoder | 23.80 M | ≈2.7 M¹ | ≈9× | Transformer blocks paired to 1152-wide stages |
| EDB ×3 | 15.68 M | 0.4 M | ≈39× | DoubleConv at concat widths up to 1728→576 |
| RDB | 11.80 M | 7.8 M | ≈1.5× | full-width grouped transposed convs |
| MMPA ×4 | 0.155 M | 1.7 M | ≈0.09× (**under**) | lean depthwise experts; no extra projections |
| Seg heads ×4 | 0.002 M | — | — | 1×1 convs |
| Encoder | 7.86 M (frozen) | 227 M (frozen) | — | placeholder, not real SAM-ViT-H |
| **Total trainable** | **185.41 M** | **≈14.8 M** | **≈12.5×** | — |

¹ Paper's 14.8 M = MMPA 1.7 + FGFF 2.2 + EDB 0.4 + RDB 7.8 = 12.1 M itemised; the ≈2.7 M remainder covers the frequency encoder, bottleneck, and heads.

**Single dominant cause:** FGFF accounts for **134 M (72 %)** of all trainable parameters, and `fgff[3]` alone (C = 1152) is **101 M**. The paper's 2.2 M FGFF budget is only achievable if fusion runs at a small unified width and/or with depthwise-separable branch convolutions — neither of which the paper states.

---

## 4. Sources of the discrepancy (ranked, every contributor)

1. **FGFF runs at full per-stage width (≤1152) with dense 3×3 convs** → +≈132 M. *Largest source.* Paper's 2.2 M implies a small unified fusion width / separable convs.
2. **Frequency encoder paired to 1152-wide stages** → +≈21 M. Paper's lightweight encoder must use a small fixed `dim`.
3. **EDB DoubleConv at native concat widths (≤1728)** → +≈15 M. Paper's 0.4 M implies a narrow decoder width (≈32–64).
4. **RDB at full grouped-transpose widths** → +≈4 M over budget (closest match; ≈1.5×).
5. **Placeholder encoder vs real frozen SAM-ViT-H** → frozen count 7.9 M vs 227 M (affects *total*, not *trainable*; non-architectural).
6. **MMPA leaner than paper** → −1.5 M (a *negative* contributor; mine is smaller, indicating the paper's MMPA carries extra parameters this reproduction omits).
7. **Under-specified depth (assumption 5)** — using 1 block/stage may *under*-count relative to "multiple levels," partially offsetting (1)–(3).

All sources except (5) and (6) stem from the same root: **channel widths the paper never specifies were set by pairing each fusion/decoder stage to the encoder stage width instead of to a small unified width.**

---

## 5. Classification

| Category | Applies? | Justification |
|---|---|---|
| Exact reproduction | **No** | A *published* quantity — the 14.8 M trainable budget — is off by ≈12.5×. |
| **Structurally faithful but dimensionally different** | **Yes** | Every module, equation (1–13), connection, I/O contract, and design choice is present and correctly wired; only the **unspecified channel dimensions** differ, inflating parameters. |
| Functionally equivalent | **Partly** | Same forward semantics, output signature `(p1..p4, rec)`, and loss; but parameter scale, FLOPs, and learned behaviour will differ — not numerically equivalent. |
| Not paper-faithful | **No** | It faithfully follows every *stated* element of the paper. |

**Verdict: Structurally faithful but dimensionally different** (with two genuinely unconstrained free choices: frequency-Transformer dimensions and RDB input source).

---

## 6. Confidence score

| Dimension | Weight | Fidelity | Notes |
|---|---:|---:|---|
| Module presence & connectivity | 30% | ~98% | all modules present and correctly wired |
| Equation correctness (1–13) | 30% | ~92% | faithful; resolves paper-internal ambiguities in Eqs. 5, 11, 13 |
| Stated hyperparameters honoured | 15% | ~95% | dilations, C→32, Γ, RDB conv specs, loss weights all matched |
| Dimensional fidelity (widths/params) | 20% | ~20% | trainable params ≈12.5× over a *published* budget; FGFF 61× |
| Free-choice gaps (freq dims, RDB src) | 5% | ~40% | unconstrained by paper |

**Overall faithfulness confidence: 70 / 100.**

Interpretation: the reproduction is a **high-fidelity structural and mathematical match** to the published MAFF-Net (the verifiable core ≈ 90%+), but it is **not dimensionally faithful**: it diverges by ≈12.5× on the one published size figure, driven overwhelmingly by full-width FGFF. Because the paper omits every width needed to reach 14.8 M and releases no code, an exact reproduction is **not attainable from the paper alone**. Reaching the published budget would require (a) a small unified fusion/decoder width (likely with depthwise-separable FGFF branches), (b) a fixed small frequency-Transformer `dim`, and (c) the real frozen SAM-ViT-H trunk — none of which can be derived from the text without author confirmation.

---

*Status: Verification v1 / No code changed / Dependencies: author confirmation of unstated widths, real SAM2 checkpoint. Recommended next step (not performed): introduce a configurable unified fusion/decoder width and depthwise-separable FGFF branches to bring trainable params toward 14.8 M, then re-verify.*
