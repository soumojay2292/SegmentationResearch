# 6. Discussion

> Framework only. The discussion is written in generic terms and contains no numerical performance; quantitative statements are deferred to Section 5 and are referenced here as placeholders to be completed once the experiments are finalized.

## 6.1 Strengths

The proposed Boundary-centric Dual Supervision acts entirely at the level of the training objective. It introduces no additional modules or parameters and leaves the inference-time network identical to the restored MAFF-Net, so any improvement it provides is obtained without increasing model size or runtime. This separation also makes the comparison clean: because the architecture is held fixed and only the supervision changes, an observed effect can be attributed to the supervision rather than to a change in capacity. The component ablation in Section 5.3 is designed to make this attribution explicit, and the magnitude of the effect will be reported there.

A second strength is coherence. Both the boundary loss and the edge-weighted reconstruction are derived from a single boundary prior taken from the ground-truth mask, and they apply that prior to two different parts of the network. The method therefore adds one consistent idea rather than several loosely related terms, and it reduces to the restored baseline when its weights are set to zero, which keeps the design interpretable and the ablation well defined. The boundary operator is fixed rather than learned, and the boundary term is introduced under a warmup schedule, which keeps the added supervision simple and stable to train.

Finally, the work is built on a faithfully restored implementation of MAFF-Net with a documented, uniform training protocol and fixed data splits. This provides a controlled basis for the study and supports reproducibility, independently of the size of the final improvement.

## 6.2 Limitations

The scope of the contribution is deliberately narrow. It is a supervision strategy applied to an existing architecture, not a new architecture, and its value depends on whether the two terms provide a benefit beyond the restored baseline and beyond each other. Whether the combined supervision improves on either term used alone is an empirical question that the ablation in Section 5.3 is intended to answer; if that separation does not hold, the contribution should be reported as the single more effective term rather than as a pair.

Several factors constrain interpretation of the results. The boundary prior is derived from the annotated mask through a fixed edge operator, so its quality is bounded by the quality and consistency of the annotations, and lesions with genuinely ambiguous or gradual borders may yield a noisy prior. The edge-weighted reconstruction modifies a branch that is auxiliary and active only during training, so its influence on the final segmentation may be limited and should not be assumed in advance. The evaluation in this study uses overlap-based metrics; a complete assessment of boundary behaviour would benefit from boundary-specific measures, and where these are not computed the boundary claims rest on overlap metrics together with qualitative comparison.

The experimental protocol also limits external comparison. All models, including the restored MAFF-Net, are trained under a single protocol whose batch size and schedule differ from those in the original MAFF-Net paper, so the restored model's numbers are not directly comparable to the originally published values. For ISIC 2017, evaluation uses a held-out test set that we constructed from the annotated training data, which means the ISIC 2017 results are not directly comparable to methods evaluated on the official 2017 test partition. The study is confined to dermoscopic skin-lesion segmentation, and generalization to other imaging modalities is not examined here.

## 6.3 Practical Implications

Because the method changes only the training objective, it can be added to an existing MAFF-Net-style pipeline without altering the deployed model. There is no additional inference cost, which keeps the deployment characteristics of the underlying framework unchanged and preserves its suitability for time-sensitive clinical use. For groups that already train such a model, adopting the supervision requires changing the loss rather than re-engineering the network.

The method is also aligned with the clinical priority that motivates the work. Diagnostic decisions in dermoscopy depend heavily on margin geometry, and a supervision signal that targets the boundary directs learning toward the region clinicians rely on, rather than toward interior fill that overlap metrics already reward. Should the ablation confirm a benefit, this alignment is the practical reason to prefer boundary-centric supervision over interior-weighted objectives alone.

Beyond the specific framework studied here, the underlying idea—deriving a single boundary prior and using it to supervise both the segmentation outputs and an auxiliary reconstruction branch—is not tied to any one architecture and could in principle be applied to other models that provide deep supervision and an auxiliary reconstruction path. We do not evaluate this broader applicability in the present work and note it only as a direction, not a claim.
