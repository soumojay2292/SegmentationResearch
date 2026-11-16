# Abstract — Paper 2

> Standalone extract from paper2/manuscript/Paper2_Manuscript.md.
> Word count: 241 (target 180-250). No citations, no equations, no numerical results.

## Title

A Dual-Stream Spatial–Spectral Fusion Pathway for Boundary Fidelity in Dermoscopic Skin Lesion Segmentation

## Abstract

Accurate delineation of lesion boundaries remains a fundamental difficulty in dermoscopic image segmentation, where gradual lesion-to-skin transitions, irregular margins, and occluding structures such as hair make the contour the least certain part of any prediction. Segmentation methods are nevertheless optimised and assessed predominantly by region-overlap criteria, which are area-based and respond only weakly to displacement of the contour; boundary-relevant content is commonly merged with contextual information in a single representation dominated by the lesion interior. This paper presents a targeted refinement of the MAFF-Net framework in which the pretrained encoder, decoder, and auxiliary reconstruction branch are retained and only the fusion pathway is redesigned. Multi-scale features from a frozen foundation backbone are processed along two parallel streams, a spatial stream that preserves contextual structure and a parameter-free frequency stream that emphasises boundary-sensitive spectral content; their statistics are reconciled before they are merged by a learned, spatially varying gate. The method was evaluated on three public dermoscopic benchmarks against four segmentation architectures under an identical protocol, using region-based and boundary-based measures in combination, and each introduced component was isolated by a leave-one-out ablation. Removing any component degraded all boundary measures while leaving region overlap substantially unchanged. The contribution is accordingly characterised as an improvement in boundary fidelity rather than region overlap, and indicates that region-overlap criteria alone are insufficient to assess methods acting upon the lesion margin. All results derive from single training runs, and no claim of statistical significance is made.

## Keywords

Skin lesion segmentation; dermoscopy; boundary fidelity; frequency-domain representation; gated feature fusion; frozen foundation model; deep supervision; ISIC.
