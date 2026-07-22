III-I. Training Objective / Loss Formulation

The Proposed MAFF-Net is trained by minimizing a single composite objective that combines a primary region-segmentation term with the two auxiliary boundary-focused terms of Boundary-centric Dual Supervision introduced in the previous subsection. Let $x$ denote the input dermoscopic image with $C$ channels, $y \in \{0,1\}^{H \times W}$ the binary ground-truth lesion mask, and $p_i$, for $i = 1,\dots,N$ with $N = 4$, the segmentation predictions (as logits) produced at the decoder stages under deep supervision. The logistic sigmoid is written $\sigma(\cdot)$, and sums indexed by $u$ run over spatial locations. The region term supervises the predicted lesion area, whereas the two Boundary-centric Dual Supervision terms supervise the lesion boundary; the region term is the primary training signal and the boundary terms are auxiliary.

The region term is built from two standard segmentation losses. The soft Dice loss measures the overlap between a predicted probability map and the ground-truth mask,
$$
L_{\mathrm{Dice}}(p,y) = 1 - \frac{2\sum_{u}\sigma(p)_u\,y_u + s}{\sum_{u}\sigma(p)_u + \sum_{u}y_u + s}, \tag{1}
$$
where $s$ is a small smoothing constant that keeps the expression well defined when the mask or prediction is empty. A value of $L_{\mathrm{Dice}}$ near zero indicates strong overlap. The binary cross-entropy loss, applied to the logits with a positive-class weight $\omega$, measures per-pixel classification agreement,
$$
L_{\mathrm{BCE}}(p,y) = -\frac{1}{|\Omega|}\sum_{u \in \Omega}\Big[\omega\,y_u\log\sigma(p)_u + (1-y_u)\log\big(1-\sigma(p)_u\big)\Big], \tag{2}
$$
where $\Omega$ is the set of spatial locations and $\omega$ up-weights lesion (positive) pixels relative to background pixels to counteract class imbalance.

The region segmentation objective aggregates these two losses over the deep-supervision predictions. Each of the $N$ decoder predictions is supervised with the sum of the Dice and cross-entropy losses, and the results are averaged uniformly across the predictions,
$$
L_{\mathrm{seg}} = \frac{1}{N}\sum_{i=1}^{N}\Big[L_{\mathrm{Dice}}(p_i,y) + L_{\mathrm{BCE}}(p_i,y)\Big]. \tag{3}
$$
The uniform average in Eq. (3) is the deep-supervision aggregation used for the region term: every prediction contributes equally, so the region objective supervises the coarse and fine decoder stages with the same weight.

Boundary-aware Deep Supervision compares the boundary structure of the predictions with that of the ground truth. Boundaries are extracted by an edge operator $S(\cdot)$ that applies the horizontal and vertical Sobel filters $G_x$ and $G_y$, forms the gradient magnitude, and bounds it to the unit interval,
$$
S(m) = \mathrm{clamp}\!\Big(\sqrt{(G_x * m)^2 + (G_y * m)^2 + \varepsilon},\,0,\,1\Big), \tag{4a}
$$
where $*$ denotes convolution and $\varepsilon$ is a small constant that keeps the gradient of the square root finite in flat regions. The Boundary-aware Deep Supervision loss is then the weighted average, over the deep-supervision predictions, of the discrepancy between the predicted and ground-truth edge maps,
$$
L_{\mathrm{BDS}} = \frac{1}{\sum_{i}w_i}\sum_{i=1}^{N} w_i \,\big\lVert\, S\big(\tilde{\sigma}(p_i)\big) - S(y)\,\big\rVert_1, \tag{4b}
$$
where $\tilde{\sigma}(p_i)$ is a smoothed sigmoid prediction and $w_i$ are per-prediction weights. In contrast to the uniform aggregation of Eq. (3), Eq. (4b) uses the weights $w_i$, so the finest predictions dominate the boundary term while the coarsest contributes only marginally.

The boundary term is introduced gradually rather than from the first iteration. Its weight follows a linear warm-up schedule,
$$
\lambda_{B}(t) = \lambda_{\mathrm{bds}}\cdot\min\!\Big(1,\ \frac{t}{T_{\mathrm{warm}}}\Big), \tag{5}
$$
where $t$ is the current epoch index, $T_{\mathrm{warm}}$ is the warm-up length, and $\lambda_{\mathrm{bds}}$ is the maximum boundary weight. According to Eq. (5), the Boundary-aware Deep Supervision weight increases from zero to $\lambda_{\mathrm{bds}}$ over the first $T_{\mathrm{warm}}$ epochs and remains constant thereafter, so the region objective governs the early training and the boundary term is phased in once the predictions have stabilized.

Edge-weighted Reconstruction supervises the auxiliary reconstruction branch with a boundary-emphasized reconstruction error. A spatial weight map is derived from the ground-truth boundary, $W = 1 + \alpha\,B(y)$, where $B(y)$ is a boundary band obtained from $S(y)$ and $\alpha \ge 0$ controls the emphasis placed on boundary regions. Writing $r$ for the reconstruction mapped to the range of the input image, the loss is a normalized weighted mean-squared error,
$$
L_{\mathrm{EWR}} = \frac{\sum_{u} W_u\,(r_u - x_u)^2}{C\sum_{u} W_u}. \tag{6}
$$
Because $W$ exceeds unity along the lesion boundary and equals unity elsewhere, Eq. (6) concentrates the reconstruction pressure on boundary regions while normalizing by the total weight and the number of channels $C$.

The complete training objective is the weighted sum of the region term and the two boundary terms,
$$
L = L_{\mathrm{seg}} + \lambda_{B}(t)\,L_{\mathrm{BDS}} + \lambda_{\mathrm{ewr}}\,L_{\mathrm{EWR}}, \tag{7}
$$
where $\lambda_{\mathrm{ewr}}$ is the reconstruction weight. In the implemented configuration, the region objective carries unit weight and is therefore the dominant term, while the boundary terms are auxiliary, with $\lambda_{\mathrm{bds}} = 0.05$ and $\lambda_{\mathrm{ewr}} = 0.05$; the remaining constants are $\omega = 2$, $\alpha = 3$, $T_{\mathrm{warm}} = 5$, and per-prediction weights $w = (1, 1, 1, 0.01)$, with $s$ and $\varepsilon$ set to small fixed values. Equation (7) thus realizes the supervision framework of the preceding subsection: a primary region objective supplemented by the boundary-focused terms of Boundary-centric Dual Supervision.

The objective in Eq. (7) is minimized over the trainable parameters of the network — the fusion-pathway modules, the decoder, the segmentation heads, and the auxiliary reconstruction branch — while the encoder remains frozen. Optimization uses the AdamW optimizer with decoupled weight decay, a learning rate that is decayed linearly toward zero over the course of training, and gradient-norm clipping; training is performed under mixed-precision autocasting, and the model used for evaluation is the one achieving the best validation Dice score. Having specified the training objective, the following section describes the datasets, implementation details, and evaluation protocol used in the experiments.
