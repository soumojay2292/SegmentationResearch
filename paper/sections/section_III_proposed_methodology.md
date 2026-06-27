# III. Proposed Methodology

The proposed framework, termed Stabilized MAFFNet, is a dual-pathway encoder–decoder network for dermoscopic skin-lesion segmentation. It is built upon a frozen foundation-model encoder and is designed so that the complementary information carried by spatial and spectral representations is extracted, aligned, and combined before decoding. The complete architecture is depicted in Fig. 1, with the frequency processing and fusion components detailed in Fig. 2 and Fig. 3, respectively. The following subsections describe the overall pipeline and each constituent module in turn, together with the training objective that is optimized.

## III-A. Overall Architecture

The network receives a dermoscopic image and produces a binary segmentation map of the lesion region. Let the input be denoted by $I \in \mathbb{R}^{3 \times H \times W}$, where $H = W = 384$ in the present configuration. The image is first standardized to the channel statistics expected by the pretrained encoder and is then passed through a frozen Segment Anything Model 2.1 (SAM2.1) Hiera-Large encoder, from which a set of multi-scale feature maps is extracted. Each feature map is subsequently processed by two parallel streams. In the spatial stream, the encoder feature is retained in the spatial domain and is normalized and projected by a feature-alignment module. In the frequency stream, the same encoder feature is transformed into the spectral domain, high-pass filtered, reconstructed by an inverse transform, and then aligned by an identical alignment operator. The two aligned representations are combined by a lightweight gated fusion module, the output of which is propagated to a convolutional decoder through projected skip connections. The decoder produces multi-scale predictions under deep supervision, and the highest-resolution prediction is taken as the final segmentation output. The overall data flow is therefore

$$
I \rightarrow \text{Encoder} \rightarrow \{\text{spatial}, \text{frequency}\} \rightarrow \text{alignment} \rightarrow \text{fusion} \rightarrow \text{decoder} \rightarrow \hat{Y}.
$$

The division of the network into spatial and frequency pathways is motivated by the manner in which boundary information is represented and degraded in convolutional encoders. Repeated downsampling behaves as a sequence of low-pass operations, so that the high-frequency components that encode edge gradients are progressively attenuated relative to the low-frequency content that describes the lesion interior. A purely spatial decoder cannot restore spectral content that has already been suppressed during encoding. The frequency pathway is introduced precisely to recover this content: by operating in the spectral domain, the high-frequency coefficients associated with lesion boundaries can be isolated and reconstructed into a boundary-sensitive spatial representation. The spatial pathway, in contrast, preserves the semantic context and the coarse regional structure required to localize the lesion. Because each pathway captures information that the other tends to lose, their explicit separation and subsequent fusion provide a more complete description of the lesion than either domain alone.

## III-B. Frozen SAM2 Encoder

The encoder of the proposed network is the image encoder of the SAM2.1 Hiera-Large model. This encoder is adopted because it is pretrained on a very large and diverse corpus of segmentation masks, and consequently its intermediate representations carry strong, transferable spatial priors that generalize across visual domains. For dermoscopic segmentation, where annotated data are comparatively scarce, such priors reduce the dependence on large task-specific training sets and provide a stable initialization for the downstream modules.

The encoder parameters are held fixed throughout training. Freezing the encoder is appropriate for two reasons. First, full adaptation of a foundation-scale encoder is computationally demanding and is prone to overfitting when the available medical data are limited; by keeping the encoder fixed, the number of trainable parameters is confined to the lightweight modules and the decoder, and the strong pretrained representations are preserved. Second, because no gradients are propagated through the encoder, its contribution to the training cost is restricted to a single forward evaluation, which improves computational efficiency relative to a fully fine-tuned backbone of comparable size.

The hierarchical structure of the Hiera trunk yields feature maps at several spatial resolutions. The encoder is expressed as

$$
\{X_1, X_2, X_3, X_4\} = \mathcal{E}_{\mathrm{SAM}}(I), \qquad X_i \in \mathbb{R}^{C \times H_i \times W_i}, \tag{1}
$$

where $i \in \{1,2,3,4\}$ indexes the scale from shallow to deep, $C$ denotes the common channel width of the extracted features, and $(H_i, W_i)$ correspond to spatial strides of $4, 8, 16,$ and $32$ with respect to the input. The same set $\{X_i\}$ is reused without modification by both the spatial and the frequency pathways, so that the encoder is evaluated only once per image and its features serve as the shared basis for all subsequent processing.

## III-C. Frequency Branch

The frequency branch, illustrated in Fig. 2, transforms an encoder feature map into a boundary-sensitive spatial representation by retaining only its high-frequency content. For a given scale, the operation is applied independently to the feature map $X_i$.

The forward transform is the two-dimensional discrete Fourier transform, which is evaluated in single precision irrespective of the precision of the surrounding computation. Single-precision evaluation is adopted so that the spectral computation remains numerically stable under mixed-precision training. Denoting the single-precision cast by $(\cdot)_{\mathrm{fp32}}$, the spectrum is obtained as

$$
\mathcal{F}_i(u,v) = \mathrm{FFT}_{2}\!\big[(X_i)_{\mathrm{fp32}}\big](u,v), \tag{2}
$$

where $(u,v)$ index the discrete frequency coordinates and $\mathcal{F}_i$ is complex-valued.

A radial high-pass mask is then applied. Let $f_y(u)$ and $f_x(v)$ denote the normalized frequencies along the vertical and horizontal axes, expressed in cycles per pixel, and let the radial frequency be

$$
r(u,v) = \sqrt{f_y(u)^2 + f_x(v)^2}. \tag{3}
$$

The high-pass mask is defined by a fixed cutoff $\rho$ as

$$
M(u,v) =
\begin{cases}
1, & r(u,v) \ge \rho, \\
0, & r(u,v) < \rho,
\end{cases} \tag{4}
$$

so that frequencies below the cutoff, including the direct-current component, are suppressed and the high-frequency coefficients that encode edge structure are retained. The filtered spectrum is

$$
\mathcal{F}_i^{\,\mathrm{hp}}(u,v) = M(u,v)\,\mathcal{F}_i(u,v). \tag{5}
$$

A boundary-sensitive spatial map is recovered by the inverse two-dimensional transform, which is constrained to reconstruct at the original spatial resolution,

$$
X_i^{\mathrm{freq}} = \mathrm{IFFT}_{2}\!\big[\mathcal{F}_i^{\,\mathrm{hp}}\big] \in \mathbb{R}^{C \times H_i \times W_i}. \tag{6}
$$

Because the mask removes the low-frequency content, the reconstructed map is approximately zero-mean and emphasizes intensity transitions, that is, the regions in which the lesion boundary is located. The explicit specification of the output size in (6) ensures that the spatial resolution of $X_i^{\mathrm{freq}}$ is identical to that of the input feature $X_i$, so that the spatial and frequency representations remain registered and can be combined pointwise. The branch contains no learnable parameters; the cutoff $\rho$ is a fixed hyperparameter.

## III-D. Feature Alignment

The spatial and frequency representations are produced by different operations and therefore exhibit different activation statistics. Before they are combined, each representation is processed by a feature-alignment module that harmonizes these statistics and projects the features into a common space. The module consists of an instance normalization, a $1\times1$ convolution, and a rectified-linear activation.

For an input feature $Z \in \mathbb{R}^{C \times H_i \times W_i}$, instance normalization is applied per channel and per sample over the spatial dimensions. With the per-channel mean and variance

$$
\mu_c = \frac{1}{H_i W_i}\sum_{h,w} Z_{c,h,w}, \qquad
\sigma_c^2 = \frac{1}{H_i W_i}\sum_{h,w}\big(Z_{c,h,w} - \mu_c\big)^2, \tag{7}
$$

the normalized feature is

$$
\hat{Z}_{c,h,w} = \frac{Z_{c,h,w} - \mu_c}{\sqrt{\sigma_c^2 + \varepsilon}}, \tag{8}
$$

where $\varepsilon$ is a small constant ensuring numerical stability. The normalization removes per-channel offsets and scale differences, so that the spatial and frequency streams are brought to a comparable dynamic range. The normalized feature is then linearly recombined across channels by a $1\times1$ convolution and passed through the activation,

$$
\mathcal{A}(Z) = \mathrm{ReLU}\!\big(W_p \ast \hat{Z} + b_p\big), \tag{9}
$$

where $W_p$ and $b_p$ denote the convolution weights and bias and $\ast$ denotes convolution. The $1\times1$ convolution performs channel harmonization, mapping each stream onto a shared channel configuration suitable for fusion, while the activation introduces a non-linearity. The same alignment operator is applied, with independent parameters, to the spatial feature and to the frequency feature, yielding the aligned representations $F_s = \mathcal{A}_s(X_i)$ and $F_f = \mathcal{A}_f(X_i^{\mathrm{freq}})$.

## III-E. Lightweight Gated Fusion

The aligned spatial and frequency features are combined by a lightweight gated fusion module, shown in Fig. 3, which learns a spatially varying, channel-wise weighting between the two streams. The two features are concatenated along the channel dimension and a single $1\times1$ convolution followed by a logistic sigmoid produces a gate,

$$
g = \sigma\!\big(W_g \ast [\,F_s \,\|\, F_f\,] + b_g\big), \qquad g \in (0,1)^{C \times H_i \times W_i}, \tag{10}
$$

where $[\cdot \,\|\, \cdot]$ denotes channel concatenation, $W_g$ and $b_g$ are the convolution parameters, and $\sigma(\cdot)$ is the sigmoid function. The fused representation is the convex combination

$$
F_{\mathrm{fused}} = g \odot F_s + (1 - g) \odot F_f, \tag{11}
$$

in which $\odot$ denotes element-wise multiplication. Because the gate lies in $(0,1)$, the fused feature is a per-element interpolation between the two streams; the output magnitude is therefore bounded by the inputs, and the gate determines, at each spatial location and channel, the relative emphasis placed on spatial context and spectral detail.

The fusion is computationally light because it avoids any pairwise interaction between spatial locations. Its dominant cost is that of a $1\times1$ convolution, which scales as $\mathcal{O}(C^2 H_i W_i)$ and is therefore linear in the number of spatial positions. In contrast, attention-based fusion forms an interaction matrix over all token pairs and scales as $\mathcal{O}((H_i W_i)^2 C)$, which is quadratic in the spatial resolution and becomes prohibitive at the resolutions required for boundary-sensitive dermoscopic analysis. The proposed gating mechanism attains an adaptive, location-dependent combination of the two domains without the projection, softmax, and quadratic-cost operations that characterize attention, and consequently incurs substantially lower computation and memory.

## III-F. Decoder and Skip Connections

The fused multi-scale features are decoded into a segmentation map by a convolutional decoder with skip connections. Prior to decoding, each fused feature that serves as a skip connection is processed by a $1\times1$ projection,

$$
S_i = W_i^{\mathrm{skip}} \ast F_{\mathrm{fused},i} + b_i^{\mathrm{skip}}, \tag{12}
$$

which adapts the channel representation of the skip path to the decoder. The decoder proceeds from the coarsest scale to the finest. At each stage, the incoming feature is upsampled, concatenated with the corresponding projected skip, and refined by two consecutive convolution–normalization–activation blocks, denoted collectively by $\mathrm{DConv}(\cdot)$,

$$
D_j = \mathrm{DConv}\!\big(\,[\,\mathrm{Up}(D_{j+1}) \,\|\, S_j\,]\big), \tag{13}
$$

where $\mathrm{Up}(\cdot)$ denotes bilinear upsampling and $D_{j+1}$ is the output of the preceding, coarser stage. The deepest stage is initialized from the deepest fused feature. At the finest stage, where no shallower encoder feature is available, the decoder feature is refined without concatenation.

Deep supervision is employed. A segmentation head, realized as a $1\times1$ convolution, is attached to each decoder stage, and its output is upsampled to the input resolution,

$$
p_k = \mathrm{Up}\!\big(W_k^{\mathrm{head}} \ast D_k + b_k^{\mathrm{head}}\big), \qquad k \in \{1,2,3,4\}, \tag{14}
$$

so that predictions are obtained at multiple scales. The prediction from the finest stage is taken as the final segmentation output, $\hat{Y} = \sigma(p_1)$. In addition, an auxiliary reconstruction decoder reconstructs an image-sized output from the fused features; this branch is used only during training and is removed at inference, so that it does not affect the deployed model. The use of multi-scale supervision provides gradient signals at several resolutions, which assists the decoder in recovering both the global lesion shape and the fine detail near the boundary.

## III-G. Loss Function

The network is trained by minimizing a composite objective that combines region-overlap supervision on the segmentation heads with a boundary term and an auxiliary reconstruction term. Let $y$ denote the ground-truth mask and $\sigma(p_k)$ the predicted probability at scale $k$. The Dice loss for a single prediction is

$$
\mathcal{L}_{\mathrm{Dice}}(\sigma(p_k), y) = 1 - \frac{2\sum \sigma(p_k)\,y + \epsilon}{\sum \sigma(p_k) + \sum y + \epsilon}, \tag{15}
$$

with $\epsilon$ a smoothing constant. The segmentation loss is the mean, over the four heads, of the Dice loss together with a binary cross-entropy term with a positive-class weight $\omega$,

$$
\mathcal{L}_{\mathrm{seg}} = \frac{1}{4}\sum_{k=1}^{4}\Big[\mathcal{L}_{\mathrm{Dice}}(\sigma(p_k), y) + \mathcal{L}_{\mathrm{BCE}}(p_k, y;\,\omega)\Big]. \tag{16}
$$

Boundary information is supervised by comparing the edges of the predictions with the edges of the ground truth. The edge operator is the gradient magnitude obtained from the horizontal and vertical Sobel responses $g_x$ and $g_y$, stabilized by a small constant $\delta$ and bounded to the unit interval,

$$
\mathcal{S}(m) = \mathrm{clamp}\!\Big(\sqrt{g_x(m)^2 + g_y(m)^2 + \delta},\,0,\,1\Big). \tag{17}
$$

The boundary loss aggregates, over the supervised heads, the mean absolute difference between the predicted and the ground-truth boundaries, with $\tilde{\sigma}(p_k)$ denoting a lightly smoothed prediction and $w_k$ the deep-supervision weights,

$$
\mathcal{L}_{\mathrm{B}} = \frac{1}{\sum_k w_k}\sum_{k} w_k\,\big\lVert \mathcal{S}(\tilde{\sigma}(p_k)) - \mathcal{S}(y) \big\rVert_1. \tag{18}
$$

The boundary term is introduced under a warmup schedule, so that its weight increases from zero to a maximum value $\lambda_{\mathrm{B}}$ over the first $T$ epochs,

$$
\lambda_{\mathrm{B}}(t) = \lambda_{\mathrm{B}}\,\min\!\Big(1,\ \tfrac{t}{T}\Big). \tag{19}
$$

The auxiliary reconstruction is supervised by a boundary-weighted reconstruction loss. A boundary prior $B(y) = \mathcal{S}(y)$ is derived from the ground-truth mask and is used to form a spatial weight $W(y) = 1 + \alpha\,B(y)$, where $\alpha$ controls the emphasis placed on boundary regions. With $r$ denoting the reconstruction mapped to the input range, $x$ the input image, and $C$ the number of channels, the reconstruction loss is the normalized weighted mean-squared error

$$
\mathcal{L}_{\mathrm{R}} = \frac{\sum W(y)\,(r - x)^2}{C\,\sum W(y)}. \tag{20}
$$

The complete training objective is the weighted sum of the three terms,

$$
\mathcal{L} = \mathcal{L}_{\mathrm{seg}} + \lambda_{\mathrm{B}}(t)\,\mathcal{L}_{\mathrm{B}} + \lambda_{\mathrm{R}}\,\mathcal{L}_{\mathrm{R}}, \tag{21}
$$

in which $\lambda_{\mathrm{R}}$ weights the reconstruction term. The objective in (21) is minimized over the trainable parameters of the alignment modules, the fusion modules, the decoder, the segmentation heads, and the auxiliary reconstruction decoder, while the SAM2.1 encoder remains fixed.

Having specified the proposed architecture and its training objective, the experimental configuration used to evaluate the framework, including the datasets, the implementation details, and the evaluation metrics, is described in the following section.
