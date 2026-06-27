# 4. Experimental Setup

## 4.1 Datasets

We evaluate on three publicly available ISIC dermoscopic skin-lesion segmentation benchmarks—ISIC 2016 [11], ISIC 2017 [12], and ISIC 2018 [4]—which differ in size, diagnostic category distribution, and annotation protocol. Each example consists of an RGB dermoscopic image in JPEG format and a binary segmentation mask in PNG format. The number of images in each split is summarized in Table 1.

| Dataset | Train | Validation | Test |
|---------|------:|-----------:|-----:|
| ISIC 2016 | 895 | 192 | 192 |
| ISIC 2017 | 1600 | 200 | 200 |
| ISIC 2018 | 1825 | 377 | 392 |

For ISIC 2016 and ISIC 2018 we use the released train/validation/test partitions. For ISIC 2017, the test partition available to us did not include released ground-truth masks, which prevents quantitative evaluation on that split. To obtain a properly held-out, fully annotated test set, we partitioned the 2000-image annotated ISIC 2017 training set into train, validation, and test subsets in an 80/10/10 ratio using a fixed random seed of 42, yielding the 1600/200/200 split reported above. This re-partitioning is applied only to ISIC 2017; the other two datasets are left unchanged.

All images and masks are resized to a fixed resolution of 384 × 384. Images are resized with bilinear interpolation and masks with nearest-neighbour interpolation, after which masks are binarized at a threshold of 0.5 to remove any interpolation artifacts. Image intensities are scaled to the range [0, 1].

## 4.2 Implementation Details

The framework is implemented in PyTorch 2.7.1 (CUDA 11.8) under Python 3.10, with NumPy 2.2.6 and timm 1.0.26. The spatial encoder is the frozen SAM2 Hiera-Large image encoder, loaded from its pretrained checkpoint; its parameters are not updated during training, and only the perception adaptors, the frequency-guided fusion modules, the segmentation decoder, and the auxiliary reconstruction decoder are trainable. The two-dimensional Fourier transform used by the frequency-guided fusion is computed in single precision within the otherwise mixed-precision pipeline, which avoids the numerical overflow that arises when the transform is applied to half-precision tensors.

Data augmentation is applied to the training split only and consists of horizontal and vertical flipping, rotation within [−30°, 30°], a random affine transform (translation up to 0.1, scaling in [0.9, 1.1], shear within ±5°), and colour jitter (brightness, contrast, and saturation 0.2, hue 0.1). The validation and test splits are resized without augmentation. For the SAM2-based model, the input is standardized with ImageNet channel statistics before the frozen encoder, consistent with the distribution on which the encoder was pretrained.

The baseline models are U-Net [1], Attention U-Net [3], UNet++ [13], and TransUNet [2]; the TransUNet baseline uses the hybrid ResNet-50 + ViT-B/16 encoder with ImageNet-pretrained weights. All baselines and the restored MAFF-Net are trained and evaluated under the same protocol described below so that the comparison is controlled.

## 4.3 Hardware

Training and evaluation are carried out on a single CUDA-capable NVIDIA GPU using PyTorch 2.7.1 with the CUDA 11.8 runtime and automatic mixed precision. The batch size is constrained to 2 by the memory footprint of the frozen SAM2 Hiera-Large encoder at 384 × 384 resolution. *(Specific GPU model to be confirmed by the authors.)*

## 4.4 Training Configuration

All models are optimized with AdamW using an initial learning rate of 1 × 10⁻³ and a weight decay of 0.01. The learning rate follows a linear decay schedule, scaling the initial rate from a factor of 1.0 to 0.0 over the course of training. Training uses automatic mixed precision with gradient scaling, and gradients are clipped to a maximum norm of 1.0. The batch size is 2 and the input resolution is 384 × 384. Models are trained for up to 40 epochs with early stopping based on validation Dice and a patience of 10 epochs, and the checkpoint with the highest validation Dice is retained for evaluation.

For the proposed Boundary-centric Dual Supervision, the default hyperparameters are a maximum boundary weight λ_b^max = 0.05 applied under a warmup of T_warm = 5 epochs, a reconstruction weight λ_r = 0.05, an edge-emphasis factor α = 3.0, a boundary band width of 1, the Sobel operator for boundary extraction, and deep-supervision weights w = {1, 1, 1, 0.01}. These values are used unless otherwise stated, and they are varied individually in the ablation study.

We note that this protocol is applied uniformly to every model, including the restored MAFF-Net, to keep the comparison controlled. It therefore differs from the training budget reported in the original MAFF-Net paper (a larger batch size and a longer schedule); the restored model is trained under our unified protocol rather than the original one, and this difference should be considered when relating our measurements to the originally published numbers.

## 4.5 Evaluation Metrics

Segmentation quality is measured by comparing the binarized prediction with the ground-truth mask at a threshold of 0.5. From the resulting counts of true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN), we report the Dice coefficient, the Intersection-over-Union (IoU), accuracy, sensitivity, specificity, and the F1 score:

$$\text{Dice} = \frac{2\,\text{TP}}{2\,\text{TP} + \text{FP} + \text{FN}}, \qquad \text{IoU} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}},$$

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{FP} + \text{TN} + \text{FN}}, \quad \text{Sensitivity} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}.$$

The F1 score is computed as the harmonic mean of precision and recall and coincides with the Dice coefficient in the binary case. Dice and IoU are the primary measures of overlap, while sensitivity and specificity characterize the balance between under- and over-segmentation. The effect of the proposed supervision on boundary regions specifically is examined through the controlled ablation in the next section.
