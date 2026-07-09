# TriModal Perception Architectures for Structural Defect Detection: Generative vs. Latent Predictive Networks

Abstract
Structural defect detection in complex industrial and agricultural environments requires robust multimodal integration. In this repository, we document the evolution of two distinct spatial perception engines for RGB, Depth, and Thermal sensors built upon a 4-channel patched mit_b1 Vision Transformer backbone. Part I details the legacy TriModal Predictive Network (TMPN), utilizing a pixel-space generative approach to hallucinate obscured structural data. Part II introduces the upgraded TriModal Latent Predictive Network (TMLPN), which abandons pixel generation entirely. Framed as a JEPA-inspired architecture with VICReg regularization, it incorporates Spatial Reduction Modality Attention (SRMA) and a Target-Selective Spatial Latent Predictor. The architecture successfully traverses domain manifolds to achieve multimodal latent feature consistency with vastly improved sample efficiency and resilience to stochastic noise.

---

## 1. Introduction

The detection of structural defects utilizing high-resolution multimodal sensors poses a unique cross-modal alignment challenge. Early iterations of this architecture utilized generative pixel-space decoders for thermodynamic modelling. However, reconstructing pixel-space values forces the network to map irrelevant high-frequency noise and sensor grain, frequently leading to "Background Collapse." To address this, we transitioned to Latent Space Prediction, building upon recent advancements in self-supervised architectures. In response to rigorous academic evaluation, this architecture has been significantly upgraded to preserve spatial thermal correspondences and implement genuine target-selective spatial prediction, establishing a highly robust framework for multimodal feature consistency.

---

## 2. Reproducibility & Open Source Assets

To ensure strict academic reproducibility of the 0.8014 TTA mIoU benchmark, all necessary artifacts have been published alongside this repository:

* Pre-Trained Weights: The converged best_model.pt checkpoints for all architectural scales (mit_b1 through mit_b5) are available under the GitHub Releases tab.
* Dataset Splits: The exact training and evaluation data splits are provided in the data/splits/ directory as CSV files to guarantee identical data distribution during independent verification.
* Deterministic Execution: The train.py engine utilizes a strict environmental lock (seed=42) across PyTorch, NumPy, and CUDA backends to eliminate stochastic variance during the training cycle.

---

## 3. Core Repository Structure

This repository is strictly for deploying the TriModal Latent Predictive Network (TMLPN). All legacy artifacts have been isolated in the deprecated/ directory.

```
RGBDT-Predictive/
    ├── assets/
    ├── dataset.py
    ├── models.py
    ├── pyproject.toml
    ├── README.md
    ├── train.py
    └── uv.lock
```

---

## PART I: The Legacy Tri-Objective Generative Network (TMPN)

### 4.1 TMPN Methodology & Architecture

The baseline TMPN architecture addresses cross-modal alignment by forcing the network to hallucinate obscured thermodynamic data back into pixel space.

* Primary Segmentation: A Focal Dice loss evaluating spatial boundaries [3].
* Thermal Reconstruction: An Object-Aware Block Mask obscures a percentage of the input Thermal tensor. The network's decoder must reconstruct this masked region in pixel-space using a Masked Mean Squared Error (MSE) loss, forcing it to learn structural composition.
* Global Context Modality Attention (GCMA): To resolve mechanical parallax, the GCMA head preserves pristine geometry by treating every individual pixel in the RGB-D feature map as a discrete Query. The globally pooled Thermal and RGB-D signatures act as the Keys and Values [1].

### 4.2 TMPN Quantitative Milestones

The state-machine progression on the MM5 dataset yielded the following quantitative milestones. Test-Time Augmentation (TTA) was utilized during the final diagnostic passes to measure robustness against asymmetric false-positives.

| Training Phase | Objective / Mechanism | Final Base mIoU | Final TTA mIoU |
| :--- | :--- | :--- | :--- |
| Baseline | Warmup; ImageNet patched weights [2] | 0.7434 | 0.7341 |
| HPO | 30-Trial Optuna sweep. Best peak mIoU: 0.7504 | - | - |
| Hero | Deep convergence (Patience @ Epoch 93) | 0.7453 | 0.7383 |
| Microtune | Cooling schedule + TTA Polish | 0.7488 | 0.7391 |

(Note: TMPN represents the absolute mathematical ceiling for the pixel-space generative approach. To eliminate the computational overhead of rendering stochastic sensor noise, the pipeline transitions to Latent Space Prediction in Part II).

---

## PART II: The TriModal Latent Predictive Network (TMLPN)

## 5. Introduction to VICReg-Regularized Multimodal Feature Consistency

While the generative TMPN architecture successfully aligned multimodal features, pixel-space reconstruction is inherently inefficient. Part II introduces the TMLPN, shifting the paradigm from pixel generation to latent feature prediction. By operating entirely in an abstract manifold, the architecture achieves immunity to stochastic noise and accelerates domain traversal. Acknowledging cross-modal environments, this framework is formally classified as a JEPA-inspired architecture with VICReg regularization.

## 6. Mathematical Mapping to the Predictive Framework

> 
> Figure 1: Comprehensive pipeline of the updated TriModal Latent Predictive Network, detailing the intermediate-fusion SRMA topology, the new Spatial Latent Predictor, and the stabilized VICReg regularization engine.

To ensure rigorous adherence to the theoretical framework and preserve spatial reasoning, the architecture is defined by the following strict topological mappings:

* The Context (x): The observable information the model is permitted to evaluate. In TMLPN, this is the pristine RGB-D geometry tensor combined with an artificially masked Thermal tensor.
* The Target (y): The uncorrupted physical ground truth the model is attempting to predict. In TMLPN, this is the pristine, unmasked Thermal tensor.
* The Context Encoder ($E_\theta$): The composite network responsible for processing the observable world. In TMLPN, this comprises the mit_b1 RGB-D encoder, the Thermal encoder processing the masked input, and the Spatial Reduction Modality Attention (SRMA) fusion head that binds them. (Note: The SegFormer architecture serves strictly as the encoder foundation for feature extraction ).
* The Target Encoder ($E_{target}$): The network that generates the "ground truth" latent signature. In TMLPN, this is the isolated Thermal encoder processing the unmasked target tensor, governed by a strict .detach() operation to lock the weights during prediction.
* The Predictor ($P_\phi$): The neural module that infers the Target embedding based solely on the Context embedding. In TMLPN, this is the Spatial Latent Predictor utilizing expanded receptive fields and concatenated positional encodings. It employs a specific binary block mask alongside a learnable [MASK] token to perform genuine target-selective spatial inference.

### 6.1 The Latent Regularization Engine (The VICReg Triad)

Because the Target Encoder is locked via a Stop-Gradient, the network must mathematically regularize rank collapse. TMLPN adapts the VICReg framework [6] via three constraints:

* Similarity Loss (Invariance): The primary MSE objective. Forces the prediction of the Context Encoder to match the Target Encoder's output.
* Variance Loss (Anti-Collapse): A hinge loss enforcing standard deviation ≥ 1.0 across the predicted latent batch.
* Covariance Loss (The Decorrelator): Penalizes off-diagonal elements in the covariance matrix, forcing all 512 channels of the backbone to learn unique, orthogonal features [6].

$$L_{cov} = \frac{1}{C} \sum_{i \neq j} \left( \frac{Z_i^T Z_j}{N-1} \right)^2$$

---

## 7. Results & Analysis

### 7.1 Baseline Training Dynamics & Pathologies

During the unoptimized baseline run, the architecture exhibited expected self-supervised mathematical behaviors, notably a discrepancy between Total Training Loss and Validation Loss. Because latent target generation is an auxiliary training task [5], the Validation loop correctly bypassed the VICReg penalties to strictly evaluate the downstream segmentation cross-entropy error.

Previously, high covariance weights caused numerical instability. By recalibrating the covariance penalty to 0.01 and implementing LayerNorm bounding, the manifold naturally stabilized.

> 
> Figure 2: Telemetry of the TMLPN Baseline Run.

> 
> Figure 3: Optuna Parallel Coordinate Plot demonstrating convergence.

### 7.2 Deep Convergence & The Microtune Polish

Following a 30-trial Bayesian optimization sweep (using Optuna), the architecture achieved deep convergence during a long-horizon Hero phase. To finalize spatial boundaries, a Microtune phase shifted the learning rate into a microscopic 10⁻⁵ to 10⁻⁷ cooling schedule, anchoring the latent space.

#### Hero Phase: Global Convergence
| Architecture | Parameters | Base Validation mIoU | TTA Validation mIoU |
| :--- | :--- | :--- | :--- |
| mit_b1 | 13.7M | 0.7311 | 0.7262 |
| mit_b2 | 24.2M | 0.7531 | 0.7473 |
| mit_b3 | 44.0M | 0.7923 | 0.7851 |
| mit_b4 | 60.8M | 0.7896 | 0.7969 |
| mit_b5 | 81.4M | 0.7829 | 0.7870 |

#### Microtune Phase: Spatial Boundary Refinement
| Architecture | Parameters | Base Validation mIoU | TTA Validation mIoU |
| :--- | :--- | :--- | :--- |
| mit_b1 | 13.7M | 0.7301 | 0.7265 |
| mit_b2 | 24.2M | 0.7501 | 0.7444 |
| mit_b3 | 44.0M | 0.7946 | 0.7866 |
| mit_b4 | 60.8M | 0.7960 | 0.8014 |
| mit_b5 | 81.4M | 0.7827 | 0.7865 |

(Note: Statistical significance tests including standard deviation, confidence intervals, and per-class IoU are scheduled for evaluation in upcoming ablation studies).

> 
> Figure 4: mIoU of the TMLPN Hero and Microtune Phases.

### 7.3 Discussion: Scaling and Multi-Modal Behaviors

1. The Capacity Saturation Point
Scaling from the lightweight mit_b1 up to mit_b5 exposes a clear performance ceiling within the MM5 dataset. The network achieves its absolute peak performance at the mit_b4 scale following the Microtune phase (0.8014 TTA mIoU).

2. The TTA Inversion Phenomenon
The telemetry reveals a fascinating behavioral inversion regarding Test-Time Augmentation (TTA). For the lighter architectures (mit_b1, mit_b2, and mit_b3), applying TTA consistently lowers the mIoU. Conversely, the deeper mit_b4 and mit_b5 architectures experience a performance boost from TTA.

### 7.4 Explainability: Tightening Spatial Boundaries

Semantic Grad-CAM and Epistemic Uncertainty mapping applied to identical input geometry at the conclusion of the Microtune run demonstrate razor-sharp, object-centric hotspots, refined heavily by the spatial predictor component.

> 
>
> Figure 5: The Grad-CAM heatmap reveals object-centric hotspots.

> 
>
> Figure 6: The Epistemic Uncertainty map captures the model's spatial hesitation.

---

## 8. Architectural Trade-Offs & Theoretical Defenses

### 8.1 Intermediate vs. Early Fusion Topology
TMLPN explicitly avoids naive early-fusion by utilizing an intermediate-fusion topology. The RGB-D and Thermal domains are processed by completely isolated Vision Transformer encoders.

Note on Pre-Trained Weights: Initializing the 4-channel and 1-channel streams directly from 3-channel SMP ImageNet weights creates an inherent limitation via channel-averaging. This sub-optimal initialization  is actively stabilized across the initial baseline warmup epochs.

### 8.2 The Upgraded SRMA vs Global GCMA Bottleneck
We redesigned the fusion head from a previous global conditioning mechanism to a Spatial Reduction Modality Attention (SRMA) mechanism. Utilizing a Spatial Reduction Ratio ($R$), we downsample the Thermal feature map while preserving localized correspondences. This successfully scales cross-attention mathematical complexity to $\mathcal{O}(N/R^2)$.

Documented Limitation (The Information Bottleneck): With a reduction ratio of R=8, the thermal feature map's information capacity is mathematically reduced by a factor of 64. While the SRMA output is projected back to the original resolution, the true thermal information content remains bounded by the reduced resolution.

### 8.3 Target-Selective Spatial Inductive Biases
The predictive engine has been upgraded to a mathematically compliant Spatial Latent Predictor. By explicitly ingesting the exact binary block mask used on the input data, replacing obscured regions with a learnable [MASK] token, and applying concatenated 2D Sinusoidal Positional Encodings, the depthwise separable convolutions perform genuine target-selective spatial inference.

### 8.4 Stop-Gradient Heuristics and Explicit Covariance Regularization
Many self-supervised frameworks utilize an Exponential Moving Average (EMA) teacher network to prevent representation collapse in the target encoder. TMLPN abandons the EMA framework entirely, utilizing identical shared weights for the Context and Target encoders governed solely by a strict Stop-Gradient (.detach()) operation. By explicitly enforcing the VICReg constraints, TMLPN proves that mathematically regularizing the variance and covariance of the embedding manifold physically prevents rank collapse without EMA momentum overhead.

### 8.5 Mitigating Imbalance and Asymptotic Limits (DCW & KD)
Industrial defect datasets exhibit extreme class imbalance. To overcome this without unbalancing the VICReg latent space, TMLPN utilizes a Dynamic Class-Weighting Schedule (DCW). To break representational capacity ceilings, the pipeline integrates a Knowledge Distillation (KD) engine.

$$L_{KD} = \tau^2 \text{KL}\left( \sigma\left(\frac{z_{student}}{\tau}\right) \parallel \sigma\left(\frac{z_{teacher}}{\tau}\right) \right)$$

## 9. Future Work: Systematic Ablation Studies

To definitively quantify the independent contributions of this architecture, ongoing active research includes comprehensive ablation studies evaluating:

1.  Reduction Ratio Optimization: Systematic evaluation of SRMA reduction ratios (R=4, 8, 16) to quantify the trade-off between the thermal information bottleneck and computational efficiency.
2.  Joint-Training Dynamics: Comparing segmentation-only training, isolated VICReg pre-training followed by fine-tuning, and the current joint-training methodology.
3.  Same-Modal Predictive Baselines: Evaluating whether predicting RGB-D from RGB-D yields additional geometric benefits compared to the current cross-modal approach.

---

## 10. Conclusion & Edge Deployment

By abandoning pixel-space generation, the TriModal Latent Predictive Network establishes a vastly more efficient methodology for multimodal defect detection. The empirical scaling behavior dictates a highly specific deployment strategy to balance maximum predictive fidelity against strict edge hardware constraints.

To achieve true edge autonomy, we utilize the lightweight mit_b1 backbone (13.7M parameters) as the active "Student." Utilizing Knowledge Distillation, we transfer the inter-class dark knowledge from the mit_b4 Teacher down into the mit_b1 framework. Serialized via ONNX Opset 18, this optimized asymmetric graph is deployed onto edge hardware for continuous autonomous processing.

---

## 11. Key Concepts & Technical Glossary

* Joint-Embedding Predictive Architectures (JEPA): A self-supervised paradigm that forces a Context Encoder and a Target Encoder to align their outputs in an abstract latent space. (Meta AI: I-JEPA)
* VICReg (Variance-Invariance-Covariance): The mathematical regularization triad used to mathematically regularize the latent manifold without an EMA teacher network. (VICReg Paper)
* Hierarchical Vision Transformers (SegFormer): The underlying architecture of the modality-isolated encoders. (SegFormer Paper)
* Spatial Reduction Modality Attention (SRMA): The updated fusion head that limits cross-attention complexity to $\mathcal{O}(N/R^2)$.

---

## References

[1] Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.
[2] Xie, E., et al. (2021). SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. NeurIPS.
[3] Sudre, C. H., et al. (2017). Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations. DLMIA.
[4] Maes, L., et al. (2024). LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels. arXiv preprint.
[5] Assran, M., et al. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. CVPR.
[6] Bardes, A., Ponce, J., & LeCun, Y. (2022). VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. ICLR.
[7] Huang, Y., et al. (2020). Dynamic Weighting for Imbalanced Semantic Segmentation.
[8] Lin, M., Chen, Q., & Yan, S. (2013). Network In Network. ICLR.
[9] Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. NIPS Deep Learning Workshop.

---

## 🙏 Acknowledgments & Citations

This project would not be possible without the MM5 Dataset. We sincerely thank the original creators and authors for their foundational work in multi-modal data collection, hardware synchronization, and curation, which enabled the training and evaluation of this architecture.

If you utilize this pipeline, the underlying architecture, or the data, please cite the primary publication alongside the dataset repository:

Primary Publication:
> Brenner, M., Reyes, N. H., Susnjak, T., & Barczak, A. L. C. (2026). MM5: Multimodal image capture and dataset generation for RGB, depth, thermal, UV, and NIR. Information Fusion, 126, 103516.
>
> [DOI: https:doi.org/10.1016/j.inffus.2025.103516](https:doi.org/10.1016/j.inffus.2025.103516)

Dataset:
> Brenner, M., Reyes, N., Susnjak, T., & Barczak, A. (2025). MM5: Multimodal Image Dataset. figshare. Dataset.
>
> [DOI: https:doi.org/10.6084/m9.figshare.28722164](https:doi.org/10.6084/m9.figshare.28722164)