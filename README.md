# TriModal Perception Architectures for Structural Defect Detection: The MM-JEPA Paradigm

Abstract
Structural defect detection in industrial and agricultural environments requires robust multimodal integration. In this repository, we document the evolution of spatial perception engines for RGB, Depth, and Thermal sensors built upon a patched `mit_b1` Vision Transformer backbone. Recognizing the theoretical limitations of pixel-space generative networks, this architecture introduces the Multimodal Joint-Embedding Predictive Architecture (MM-JEPA). By decoupling representation learning from downstream segmentation, the framework utilizes pure self-supervised prediction in the latent space to build a robust structural foundation model. This allows for rapid supervised fine-tuning and deployment on edge computational devices.

---

## 1. Introduction

The integration of high-resolution, unaligned multimodal sensors poses a unique challenge. Early iterations of this architecture utilized generative pixel-space decoders. However, reconstructing pixel-space values forces the network to map irrelevant high-frequency noise. To address this, we transitioned to Latent Space Prediction, building upon recent advancements in self-supervised architectures (e.g., Meta AI's I-JEPA). This repository completely deprecates the legacy generative network in favor of a 2-Phase MM-JEPA paradigm, executing genuine target-selective spatial inference via mathematically rigorous spatial constraints.

---

## 2. Reproducibility & Open Source Assets

To ensure strict academic reproducibility of our evaluation benchmarks, all necessary artifacts will be published alongside this repository following the conclusion of the training cycle:

* **Pre-Trained Weights:** The converged `best_model.pt` checkpoints for all architectural scales will be made available under the GitHub Releases tab.
* **Dataset Splits:** The exact training and evaluation data splits are provided in the `data/splits/` directory as CSV files to guarantee identical data distribution.
* **Deterministic Execution:** The execution engines utilize strict environmental locking (`seed=42`) across PyTorch, NumPy, and CUDA backends to eliminate stochastic variance.

---

## 3. Core Repository Structure
```
RGBDT-Predictive/
├── assets/
├── dataset_jepa.py         # Unified 5-Channel dataloaders for pre-training and finetuning.
├── models.py               # MM-JEPA architecture definitions.
├── pyproject.toml
├── README.md
├── pretrain_jepa.py        # Phase 1: Self-Supervised execution loop.
├── train_downstream.py     # Phase 2: Supervised fine-tuning execution loop.
└── uv.lock
```

---

## 4. Phase 1: Self-Supervised MM-JEPA Pre-Training

To build a true foundation model of the physical environment, the network must decouple feature extraction from human-annotated labels. Phase 1 achieves this through pure self-supervised spatial inference.

### 4.1 The MM-JEPA Topology

To satisfy the theoretical mandates of the Joint-Embedding Predictive Architecture, the network executes the following mathematical constraints:

* **Unified Same-Modal Inference:** The Context and Target Encoders operate on identical architectural modalities. The sensor streams are fused into a unified 5-channel block. The network is forced to predict masked properties from the *same* multimodal manifold.
* **Multi-Block Masking Strategy:** Standard JEPA avoids single-block occlusion. The architecture samples 4 independent overlapping blocks with varying scales (0.15–0.20) and aspect ratios (0.75–1.5) to force multi-scale semantic reasoning.
* **Target-Conditioned Spatial Predictor:** The Predictor module is explicitly conditioned on the target region. By concatenating the contextual feature map, the target mask, and pure 2D Positional Encodings, the CNN explicitly knows *where* to predict without relying on unconstrained additive degradation.
  * *Architectural Defense of the CNN Predictor:* Standard implementations of I-JEPA utilize isotropic Vision Transformers (e.g., ViT-Huge) and sequence-based Transformer predictors. However, flattening high-resolution 2D spatial maps into a 1D sequence introduces an $\mathcal{O}(N^2)$ self-attention bottleneck that is mathematically incompatible with bounded edge-hardware deployment. Because TMLPN utilizes a hierarchical encoder (`mit_b1`), which preserves the 2D grid structure essential for downstream dense semantic segmentation, the predictor must operate natively in 2D space. Aligning with recent literature on hierarchical masked modeling (e.g., ConvNeXt-V2, CNN-JEPA), TMLPN employs a fully convolutional predictor utilizing Token Replacement to perform genuine, target-selective spatial inference.
* **EMA Momentum Teacher:** The Target Encoder is governed strictly by an Exponential Moving Average (EMA) update schedule, maintaining a smooth, historically stable target manifold to prevent representation collapse.

$$\theta_{target} \leftarrow \tau \theta_{target} + (1 - \tau) \theta_{context}$$

* **EMA Cosine Annealing and Target Isolation:**Early training epochs produce highly volatile context embeddings. Utilizing a static Exponential Moving Average (EMA) schedule (e.g., $\tau = 0.996$) forces the target network to absorb this chaotic initialization, risking representational collapse. TMLPN mitigates this via a Cosine Annealing Schedule, where the momentum parameter dynamically asymptotes from $0.996$ to $1.0$ over the total training steps, allowing the target manifold to stabilize as training concludes.

$$\tau_i = 1 - (1 - \tau_{base}) \frac{\cos(\pi \cdot i / N) + 1}{2}$$

Furthermore, to prevent gradient leakage from external operational hooks or distributed training wrappers, the target network's outputs are explicitly severed from the computational graph via `.detach()`, providing a strict mathematical guarantee of isolation beyond standard `.eval()` and `no_grad()` contexts.

### 4.2 The Latent Predictive Objective & Variance Regularization

To optimize throughput for edge inference, TMLPN utilizes a dense CNN predictor that processes the entire spatial map. The resulting feature manifolds are optimized via a unified, three-part objective function designed to ensure gradient stability and prevent representational collapse:

**1. $L_2$ Normalized Target Inference:** To prevent unbound gradient scaling as latent dimensions increase, both the predicted and target features are $L_2$-normalized along the channel dimension before MSE calculation. This projects the continuous multimodal representations onto a unit hypersphere, guaranteeing that the spatial prediction loss strictly measures cosine-equivalent directional alignment rather than arbitrary magnitude scaling.

**2. Context Consistency:**
A down-weighted auxiliary objective evaluated on the visible (unmasked) coordinates to explicitly enforce representational consistency across the spatial grid, neutralizing the representational drift inherent in dense convolutional prediction.

**3. Variance Hinge Regularization:**
While standard I-JEPA relies solely on EMA momentum to prevent representation collapse, high-dimensional unaligned multimodal data is highly susceptible to dimensional shrinkage. Rather than utilizing computationally hostile covariance mechanisms (e.g., SIGReg) which violate edge-hardware constraints, TMLPN integrates a lightweight Variance Hinge Penalty (adapted from VICReg). By explicitly penalizing the network if the channel-wise standard deviation $\sigma$ falls below $1.0$, the architecture mathematically guarantees a diverse, non-collapsed embedding space with minimal computational overhead.

$$\mathcal{L}_{Total} = \mathcal{L}_{Target} + \alpha \mathcal{L}_{Context} + \lambda \mathcal{L}_{Var}$$

$$\mathcal{L}_{Var} = \frac{1}{C} \sum_{c=1}^{C} \max(0, 1 - \sqrt{\text{Var}(z_{target}^{(c)}) + \epsilon})$$

*(Where $\alpha = 0.1$ and $\lambda = 0.1$ balance consistency and spatial diversity without overpowering the primary target inference).*

---

## 5. Phase 2: Supervised Semantic Fine-Tuning

Following deep convergence in Phase 1, the Context Encoder possesses a pre-trained understanding of structural defects and thermodynamic boundaries—learned entirely without human annotation. Phase 2 transitions to the task of semantic segmentation.

### 5.1 Downstream Architecture & Multi-Scale Fine-Tuning

The pre-trained weights from Phase 1 (`jepa_context_encoder.pt`) are injected into the downstream network. The encoder backbone is temporarily frozen, and a decoding head is attached to process the downstream semantic annotations using an optimized Focal Dice objective.

**Modality-Specific Tokenizers:** Initializing a multi-channel stream directly from 3-channel ImageNet weights inherently creates representational interference. To mitigate this, TMLPN physically isolates modality ingestion at the stem. The network utilizes a custom `ModalityIsolatedPatchEmbed` module: the RGB stream inherits pristine, unmodified 3-channel ImageNet kernels, while the Depth and Thermal streams are processed by independent, Kaiming-initialized convolutional filters. These features are fused strictly via summation within the latent embedding dimension, completely eliminating low-level kernel corruption.

**The Multi-Scale All-MLP Decoder:** Relying strictly on the deepest latent feature map (e.g., $1/32$ resolution) destroys high-frequency spatial details crucial for microscopic defect segmentation. However, utilizing heavy transposed convolution decoders (e.g., U-Net topologies) violates edge-hardware latency constraints. To synthesize sub-pixel spatial boundaries with deep thermodynamic semantics, the downstream architecture adopts a Multi-Scale All-MLP Decoder. By projecting the $1/4$, $1/8$, $1/16$, and $1/32$ hierarchical feature grids to a unified embedding dimension, upsampling them to a common $1/4$ resolution, and concatenating them, the network achieves razor-sharp boundary delineation while maintaining a strict, edge-compliant computational footprint.

### 5.2 Evaluation Benchmarks [PENDING]

The downstream execution engine (`train_downstream.py`) utilizes a multi-phase state machine governed by Bayesian optimization (Optuna).

*Note: The architecture is currently undergoing empirical evaluation across the `mit` Vision Transformer series (`mit_b1` through `mit_b5`). Quantitative milestones, including Base Validation mIoU, Test-Time Augmentation (TTA) robustness, and Expected Calibration Error (ECE), will be populated upon the conclusion of the training cycle.*

### 5.3 Mitigating Imbalance and Asymptotic Limits ($\alpha$-Balancing & GDL)

Industrial defect datasets exhibit extreme foreground-background class imbalance. To overcome this without resorting to static heuristics or unvalidated inter-epoch validation shifting, the downstream engine utilizes a bipartite loss objective rooted in formal imbalance mitigation literature:

**1. $\alpha$-Balanced Focal Loss:**
To mitigate the dominance of the background class, the primary segmentation objective is governed by the $\alpha$-balanced variant of Focal Loss (Lin et al., 2017). The parameter $\alpha$ explicitly weights classes according to their empirical dataset frequencies (computed via Median Frequency Balancing), ensuring mathematical suppression of the background without arbitrary tuning.

$$\mathcal{L}_{Focal} = - \alpha_t (1 - p_t)^\gamma \log(p_t)$$

**2. Generalized Dice Loss (GDL):**
To address intra-batch scale variance between large structural components and microscopic defects, TMLPN utilizes Generalized Dice Loss (Sudre et al., 2017). GDL computes dynamic weights strictly within the forward pass by scaling each class's intersection by the inverse square of its volume ($w_c = 1 / (\sum g_{nc})^2$). This mathematically guarantees that microscopic defects receive massive gradient scaling naturally, eliminating the requirement for unvalidated momentum heuristics or temperature parameters, and strictly preserving the Markovian property of Stochastic Gradient Descent.

$$\mathcal{L}_{GDL} = 1 - 2 \frac{\sum_{c=1}^C w_c \sum_{n=1}^N p_{nc} g_{nc}}{\sum_{c=1}^C w_c \sum_{n=1}^N (p_{nc} + g_{nc})}$$

To break representational capacity ceilings during edge deployment, the pipeline integrates a Knowledge Distillation (KD) engine. By forcing the lightweight `mit_b1` Student to minimize the Kullback-Leibler (KL) Divergence against a massive Teacher's soft probabilities ("Dark Knowledge"), the edge-deployed model inherits advanced stochastic noise suppression.

**2. Dynamic Class-Weighting (DCW) as Class-Level OHEM:**
While Focal Loss targets pixel-level hesitation, TMLPN targets class-level failure modes using a continuous Dynamic Class-Weighting (DCW) schedule (Huang et al., 2020). Operating as a differentiable, class-level analog to Online Hard Example Mining (OHEM) (Shrivastava et al., 2016), DCW tracks an Exponential Moving Average of the validation IoU. The downstream Dice penalty is exponentially scaled on the fly via $W_c = \exp(\tau \cdot (1 - \text{IoU}_c))$. This applies a smooth, non-linear amplification to lagging minority classes, prioritizing convergence on complex structural defects without inducing the gradient shocks common to discrete OHEM step-functions.

Algorithmic Stability in Bounded Edge-Hardware:Operating under strict memory constraints necessitates small batch sizes. This invalidates standard batch-wise Generalized Dice Loss (GDL), as structural defect classes are frequently absent from individual iterations, causing destructive gradient sparsity. TMLPN mitigates this via Global Volume Anchoring. Rather than scaling intersections dynamically per batch, GDL weights are anchored to the inverse square of the global dataset frequencies.Furthermore, arbitrary bounds on class-imbalance modifiers are replaced with Additive Laplace Smoothing. By injecting a uniform pseudo-count $\epsilon$ across the active pixel distribution prior to Median Frequency Balancing, theoretical limits are naturally bounded by the dataset's native volume. This provides mathematically rigorous suppression of the background class without resorting to unjustified heuristic clipping, while explicit tensor clamping within the Focal computation guarantees absolute safety against floating-point overflow during unaligned gradient shocks.

---

## 6. Edge Deployment (Jetson Orin Nano)

The ultimate objective of the MM-JEPA paradigm is real-time autonomous inspection.

The isolated, finetuned architecture is optimized and serialized to an ONNX artifact (opset_version=18) natively through the execution engine. By deploying this distilled TensorRT engine onto a Jetson Orin Nano functioning as a companion computer aboard a UAV or inspection rover, the theoretical aim of the system is to achieve sub-pixel structural segmentation and real-time autonomous predictions directly at the sensor source.

---

## 7. Key Concepts & Technical Glossary

* **Joint-Embedding Predictive Architectures (JEPA):** A self-supervised paradigm that forces a Context Encoder and a Target Encoder to align their outputs in an abstract latent space via target-selective spatial prediction. (Meta AI: I-JEPA)
* **Exponential Moving Average (EMA):** The momentum-based mathematical update schedule used to stabilize the Target Encoder weights during Phase 1 pre-training, preventing representational collapse.
* **Hierarchical Vision Transformers (SegFormer):** The underlying architecture of the Context Encoder, utilizing an overlap-patching mechanism to process high-resolution geometry without losing 2D grid structure. (SegFormer Paper)
* **Knowledge Distillation (KL Divergence):** The model compression strategy used to transfer the complex inter-class similarities of a massive workstation model into a lightweight edge-deployable footprint. (Distilling Knowledge)
* **Test-Time Augmentation (TTA) Uncertainty:** The mathematical evaluation of spatial hesitation and out-of-distribution (OOD) anomalies by measuring prediction variance across augmented geometric orientations.

---

## References

[1] Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.

[2] Xie, E., et al. (2021). SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. NeurIPS.

[3] Sudre, C. H., et al. (2017). Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations. DLMIA.

[4] Assran, M., et al. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. CVPR.

[5] Huang, Y., et al. (2020). Dynamic Weighting for Imbalanced Semantic Segmentation.

[6] Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. NIPS Deep Learning Workshop.

---

## 🙏 Acknowledgments & Citations

This project would not be possible without the MM5 Dataset. We sincerely thank the original creators and authors for their foundational work in multi-modal data collection, hardware synchronization, and curation, which enabled the training and evaluation of this architecture.

If you utilize this pipeline, the underlying architecture, or the data, please cite the primary publication alongside the dataset repository:

**Primary Publication:**
> Brenner, M., Reyes, N. H., Susnjak, T., & Barczak, A. L. C. (2026). MM5: Multimodal image capture and dataset generation for RGB, depth, thermal, UV, and NIR. Information Fusion, 126, 103516.
>
> [DOI: https:doi.org/10.1016/j.inffus.2025.103516](https:www.google.com/search?q=https:doi.org/10.1016/j.inffus.2025.103516)

**Dataset:**
> Brenner, M., Reyes, N., Susnjak, T., & Barczak, A. (2025). MM5: Multimodal Image Dataset. figshare. Dataset.
>
> [DOI: https:doi.org/10.6084/m9.figshare.28722164](https:www.google.com/search?q=https:doi.org/10.6084/m9.figshare.28722164)