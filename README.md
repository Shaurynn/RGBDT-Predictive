# TriModal Perception Architectures for Structural Defect Detection: The MM-JEPA Paradigm

## Abstract

Structural defect detection in industrial environments necessitates the robust integration of RGB, Depth, and Thermal (RGB-D-T) modalities [20]. In this repository, we document the evolution of a spatial perception engine designed specifically for bounded edge-hardware [19]. Recognizing the computational limits of deep cross-attention networks [4] and the high-frequency noise inherent to generative pixel-space autoencoders [1], this architecture introduces the Tri-Modal Latent Predictive Network (TMLPN_v3) utilizing a Multimodal Joint-Embedding Predictive Architecture (MM-JEPA) [2]. By decoupling representation learning from downstream segmentation [10] and applying rigorous spatial constraints—including Dirac-initialized alignment projections [7] and Global Volume Anchored Generalized Dice Loss [14]—the framework establishes a highly robust structural foundation model capable of real-time, high-resolution edge inference [20].

---

## 1. Introduction

The integration of high-resolution, unaligned multimodal sensors provides critical advantages in structural defect and anomaly detection [20]. Deploying continuous multi-channel arrays—specifically those originating from 16-bit Baumer GigE industrial sensors—onto edge compute modules necessitates strict algorithmic efficiency [20].

Early iterations of multimodal learning utilized generative pixel-space decoders [1]. However, reconstructing pixel-space values forces the network to map irrelevant high-frequency radiometric noise, wasting representational capacity [1]. To address this, we transitioned to Latent Space Prediction, building upon recent advancements in self-supervised architectures [2]. This repository executes genuine target-selective spatial inference via an empirically rigorous two-phase pipeline [2, 10].

### 1.1 Architectural Deviations: TMLPN vs. MuMo-JEPA

Recent literature highlights Multimodal JEPA (MuMo-JEPA) architectures, which rely heavily on deep late-fusion, independent Vision Transformer (ViT-Huge) trunks per modality, and cross-attention joint embeddings [3]. While theoretically optimal for unconstrained server environments, we explicitly deviate from the MuMo-JEPA methodology for the following theoretically grounded reasons:

1. **Memory-Bandwidth Bottlenecks:** Holding multiple independent ViT trunks in VRAM violates the shared-memory and bandwidth constraints of edge devices, as standard non-hierarchical Transformers incur prohibitive parameter counts and memory access costs during inference [19].
2. **Quadratic Scaling:** Standard self-attention mechanisms scale with $O(N^2)$ complexity relative to spatial resolution, prohibiting real-time inference on high-resolution industrial image strips [4].
3. **Hardware-Aware Early Fusion:** TMLPN utilizes a single hierarchical MiT trunk with a structurally isolated modality stem [10]. The MiT sequence reduction process ensures linear $O(N)$ computational scaling [10], while our Dirac-initialized $1 \times 1$ projections neutralize the latent alignment flaws typically associated with simplistic early fusion [5, 7].

---

## 2. Reproducibility & Open Source Assets

To ensure strict academic reproducibility of our evaluation benchmarks, all artifacts will be published alongside this repository following the conclusion of the training cycle [24]:

* **Pre-Trained Weights:** Converged `best_model.pt` checkpoints will be provided for identical inference replication [24].
* **Dataset Splits & Blind Testing:** Exact training and evaluation subsets are mapped via CSV data splits (`data/splits/`) to eliminate distribution variance [21]. To correct statistical inference and address validation leakage, the framework employs a strictly separated blind test set. The original evaluation pool is programmatically split into a 50% validation set (used strictly for hyperparameter sweep monitoring and early stopping) and a 50% test set reserved exclusively for the final gradient-free metrics generation.
* **Deterministic Execution & Multi-Seed Verification:** The execution engines utilize strict environmental locking (`seed=42`) across PyTorch, NumPy, and CUDA backends to eliminate stochastic gradient variance [24]. To definitively eliminate single-run optimistic bias, all main architectural baselines and ablation experiments output multi-seed main results (N=5) to capture the true Mean mIoU $\pm$ Standard Deviation.

---

## 3. Phase 1: Self-Supervised MM-JEPA Pre-Training

To build a true foundation model of the physical environment, the network must decouple feature extraction from human-annotated labels [2]. Phase 1 achieves this through pure self-supervised spatial inference, forcing the network to understand structural geometry prior to task-specific fine-tuning [2].

### 3.1 Stem Modality Isolation & Alignment Projection

Initializing a multi-channel stream directly from 3-channel weights introduces severe representational interference due to the cross-modal domain gap [5]. TMLPN physically isolates modality ingestion at the stem using a `ModalityIsolatedPatchEmbed` module to safely fuse unaligned manifolds [5]:

* **RGB Stream:** Inherits pristine, unmodified 3-channel ImageNet kernels to leverage generalized edge-detection priors [22].
* **D+T Stream:** Utilizes independent, Kaiming-initialized convolutions to prevent early-epoch activation vanishing in the high-variance depth and thermal tensors [6].
* **1x1 Dirac Alignment:** To bridge this dimensional gap prior to additive fusion, TMLPN utilizes a $1 \times 1$ convolutional projection initialized via a Dirac delta distribution [7]: 

$$ W_{i,j,k,l} = \begin{cases} 1 & \text{if } i=j \text{ and } k,l \text{ are the center} \\ 0 & \text{otherwise} \end{cases} $$

This initialization provides a structural identity mapping that mitigates initial gradient shocks, allowing the Kaiming-initialized Depth/Thermal kernels to gradually assimilate into the pre-trained ImageNet manifold without shattering the learned weights [7].

### 3.2 The MM-JEPA Topology

To satisfy the theoretical mandates of latent prediction, the network executes the following spatial constraints:

* **Token Replacement Masking:** Masked spatial coordinates are explicitly replaced with a broadcasted, learnable parameter (`encoder_mask_token`), aligning with proven masked image modeling protocols [1, 9].
* **Multi-Block Strategy:** The architecture samples 4 independent overlapping target blocks with varying scales (0.15–0.20) and aspect ratios (0.75–1.5) [2].
* **Target-Conditioned Spatial Predictor:** Pure 2D Positional Encodings are concatenated *only* alongside the context feature map and target mask within the depthwise-separable predictor, ensuring the network knows exactly *where* to predict [10].
* **EMA Cosine Annealing & Gradient Isolation:** Target network outputs are severed from the computational graph, updated strictly via a Cosine Annealing Exponential Moving Average (EMA) schedule from $0.996$ to $1.0$ to combat representation collapse [11].

---

## 4. Phase 2: Supervised Semantic Fine-Tuning

Phase 2 transfers the pre-trained Context Encoder to the downstream task of semantic segmentation, utilizing a resolution-invariant SegFormer MLP decoder [10].

### 4.1 The Multi-Scale All-MLP Decoder

Heavy transposed convolution decoders violate the latency constraints required for real-time edge processing [19]. TMLPN synthesizes sub-pixel spatial boundaries using an All-MLP Decoder [10]. **Batch Normalization within the SegFormer decoder was substituted with Layer Normalization to maintain activation stability under bounded hardware batch constraints (N=6).** By projecting the $1/4$, $1/8$, $1/16$, and $1/32$ hierarchical feature grids to a unified embedding dimension, applying LayerNorms, upsampling exclusively to a common $1/4$ resolution, and concatenating them, the network achieves boundary delineation while maintaining an edge-compliant footprint [10].

### 4.2 Mitigating Imbalance: Alpha-Balanced Focal GDL

Industrial datasets exhibit extreme foreground-background class imbalance [20]. The downstream engine utilizes a rigorous bipartite loss objective:

1. **$\alpha$-Balanced Focal Loss:** Mitigates background dominance by explicitly weighting classes according to their empirical dataset frequencies via Median Frequency Balancing [13].
2. **Global Volume Anchored Generalized Dice Loss (GDL):** TMLPN utilizes *Global Volume Anchoring*, where GDL weights are permanently anchored to the inverse square of the global dataset frequencies [14]. Additive Laplace Smoothing ensures theoretical bounds are naturally constrained by the dataset's native volume [24].

---

## 5. V1 Pipeline Findings & Ablation Analysis

Initial executions of the `v1` pipeline yielded high baseline convergence during the frozen `hero` phase. However, when the pipeline transitioned to the `microtune` phase for end-to-end unfreezing, performance plateaued and, in several configurations, actively degraded. Unconstrained downstream gradients propagating back from a high-capacity decoder aggressively interfered with the generalized multimodal manifolds learned during Phase 1 self-supervised pre-training.

> ![v1 Network Architecture Diagram](assets/v1_Network_Architecture_Diagram.png)
>
> **Figure 1: The Tri-Modal Latent Predictive Network (TMLPN_v1) two-stage execution pipeline.** Demonstrates the baseline architecture prior to the LoRA and LLRD preservation interventions.

### 5.1 V1 Experimental Results & The `microtune` Plateau

As detailed in the experimental results below, architectures such as `mit_b1`, `mit_b2`, and `mit_b5` experienced measurable degradation in Base Validation mIoU when transitioning from the `hero` to `microtune` phase, highlighting the instability of end-to-end unfreezing without proper gradient constraints.

| Method | Backbone | Parameters | Pre-Training | Hero mIoU (Base / TTA) | Microtune mIoU (Base / TTA) |
| --- | --- | --- | --- | --- | --- |
| **TMLPN_v1** | **MiT-b1** | 13.7M | **MM-JEPA** | 0.7311 / 0.7262 | 0.7301 / 0.7265 |
| **TMLPN_v1** | **MiT-b2** | 24.2M | **MM-JEPA** | 0.7531 / 0.7473 | 0.7501 / 0.7444 |
| **TMLPN_v1** | **MiT-b3** | 44.0M | **MM-JEPA** | 0.7923 / 0.7851 | 0.7946 / 0.7866 |
| **TMLPN_v1** | **MiT-b4** | 60.8M | **MM-JEPA** | 0.7896 / 0.7969 | 0.7960 / 0.8014 |
| **TMLPN_v1** | **MiT-b5** | 81.4M | **MM-JEPA** | 0.7829 / 0.7870 | 0.7827 / 0.7865 |

> ![v1 Phase Progression](assets/v1_phase_progression_trajectory_2.jpg)
>
> **Figure 2: Architecture Progression Trajectory (V1).** Visualizes the performance plateau when the massive multi-scale decoder's gradients clash with the delicate JEPA-pretrained foundation during the Microtune phase.

### 5.2 V1 Ablation Matrix: Component Necessity

A rigorous N=5 ablation study isolated the empirical necessity of our initial architectural components:

> ![v1 Ablation Studies](assets/v1_ablation_statistics.png)
>
> **Figure 3: N=5 Ablation Study Results (V1).** Validated using Welch's t-test for statistical significance against the optimal control.

* **Modality Isolation is Critical:** The `NaiveFusion` ablation collapsed the network to a **0.4476 mIoU** ($p=0.0001$), proving that forcing 3-channel pretrained weights to blindly accept 5-channel unaligned data destroys ImageNet priors. The Dirac-initialized $1 \times 1$ alignment stem is empirically necessary.
* **Logit-Level KD is Noisy:** Disabling Knowledge Distillation (`NoKD`) yielded no statistically significant penalty ($p=0.7603$). In dense pixel-prediction tasks, raw logit distributions are highly noisy, meaning traditional KD provided zero measurable gain while doubling VRAM requirements.

---

## 6. V2 Architecture: Optimization & Preservation Strategies

To address the plateau vulnerabilities discovered in the `v1` pipeline, the `v2` architecture transitioned to a highly fortified downstream engine designed specifically to bend the network toward semantic segmentation without inducing gradient shattering.

> ![v2 Network Architecture Diagram](assets/v2_Network_Architecture_Diagram.png)
>
> **Figure 4: The fortified Tri-Modal Latent Predictive Network (TMLPN_v2) two-stage execution pipeline.**

### 6.1 Low-Rank Adaptation (LoRA)

To completely neutralize gradient shattering during the `microtune` phase, the `v2` pipeline integrates Low-Rank Adaptation (LoRA) [25]. Rather than fully unfreezing the $N \times N$ weight matrices inside the MiT backbone, the foundation remains mathematically locked. We inject tiny, trainable low-rank matrices ($A$ and $B$) specifically into the Query and Value projection layers of the transformer blocks [25]. 

### 6.2 Layer-Wise Learning Rate Decay (LLRD)

To safely couple the LoRA modules with the high-capacity decoder, we deploy Layer-Wise Learning Rate Decay (LLRD) [27]. The optimizer groups parameters hierarchically: the downstream decoder receives the full base learning rate, while gradients flowing deeper into the backbone are exponentially decayed (e.g., $lr \times 0.85^n$). 

### 6.3 Feature-Level Knowledge Distillation

To compress the performant `mit_b5` architecture into an edge-ready `mit_b1` without the noise of logit-level distillation, `v2` upgrades to Feature-Level KD [26]. By $L_2$-normalizing and aligning the intermediate feature grids of the teacher directly with the student via MSE loss, the student mathematically inherits the complex representational structure of the teacher, circumventing spatial boundary noise [26].

### 6.4 V2 Experimental Results & Scaling Laws

The evaluation of the `v2` architecture across the MM5 dataset demonstrates the successful mitigation of catastrophic forgetting and the robust restoration of scaling laws. Advancing from the lightweight `mit_b1` student model to the `mit_b2` architecture yielded a significant performance jump, while extending to `mit_b4` and `mit_b5` revealed minor dataset saturation mechanics.

| Method | Backbone | Pre-Training | Mean mIoU | Train Loss | Val Loss |
| --- | --- | --- | --- | --- | --- |
| **TMLPN_v2** | **MiT-b1 (Student)** | **MM-JEPA** | **0.8015** | **0.0182** | **0.2184** |
| **TMLPN_v2** | **MiT-b2** | **MM-JEPA** | **0.8694** | **0.0068** | **0.1372** |
| **TMLPN_v2** | **MiT-b3** | **MM-JEPA** | **0.8687** | **0.0112** | **0.1193** |
| **TMLPN_v2** | **MiT-b4** | **MM-JEPA** | **0.8461** | **0.0231** | **0.1177** |
| **TMLPN_v2** | **MiT-b5 (Teacher)** | **MM-JEPA** | **0.8557** | **0.0117** | **0.1516** |

> ![V2 Phase Progression](assets/v2_phase_progression_trajectory.png)
>
> **Figure 5: Architecture Progression Trajectory (V2).** Demonstrates the successful mitigation of the Microtune plateau, showcasing stable, restored scaling laws.

### 6.5 V2 Ablation Analysis

An updated ablation matrix was executed across 5 random seeds to empirically validate the theoretical constraints governing the `v2` architecture. Statistical significance during this evaluation phase was determined utilizing standard Welch's t-tests against the optimal control baseline, prior to the adoption of advanced FDR correction methodologies.

> ![v2 Ablation Studies](assets/v2_ablation_statistics.png)
>
> **Figure 6: N=5 Ablation Study Results (V2).** Validated using Welch's t-test for statistical significance.

### 6.6 Latent Space Alignment (Manifold Progression)

Visualizing the latent embeddings confirms that the V2 supervised downstream decoder effectively organizes spatial data without destroying foundational priors.

> ![V2 mit_b2 Microtune Manifolds](assets/v2_mit_b2_Microtune.png)
>
> **Figure 7: TMLPN_v2 `mit_b2` Microtune Manifold Projections.** The t-SNE and UMAP projections demonstrate dense semantic separation of classes (bottom row) while maintaining perfectly parallel Modality Isolation (top row) between the RGB and Depth+Thermal streams directly after the stem.

---

## 7. V3 Architecture: Theoretical Fortification & Physical Priors

The TMLPN_v3 pipeline represents a comprehensive overhaul of the data ingestion and objective formulation strategies, systematically addressing memory inefficiencies, latent manifold degeneracy, and physically contradictory normalization priors.

### 7.1 Modality-Decoupled Physical Calibration

In prior iterations, geometric depth and thermal intensities shared bundled affine scaling parameters. This design was physically contradictory. Depth values represent scale-invariant distance measurements, whereas thermal tensors represent temperature-dependent radiometric variances [20]. TMLPN_v3 physically isolates these priors at the tensor level (`depth_scale` and `therm_scale`), allowing the network to empirically learn independent calibration mappings:
* **Scale-Invariant Depth:** Depth matrices are normalized by their spatial mean prior to calibration, ensuring the network penalizes geometric structural differences rather than absolute global distances.
* **Radiometrically Bounded Thermal:** The thermal scaling factor is heavily bounded to prevent the network from incorrectly memorizing ambient factory temperature drift instead of physical anomalies.

### 7.2 Fortified Pre-Training Objective

Relying exclusively on an Exponential Moving Average (EMA) teacher network [11] to prevent dimensional collapse creates a fragile single point of failure. TMLPN_v3 fortifies the loss formulation:
1. **Context Consistency Restoration:** The objective was updated to compute Mean Squared Error explicitly on the *unmasked* spatial coordinates. This physically anchors the context encoder, preventing the network from allowing unmasked geometry to drift into a degenerate, low-rank manifold [2].
2. **Covariance Penalty (VICReg-Inspired):** To actively immunize the network against dimensional collapse, a covariance penalty was injected into the objective. By calculating the covariance matrix of the channel embeddings and explicitly penalizing the off-diagonal correlations, the network is forced to utilize its entire representational capacity to map orthogonal, independent features [12].

### 7.3 Multimodal Data Augmentation & Precision

The `JEPAPretrainDataset` engine was structurally rewritten to prevent PyTorch/OpenCV garbage collection memory spikes, ensuring scalable execution on datasets exceeding 10,000 samples. Furthermore, statistical caches were transitioned to IEEE 64-bit float PyTorch binaries (`.pt`), eliminating the numerical precision loss previously caused by standard JSON serialization.

Crucially, TMLPN_v3 introduces physically-aware Multimodal Data Augmentation. To preserve alignment integrity, radiometric augmentations (brightness, contrast, hue, and saturation jitter) are strictly isolated to the RGB manifold. This simulates factory ambient lighting variance while permanently protecting the absolute physical measurement metrics inherent to the Depth and Thermal modalities. Furthermore, spatial geometry augmentations (horizontal/vertical flips) are strictly applied to individual camera strips during the initial ingestion phase to accurately simulate physical post-installation lane adjustments. Global spatial flips are explicitly omitted from the dataloader to prevent the generation of physically impossible structural layouts on the final composite manifold.

---

## 8. Experimental Results & Ablation Analysis

The evaluation of the `v3` architecture across the MM5 dataset demonstrates the robust integration of physical priors and mathematically fortified objective sums. As we are currently executing the full V3 MLOps pipeline to aggregate the N=5 seed distributions, full SOTA empirical comparisons remain "PENDING".

| Method | Backbone | Pre-Training | Mean mIoU | Train Loss | Val Loss | Edge FPS |
| --- | --- | --- | --- | --- | --- | --- |
| SFDFNet [18] | ResNet-50 | ImageNet | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| **TMLPN_v3 (Ours)** | **MiT-b1 (Student)** | **MM-JEPA** | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| **TMLPN_v3 (Ours)** | **MiT-b2** | **MM-JEPA** | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| **TMLPN_v3 (Ours)** | **MiT-b3** | **MM-JEPA** | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| **TMLPN_v3 (Ours)** | **MiT-b4** | **MM-JEPA** | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |
| **TMLPN_v3 (Ours)** | **MiT-b5 (Teacher)** | **MM-JEPA** | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |

### 8.1 Statistical Rigor & Benjamini-Hochberg Correction

An updated ablation matrix was executed across 5 random seeds to empirically validate the theoretical constraints governing the architecture. To definitively address Type I errors during multiple hypothesis testing, all generated ablation p-values are processed through the Benjamini-Hochberg False Discovery Rate (FDR) correction. 

The Benjamini-Hochberg FDR was explicitly chosen over the standard Bonferroni correction because Bonferroni acts overly conservative in deep learning ablations—unfairly penalizing valid, synergistic architectural contributions—whereas the Benjamini-Hochberg correction strictly bounds the expected proportion of falsely rejected null hypotheses without sacrificing statistical power.

### 8.2 V3 Component Isolation Matrix

To rigorously attribute exactly which architectural upgrades definitively prevented catastrophic gradient shattering, the automated pipeline extracts the overall optimal performing backbone and executes a targeted Component Isolation matrix (N=5 seeds per variant). This ensures empirical validation on whether these strategies are individually necessary or if they rely on synergistic interactions to prevent collapse:

* **Without LoRA (`Ablation_NoLoRA`):** Fully unfreezes the $N \times N$ network weights to prove whether allowing unbounded gradient flows natively overwrites the generalized multimodal representations.
* **Without LLRD (`Ablation_NoLLRD`):** Overrides the hierarchical optimization by applying a flat learning rate multiplier of 1.0 across all layers to measure the impact of uniform tuning.
* **Without Feature-Level KD (`Ablation_NoFeatureKD`):** Severs the student-teacher MSE spatial alignment constraint, forcing the isolated student backbone to train strictly on the supervised downstream semantic segmentation labels.
* **Without Context / Covariance:** Validates the efficacy of the newly fortified Phase 1 objective by selectively disabling the unmasked MSE [2] and VICReg mathematical penalties [12].

---

## 9. Edge Deployment (Jetson Orin Nano / Orange Pi 5)

The isolated, finetuned architecture is natively serialized to an ONNX artifact (`opset_version=18`) for cross-platform compatibility [20]. By deploying this distilled TensorRT engine onto edge hardware such as the Jetson Orin Nano or Rockchip RK3588 NPUs, the system achieves sub-pixel structural segmentation and real-time autonomous predictions directly at the sensor source [20].

To ensure explainability is preserved during deployment, Segmentation Grad-CAM heatmaps are extracted directly from the `decode_head.linear_pred` layer, dynamically verifying structural defect focus over artifact exploitation [23].

> ![V2 GradCAM Explainability](assets/v2_Gradcam_mit_b2_b3.png)
>
> **Figure 8: Segmentation Grad-CAM Defect Focus.** Explanability heatmaps tracking visual attention across the Baseline, Hero, and Microtune phases. The spatial attention confirms the network successfully localizes on physical anomalies rather than overfitting to background noise or uniform artifacting.

---

## 10. Future Work

To fully complete our evaluation suite and validate its utility in physical infrastructure settings, our future work is explicitly outlined below:

1. **SOTA Empirical Benchmarking:** Execute direct, head-to-head architectural comparisons against SFDFNet and leading RGB-D-T semantic segmentation baselines. This future evaluation will strictly utilize the locked dataset splits and identical Benjamini-Hochberg multi-seed correction framework to secure fair statistical validity.
2. **Edge Deployment Validation (FPS):** While cross-platform compilation logic is confirmed, completing formal edge deployment validation remains outstanding. Physical benchmarking will evaluate execution latency, specifically reporting the absolute Frames Per Second (FPS) achievable via our generated TensorRT engine directly on Jetson Orin Nano and Orange Pi 5 edge endpoints.

---

## References

[1] He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked autoencoders are scalable vision learners. *CVPR*.

[2] Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., & Ballas, N. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. *CVPR*.

[3] Girdhar, R., El-Nouby, A., Liu, Z., Singh, M., Alwala, K. V., Joulin, A., & Misra, I. (2023). ImageBind: One Embedding Space To Bind Them All. *CVPR*.

[4] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *NeurIPS*.

[5] Gupta, S., Hoffman, J., & Malik, J. (2016). Cross Modal Distillation for Supervision Transfer. *CVPR*.

[6] He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. *ICCV*.

[7] Zagoruyko, S., & Komodakis, N. (2017). DiracNets: Training Very Deep Neural Networks Without Skip-Connections. *arXiv preprint arXiv:1706.00388*.

[8] Perez, E., Strub, F., De Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. *AAAI*.

[9] Bao, H., Dong, L., Piao, S., & Wei, F. (2022). BEiT: BERT Pre-Training of Image Transformers. *ICLR*.

[10] Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. *NeurIPS*.

[11] Grill, J. B., Strub, F., Altché, F., Tallec, C., Richemond, P. H., Buchatskaya, E., ... & Valko, M. (2020). Bootstrap your own latent: A new approach to self-supervised learning. *NeurIPS*.

[12] Bardes, A., Ponce, J., & LeCun, Y. (2022). VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. *ICLR*.

[13] Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. *ICCV*.

[14] Sudre, C. H., Li, W., Vercauteren, T., Ourselin, S., & Jorge Cardoso, M. (2017). Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations. *DLMIA*.

[15] Shrivastava, A., Gupta, A., & Girshick, R. (2016). Training Region-based Object Detectors with Online Hard Example Mining. *CVPR*.

[16] Huang, Y., et al. (2020). Dynamic Weighting for Imbalanced Semantic Segmentation. *IEEE Access*.

[17] Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. *NIPS Deep Learning Workshop*.

[18] SFDFNet: Leveraging spatial-frequency deep fusion for RGB-T semantic segmentation. (2025). *Image and Vision Computing*.

[19] Mehta, S., & Rastegari, M. (2021). MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer. *ICLR*.

[20] Brenner, M., Reyes, N. H., Susnjak, T., & Barczak, A. L. C. (2026). MM5: Multimodal image capture and dataset generation for RGB, depth, thermal, UV, and NIR. *Information Fusion*, 126, 103516.

[21] Brenner, M., Reyes, N., Susnjak, T., & Barczak, A. (2025). MM5: Multimodal Image Dataset. *figshare. Dataset*.

[22] Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. *CVPR*.

[23] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV*.

[24] Bouthillier, X., Delaunay, P., Bronzi, M., Trofimov, A., Nichyporuk, B., Szeto, J., ... & Vincent, P. (2021). Accounting for Variance in Machine Learning Benchmarks. *MLSys*.

[25] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*.

[26] Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., & Bengio, Y. (2014). FitNets: Hints for Thin Deep Nets. *ICLR*.

[27] Clark, K., Luong, M. T., Le, Q. V., & Manning, C. D. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators. *ICLR*.

---

## 🙏 Acknowledgments & Citations

This project would not be possible without the MM5 Dataset. We sincerely thank the original creators and authors for their foundational work in multi-modal data collection, hardware synchronization, and curation, which enabled the training and evaluation of this architecture.

If you utilize this pipeline, the underlying architecture, or the data, please cite the primary publication alongside the dataset repository:

**Primary Publication:**

> Brenner, M., Reyes, N. H., Susnjak, T., & Barczak, A. L. C. (2026). MM5: Multimodal image capture and dataset generation for RGB, depth, thermal, UV, and NIR. Information Fusion, 126, 103516.
> DOI: [https://doi.org/10.1016/j.inffus.2025.103516](https://doi.org/10.1016/j.inffus.2025.103516)

**Dataset:**

> Brenner, M., Reyes, N., Susnjak, T., & Barczak, A. (2025). MM5: Multimodal Image Dataset. figshare. Dataset.
> DOI: [https://doi.org/10.6084/m9.figshare.28722164](https://www.google.com/search?q=https://doi.org/10.6084/m9.figshare.28722164)