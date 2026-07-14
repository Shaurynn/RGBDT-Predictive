# TriModal Perception Architectures for Structural Defect Detection: The MM-JEPA Paradigm

## Abstract

Structural defect detection in industrial environments necessitates the robust integration of RGB, Depth, and Thermal (RGB-D-T) modalities [20]. In this repository, we document the evolution of a spatial perception engine designed specifically for bounded edge-hardware [19]. Recognizing the computational limits of deep cross-attention networks [4] and the high-frequency noise inherent to generative pixel-space autoencoders [1], this architecture introduces the Tri-Modal Latent Predictive Network (TMLPN) utilizing a Multimodal Joint-Embedding Predictive Architecture (MM-JEPA) [2]. By decoupling representation learning from downstream segmentation [10] and applying rigorous spatial constraints—including Dirac-initialized alignment projections [7] and Global Volume Anchored Generalized Dice Loss [14]—the framework establishes a highly robust structural foundation model capable of real-time, high-resolution edge inference [20].

---

## 1. Introduction

The integration of high-resolution, unaligned multimodal sensors provides critical advantages in structural defect and anomaly detection [20]. Deploying continuous multi-channel arrays—specifically those originating from 16-bit Baumer GigE industrial sensors—onto edge compute modules necessitates strict algorithmic efficiency [20].

Early iterations of multimodal learning utilized generative pixel-space decoders [1]. However, reconstructing pixel-space values forces the network to map irrelevant high-frequency radiometric noise, wasting representational capacity [1]. To address this, we transitioned to Latent Space Prediction, building upon recent advancements in self-supervised architectures [2]. This repository executes genuine target-selective spatial inference via a mathematically rigorous two-phase pipeline [2, 10].

### 1.1 Architectural Deviations: TMLPN vs. MuMo-JEPA

Recent literature highlights Multimodal JEPA (MuMo-JEPA) architectures, which rely heavily on deep late-fusion, independent Vision Transformer (ViT-Huge) trunks per modality, and cross-attention joint embeddings [3]. While theoretically optimal for unconstrained server environments, we explicitly deviate from the MuMo-JEPA methodology for the following mathematically grounded reasons:

1. **Memory-Bandwidth Bottlenecks:** Holding multiple independent ViT trunks in VRAM violates the shared-memory and bandwidth constraints of edge devices, as standard non-hierarchical Transformers incur prohibitive parameter counts and memory access costs during inference [19].
2. **Quadratic Scaling:** Standard self-attention mechanisms scale with $O(N^2)$ complexity relative to spatial resolution, prohibiting real-time inference on high-resolution industrial image strips [4].
3. **Hardware-Aware Early Fusion:** TMLPN utilizes a single hierarchical MiT trunk with a mathematically protected modality-isolated stem [10]. The MiT sequence reduction process ensures linear $O(N)$ computational scaling [10], while our Dirac-initialized $1 \times 1$ projections successfully neutralize the latent alignment flaws typically associated with simplistic early fusion [5, 7].

---

## 2. Reproducibility & Open Source Assets

To ensure strict academic reproducibility of our evaluation benchmarks, all artifacts will be published alongside this repository following the conclusion of the training cycle [24]:

* **Pre-Trained Weights:** Converged `best_model.pt` checkpoints will be provided for identical inference replication [24].
* **Dataset Splits:** Exact training and evaluation subsets are mapped via CSV data splits (`data/splits/`) to eliminate distribution variance [21].
* **Deterministic Execution:** The execution engines utilize strict environmental locking (`seed=42`) across PyTorch, NumPy, and CUDA backends to eliminate stochastic gradient variance [24].

---

## 3. Phase 1: Self-Supervised MM-JEPA Pre-Training

To build a true foundation model of the physical environment, the network must decouple feature extraction from human-annotated labels [2]. Phase 1 achieves this through pure self-supervised spatial inference, forcing the network to understand structural geometry prior to task-specific fine-tuning [2].

> ![Network Architecture](assets/Network_Architecture_Diagram.png)
>
> Figure 1: The Tri-Modal Latent Predictive Network (TMLPN_v1) two-stage execution pipeline.
>
> (Left) Phase 1: Self-Supervised MM-JEPA Pre-Training. Unaligned 5-channel multi-modal inputs are processed through a ModalityIsolatedPatchEmbed stem. To mathematically bridge the cross-modal domain gap, the Kaiming-initialized Depth+Thermal stream passes through a $1 \times 1$ Dirac-initialized alignment projection and learnable affine calibration priors before additive fusion with the ImageNet-initialized RGB manifold. The masked input (utilizing explicit token replacement) is processed by the Context Encoder, while the unmasked input is processed by the Target Encoder. To prevent representation collapse, the Target Encoder is computationally isolated from the gradient graph and updated strictly via a Cosine Annealing Exponential Moving Average (EMA) schedule. A depthwise-separable predictor aligns the representations in the latent space, optimized via an $L_2$-normalized MSE loss and spatial variance regularization.
>
> (Center) Phase 2: Downstream Semantic Fine-Tuning. The converged Context Encoder is transferred to the downstream task as a foundation backbone. Hierarchical feature stages are extracted, passed through LayerNorms, and upsampled to a unified $1/4$ resolution via a resolution-invariant SegFormer All-MLP Decoder to synthesize sub-pixel spatial boundaries. The final segmentation maps are optimized using an $\alpha$-Balanced Focal Loss and Global Volume Anchored Generalized Dice Loss.
>
> (Far Right) Explainability & Validation. Downstream inference is validated via Segmentation Grad-CAM heatmaps extracted directly from the linear prediction head, empirically verifying the network's localization on physical structural defects rather than high-frequency radiometric artifacts.

### 3.1 Stem Modality Isolation & Alignment Projection

Initializing a multi-channel stream directly from 3-channel weights introduces severe representational interference due to the cross-modal domain gap [5]. TMLPN physically isolates modality ingestion at the stem using a `ModalityIsolatedPatchEmbed` module to safely fuse unaligned manifolds [5]:

* **RGB Stream:** Inherits pristine, unmodified 3-channel ImageNet kernels to leverage generalized edge-detection priors [22].
* **D+T Stream:** Utilizes independent, Kaiming-initialized convolutions to prevent early-epoch activation vanishing in the high-variance depth and thermal tensors [6].
* **1x1 Dirac Alignment:** To bridge this dimensional and statistical gap prior to additive fusion, TMLPN utilizes a $1 \times 1$ convolutional projection initialized via a Dirac delta distribution [7]. This mathematically prevents gradient shattering while allowing the Kaiming-initialized Depth/Thermal kernels to gradually align with the ImageNet manifold [7].

### 3.2 The MM-JEPA Topology

To satisfy the theoretical mandates of latent prediction, the network executes the following spatial constraints:

* **Token Replacement Masking:** Masked spatial coordinates are explicitly replaced with a broadcasted, learnable parameter (`encoder_mask_token`), aligning with proven masked image modeling protocols [9].
* **Multi-Block Strategy:** The architecture samples 4 independent overlapping target blocks with varying scales (0.15–0.20) and aspect ratios (0.75–1.5) [2].
* **Target-Conditioned Spatial Predictor:** Pure 2D Positional Encodings are concatenated *only* alongside the context feature map and target mask within the depthwise-separable predictor, ensuring the network knows exactly *where* to predict [10].
* **EMA Cosine Annealing & Gradient Isolation:** Target network outputs are severed from the computational graph, updated strictly via a Cosine Annealing Exponential Moving Average (EMA) schedule from $0.996$ to $1.0$ to prevent representation collapse [11].

---

## 4. Phase 2: Supervised Semantic Fine-Tuning

Phase 2 transfers the pre-trained Context Encoder to the downstream task of semantic segmentation, utilizing a resolution-invariant SegFormer MLP decoder [10].

### 4.1 The Multi-Scale All-MLP Decoder

Heavy transposed convolution decoders violate the latency constraints required for real-time edge processing [19]. TMLPN synthesizes sub-pixel spatial boundaries using an All-MLP Decoder [10]. By projecting the $1/4$, $1/8$, $1/16$, and $1/32$ hierarchical feature grids to a unified embedding dimension, applying LayerNorms, upsampling exclusively to a common $1/4$ resolution, and concatenating them, the network achieves boundary delineation while maintaining an edge-compliant footprint [10].

### 4.2 Mitigating Imbalance: Alpha-Balanced Focal GDL

Industrial datasets exhibit extreme foreground-background class imbalance [20]. The downstream engine utilizes a mathematically rigorous bipartite loss objective:

1. **$\alpha$-Balanced Focal Loss:** Mitigates background dominance by explicitly weighting classes according to their empirical dataset frequencies via Median Frequency Balancing [13].
2. **Global Volume Anchored Generalized Dice Loss (GDL):** TMLPN utilizes *Global Volume Anchoring*, where GDL weights are permanently anchored to the inverse square of the global dataset frequencies [14]. Additive Laplace Smoothing ensures theoretical bounds are naturally constrained by the dataset's native volume [24].

---

## 5. V1 Pipeline Findings & Ablation Analysis

Initial execution of the `v1` pipeline yielded high baseline performance but revealed critical structural vulnerabilities during end-to-end unfreezing.

### 5.1 The `microtune` Collapse: Gradient Shattering

As illustrated in Figure 3, the models achieved exceptional peaks during the `hero` phase (with `mit_b5` reaching an impressive **0.8397 mIoU** with a frozen foundation). However, when the pipeline transitioned to the `microtune` phase for end-to-end tuning, performance suffered a catastrophic collapse (dropping to ~0.61–0.68 mIoU).

> ![v1 Phase Progression](assets/v1_phase_progression_trajectory.png)
>
> **Figure 3: Architecture Progression Trajectory (V1).** Demonstrates severe catastrophic forgetting when the massive multi-scale decoder's gradients overwrite the delicate JEPA-pretrained foundation during the Microtune phase.

This phenomenon is a textbook case of gradient shattering. Unconstrained downstream gradients propagating back from a high-capacity decoder aggressively overwrote the generalized multimodal manifolds learned during Phase 1 self-supervised pre-training, destroying the network's spatial awareness.

### 5.2 Ablation Matrix: Component Necessity

A rigorous N=5 ablation study (Figure 3) isolated the empirical necessity of our architectural components:

> ![v1 Ablation Studies](assets/v1_ablation_statistics.png)
>
> **Figure 3: N=5 Ablation Study Results.** Validated using Welch's t-test for statistical significance against the optimal control.

* **Modality Isolation is Critical:** The `NaiveFusion` ablation collapsed the network to a **0.4476 mIoU** ($p=0.0001$), proving that forcing 3-channel pretrained weights to blindly accept 5-channel unaligned data destroys ImageNet priors. The Dirac-initialized $1 \times 1$ alignment stem is mathematically necessary.
* **Logit-Level KD is Noisy:** Disabling Knowledge Distillation (`NoKD`) yielded no statistically significant penalty ($p=0.7603$). In dense pixel-prediction tasks, raw logit distributions are highly noisy, meaning traditional KD provided zero measurable gain while doubling VRAM requirements.
* **Variance Regularization is Redundant:** Removing variance regularization (`NoVariance`) showed no degradation ($p=0.8297$), confirming that the EMA asymmetric target updates naturally prevent dimensional collapse without explicit spatial variance penalties [11].

---

## 6. V2 Architecture: Optimization & Preservation Strategies

To address the vulnerabilities discovered in the `v1` pipeline, the `v2` architecture transitions to a highly fortified downstream engine designed specifically to bend the network toward semantic segmentation without inducing catastrophic forgetting.

> ![v2 Network Architecture Diagram](assets/v2_Network_Architecture_Diagram.png)
>
> **Figure 4: The finalized Tri-Modal Latent Predictive Network (TMLPN_v2) two-stage execution pipeline.**
>
> (Top) Phase 1: Self-Supervised MM-JEPA Pre-Training. Reorganized and streamlined for improved visual clarity, removing Predictor implementation notes and internal citations. Arrows flow from unaligned inputs (with mask token replacement) through a `ModalityIsolatedPatchEmbed` stem (with Dirac alignment and learnable affine priors) to the Context Encoder (student MiT backbone, with LLRD decay) and Target Encoder (teacher MiT backbone, frozen EMA with cosine schedule). Arrows point from context and target representations to the Unified Phase 1 JEPA Objective, now strictly optimized via an $L_2$-normalized MSE Latent >Predictive Loss on masked target coordinates]. The redundant master 'Context Consistency' and 'Variance Regularization' loss terms have been masterfully pruned].
>
> (Center) Phase 2: Supervised Segmentation & Feature-Level Distillation (N=5 Matrix). The converged student and locked teacher models run in parallel. Arrows flow from Student segmentation logits to a combined Segmentation Loss block (GDL + Focal + DCW). Arrows flow from four intermediate hierarchical feature stages to a central MSE Feature-Alignment Loss block]. The diagram visualizes the bipartite objective: [SegLoss] + [FD Loss]. Text labels explicitly state "Trainable: Only LoRA & Decoder"].
>
> (Right) Explainability & Deployment. Validated via Segmentation Grad-CAM overlays. A summarized MLOps M-JEPA Research Rigor box details the component necessity.

### 6.1 Low-Rank Adaptation (LoRA)

To completely neutralize gradient shattering during the `microtune` phase, the `v2` pipeline integrates Low-Rank Adaptation (LoRA) [25]. Rather than fully unfreezing the $N \times N$ weight matrices inside the MiT backbone, the foundation remains mathematically locked. We inject tiny, trainable low-rank matrices ($A$ and $B$) specifically into the Query and Value projection layers of the transformer blocks [25]. This guarantees zero catastrophic forgetting of the Phase 1 JEPA pre-training while radically reducing VRAM consumption.

### 6.2 Layer-Wise Learning Rate Decay (LLRD)

To safely couple the LoRA modules with the high-capacity decoder, we deploy Layer-Wise Learning Rate Decay (LLRD) [27]. The optimizer groups parameters hierarchically: the downstream decoder receives the full base learning rate, while gradients flowing deeper into the backbone are exponentially decayed (e.g., $lr \times 0.85^n$). This ensures deep foundation layers adapt smoothly while protecting the fragile structural manifolds at the modality stem [27].

### 6.3 Feature-Level Knowledge Distillation

To compress the performant `mit_b5` architecture into an edge-ready `mit_b1` without the noise of logit-level distillation, `v2` upgrades to Feature-Level KD [26]. By $L_2$-normalizing and aligning the intermediate feature grids of the teacher directly with the student via MSE loss, the student mathematically inherits the complex representational structure of the teacher, circumventing spatial boundary noise [26].

---

## 7. V2 Experimental Results [PENDING]

*The `v2` architecture is actively undergoing its N=5 statistical evaluation.*

| Method | Backbone | Pre-Training | Mean mIoU | Std Dev | Params (M) | Edge FPS |
| --- | --- | --- | --- | --- | --- | --- |
| SFDFNet [18] | ResNet-50 | ImageNet | TBD | --- | TBD | TBD |
| **TMLPN_v2 (Ours)** | **MiT-b1 (Student)** | **MM-JEPA** | **[PENDING]** | **[PENDING]** | **13.7** | **[PENDING]** |
| **TMLPN_v2 (Ours)** | **MiT-b5 (Teacher)** | **MM-JEPA** | **[PENDING]** | **[PENDING]** | **82.0** | **[PENDING]** |

---

## 8. Edge Deployment (Jetson Orin Nano / Orange Pi 5)

The isolated, finetuned architecture is natively serialized to an ONNX artifact (`opset_version=18`) for cross-platform compatibility [20]. By deploying this distilled TensorRT engine onto edge hardware such as the Jetson Orin Nano or Rockchip RK3588 NPUs, the system achieves sub-pixel structural segmentation and real-time autonomous predictions directly at the sensor source [20].

To ensure explainability is preserved during deployment, Segmentation Grad-CAM heatmaps are extracted directly from the `decode_head.linear_pred` layer, dynamically verifying structural defect focus over artifact exploitation [23].

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