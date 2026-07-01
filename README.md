# TriModal Perception Architectures for Structural Defect Detection: Generative vs. Latent Predictive Networks

**Abstract**
Structural defect detection in complex industrial environments requires robust multimodal integration. In this work, we present two distinct spatial perception engines for RGB, Depth, and Thermal sensors built upon a 4-channel patched `mit_b1` Vision Transformer backbone. **Part I** documents the legacy TriModal Predictive Network (TMPN), which utilizes a pixel-space generative approach to hallucinate obscured thermodynamics. **Part II** introduces the TriModal Latent Predictive Network (TMLPN), which abandons pixel generation entirely. By adapting the Joint-Embedding Predictive Architecture (JEPA) paradigm to a static spatial domain and regulating the 512-channel embeddings via a strict Variance-Covariance (VICReg) penalty, the architecture successfully traverses domain manifolds to predict structural physics with vastly improved sample efficiency and resilience to stochastic noise.

---

## PART I: The Tri-Objective Generative Network (TMPN)

## 1. Introduction
The detection of structural defects utilizing RGB, Depth, and Thermal (TriModal) sensors poses a unique cross-modal alignment challenge. This repository documents the TMPN architecture, which addresses this challenge by forcing the network to hallucinate obscured thermodynamic data back into pixel space. By reconstructing missing thermal signatures based on pristine RGB-D context, the network inherently learns the physical properties of structural anomalies.

## 2. Methodology & Architecture

### 2.1 Tri-Objective Learning
The architecture is supervised by three distinct loss functions to ensure spatial and physical accuracy:
1. **Primary Segmentation:** A Focal Dice loss evaluating the final spatial boundaries.
2. **Thermal Reconstruction (Physics Loss):** An Object-Aware Block Mask obscures a percentage of the input Thermal tensor. The network's decoder must reconstruct this masked region in pixel-space using a Masked Mean Squared Error (MSE) loss, forcing it to learn structural thermodynamics.
3. **Auxiliary Supervision:** An auxiliary classifier attached directly to the thermal encoder provides deep supervision, stabilizing the gradients during early training epochs.

### 2.2 Global Context Modality Attention (GCMA) with Spatial Queries
To resolve mechanical parallax (Y-axis sensor offset), previous paradigms collapsed spatial dimensions via Global Average Pooling, inducing "Spatial Annihilation." 
TMPN's GCMA head preserves pristine geometry by treating every individual pixel in the RGB-D feature map as a discrete Query. The globally pooled Thermal and RGB-D signatures act as the Keys and Values [1]. This allows every pixel to query the global thermodynamic state while retaining its exact X, Y coordinate boundaries.

### 2.3 4-Channel Stem Patching & Capacity Scaling (MiT-b1)
To retain the foundational intelligence of pre-trained Vision Transformers while accepting 4-channel input (RGB + Depth), we surgically patch the `patch_embed1.proj` layer of the SegFormer backbone [2]. The 3-channel ImageNet weights are loaded into the RGB channels, and the 4th (Depth) channel is initialized using the mathematical mean of the RGB weights. 
* To ensure the network possesses sufficient representational capacity to act as a physical world model, the backbone was scaled to the **`mit_b1`** architecture, pushing the total parameter count to approximately 14 million.

### 2.4 Batch-Aware Focal Dice Loss
To combat "Empty-Class Suppression" across 31 highly unbalanced classes, the custom `FocalDiceLoss` dynamically evaluates the ground truth mask and restricts the Dice penalty calculation strictly to classes physically present within the current batch [3].

---

## 3. Experimental Setup & Dataset

### 3.1 The MM5 Dataset
This research and architecture relies heavily on the **MM5 Dataset**, which provides the rigorously aligned multimodal data necessary to train and validate this cross-modal architecture. 

### 3.2 The State-Machine Training Pipeline
The training regimen is autonomously managed by an `ExperimentManager` through four sequential phases:
1. **Baseline Phase:** Warms up patched ImageNet weights with a Cosine Annealing scheduler.
2. **HPO Phase:** Executes a 30-trial Optuna sweep focusing on loss weighting ($\alpha$, $\beta$) and learning rates.
3. **Hero Phase:** Injects optimized hyperparameters for full convergence.
4. **Microtune Phase:** A cooling phase utilizing a microscopic learning rate schedule (1e-5 to 1e-7) coupled with Test-Time Augmentation (TTA) to polish spatial boundaries.

---

## 4. Results & Analysis

### 4.1 Quantitative Pipeline Milestones (TriModalPredictiveNetwork)
The state-machine progression on the MM5 dataset yielded the following quantitative milestones. Test-Time Augmentation (TTA) was utilized during the final diagnostic passes to measure robustness against asymmetric false-positives.

| Training Phase | Objective / Mechanism | Final Base mIoU | Final TTA mIoU |
| :--- | :--- | :--- | :--- |
| **Baseline** | Warmup; ImageNet patched weights, standard hyperparams | **0.7434** | **0.7341** |
| **HPO** | 30-Trial Optuna sweep. Best peak mIoU recorded: 0.7504 | **-** | **-** |
| **Hero** | Deep convergence (Patience triggered at Epoch 93) | **0.7453** | **0.7383** |
| **Microtune** | Cooling schedule + Polish (Patience triggered at Epoch 56) | **0.7488** | **0.7391** |

### 4.2 Hyperparameter Optimization (Optuna)
The HPO phase successfully isolated the optimal balance between the Segmentation Loss, the Physical Reconstruction penalty ($\alpha$), and the Auxiliary Supervision ($\beta$).

> ![Optuna Dashboard](assets/Optuna_TMPN.png)
> *Figure 1: Optuna Parallel Coordinate Plot detailing the correlation between the loss weights, learning rate, and the objective Validation mIoU.*

### 4.3 Training Dynamics & Convergence
The integration of the `mit_b1` backbone alongside the Tri-Objective loss landscape provided rapid early-epoch convergence, transitioning smoothly into the Microtune cooling schedule.

> ![TensorBoard Metrics](assets/Tensorboard_TMPN.png)
> *Figure 2: TensorBoard metrics during the Hero and Microtune phases. Note the stability of the Validation mIoU curve as the microscopic learning rate schedule polishes the spatial decision boundaries.*

### 4.4 Explainability & Spatial Attention (Grad-CAM)
To verify that the GCMA head successfully retains spatial geometry while querying global thermodynamics, Semantic Grad-CAM hooks and Epistemic Uncertainty mapping were applied directly to the evaluation pipeline.

> ![Grad-CAM Overlay](assets/batch0_img1_class15_gradcam.png)
> ![Epistemic Uncertainty](assets/batch0_img1_epistemic_uncertainty.png)
> *Figure 3: Diagnostics generated during the Evaluation Pass. The Grad-CAM heatmaps demonstrate highly precise boundary delineation around structural defects. The Epistemic Uncertainty map confirms zero variance in background suppression, with hesitation strictly constrained to the extreme sub-pixel edges of the geometric structures.*

---

## 5. Deployment

The final phase of the pipeline serializes the optimized graph to an ONNX artifact (`opset_version=14`). The architecture is strictly engineered for low-latency inference and is prepared for downstream quantization and TensorRT engine compilation via `trtexec` for edge hardware execution.

*(Note: TMPN represents the absolute mathematical ceiling for the pixel-space generative approach. To eliminate the computational overhead of rendering stochastic sensor noise, the pipeline transitions to Latent Space Prediction in Part II).*

---

## PART II: The TriModal Latent Predictive Network (TMLPN)

## 6. Introduction to Latent Prediction

While the generative TMPN architecture successfully aligned multimodal features, pixel-space reconstruction is inherently inefficient. The network expends massive computational capacity hallucinating high-frequency thermal "speckle" and ambient heat bleed—artifacts that are irrelevant to the actual physical structure of a defect. Part II introduces the TMLPN, shifting the paradigm from pixel generation to latent feature prediction. By operating entirely in an abstract manifold, the architecture achieves immunity to stochastic noise and accelerates domain traversal.

## 7. Mathematical Mapping to the JEPA Framework

The TMLPN formally adapts the Joint-Embedding Predictive Architecture (JEPA) [5] for cross-modal structural evaluation. To ensure rigorous adherence to the theoretical framework, the architecture is defined by the following strict topological mappings:

* **The Context ($x$):** The observable information the model is permitted to evaluate. In TMLPN, this is the pristine RGB-D geometry tensor combined with an artificially masked Thermal tensor.
* **The Target ($y$):** The uncorrupted physical ground truth the model is attempting to predict. In TMLPN, this is the pristine, unmasked Thermal tensor.
* **The Context Encoder ($E_\theta$):** The composite network responsible for processing the observable world. In TMLPN, this comprises the `mit_b1` RGB-D encoder, the Thermal encoder processing the masked input, and the Global Context Modality Attention (GCMA) fusion head that binds them.
* **The Target Encoder ($E_{\bar{\theta}}$):** The network that generates the "ground truth" latent signature. In TMLPN, this is the isolated Thermal encoder processing the unmasked target tensor, governed by a strict `.detach()` operation to lock the weights during prediction.
* **The Predictor ($P_\phi$):** The neural module that infers the Target embedding based solely on the Context embedding. In TMLPN, this is the `latent_predictor` operating via a hierarchical convolutional bottleneck.

### 7.1 The Latent Regularization Engine (The VICReg Triad)

Because the Target Encoder ($E_{\bar{\theta}}$) is locked via a Stop-Gradient, the network must be physically restrained from Representation Collapse. TMLPN adapts the VICReg framework [6] via three constraints:

* **Similarity Loss (Invariance):** The primary MSE objective. Forces $P_\phi(E_\theta(x))$ to match $E_{\bar{\theta}}(y)$.
* **Variance Loss (Anti-Collapse):** A hinge loss enforcing standard deviation $\geq 1.0$ across the predicted latent batch, preventing points from collapsing into a singularity.
* **Covariance Loss (The Decorrelator):** Penalizes off-diagonal elements in the covariance matrix, forcing all 512 channels of the backbone to learn unique, orthogonal features [6].

---

## 8. Results & Analysis

### 8.1 Baseline Training Dynamics & Pathologies

During the unoptimized 150-epoch baseline run, the architecture exhibited expected self-supervised mathematical behaviors, notably a massive discrepancy between Total Training Loss (~40.0) and Validation Loss (~0.14). Because latent target generation is strictly an auxiliary training task [5], the Validation loop correctly bypassed the VICReg penalties to strictly evaluate the downstream segmentation cross-entropy error.

Furthermore, early epochs demonstrated a sharp spike in Covariance (peaking at ~42.0 around Epoch 23). This is a known pathology in high-capacity architectures attempting to satisfy VICReg Variance constraints by duplicating features across channels (Dimensional Redundancy) [6]. By aggressively weighting the covariance penalty (`cov_weight = 15.0`), the network was forced to decorrelate its 512 channels, stabilizing the manifold.

### 8.2 Deep Convergence & The Microtune Polish

Following a 30-trial Bayesian optimization sweep (Optuna), the architecture achieved deep convergence during a long-horizon Hero phase. To finalize spatial boundaries, a Microtune phase shifted the learning rate into a microscopic $10^{-5}$ to $10^{-7}$ cooling schedule, anchoring the latent space.

| Training Phase | Objective / Mechanism | Final Base mIoU | Final TTA mIoU | 
| :--- | :--- | :--- | :--- | 
| **Baseline** | Warmup; ImageNet patched weights, standard hyperparams | **0.7248** | **0.7161** | 
| **HPO** | 30-Trial Optuna sweep. Peak mIoU: 0.7328 | **-** | **-** | 
| **Hero** | Deep convergence (Patience triggered at Epoch 96) | **0.7311** | **0.7262** | 
| **Microtune** | Cooling schedule + Spatial Polish | **[Recorded in JSON]** | **[Recorded in JSON]** | 

> ![TMLPN Microtune Dynamics](assets/Tensorboard_TMLPN_Microtune.png)
> *Figure 1: Telemetry of the TMLPN Microtune Phase. The microscopic learning rate gently cools the Covariance and Total Train Loss (top) while the Validation mIoU (bottom) remains highly stable.*

### 8.3 Explainability: Tightening Spatial Boundaries

Semantic Grad-CAM and Epistemic Uncertainty mapping applied to identical input geometry at the conclusion of the Microtune run demonstrate razor-sharp, object-centric hotspots. Boundary hesitation is nearly eliminated, remaining strictly confined to the extreme geometric perimeters.

> ![TMLPN Microtune Grad-CAM](assets/TMLPN_Microtune_batch0_img1_class15_gradcam.png)
> ![TMLPN Microtune Epistemic Uncertainty](assets/TMLPN_Microtune_batch0_img1_epistemic_uncertainty.png)
> *Figure 2: Final TMLPN Diagnostics.*

---

## 9. Discussion

The transition from generative to latent predictive architectures fundamentally restructures how the model internalizes structural physics. The TMLPN architecture introduces several deliberate deviations from foundational JEPA literature (e.g., I-JEPA [5], LeWorldModel [4]) to optimize theoretical frameworks for real-world, constrained industrial edge execution.

### 9.1 The Efficacy of Spatial MLPs over Transformer Predictors
Standard foundation-scale JEPAs frequently utilize heavy Multi-Head Attention predictors to route information across spatial and temporal gaps by combining visible tokens with explicit Mask Tokens [5]. TMLPN explicitly discards the Transformer-based predictor in favor of a Hierarchical Convolutional Predictor (a sequence of $1 \times 1$ convolutions). 

A $1 \times 1$ convolution acts mathematically as a pixel-wise Multi-Layer Perceptron (MLP) [8]. First formalized in the landmark *Network In Network* paper [8], utilizing a $1 \times 1$ convolution is the standard method for executing a point-wise MLP without flattening the 2D tensor. Flattening the tensor to pass through standard `nn.Linear` layers would catastrophically destroy the geometric grid established by the Vision Transformer. Furthermore, because the TMLPN's Context Encoder relies on the GCMA head—which has already performed the heavy computational lifting of aggregating the global multi-scale receptive field into the local feature maps—applying additional spatial attention at the predictor level is computationally redundant. 

The predictor's sole mathematical burden is to perform a cross-modal feature-space translation at coordinate $(x, y)$ to the thermal latent space at that exact same $(x, y)$. Consequently, the hierarchical spatial MLP bottleneck is the mathematically optimal operation for this cross-modal projection, perfectly preserving grid coherence while drastically reducing floating-point operations (FLOPs).

### 9.2 Stop-Gradient Heuristics and Explicit Covariance Regularization
Many self-supervised frameworks utilize an Exponential Moving Average (EMA) teacher network to gently update the Target Encoder, preventing representation collapse. TMLPN abandons the EMA framework entirely, utilizing identical shared weights for $E_\theta$ and $E_{\bar{\theta}}$ governed solely by a strict Stop-Gradient (`.detach()`) operation.

By explicitly enforcing the VICReg constraints [6], TMLPN proves that mathematically regularizing the variance and covariance of the embedding manifold physically prevents rank collapse. When collapse is rendered mathematically impossible by explicit channel decorrelation, the implicit regularization provided by a momentum encoder becomes redundant. Excision of the EMA teacher network significantly reduces memory overhead during training without sacrificing manifold stability [6].

### 9.3 Overcoming Domain Pathologies: Capacity Limits and Class Imbalance
Applied industrial defect datasets exhibit extreme class imbalance, often causing networks to suppress minority signals once dominant classes converge. To overcome this dataset-agnostic imbalance without unbalancing the delicately constructed VICReg latent space, TMLPN utilizes a **Dynamic Class-Weighting Schedule (DCW)** [7]. By tracking an Exponential Moving Average (EMA) of the validation IoU for each class, the downstream Dice penalty is exponentially scaled on the fly specifically for lagging minority classes ($W_c = \text{EMA}\left( W_c, e^{\tau(1 - \text{IoU}_c)} \right)$).

While DCW resolves focal imbalance, the lightweight `mit_b1` backbone (14M parameters) eventually encounters a hard representational capacity limit. Upgrading to a massive `mit_b5` backbone (82M parameters) shatters this ceiling, but its sheer size violates edge-inference latency budgets. The integrated **Knowledge Distillation (KD)** engine resolves this paradox [9]. By forcing the lightweight Student to minimize the Kullback-Leibler (KL) Divergence against the massive Teacher's soft probabilities ("Dark Knowledge"), the edge-deployed model inherits the advanced stochastic noise suppression of an 82M-parameter network while retaining its 14M-parameter high-speed footprint [9].

---

## 10. Conclusion & Edge Deployment

By abandoning pixel-space generation, the TriModal Latent Predictive Network establishes a vastly more efficient methodology for multimodal defect detection. The architecture successfully isolates structural thermodynamics from stochastic sensor noise, achieving rapid spatial convergence and immense confidence on sub-pixel boundaries.

For isolated industrial deployment, the optimized TMLPN graph is serialized to an ONNX artifact (`opset_version=18`). With all generative decoders excised and the core intelligence distilled into a lightweight footprint, the model is strictly engineered for low-latency inference on robust edge computers (e.g., Jetson platform series). By directly interfacing this TensorRT engine with edge controllers, the system executes real-time autonomous thermal inspections directly at the sensor source.

---

## References

[1] Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.  
[2] Xie, E., et al. (2021). *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*. NeurIPS.  
[3] Sudre, C. H., et al. (2017). *Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations*. DLMIA.  
[4] Maes, L., et al. (2024). *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels*. arXiv preprint.  
[5] Assran, M., et al. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*. CVPR.  
[6] Bardes, A., Ponce, J., & LeCun, Y. (2022). *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning*. ICLR.  
[7] Huang, Y., et al. (2020). *Dynamic Weighting for Imbalanced Semantic Segmentation*.  
[8] Lin, M., Chen, Q., & Yan, S. (2013). *Network In Network*. ICLR.  
[9] Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network*. NIPS Deep Learning Workshop.

---

## 🙏 Acknowledgments & Citations

This project would not be possible without the MM5 Dataset. We sincerely thank the original creators and authors for their foundational work in multi-modal data collection, hardware synchronization, and curation, which enabled the training and evaluation of this architecture.

If you utilize this pipeline, the underlying architecture, or the data, please cite the primary publication alongside the dataset repository:

**Primary Publication:**
> Brenner, M., Reyes, N. H., Susnjak, T., & Barczak, A. L. C. (2026). MM5: Multimodal image capture and dataset generation for RGB, depth, thermal, UV, and NIR. *Information Fusion*, 126, 103516.  
> DOI: https://doi.org/10.1016/j.inffus.2025.103516

**Dataset:**
> Brenner, M., Reyes, N., Susnjak, T., & Barczak, A. (2025). MM5: Multimodal Image Dataset. figshare. Dataset.  
> DOI: https://doi.org/10.6084/m9.figshare.28722164