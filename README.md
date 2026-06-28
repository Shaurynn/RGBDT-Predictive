# TriModal Predictive Network (TMPN): A Tri-Objective Spatial Architecture for Structural Defect Detection

**Abstract**
Structural defect detection in complex industrial environments requires robust multimodal integration. In this work, we present the TriModal Predictive Network (TMPN), a static, spatial perception engine for RGB, Depth, and Thermal sensors. This architecture utilizes a Tri-Objective learning paradigm: Semantic Segmentation, Pixel-Space Thermal Reconstruction, and Auxiliary Feature Supervision. To overcome mechanical parallax without annihilating geometric boundaries, we introduce a Global Context Modality Attention (GCMA) head utilizing spatial queries. Built upon a 4-channel patched `mit_b1` Vision Transformer backbone, the resulting pipeline is rigorously evaluated through Test-Time Augmentation (TTA) and optimized for edge deployment via TensorRT.

---

## 1. Introduction
The detection of structural defects utilizing RGB, Depth, and Thermal (TriModal) sensors poses a unique cross-modal alignment challenge. This repository documents the TMPN architecture, which addresses this challenge by forcing the network to hallucinate obscured thermodynamic data back into pixel space. By reconstructing missing thermal signatures based on pristine RGB-D context, the network inherently learns the physical properties of structural anomalies.

*(Note: This repository is currently transitioning from this pixel-space generative approach to a Latent-Space Predictive approach to further reduce computational overhead. The metrics below represent the finalized baseline ceiling of the generative TMPN model).*

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

---

## 6. References
[1] Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.  
[2] Xie, E., et al. (2021). *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*. NeurIPS.  
[3] Sudre, C. H., et al. (2017). *Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations*. DLMIA.

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