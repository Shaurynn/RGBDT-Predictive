# TriModal Perception Architectures for Structural Defect Detection: The MM-JEPA Paradigm

Abstract
Structural defect detection in industrial and agricultural environments requires robust multimodal integration. In this repository, we document the evolution of spatial perception engines for RGB, Depth, and Thermal sensors—captured via specialized Baumer GigE sensor arrays—built upon a patched `mit_b1` Vision Transformer backbone. Recognizing the theoretical limitations of pixel-space generative networks, this architecture introduces the Multimodal Joint-Embedding Predictive Architecture (MM-JEPA). By decoupling representation learning from downstream segmentation, the framework utilizes pure self-supervised prediction in the latent space to build a robust structural foundation model. This allows for rapid supervised fine-tuning and deployment on edge computational devices.

---

## 1. Introduction

The integration of high-resolution, unaligned multimodal sensors poses a unique challenge. Early iterations of this architecture utilized generative pixel-space decoders for thermal representation. However, reconstructing pixel-space values forces the network to map irrelevant high-frequency noise, leading to mathematical instability. To address this, we transitioned to Latent Space Prediction, building upon recent advancements in self-supervised architectures (e.g., Meta AI's I-JEPA). This repository completely deprecates the legacy generative network in favor of a 2-Phase MM-JEPA paradigm, executing genuine target-selective spatial inference via pure mathematical modeling.

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

* **Unified Same-Modal Inference:** The Context and Target Encoders operate on identical architectural modalities. The Baumer GigE sensor streams (RGB, Depth, Thermal) are fused into a unified 5-channel block. The network is forced to predict masked properties from the *same* multimodal manifold.
* **The Information Bottleneck:** The Predictor module is explicitly isolated from the block mask tensor. It infers the missing spatial data relying strictly on the context latent space and pure additive 2D Positional Encodings, guaranteeing that the network does not receive the "answer key" prior to inference.
* **EMA Momentum Teacher:** The framework prevents Representation Collapse without relying on loss regularizers. The Target Encoder is governed strictly by an Exponential Moving Average (EMA) update schedule, maintaining a smooth, historically stable target manifold.

$$\theta_{target} \leftarrow \tau \theta_{target} + (1 - \tau) \theta_{context}$$

### 4.2 The Latent Predictive Objective

The Spatial Latent Predictor minimizes the Mean Squared Error (MSE) strictly within the coordinate bounds of the masked patches.

$$\mathcal{L}_{JEPA} = \frac{1}{N_{mask}} \sum_{i \in mask} \| s_{\phi}(z_{context})_i - z_{target, i} \|_2^2$$

---

## 5. Phase 2: Supervised Semantic Fine-Tuning

Following deep convergence in Phase 1, the Context Encoder possesses a pre-trained understanding of structural defects and thermodynamic boundaries—learned entirely without human annotation. Phase 2 transitions to the task of semantic segmentation.

### 5.1 Downstream Architecture & Transfer Learning

The pre-trained weights from Phase 1 (`jepa_context_encoder.pt`) are injected into the downstream network. The encoder backbone is frozen (Linear Probing), and a lightweight multi-layer perceptron (MLP) decoding head is attached to process the downstream semantic annotations using an optimized Focal Dice objective.

### 5.2 Deep Convergence & Evaluation [PENDING]

The downstream execution engine (`train_downstream.py`) utilizes a multi-phase state machine governed by Bayesian optimization (Optuna).

*Note: The architecture is currently undergoing empirical evaluation across the `mit` Vision Transformer series (`mit_b1` through `mit_b5`). Quantitative milestones, including Base Validation mIoU, Test-Time Augmentation (TTA) robustness, and Expected Calibration Error (ECE), will be populated upon the conclusion of the training cycle.*

### 5.3 Mitigating Imbalance and Asymptotic Limits (DCW & KD)

Industrial defect datasets exhibit extreme class imbalance. To overcome this, the downstream engine utilizes a Dynamic Class-Weighting Schedule (DCW). By tracking an Exponential Moving Average of the validation IoU, the downstream Dice penalty is exponentially scaled on the fly for lagging minority classes.

To break representational capacity ceilings during edge deployment, the pipeline integrates a Knowledge Distillation (KD) engine. By forcing the lightweight `mit_b1` Student to minimize the Kullback-Leibler (KL) Divergence against a massive Teacher's soft probabilities ("Dark Knowledge"), the edge-deployed model inherits advanced stochastic noise suppression.

$$\mathcal{L}_{KD} = \tau^2 \text{KL}\left( \sigma\left(\frac{z_{student}}{\tau}\right) \parallel \sigma\left(\frac{z_{teacher}}{\tau}\right) \right)$$

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