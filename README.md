# RGBDT-Predictive: JEPA-Inspired Agricultural Segmentation

An advanced, physics-aware predictive architecture for 5-channel multi-modal sensor fusion (RGB, Depth, Thermal). Designed for high-precision agricultural anomaly detection (e.g., fruit freshness, bruising, rot) using the MM5 dataset.

## 🧠 Architectural Paradigm Shift
This repository represents a fundamental upgrade from early-fusion Convolutional Neural Networks to a late-fusion Vision Transformer (ViT) paradigm. Inspired by Joint Embedding Predictive Architectures (JEPA), the model learns thermodynamic physics rather than memorizing pixel textures.

1.  **Independent Modality Encoders:** Dual `MiT-B0` (SegFormer) backbones process RGB-D geometry and Thermal data in strict isolation. This prevents the "spatial entanglement" that causes early-fusion models to fail under sensor parallax and mechanical vibration.
2.  **Cross-Modal Attention (GCMA):** The pipeline fuses the independent streams at the pre-logit stage using Global Context Modality Attention. The Transformer mathematically correlates thermodynamic signatures (Thermal Keys/Values) with physical structures (RGB-D Queries) without requiring pixel-perfect Cartesian alignment.
3.  **Predictive Learning:** The pre-training objective forces the network to predict masked thermal patches using only surrounding RGB-D geometry, forcing the weights to learn the physics of cellular decay and evaporative cooling.

## 🗄️ Data Lineage
Training is strictly bound to the **MM5 dataset**. The active dataset pointer is located at `dataset/MM5.dvc`.