# WhoFi: Deep Person Re-Identification via Wi-Fi Channel Signal Encoding

This project is a PyTorch-based re-implementation of the deep learning pipeline described in the paper "WhoFi: Deep Person Re-Identification via Wi-Fi Channel Signal Encoding" (arXiv:2507.12869v2). The goal is to provide a robust and reproducible baseline for Wi-Fi-based biometric research.

## System Overview

The WhoFi system processes Wi-Fi Channel State Information (CSI) data to generate unique biometric signatures for individuals. The pipeline consists of four main stages:

1.  **Data Ingestion & Pre-processing:** Loading and cleaning CSI data from the NTU-Fi Human Identification (HID) dataset.
2.  **Data Augmentation:** Artificially expanding the training dataset to improve model robustness.
3.  **Signature Generation (Model):** A Transformer Encoder processes the sequential CSI data to produce a fixed-size embedding vector (the signature).
4.  **Training & Inference:** The model is trained to distinguish between different individuals using an in-batch negative loss function.

## Technology Stack

*   **Language:** Python
*   **Deep Learning Framework:** PyTorch
*   **Scientific Computing:** NumPy

This project uses `uv` for Python environment and package management.
