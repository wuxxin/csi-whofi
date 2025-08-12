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

## Setup & Usage

### 1. Environment Setup

This project uses `uv` for fast environment and package management. First, create a virtual environment and install the required dependencies.

```bash
# Create a virtual environment
uv venv

# Install dependencies (including PyTorch for CPU)
uv pip install . --extra-index-url https://download.pytorch.org/whl/cpu
```

### 2. Download the Dataset

The dataset can be downloaded and unzipped automatically by running the provided script. This will download the limited version of the NTU-Fi dataset (~200MB) and place it in the `data/` directory.

```bash
./.venv/bin/python scripts/download_dataset.py
```

### 3. Training the Model

To train the model, run the `train.py` script. The script provides several command-line arguments to configure the training process and model hyperparameters.

**Note on the Dataset:** The default download script fetches a limited version of the NTU-Fi dataset which only contains samples for a single person. While this is useful for quickly verifying that the pipeline runs, it is not suitable for meaningful training. To replicate the paper's results, you will need to download the full dataset.

**Example Training Command:**

```bash
./.venv/bin/python train.py --epochs 300 --batch_size 16 --learning_rate 0.0001
```

The script will periodically evaluate the model on the test set and save the checkpoint with the best Rank-1 accuracy to `best_model.pth`.

To see all available training options, run:
```bash
./.venv/bin/python train.py --help
```

### 4. Evaluating a Trained Model

To evaluate a trained model checkpoint, use the `evaluate.py` script. You must provide the path to the saved model weights.

**Example Evaluation Command:**

```bash
./.venv/bin/python evaluate.py --model_path best_model.pth
```

This will load the model, run it on the test set, and print the final re-identification metrics (Rank-k and mAP).

## Evaluation

The model's performance is measured using standard person re-identification metrics:

*   **Rank-k Accuracy (k=1, 3, 5):** Measures if the correct person is identified within the top-k most similar results.
*   **Mean Average Precision (mAP):** Provides a comprehensive score of the model's ability to rank the correct person higher than incorrect ones.

For a detailed explanation of these metrics and how they are calculated in this project, please see the [Evaluation Metrics Explained](explanation.md) document.
