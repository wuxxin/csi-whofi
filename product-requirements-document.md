# **Product Requirements Document: WhoFi Re-implementation**

## **1\. Introduction & Objective**

This document outlines the requirements for the reimplementation of **WhoFi**, a deep learning pipeline for person re-identification (Re-ID) using Wi-Fi Channel State Information (CSI). The primary objective is to create a functional software package that replicates the Transformer-based model, data processing, and training methodology described in the paper "WhoFi: Deep Person Re-Identification via Wi-Fi Channel Signal Encoding" (arXiv:2507.12869v2).

The goal is to provide a robust and reproducible baseline for Wi-Fi-based biometric research.

## **2\. System Overview**

The WhoFi system processes Wi-Fi CSI data to generate unique biometric signatures for individuals. The pipeline consists of four main stages:

1. **Data Ingestion & Pre-processing:** Loading CSI data and cleaning it to remove noise.
2. **Data Augmentation:** Artificially expanding the training dataset to improve model robustness.
3. **Signature Generation (Model):** A deep neural network, specifically a Transformer Encoder, processes the sequential CSI data to produce a fixed-size embedding vector (the signature).
4. **Training & Inference:** The model is trained to distinguish between different individuals using an in-batch negative loss function. During inference, it compares a query signature against a gallery of known signatures to find a match.

## **3\. Functional Requirements**

### **3.1. Data Management**

| ID | Requirement | Details | Priority |
| :---- | :---- | :---- | :---- |
| FR-01 | **Dataset Compatibility** | The system must be able to load and process the **NTU-Fi Human Identification (HID) dataset**. | Must-have |
| FR-02 | **Input Data Shape** | The system must handle input samples with the dimensionality specified in the paper: 3 (antennas) x 114 (subcarriers) x 2000 (packets). | Must-have |
| FR-03 | **Data Flattening** | Before feeding into the model, the input data (3 x 114 x P) should be flattened along the antenna and subcarrier dimensions to create a sequence of shape (P x 342), where P is the number of packets. | Must-have |

### **3.2. Pre-processing & Augmentation**

| ID | Requirement | Details | Priority |
| :---- | :---- | :---- | :---- |
| FR-04 | **Amplitude Filtering (Optional)** | Implement the Hampel filter to identify and remove outliers from the CSI amplitude data. This should be a configurable step, as the paper notes Transformers perform better without it. | Should-have |
| FR-05 | **Data Augmentation (Optional)** | The system should provide optional data augmentation techniques to be applied during training: \<br\> \- **Gaussian Noise:** Add noise N(0, 0.02^2). \<br\> \- **Scaling:** Scale amplitude by a random factor in \[0.9, 1.1\]. \<br\> \- **Time Shift:** Shift the sequence by a random integer t' in \[-5, 5\]. | Should-have |

### **3.3. Model Architecture**

| ID | Requirement | Details | Priority |
| :---- | :---- | :---- | :---- |
| FR-06 | **Transformer Encoder** | Implement the Transformer encoder module. It should be configurable in terms of the number of layers, heads, and dropout probability. It must accept sequential input and produce a fixed-size encoding. | Must-have |
| FR-07 | **Positional Encoding** | Standard sinusoidal positional encodings must be added to the input embeddings before they are passed to the Transformer encoder. | Must-have |
| FR-08 | **Signature Module** | Implement the signature module, which consists of a final linear layer that maps the encoder's output to the signature space, followed by an l2-normalization layer. | Must-have |

### **3.4. Training & Evaluation**

| ID | Requirement | Details | Priority |
| :---- | :---- | :---- | :---- |
| FR-09 | **Custom Batch Sampling** | A custom batch sampler must be used to create batches where each batch contains a query list B\_q and a gallery list B\_g of size N. The i-th sample in B\_q and B\_g must belong to the same person. | Must-have |
| FR-10 | **In-Batch Negative Loss** | The model must be trained using an in-batch negative loss. This involves computing a cosine similarity matrix between query and gallery signatures and applying a cross-entropy loss to push the model towards an identity matrix. | Must-have |
| FR-11 | **Evaluation Metrics** | The system must compute and report **Rank-k accuracy (k=1, 3, 5\)** and **mean Average Precision (mAP)** to evaluate re-identification performance. | Must-have |
| FR-12 | **Hyperparameter Config** | All key hyperparameters (learning rate, batch size, epochs, optimizer settings, scheduler settings) must be easily configurable. | Must-have |

## **4\. Non-Functional Requirements**

| ID | Requirement | Details | Priority |
| :---- | :---- | :---- | :---- |
| NFR-01 | **Performance Benchmark** | The reimplemented Transformer model should achieve performance metrics comparable to those reported in the paper (Rank-1: **95.5%**, mAP: **88.4%**) on the NTU-Fi test set. | Must-have |
| NFR-02 | **Technology Stack** | The primary implementation must use **Python** and the **PyTorch** deep learning framework. | Must-have |
| NFR-03 | **Code Quality** | The code must be well-documented, with clear comments explaining the implementation of each component, especially the model architecture and the loss function. | Must-have |
| NFR-04 | **Reproducibility** | The training script should allow for setting a random seed to ensure experiments are reproducible. | Should-have |

## **5\. Out of Scope**

The following features will not be included in this initial implementation:

* A graphical user interface (GUI).
* Real-time data capture from Wi-Fi hardware.
* Implementation of the LSTM and Bi-LSTM encoders.
* Phase sanitization (as the public NTU-Fi dataset only contains amplitude data).
* Deployment to a production environment.