# **Planning Document: WhoFi Re-implementation Project**

## **1\. Vision & Mission**

### **Vision**

To create a world where human identification can be performed accurately and reliably without compromising individual privacy, using ubiquitous technologies like Wi-Fi.

### **Mission**

Our mission is to develop and open-source a faithful and high-performance reimplementation of the file "research-paper.md" describing the **WhoFi** research project.

This project will serve as a foundational, reproducible benchmark for the academic and research communities, accelerating innovation in privacy-preserving biometric sensing and person re-identification.

## **2\. System Architecture**

The WhoFi system is designed as a modular pipeline that transforms raw Wi-Fi signal data into a unique biometric identifier. The architecture is broken down into four primary layers:

### **Layer 1: Data Ingestion & Preparation**

* **Source:** The system will ingest data from the **NTU-Fi Human Identification (HID) dataset**.
* **Process:**
  1. Load the raw sample data, which has a shape of (3, 114, P), where P is the number of packets.
  2. Flatten the data along the antenna and subcarrier dimensions to create a sequence of shape (P, 342). This prepares the data for the sequence-based Transformer model.

### **Layer 2: Pre-processing & Augmentation (Optional)**

* **Purpose:** To clean the data and artificially expand the dataset to improve model robustness. This layer is configurable and can be bypassed, as the original paper found the Transformer performed best on unfiltered data.
* **Components:**
  * **Hampel Filter:** An optional filter to remove outliers from the amplitude signal.
  * **Augmentation Engine:** Applies a chain of random transformations during training, including adding Gaussian noise, random scaling, and time-shifting the sequence.

### **Layer 3: Core Model \- Signature Generation**

* **Engine:** A PyTorch-based Deep Neural Network.

* **Architecture:**
  1. **Input Projection & Positional Encoding:** The input sequence is first projected into the model's embedding space and infused with sinusoidal positional information to retain temporal order.
  2. **Transformer Encoder:** The core of the model. A single-layer Transformer with multi-head self-attention processes the sequence, capturing long-range dependencies and discriminative patterns in the Wi-Fi signal.
  3. **Signature Module:** The output from the Transformer is passed to a final linear layer, which maps it to the target signature dimension (e.g., 128). This is followed by an L2-normalization step to ensure all signatures lie on a unit hypersphere, simplifying similarity calculations.

### **Layer 4: Training & Evaluation**

* **Training Loop:**
  * **Custom Sampler:** A specialized InBatchSampler prepares batches by selecting two distinct samples for N unique individuals.
  * **Loss Function:** The model is trained using an **in-batch negative loss**. It computes a cosine similarity matrix between query and gallery signatures and optimizes a cross-entropy loss to maximize the similarity of matching pairs (diagonal) and minimize all others (off-diagonal).
* **Evaluation Engine:**
  * **Metrics:** The model's performance is benchmarked using standard re-identification metrics: **Rank-k Accuracy (k=1, 3, 5\)** and **mean Average Precision (mAP)**.

## **3\. Technology Stack & Tools**

| Component | Technology | Purpose |
| :---- | :---- | :---- |
| **Language** | Python | Primary development language. |
| **Deep Learning** | PyTorch | Core framework for model building, training, and inference. |
| **Scientific Computing** | NumPy | For efficient numerical operations. |

| Tool | Purpose |
| :---- | :---- |
| **Version Control** | Git & GitHub |
| **Python Environment Mgmt.** | uv |
| **Dataset** | NTU-Fi HID Dataset |
| **Documentation** | Markdown |

To work within a constraint storage space, only install and use the cpu version of pytorch,
avoid pulling in big GPU requirements like CUDA or ROCM or others.
