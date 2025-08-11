# **Project Tasks & Milestones: WhoFi Re-implementation**

This document outlines the development plan for the WhoFi project, breaking it down into sequential milestones and specific tasks.

### **Milestone 1: Project Foundation & Data Pipeline**

**Goal:** Set up the development environment and create a robust data loading and processing pipeline for the NTU-Fi dataset.

* [x] **Task 1.1: Environment Setup**
  * [x] Create a README.md with a project overview.
  * [x] Set up venv isolated Python environment.
  * [x] Install core dependencies: pytorch, numpy.
* [x] **Task 1.2: Data Ingestion**
  * [x] Write a script to download and unpack the NTU-Fi HID dataset from a google drive location.
    * the full dataset is at google drive id=1IKTg5M7vDdZPnt6649i2Z3jiR4Fivsy1
    * a limited (only 1 out of 15 parts) version of the dataset is at google drive id=1ZnDpn9CxoT0hrLSeGV1hcif2DGjg6Xlk
    * per default, only download the limited dataset because of limited storage space
  * [x] Implement a PyTorch Dataset class (NTUFiDataset) to load samples and labels.
* [x] **Task 1.3: Data Processing & Preparation**
  * [x] Implement the data flattening logic ((3, 114, P) \-\> (P, 342)) within the Dataset class.
  * [x] Implement the optional Hampel filter for amplitude cleaning.
  * [x] Implement the data augmentation engine (Gaussian Noise, Scaling, Time Shift).
  * [x] Add configuration flags to enable/disable filtering and augmentation.

### **Milestone 2: Core Model Architecture**

**Goal:** Implement the complete WhoFi Transformer model in PyTorch.

* [x] **Task 2.1: Implement Transformer Components**
  * [x] Create the PositionalEncoding module.
  * [x] Build the main TransformerEncoder layer using PyTorch's built-in modules.
* [x] **Task 2.2: Implement Signature Module**
  * [x] Add the final fully connected layer (signature\_fc) to map encoder output to the signature dimension.
  * [x] Add the L2-normalization step to the forward pass.
* [x] **Task 2.3: Assemble the Full Model (WhoFiTransformer)**
  * [x] Combine the input projection, positional encoding, encoder, and signature module into a single nn.Module.
  * [x] Write unit tests to ensure the model accepts correct input shapes and produces the expected output shape.

### **Milestone 3: Training & Evaluation Pipeline**

**Goal:** Develop the complete training and evaluation loop.

* \[ \] **Task 3.1: Implement Custom Sampler**
  * \[ \] Create the InBatchSampler class to generate query/gallery pairs for the loss function.
* \[ \] **Task 3.2: Implement Loss and Metrics**
  * \[ \] Write the logic for the in-batch negative loss using the similarity matrix and nn.CrossEntropyLoss.
  * \[ \] Create a function calculate\_metrics to compute Rank-1, Rank-3, Rank-5, and mAP.
* \[ \] **Task 3.3: Build Training & Evaluation Scripts**
  * \[ \] Create train.py with the main training loop, optimizer, and learning rate scheduler.
  * \[ \] Integrate the evaluation function to run at the end of each epoch.
  * \[ \] Add logging to track loss and metrics over time (e.g., to console or a log file).
  * \[ \] Implement checkpointing to save the best performing model weights.

### **Milestone 4: Benchmarking & Finalization**

**Goal:** Replicate the paper's results, perform ablation studies, and finalize the codebase.

* \[ \] **Task 4.1: Replicate Paper Results**
  * \[ \] Run the full training script on the NTU-Fi dataset with the optimal configuration (1-layer Transformer, no filtering/augmentation).
  * \[ \] Compare results against the paper's benchmark (Rank-1: 95.5%, mAP: 88.4%).
  * \[ \] Tune hyperparameters if necessary to match performance.
* \[ \] **Task 4.2: Conduct Ablation Studies**
  * \[ \] Run experiments with and without amplitude filtering.
  * \[ \] Run experiments with and without data augmentation.
  * \[ \] Run experiments with different packet sizes (e.g., 100, 500, 1000).
  * \[ \] Document all results in a new results.md file.
* \[ \] **Task 4.3: Code Refactoring & Cleanup**
  * \[ \] Refactor code for clarity, efficiency, and adherence to PEP 8 standards.
  * \[ \] Add comprehensive docstrings and comments to all functions and classes.
  * \[ \] Create a requirements.txt file.
* \[ \] **Task 4.4: Finalize Documentation**
  * \[ \] Update README.md with final results and detailed instructions on how to set up the environment, train the model, and run the evaluation.
  * \[ \] Create a license file (e.g., MIT, Apache 2.0).
