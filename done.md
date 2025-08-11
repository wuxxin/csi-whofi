### Task: 1.1 Environment Setup (2025-08-11)

- **Action:** Completed the initial project environment setup.
- **Changes:**
  - Created `README.md` with a project overview.
  - Initialized a `uv` virtual environment in `.venv`.
  - Installed `pytorch` (CPU) and `numpy` dependencies.
  - Updated `tasks.md` to mark the task as complete.
- **Outcome:** The project foundation is now in place, ready for the data ingestion pipeline development.
- **Submission:** Changes were submitted for review.

### Task: 1.2 & 1.3 Data Pipeline Implementation (2025-08-11)
- **Action:** Developed the complete data ingestion and processing pipeline for the NTU-Fi dataset.
- **Changes:**
  - Updated `scripts/download_dataset.py` to use the limited dataset ID by default.
  - Created the `who_fi` package directory.
  - Implemented the `NTUFiDataset` class in `who_fi/data.py`, which handles data loading, discovery, and processing.
  - Integrated the required data flattening logic.
  - Added optional Hampel filtering for outlier removal (`hampel` dependency added).
  - Implemented optional data augmentation techniques (Gaussian noise, scaling, time-shift).
  - Updated `pyproject.toml` to handle package discovery correctly and add new dependencies.
- **Outcome:** The project now has a fully functional data pipeline, capable of loading raw CSI data and preparing it for model training with optional cleaning and augmentation.

### Task: 2.1, 2.2 & 2.3 Core Model Implementation (2025-08-11)
- **Action:** Implemented the core WhoFiTransformer model architecture in PyTorch.
- **Changes:**
  - Created `who_fi/model.py` to house the model components.
  - Implemented a standard `PositionalEncoding` module.
  - Built the `WhoFiTransformer` class, which includes an input projection layer, the positional encoding, a `TransformerEncoder`, and a final signature layer with L2 normalization.
  - Created `tests/test_model.py` with unit tests for the model.
  - The tests verify model instantiation, correct output shape, and proper L2 normalization of the output signatures.
- **Outcome:** The project now has a complete and tested implementation of the Transformer model, ready for the training and evaluation pipeline.

### Task: 3.1, 3.2 & 3.3 Training & Evaluation Pipeline (2025-08-11)
- **Action:** Implemented the complete training and evaluation pipeline.
- **Changes:**
  - Created `who_fi/sampler.py` with an `InBatchSampler` for metric learning.
  - Created `who_fi/loss.py` with the `InBatchNegativeLoss` function.
  - Created `who_fi/metrics.py` with a function to calculate Rank-k accuracy and mAP.
  - Developed the main `train.py` script, which includes:
    - Argument parsing for hyperparameters.
    - Instantiation of all components (dataset, model, loss, etc.).
    - A full training and evaluation loop.
    - Logging of results and checkpointing for the best model.
  - Added unit tests for the sampler, loss, and metrics modules.
- **Outcome:** The project is now capable of end-to-end training and evaluation of the WhoFi model.
