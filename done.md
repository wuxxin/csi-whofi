### Task: 1.1 Environment Setup (2025-08-11)

- **Action:** Completed the initial project environment setup.
- **Changes:**
  - Created `README.md` with a project overview.
  - Initialized a `uv` virtual environment in `.venv`.
  - Installed `pytorch` (CPU) and `numpy` dependencies.
  - Updated `tasks.md` to mark the task as complete.
- **Outcome:** The project foundation is now in place, ready for the data ingestion pipeline development.

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

### Task: 4.0 Finalization (2025-08-11)
- **Action:** Finalized the project by cleaning up the codebase, improving documentation, and adding final required files.
- **Changes:**
  - Created a standalone `evaluate.py` script for model evaluation.
  - Updated `README.md` with detailed setup, training, and evaluation instructions.
  - Added a note to the README clarifying the limitations of the default dataset for full training.
  - Added a standard Python `.gitignore` configuration.
  - Generated a `requirements.txt` file for the project.
  - Added an MIT `LICENSE` file.
  - Performed a final code review and cleanup.
- **Outcome:** The project is now complete and fully documented. While benchmarking could not be performed due to dataset limitations, the codebase is a robust and reproducible implementation of the WhoFi paper.

### Task: 4.1 Initial Training Run (2025-08-12)
- **Action:** Ran the initial training script to verify the pipeline.
- **Changes:**
  - Installed all dependencies using `uv`.
  - Downloaded and extracted the full dataset.
  - Encountered and resolved a CUDA-related PyTorch installation issue by reinstalling the CPU-only version.
  - Encountered and resolved an out-of-memory error during training by reducing the batch size.
  - Successfully ran the training script for 2 epochs.
  - Updated `tasks.md` to reflect the completion of the initial training run.
- **Outcome:** The training pipeline is confirmed to be working on the full dataset. The system is now ready for full-scale training and benchmarking.
