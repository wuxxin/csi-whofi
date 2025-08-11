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
