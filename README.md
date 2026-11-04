# Spacecraft Anomaly Detection

A machine learning project focused on detecting anomalous behavior in spacecraft telemetry data using NASA's SMAP and MSL datasets.

## 1. Project Overview

Spacecraft continuously generate telemetry data through onboard sensors and subsystems. This telemetry contains information about the operational state and behavior of different spacecraft components.

Unexpected deviations in telemetry patterns may indicate hardware failures, sensor malfunctions, subsystem abnormalities, or unusual operational conditions.

The objective of this project is to develop a machine learning-based anomaly detection system that analyzes spacecraft telemetry time-series data, identifies abnormal patterns, and evaluates detection performance using labeled anomalies.

The project explores traditional machine learning and deep learning approaches for time-series anomaly detection.

## 2. Problem Statement

Spacecraft telemetry is inherently multivariate and time-dependent. Different sensors produce measurements that vary over time, and their normal operating behavior may change depending on spacecraft operating conditions.

Detecting anomalies is challenging because:

- Normal telemetry behavior can be highly dynamic.
- Anomalies may be rare compared to normal observations.
- Abnormal patterns may develop gradually or occur suddenly.
- Different telemetry channels may have different statistical distributions.
- Anomalies may involve relationships between multiple telemetry channels.
- Labeled anomalous observations are limited.

A reliable anomaly detection pipeline must understand temporal patterns, account for variations in telemetry, and distinguish abnormal behavior from normal operational fluctuations.

## 3. Project Objectives

1. Acquire and understand NASA SMAP and MSL telemetry datasets.
2. Explore the structure and statistical properties of telemetry time series.
3. Visualize normal and anomalous telemetry patterns.
4. Investigate missing values, distributions, and data quality.
5. Develop suitable preprocessing and feature engineering pipelines.
6. Implement baseline anomaly detection algorithms.
7. Investigate deep learning approaches such as LSTM-based models and Autoencoders.
8. Evaluate anomaly detection performance using appropriate metrics.
9. Compare different models and their ability to identify anomalous behavior.
10. Build a reproducible and modular anomaly detection pipeline.

## 4. Dataset Information

This project uses two spacecraft telemetry datasets released by NASA.

### 4.1 SMAP — Soil Moisture Active Passive

SMAP is a NASA Earth-observation satellite mission.

The SMAP telemetry dataset contains spacecraft sensor measurements and labeled anomalous intervals associated with spacecraft telemetry channels.

### 4.2 MSL — Mars Science Laboratory

MSL refers to the Mars Science Laboratory mission, which operates the Curiosity rover on Mars.

The MSL telemetry dataset contains operational measurements from rover subsystems and associated anomalous intervals.

### Dataset Characteristics

- Time-series telemetry measurements.
- Multiple sensor channels.
- Channel-specific data.
- Training and testing data.
- Labeled anomalous intervals.
- Temporal dependencies between observations.

**Note:** Dataset-specific formats, channel dimensions, sampling intervals, and label representations will be documented during Phase 1 after inspecting the actual files.

## 5. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Primary programming language |
| NumPy | Numerical computing |
| Pandas | Data manipulation |
| Matplotlib | Data visualization |
| Scikit-learn | Preprocessing, baseline ML and evaluation |
| SciPy | Scientific computing |
| Jupyter / VS Code | Interactive experimentation |
| Git | Version control |

Potential future additions:

- PyTorch or TensorFlow — Deep learning.
- Seaborn — Statistical visualization.
- MLflow — Experiment tracking.
- Streamlit — Interactive anomaly detection dashboard.

These will be introduced only when required.

## 6. Project Directory Structure

```text
space_anamoly_detection/
│
├── .gitignore
├── requirements.txt
├── tasks.md
├── README.md
│
├── configs/
│
├── artifacts/
│
├── tests/
│
├── data/
│   ├── labeled_anomalies.csv
│   ├── train/
│   ├── test/
│   │
│   └── 2018-05-19_15.00.10/
│       ├── models/
│       ├── smoothed_errors/
│       └── y_hat/
│
├── reports/
│   └── figures/
│
├── src/
│   ├── preprocessing/
│   ├── visualization/
│   ├── features/
│   ├── models/
│   └── evaluation/
│
└── .venv/
```

### Directory Responsibilities

| Directory/File | Responsibility |
|---|---|
| configs/ | Configuration files and experiment parameters |
| artifacts/ | Saved models and generated artifacts |
| tests/ | Unit and integration tests |
| data/ | Dataset files and associated metadata |
| data/train/ | Training telemetry data |
| data/test/ | Testing telemetry data |
| data/labeled_anomalies.csv | Anomaly label information |
| reports/figures/ | Generated plots and visualizations |
| src/preprocessing/ | Data cleaning, normalization and transformations |
| src/visualization/ | Telemetry and anomaly visualization |
| src/features/ | Feature extraction and engineering |
| src/models/ | Anomaly detection algorithms |
| src/evaluation/ | Model evaluation and performance metrics |
| requirements.txt | Python dependencies |
| tasks.md | Development tasks and progress |
| .venv/ | Local Python virtual environment |

**Note:** The `2018-05-19_15.00.10/` directory currently contains existing model-related outputs. Its exact provenance and relationship to the current experiment will be investigated during dataset inspection.

The `.venv/` directory is local and should not be committed to GitHub.

## 7. Development Roadmap

The project will be developed incrementally.

| Phase | Description | Status |
|---|---|---|
| Phase 0 | Project setup and environment preparation | In progress |
| Phase 1 | Dataset acquisition and exploration | Pending |
| Phase 2 | Data preprocessing and preparation | Pending |
| Phase 3 | Statistical analysis and feature engineering | Pending |
| Phase 4 | Baseline anomaly detection models | Pending |
| Phase 5 | Deep learning models (LSTM, Autoencoder) | Pending |
| Phase 6 | Model evaluation and comparison | Pending |
| Phase 7 | Optimization and experimentation | Pending |
| Phase 8 | Final pipeline and documentation | Pending |

## 8. Installation and Setup

### Prerequisites

- Python 3.10 or 3.11
- Git
- VS Code or another Python IDE

### Step 1: Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd space_anamoly_detection
```

### Step 2: Create virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Ubuntu/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify installation

```bash
python -c "import numpy, pandas, matplotlib, sklearn, ipykernel; print('Environment setup successful!')"
```

## 9. Current Development Status

**Current Phase: Phase 0 — Project Setup & Environment Preparation**

Completed or established:

- Initial project directory structure.
- Python virtual environment.
- Initial dependency selection.
- Project task tracking through `tasks.md`.

Pending:

- Finalize README documentation.
- Verify all dependencies.
- Complete environment verification.
- Commit and push initial project setup to GitHub.

**Next milestone:** Begin Phase 1 — NASA SMAP/MSL dataset acquisition and exploratory data analysis.

## 10. Development Principles

This project follows these engineering principles:

- Understand the data before implementing models.
- Keep preprocessing separate from model implementation.
- Avoid data leakage between training and testing.
- Use reproducible experiments and configurations.
- Evaluate models using appropriate anomaly detection metrics.
- Keep code modular, testable, and maintainable.
- Document experimental findings and design decisions.

---

**Project Goal:** Build an end-to-end, reproducible spacecraft telemetry anomaly detection system grounded in sound machine learning principles and practical engineering.