# 🏥 MIMIC-IV Clinical Data Analysis & Prediction System

A production-ready, end-to-end machine learning pipeline for analyzing clinical data from the MIMIC-IV database and predicting in-hospital mortality.

## Overview

This project processes the [MIMIC-IV Clinical Database Demo](https://www.kaggle.com/datasets/ashishkumarak/mimic-iv-clinical-database-demo) to:

- **Clean and preprocess** patient, admission, diagnosis, and lab event data
- **Engineer features** including age at admission, length of stay, mortality flags, readmission flags, lab aggregates, and diagnosis category encodings
- **Perform exploratory data analysis** with publication-quality visualizations
- **Train and evaluate** multiple ML models for in-hospital mortality prediction
- **Serve predictions** through an interactive Streamlit dashboard

## Project Structure

```
mimic-iv-project/
├── data/                    # Raw CSV files (auto-downloaded or manual)
├── notebooks/               # Jupyter notebooks (optional exploration)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     # Data loading, cleaning, date parsing
│   ├── features.py          # Feature engineering (age, LOS, labs, diagnoses)
│   ├── train.py             # Model training (LR, RF, GB) + hyperparameter tuning
│   └── evaluate.py          # Metrics, ROC curves, confusion matrices, plots
├── app/
│   └── streamlit_app.py     # Interactive dashboard
├── outputs/
│   ├── plots/               # Generated EDA and evaluation plots
│   ├── models/              # Saved model files (.joblib)
│   ├── feature_matrix.csv   # Computed feature matrix
│   └── model_metrics.csv    # Model evaluation results
├── main.py                  # CLI entry point — runs entire pipeline
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

**Option A — Automatic download via kagglehub** (requires Kaggle credentials):

```bash
python main.py
```

**Option B — Manual download:**

1. Download from [Kaggle](https://www.kaggle.com/datasets/ashishkumarak/mimic-iv-clinical-database-demo)
2. Place `patients.csv`, `admissions.csv`, `diagnoses_icd.csv`, and `labevents.csv` into the `data/` directory
3. Run: `python main.py --data-dir data`

### 3. Run the Full Pipeline

```bash
# Default run
python main.py

# With hyperparameter tuning (slower, better results)
python main.py --tune

# Skip EDA plots
python main.py --skip-eda

# Custom paths
python main.py --data-dir /path/to/csvs --output-dir /path/to/results
```

### 4. Launch the Dashboard

```bash
streamlit run app/streamlit_app.py
```

## CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--data-dir` | Path to CSV data directory | `data` |
| `--output-dir` | Path to save outputs | `outputs` |
| `--tune` | Enable GridSearchCV hyperparameter tuning | `False` |
| `--skip-eda` | Skip EDA plot generation | `False` |
| `--test-size` | Train/test split ratio | `0.2` |
| `--random-state` | Random seed for reproducibility | `42` |

## Features Engineered

| Feature | Description |
|---------|-------------|
| `age_at_admission` | Patient age computed from anchor year/age |
| `los_days` | Length of stay in days (admittime → dischtime) |
| `hospital_expire_flag` | Binary mortality target (1 = died in hospital) |
| `readmission_30d` | Whether readmitted within 30 days |
| `gender_numeric` | Binary gender encoding (M=1, F=0) |
| `lab_mean/std/min/max` | Overall lab value aggregates per admission |
| `glucose_mean`, etc. | Per-lab-item statistics (9 key labs) |
| `diag_Circulatory`, etc. | ICD diagnosis category counts per admission |

## Models

Three classifiers are trained with class-weight balancing:

1. **Logistic Regression** — baseline linear model with L2 regularization
2. **Random Forest** — ensemble of 200 decision trees
3. **Gradient Boosting** — sequential boosting with 200 estimators

All models are evaluated on:

- Accuracy, Precision, Recall, F1 Score
- ROC-AUC and Average Precision
- Confusion matrices and ROC curves

## Streamlit Dashboard

The interactive dashboard provides four views:

- **Overview** — Key statistics, admission types, insurance distribution
- **EDA Visualizations** — Age, gender, LOS, diagnoses, and lab value charts
- **Model Performance** — Metrics table, ROC curves, confusion matrices, feature importance
- **Predict Mortality** — Enter patient parameters and get real-time mortality risk estimates

## Output Files

After running the pipeline:

- `outputs/plots/` — PNG plots (age distribution, gender, LOS, ROC curves, etc.)
- `outputs/models/` — Serialized models (`.joblib`) + feature names + metadata
- `outputs/feature_matrix.csv` — Complete feature matrix (used by Streamlit)
- `outputs/model_metrics.csv` — Evaluation metrics for all models

## Technical Notes

- **Class imbalance** is handled via `class_weight="balanced"` in all models
- **Date de-identification**: MIMIC-IV uses anchor years; ages are capped at 91
- **Lab outliers** are clipped at 1st/99th percentiles per item before aggregation
- **Missing values** are filled with column medians (numeric) or "UNKNOWN" (categorical)

## License

This project is for educational and research purposes. The MIMIC-IV dataset is subject to its own data use agreement via PhysioNet.
