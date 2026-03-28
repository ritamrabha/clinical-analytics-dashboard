"""
preprocessing.py - Data loading, cleaning, and preprocessing for MIMIC-IV dataset.
Handles missing values, date conversions, and data type corrections.
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")


def download_dataset(data_dir: str = "data") -> str:
    """
    Download MIMIC-IV Clinical Database Demo via kagglehub.
    Falls back to local data directory if kagglehub is unavailable or fails.
    
    Returns:
        str: Path to the dataset directory containing CSV files.
    """
    os.makedirs(data_dir, exist_ok=True)

    # Check if data already exists locally
    required_files = ["patients.csv", "admissions.csv", "diagnoses_icd.csv", "labevents.csv"]
    local_files = [f for f in required_files if os.path.exists(os.path.join(data_dir, f))]
    if len(local_files) == len(required_files):
        print(f"[INFO] All dataset files found in '{data_dir}'. Skipping download.")
        return data_dir

    try:
        import kagglehub
        print("[INFO] Downloading MIMIC-IV dataset via kagglehub...")
        path = kagglehub.dataset_download("ashishkumarak/mimic-iv-clinical-database-demo")
        print(f"[INFO] Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"[WARNING] kagglehub download failed: {e}")
        print(f"[INFO] Please place CSV files manually in '{data_dir}/' directory.")
        return data_dir


def find_csv(base_path: str, filename: str) -> str:
    """Recursively search for a CSV file in the base path."""
    for root, dirs, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    return os.path.join(base_path, filename)


def load_data(data_path: str) -> dict:
    """
    Load all MIMIC-IV CSV files into DataFrames.

    Args:
        data_path: Root directory containing CSV files (possibly nested).

    Returns:
        dict: Dictionary with keys 'patients', 'admissions', 'diagnoses', 'labevents'.
    """
    print("[INFO] Loading CSV files...")

    files = {
        "patients": "patients.csv",
        "admissions": "admissions.csv",
        "diagnoses": "diagnoses_icd.csv",
        "labevents": "labevents.csv",
    }

    data = {}
    for key, fname in files.items():
        filepath = find_csv(data_path, fname)
        if os.path.exists(filepath):
            # Use low_memory=False for labevents which can have mixed types
            data[key] = pd.read_csv(filepath, low_memory=False)
            print(f"  ✓ Loaded {key}: {data[key].shape[0]:,} rows × {data[key].shape[1]} cols")
        else:
            print(f"  ✗ File not found: {filepath}")
            data[key] = pd.DataFrame()

    return data


def clean_patients(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the patients table."""
    df = df.copy()

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    # Parse date columns
    date_cols = ["dod"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Standardize gender
    if "gender" in df.columns:
        df["gender"] = df["gender"].str.upper().str.strip()

    # Parse anchor_year as integer
    if "anchor_year" in df.columns:
        df["anchor_year"] = pd.to_numeric(df["anchor_year"], errors="coerce")
    if "anchor_age" in df.columns:
        df["anchor_age"] = pd.to_numeric(df["anchor_age"], errors="coerce")

    print(f"[INFO] Cleaned patients: {df.shape[0]:,} rows")
    return df


def clean_admissions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the admissions table."""
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    # Parse datetime columns
    datetime_cols = ["admittime", "dischtime", "deathtime", "edregtime", "edouttime"]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Fill categorical missing values
    cat_cols = ["admission_type", "admission_location", "discharge_location",
                "insurance", "language", "marital_status", "race"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN")

    # Remove duplicate admissions
    if "hadm_id" in df.columns:
        df = df.drop_duplicates(subset=["hadm_id"], keep="first")

    print(f"[INFO] Cleaned admissions: {df.shape[0]:,} rows")
    return df


def clean_diagnoses(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the diagnoses table."""
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    # Ensure icd_code is string
    if "icd_code" in df.columns:
        df["icd_code"] = df["icd_code"].astype(str).str.strip()
    if "icd_version" in df.columns:
        df["icd_version"] = pd.to_numeric(df["icd_version"], errors="coerce")

    # Drop rows with missing essential columns
    df = df.dropna(subset=["subject_id", "hadm_id", "icd_code"])

    print(f"[INFO] Cleaned diagnoses: {df.shape[0]:,} rows")
    return df


def clean_labevents(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the labevents table."""
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    # Parse datetime
    if "charttime" in df.columns:
        df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")

    # Ensure numeric value
    if "valuenum" in df.columns:
        df["valuenum"] = pd.to_numeric(df["valuenum"], errors="coerce")

    # Drop rows without a numeric value (unusable for ML)
    df = df.dropna(subset=["valuenum"])

    # Remove extreme outliers (beyond 1st and 99th percentile per itemid)
    if "itemid" in df.columns and "valuenum" in df.columns:
        # Only clip the most common lab items to save time
        top_items = df["itemid"].value_counts().head(50).index
        mask = df["itemid"].isin(top_items)
        clipped = df.loc[mask].copy()
        bounds = clipped.groupby("itemid")["valuenum"].quantile([0.01, 0.99]).unstack()
        for item_id in bounds.index:
            lo, hi = bounds.loc[item_id, 0.01], bounds.loc[item_id, 0.99]
            idx = df[(df["itemid"] == item_id) & ((df["valuenum"] < lo) | (df["valuenum"] > hi))].index
            df.loc[idx, "valuenum"] = np.nan
        df = df.dropna(subset=["valuenum"])

    print(f"[INFO] Cleaned labevents: {df.shape[0]:,} rows")
    return df


def preprocess_all(data_path: str) -> dict:
    """
    Full preprocessing pipeline: load → clean all tables.

    Args:
        data_path: Path to directory containing CSV files.

    Returns:
        dict: Dictionary of cleaned DataFrames.
    """
    raw = load_data(data_path)

    cleaned = {
        "patients": clean_patients(raw["patients"]),
        "admissions": clean_admissions(raw["admissions"]),
        "diagnoses": clean_diagnoses(raw["diagnoses"]),
        "labevents": clean_labevents(raw["labevents"]),
    }

    return cleaned
