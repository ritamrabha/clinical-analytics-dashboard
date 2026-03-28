"""
features.py - Feature engineering for MIMIC-IV mortality prediction.
Computes derived features: age, LOS, mortality flag, readmission flag,
lab aggregates, and diagnosis category encoding.
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


# ─── ICD-9/10 top-level category mapping ────────────────────────────────────

ICD9_CATEGORIES = {
    "001-139": "Infectious",
    "140-239": "Neoplasms",
    "240-279": "Endocrine",
    "280-289": "Blood",
    "290-319": "Mental",
    "320-389": "Nervous",
    "390-459": "Circulatory",
    "460-519": "Respiratory",
    "520-579": "Digestive",
    "580-629": "Genitourinary",
    "630-679": "Pregnancy",
    "680-709": "Skin",
    "710-739": "Musculoskeletal",
    "740-759": "Congenital",
    "760-779": "Perinatal",
    "780-799": "Symptoms",
    "800-999": "Injury",
    "V01-V91": "Supplementary_V",
    "E000-E999": "Supplementary_E",
}


def icd9_to_category(code: str) -> str:
    """Map an ICD-9 code to a broad diagnostic category."""
    code = str(code).strip()
    if code.startswith("E"):
        return "Injury_External"
    if code.startswith("V"):
        return "Supplementary"
    try:
        num = int(code[:3])
    except ValueError:
        return "Other"

    if num <= 139:
        return "Infectious"
    elif num <= 239:
        return "Neoplasms"
    elif num <= 279:
        return "Endocrine"
    elif num <= 289:
        return "Blood"
    elif num <= 319:
        return "Mental"
    elif num <= 389:
        return "Nervous"
    elif num <= 459:
        return "Circulatory"
    elif num <= 519:
        return "Respiratory"
    elif num <= 579:
        return "Digestive"
    elif num <= 629:
        return "Genitourinary"
    elif num <= 679:
        return "Pregnancy"
    elif num <= 709:
        return "Skin"
    elif num <= 739:
        return "Musculoskeletal"
    elif num <= 759:
        return "Congenital"
    elif num <= 779:
        return "Perinatal"
    elif num <= 799:
        return "Symptoms"
    else:
        return "Injury"


def icd10_to_category(code: str) -> str:
    """Map an ICD-10 code to a broad diagnostic category by first letter."""
    code = str(code).strip().upper()
    mapping = {
        "A": "Infectious", "B": "Infectious",
        "C": "Neoplasms", "D": "Blood_Neoplasms",
        "E": "Endocrine",
        "F": "Mental",
        "G": "Nervous",
        "H": "Eye_Ear",
        "I": "Circulatory",
        "J": "Respiratory",
        "K": "Digestive",
        "L": "Skin",
        "M": "Musculoskeletal",
        "N": "Genitourinary",
        "O": "Pregnancy",
        "P": "Perinatal",
        "Q": "Congenital",
        "R": "Symptoms",
        "S": "Injury", "T": "Injury",
        "V": "External_Causes", "W": "External_Causes",
        "X": "External_Causes", "Y": "External_Causes",
        "Z": "Health_Services",
        "U": "Special_Purpose",
    }
    if len(code) > 0 and code[0] in mapping:
        return mapping[code[0]]
    return "Other"


def compute_age_at_admission(patients: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute age at admission using anchor_age and anchor_year.
    age_at_admission = anchor_age + (admit_year - anchor_year)
    """
    merged = admissions.merge(
        patients[["subject_id", "anchor_age", "anchor_year", "gender"]],
        on="subject_id",
        how="left",
    )

    if "admittime" in merged.columns and "anchor_year" in merged.columns:
        merged["admit_year"] = merged["admittime"].dt.year
        merged["age_at_admission"] = merged["anchor_age"] + (
            merged["admit_year"] - merged["anchor_year"]
        )
        # Cap age at 91 (MIMIC de-identification threshold) and floor at 0
        merged["age_at_admission"] = merged["age_at_admission"].clip(lower=0, upper=91)
    else:
        merged["age_at_admission"] = np.nan

    return merged


def compute_length_of_stay(df: pd.DataFrame) -> pd.DataFrame:
    """Compute length of stay in days from admittime to dischtime."""
    df = df.copy()
    if "admittime" in df.columns and "dischtime" in df.columns:
        df["los_days"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 86400.0
        df["los_days"] = df["los_days"].clip(lower=0)  # No negative LOS
    else:
        df["los_days"] = np.nan
    return df


def compute_mortality_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary mortality flag.
    hospital_expire_flag = 1 if patient died during admission.
    """
    df = df.copy()
    if "hospital_expire_flag" not in df.columns:
        if "deathtime" in df.columns:
            df["hospital_expire_flag"] = (~df["deathtime"].isna()).astype(int)
        elif "discharge_location" in df.columns:
            df["hospital_expire_flag"] = (
                df["discharge_location"].str.contains("DIED|DEAD|EXPIRED|HOSPICE", case=False, na=False)
            ).astype(int)
        else:
            df["hospital_expire_flag"] = 0
    else:
        df["hospital_expire_flag"] = df["hospital_expire_flag"].fillna(0).astype(int)
    return df


def compute_readmission_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag whether a patient was readmitted within 30 days.
    Requires admissions sorted by admittime per subject_id.
    """
    df = df.copy()
    df = df.sort_values(["subject_id", "admittime"])
    df["next_admittime"] = df.groupby("subject_id")["admittime"].shift(-1)
    df["days_to_readmit"] = (df["next_admittime"] - df["dischtime"]).dt.total_seconds() / 86400.0
    df["readmission_30d"] = ((df["days_to_readmit"] >= 0) & (df["days_to_readmit"] <= 30)).astype(int)
    df.drop(columns=["next_admittime", "days_to_readmit"], inplace=True, errors="ignore")
    return df


def compute_lab_aggregates(labevents: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate lab values per admission: mean, std, min, max, count.
    Returns one row per (subject_id, hadm_id) with columns like:
      lab_mean, lab_std, lab_min, lab_max, lab_count
    And for key lab items (glucose=50931, creatinine=50912, hemoglobin=51222):
      glucose_mean, creatinine_mean, etc.
    """
    if labevents.empty or "hadm_id" not in labevents.columns:
        return pd.DataFrame()

    labs = labevents.dropna(subset=["hadm_id", "valuenum"]).copy()
    labs["hadm_id"] = labs["hadm_id"].astype(int)

    # Overall lab aggregates per admission
    overall = labs.groupby(["subject_id", "hadm_id"])["valuenum"].agg(
        lab_mean="mean",
        lab_std="std",
        lab_min="min",
        lab_max="max",
        lab_count="count",
    ).reset_index()
    overall["lab_std"] = overall["lab_std"].fillna(0)

    # Key lab items
    KEY_LABS = {
        50931: "glucose",
        50912: "creatinine",
        51222: "hemoglobin",
        50882: "bicarbonate",
        50983: "sodium",
        50971: "potassium",
        51006: "bun",
        51265: "platelet",
        51301: "wbc",
    }

    for item_id, name in KEY_LABS.items():
        item_df = labs[labs["itemid"] == item_id].groupby(["subject_id", "hadm_id"])["valuenum"].agg(
            **{f"{name}_mean": "mean", f"{name}_std": "std", f"{name}_min": "min", f"{name}_max": "max"}
        ).reset_index()
        item_df[f"{name}_std"] = item_df[f"{name}_std"].fillna(0)
        overall = overall.merge(item_df, on=["subject_id", "hadm_id"], how="left")

    print(f"[INFO] Computed lab aggregates: {overall.shape[0]:,} admission-level rows")
    return overall


def compute_diagnosis_features(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """
    Create diagnosis category counts per admission.
    Returns one-hot-like counts of each diagnostic category per (subject_id, hadm_id).
    """
    if diagnoses.empty:
        return pd.DataFrame()

    diag = diagnoses.copy()

    # Map ICD codes to categories
    if "icd_version" in diag.columns and "icd_code" in diag.columns:
        diag["diag_category"] = diag.apply(
            lambda row: icd9_to_category(row["icd_code"]) if row.get("icd_version") == 9
            else icd10_to_category(row["icd_code"]),
            axis=1,
        )
    elif "icd_code" in diag.columns:
        diag["diag_category"] = diag["icd_code"].apply(icd10_to_category)
    else:
        return pd.DataFrame()

    # Count diagnoses per category per admission
    diag_counts = (
        diag.groupby(["subject_id", "hadm_id", "diag_category"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    # Prefix columns
    cat_cols = [c for c in diag_counts.columns if c not in ("subject_id", "hadm_id")]
    diag_counts = diag_counts.rename(columns={c: f"diag_{c}" for c in cat_cols})

    # Total diagnosis count
    diag_total = diag.groupby(["subject_id", "hadm_id"]).size().reset_index(name="diag_total_count")
    diag_counts = diag_counts.merge(diag_total, on=["subject_id", "hadm_id"], how="left")

    print(f"[INFO] Computed diagnosis features: {diag_counts.shape[1] - 2} categories")
    return diag_counts


def build_feature_matrix(cleaned_data: dict) -> pd.DataFrame:
    """
    Build the complete feature matrix for ML modeling.
    Merges all tables and computes all derived features.

    Args:
        cleaned_data: Dict with keys 'patients', 'admissions', 'diagnoses', 'labevents'.

    Returns:
        pd.DataFrame: Final feature matrix with one row per admission.
    """
    patients = cleaned_data["patients"]
    admissions = cleaned_data["admissions"]
    diagnoses = cleaned_data["diagnoses"]
    labevents = cleaned_data["labevents"]

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    # Step 1: Age at admission
    df = compute_age_at_admission(patients, admissions)
    print(f"  ✓ Age at admission computed")

    # Step 2: Length of stay
    df = compute_length_of_stay(df)
    print(f"  ✓ Length of stay computed")

    # Step 3: Mortality flag
    df = compute_mortality_flag(df)
    print(f"  ✓ Mortality flag: {df['hospital_expire_flag'].sum()} deaths / {len(df)} admissions")

    # Step 4: Readmission flag
    df = compute_readmission_flag(df)
    print(f"  ✓ Readmission flag: {df['readmission_30d'].sum()} readmissions within 30 days")

    # Step 5: Lab aggregates
    lab_agg = compute_lab_aggregates(labevents)
    if not lab_agg.empty:
        df = df.merge(lab_agg, on=["subject_id", "hadm_id"], how="left")
        print(f"  ✓ Lab aggregates merged")

    # Step 6: Diagnosis features
    diag_feats = compute_diagnosis_features(diagnoses)
    if not diag_feats.empty:
        df = df.merge(diag_feats, on=["subject_id", "hadm_id"], how="left")
        print(f"  ✓ Diagnosis features merged")

    # Step 7: Encode gender
    if "gender" in df.columns:
        df["gender_numeric"] = (df["gender"] == "M").astype(int)

    # Step 8: Age groups for EDA
    if "age_at_admission" in df.columns:
        bins = [0, 30, 50, 65, 75, 91]
        labels = ["18-30", "31-50", "51-65", "66-75", "76+"]
        df["age_group"] = pd.cut(df["age_at_admission"], bins=bins, labels=labels, right=True)

    # Fill remaining NaNs in numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    print(f"\n[INFO] Final feature matrix: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def get_ml_features(df: pd.DataFrame) -> list:
    """Return the list of feature column names used for ML modeling."""
    base_features = ["age_at_admission", "gender_numeric", "los_days"]

    # Lab aggregates
    lab_features = [c for c in df.columns if c.startswith("lab_") or
                    c.endswith("_mean") or c.endswith("_std") or
                    c.endswith("_min") or c.endswith("_max")]
    # Remove target-leaky columns
    lab_features = [c for c in lab_features if "expire" not in c.lower()]

    # Diagnosis features
    diag_features = [c for c in df.columns if c.startswith("diag_")]

    all_features = base_features + lab_features + diag_features
    # Only keep features that actually exist
    all_features = [f for f in all_features if f in df.columns]

    return list(set(all_features))  # deduplicate
