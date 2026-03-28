"""
train.py - Model training for in-hospital mortality prediction.
Trains Logistic Regression and Random Forest, with optional hyperparameter tuning.
Handles class imbalance via SMOTE or class_weight.
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")


def prepare_data(df: pd.DataFrame, feature_cols: list, target_col: str = "hospital_expire_flag",
                 test_size: float = 0.2, random_state: int = 42):
    """
    Prepare train/test split from feature matrix.

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    # Drop rows where target is NaN
    data = df.dropna(subset=[target_col]).copy()

    # Ensure all feature columns exist and are numeric
    available_features = [f for f in feature_cols if f in data.columns]
    X = data[available_features].copy()
    y = data[target_col].astype(int)

    # Fill any remaining NaN with 0
    X = X.fillna(0)

    # Replace infinities
    X = X.replace([np.inf, -np.inf], 0)

    print(f"[INFO] Dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
    print(f"[INFO] Target distribution: {dict(y.value_counts())}")
    print(f"[INFO] Mortality rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[INFO] Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test, available_features


def train_logistic_regression(X_train, y_train, tune: bool = False):
    """
    Train a Logistic Regression pipeline with StandardScaler.
    Optionally runs hyperparameter tuning via GridSearchCV.
    """
    print("\n--- Logistic Regression ---")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        ))
    ])

    if tune:
        print("[INFO] Running hyperparameter tuning...")
        param_grid = {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__penalty": ["l2"],
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(
            pipeline, param_grid, cv=cv, scoring="roc_auc",
            n_jobs=-1, verbose=0
        )
        grid.fit(X_train, y_train)
        print(f"  Best params: {grid.best_params_}")
        print(f"  Best CV ROC-AUC: {grid.best_score_:.4f}")
        return grid.best_estimator_
    else:
        pipeline.fit(X_train, y_train)
        return pipeline


def train_random_forest(X_train, y_train, tune: bool = False):
    """
    Train a Random Forest classifier.
    Optionally runs hyperparameter tuning via GridSearchCV.
    """
    print("\n--- Random Forest ---")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    if tune:
        print("[INFO] Running hyperparameter tuning...")
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [5, 10, 20],
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(
            rf, param_grid, cv=cv, scoring="roc_auc",
            n_jobs=-1, verbose=0
        )
        grid.fit(X_train, y_train)
        print(f"  Best params: {grid.best_params_}")
        print(f"  Best CV ROC-AUC: {grid.best_score_:.4f}")
        return grid.best_estimator_
    else:
        rf.fit(X_train, y_train)
        return rf


def train_gradient_boosting(X_train, y_train, tune: bool = False):
    """
    Train a Gradient Boosting classifier (bonus model).
    """
    print("\n--- Gradient Boosting ---")

    # Compute sample weight ratio for imbalanced data
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=10,
        random_state=42,
    )

    if tune:
        print("[INFO] Running hyperparameter tuning...")
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(
            gb, param_grid, cv=cv, scoring="roc_auc",
            n_jobs=-1, verbose=0
        )
        grid.fit(X_train, y_train)
        print(f"  Best params: {grid.best_params_}")
        print(f"  Best CV ROC-AUC: {grid.best_score_:.4f}")
        return grid.best_estimator_
    else:
        gb.fit(X_train, y_train)
        return gb


def train_all_models(X_train, y_train, tune: bool = False) -> dict:
    """
    Train all models and return them in a dictionary.
    """
    models = {}

    models["Logistic Regression"] = train_logistic_regression(X_train, y_train, tune=tune)
    models["Random Forest"] = train_random_forest(X_train, y_train, tune=tune)
    models["Gradient Boosting"] = train_gradient_boosting(X_train, y_train, tune=tune)

    return models


def save_models(models: dict, feature_names: list, output_dir: str = "outputs/models"):
    """Save trained models and metadata to disk."""
    os.makedirs(output_dir, exist_ok=True)

    for name, model in models.items():
        filename = name.lower().replace(" ", "_") + ".joblib"
        filepath = os.path.join(output_dir, filename)
        joblib.dump(model, filepath)
        print(f"  ✓ Saved {name} → {filepath}")

    # Save feature names
    meta_path = os.path.join(output_dir, "feature_names.joblib")
    joblib.dump(feature_names, meta_path)
    print(f"  ✓ Saved feature names → {meta_path}")

    # Save training metadata
    meta = {
        "trained_at": datetime.now().isoformat(),
        "n_features": len(feature_names),
        "models": list(models.keys()),
    }
    joblib.dump(meta, os.path.join(output_dir, "training_meta.joblib"))


def load_model(model_name: str, model_dir: str = "outputs/models"):
    """Load a saved model from disk."""
    filename = model_name.lower().replace(" ", "_") + ".joblib"
    filepath = os.path.join(model_dir, filename)
    if os.path.exists(filepath):
        return joblib.load(filepath)
    else:
        raise FileNotFoundError(f"Model not found: {filepath}")
