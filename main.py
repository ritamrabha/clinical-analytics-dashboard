#!/usr/bin/env python3
"""
main.py - MIMIC-IV Clinical Data Analysis & Prediction System
=============================================================
End-to-end pipeline: data download → preprocessing → feature engineering →
EDA → model training → evaluation → export.

Usage:
    python main.py                          # Full pipeline with defaults
    python main.py --data-dir data          # Specify data directory
    python main.py --tune                   # Enable hyperparameter tuning
    python main.py --skip-eda               # Skip EDA plot generation
    python main.py --output-dir outputs     # Custom output directory
"""

import argparse
import os
import sys
import time
import warnings

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import download_dataset, preprocess_all
from src.features import build_feature_matrix, get_ml_features
from src.train import prepare_data, train_all_models, save_models
from src.evaluate import (
    run_eda, evaluate_all_models, plot_roc_curves,
    plot_confusion_matrices, plot_feature_importance, plot_metrics_comparison
)

warnings.filterwarnings("ignore")


def banner():
    """Print project banner."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║   MIMIC-IV Clinical Data Analysis & Prediction System       ║
║   In-Hospital Mortality Prediction Pipeline                 ║
╚══════════════════════════════════════════════════════════════╝
    """)


def main():
    parser = argparse.ArgumentParser(
        description="MIMIC-IV Clinical Data Analysis & Mortality Prediction"
    )
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Path to data directory (default: data)")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Path to output directory (default: outputs)")
    parser.add_argument("--tune", action="store_true",
                        help="Enable hyperparameter tuning (slower but better)")
    parser.add_argument("--skip-eda", action="store_true",
                        help="Skip EDA plot generation")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test split ratio (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    banner()
    start_time = time.time()

    plots_dir = os.path.join(args.output_dir, "plots")
    models_dir = os.path.join(args.output_dir, "models")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # ─── Step 1: Download / Locate Data ──────────────────────────────────────
    print("=" * 60)
    print("STEP 1: DATA ACQUISITION")
    print("=" * 60)
    data_path = download_dataset(args.data_dir)

    # ─── Step 2: Preprocess ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 60)
    cleaned_data = preprocess_all(data_path)

    # ─── Step 3: Feature Engineering ─────────────────────────────────────────
    feature_df = build_feature_matrix(cleaned_data)

    # Save feature matrix for Streamlit
    feature_path = os.path.join(args.output_dir, "feature_matrix.csv")
    feature_df.to_csv(feature_path, index=False)
    print(f"[INFO] Feature matrix saved → {feature_path}")

    # ─── Step 4: EDA ─────────────────────────────────────────────────────────
    if not args.skip_eda:
        eda_plots = run_eda(feature_df, cleaned_data, output_dir=plots_dir)

    # ─── Step 5: Model Training ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: MODEL TRAINING")
    print("=" * 60)

    feature_cols = get_ml_features(feature_df)
    print(f"[INFO] Using {len(feature_cols)} features for modeling")

    X_train, X_test, y_train, y_test, used_features = prepare_data(
        feature_df, feature_cols,
        target_col="hospital_expire_flag",
        test_size=args.test_size,
        random_state=args.random_state,
    )

    models = train_all_models(X_train, y_train, tune=args.tune)

    # ─── Step 6: Evaluation ──────────────────────────────────────────────────
    results_df = evaluate_all_models(models, X_test, y_test)

    # Save metrics
    results_path = os.path.join(args.output_dir, "model_metrics.csv")
    results_df.to_csv(results_path)
    print(f"\n[INFO] Metrics saved → {results_path}")

    # Evaluation plots
    print("\n[INFO] Generating evaluation plots...")
    plot_roc_curves(models, X_test, y_test, output_dir=plots_dir)
    plot_confusion_matrices(models, X_test, y_test, output_dir=plots_dir)
    plot_feature_importance(models, used_features, output_dir=plots_dir)
    plot_metrics_comparison(results_df, output_dir=plots_dir)

    # ─── Step 7: Save Models ─────────────────────────────────────────────────
    print("\n[INFO] Saving trained models...")
    save_models(models, used_features, output_dir=models_dir)

    # ─── Summary ─────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Admissions processed: {len(feature_df):,}")
    print(f"  Features used: {len(used_features)}")
    print(f"  Models trained: {len(models)}")
    print(f"\n  Best model by ROC-AUC: {results_df['roc_auc'].idxmax()} "
          f"({results_df['roc_auc'].max():.4f})")
    print(f"\n  Outputs directory: {os.path.abspath(args.output_dir)}")
    print(f"  Plots: {plots_dir}")
    print(f"  Models: {models_dir}")
    print()

    return results_df


if __name__ == "__main__":
    main()
