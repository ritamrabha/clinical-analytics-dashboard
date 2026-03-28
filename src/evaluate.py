"""
evaluate.py - Model evaluation and visualization for MIMIC-IV mortality prediction.
Generates EDA plots, evaluation metrics, confusion matrices, ROC curves,
and feature importance charts.
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)

# ─── Plot style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor": "#FAFAFA",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "text.color": "#333333",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})

PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B", "#44BBA4"]


def save_plot(fig, filename: str, output_dir: str = "outputs/plots"):
    """Save a matplotlib figure to disk."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved plot: {filepath}")
    return filepath


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPLORATORY DATA ANALYSIS PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_age_distribution(df: pd.DataFrame, output_dir: str = "outputs/plots"):
    """Plot age distribution of patients at admission."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["age_at_admission"].dropna(), bins=40, color=PALETTE[0],
            edgecolor="white", alpha=0.85)
    ax.set_xlabel("Age at Admission")
    ax.set_ylabel("Number of Admissions")
    ax.set_title("Age Distribution at Admission")
    ax.axvline(df["age_at_admission"].median(), color=PALETTE[1],
               linestyle="--", linewidth=1.5, label=f'Median: {df["age_at_admission"].median():.0f}')
    ax.legend()
    return save_plot(fig, "age_distribution.png", output_dir)


def plot_gender_distribution(df: pd.DataFrame, output_dir: str = "outputs/plots"):
    """Plot gender distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    counts = df["gender"].value_counts()
    axes[0].bar(counts.index, counts.values, color=[PALETTE[0], PALETTE[1]], edgecolor="white")
    axes[0].set_title("Gender Distribution")
    axes[0].set_ylabel("Count")
    for i, (idx, val) in enumerate(counts.items()):
        axes[0].text(i, val + 50, f"{val:,}", ha="center", fontweight="bold")

    # Mortality by gender
    mort_by_gender = df.groupby("gender")["hospital_expire_flag"].mean() * 100
    axes[1].bar(mort_by_gender.index, mort_by_gender.values,
                color=[PALETTE[0], PALETTE[1]], edgecolor="white")
    axes[1].set_title("Mortality Rate by Gender")
    axes[1].set_ylabel("Mortality Rate (%)")
    for i, (idx, val) in enumerate(mort_by_gender.items()):
        axes[1].text(i, val + 0.3, f"{val:.1f}%", ha="center", fontweight="bold")

    fig.tight_layout()
    return save_plot(fig, "gender_distribution.png", output_dir)


def plot_los_vs_mortality(df: pd.DataFrame, output_dir: str = "outputs/plots"):
    """Plot length of stay distribution by mortality status."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot
    data_alive = df[df["hospital_expire_flag"] == 0]["los_days"].dropna()
    data_dead = df[df["hospital_expire_flag"] == 1]["los_days"].dropna()
    bp = axes[0].boxplot(
        [data_alive.clip(upper=data_alive.quantile(0.95)),
         data_dead.clip(upper=data_dead.quantile(0.95))],
        labels=["Survived", "Died"],
        patch_artist=True,
        boxprops=dict(facecolor=PALETTE[0], alpha=0.7),
        medianprops=dict(color=PALETTE[2], linewidth=2),
    )
    bp["boxes"][1].set_facecolor(PALETTE[1])
    axes[0].set_ylabel("Length of Stay (days)")
    axes[0].set_title("LOS by Mortality Status")

    # KDE plot
    for flag, color, label in [(0, PALETTE[0], "Survived"), (1, PALETTE[1], "Died")]:
        subset = df[df["hospital_expire_flag"] == flag]["los_days"].dropna()
        subset = subset[subset <= subset.quantile(0.95)]
        axes[1].hist(subset, bins=50, color=color, alpha=0.5, density=True, label=label, edgecolor="white")
    axes[1].set_xlabel("Length of Stay (days)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("LOS Distribution by Outcome")
    axes[1].legend()

    fig.tight_layout()
    return save_plot(fig, "los_vs_mortality.png", output_dir)


def plot_los_by_age_group(df: pd.DataFrame, output_dir: str = "outputs/plots"):
    """Plot LOS by age groups."""
    if "age_group" not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    age_los = df.groupby("age_group")["los_days"].agg(["mean", "median", "std"]).reset_index()
    x = range(len(age_los))
    ax.bar([i - 0.15 for i in x], age_los["mean"], width=0.3, color=PALETTE[0],
           label="Mean", edgecolor="white")
    ax.bar([i + 0.15 for i in x], age_los["median"], width=0.3, color=PALETTE[1],
           label="Median", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(age_los["age_group"], rotation=0)
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Length of Stay (days)")
    ax.set_title("Length of Stay by Age Group")
    ax.legend()
    fig.tight_layout()
    return save_plot(fig, "los_by_age_group.png", output_dir)


def plot_top_diagnoses(diagnoses: pd.DataFrame, n: int = 15, output_dir: str = "outputs/plots"):
    """Plot top N most common ICD diagnosis codes."""
    if diagnoses.empty or "icd_code" not in diagnoses.columns:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    top = diagnoses["icd_code"].value_counts().head(n)
    bars = ax.barh(range(len(top)), top.values, color=PALETTE[0], edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index)
    ax.invert_yaxis()
    ax.set_xlabel("Frequency")
    ax.set_title(f"Top {n} Most Common Diagnoses (ICD Codes)")
    for i, v in enumerate(top.values):
        ax.text(v + 10, i, f"{v:,}", va="center", fontsize=9)
    fig.tight_layout()
    return save_plot(fig, "top_diagnoses.png", output_dir)


def plot_lab_trends(labevents: pd.DataFrame, item_id: int = 50931,
                    lab_name: str = "Glucose", output_dir: str = "outputs/plots"):
    """Plot distribution and trend for a specific lab value."""
    if labevents.empty or "itemid" not in labevents.columns:
        return None

    subset = labevents[labevents["itemid"] == item_id].copy()
    if subset.empty:
        print(f"  [WARN] No data for lab item {item_id} ({lab_name})")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution
    vals = subset["valuenum"].dropna()
    lo, hi = vals.quantile(0.01), vals.quantile(0.99)
    vals_clipped = vals[(vals >= lo) & (vals <= hi)]
    axes[0].hist(vals_clipped, bins=50, color=PALETTE[0], edgecolor="white", alpha=0.85)
    axes[0].axvline(vals_clipped.median(), color=PALETTE[2], linestyle="--",
                    label=f"Median: {vals_clipped.median():.1f}")
    axes[0].set_xlabel(lab_name)
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"{lab_name} Value Distribution")
    axes[0].legend()

    # Trend over time (if charttime exists)
    if "charttime" in subset.columns:
        trend = subset.dropna(subset=["charttime"]).copy()
        trend["date"] = trend["charttime"].dt.to_period("M").dt.to_timestamp()
        monthly = trend.groupby("date")["valuenum"].agg(["mean", "std"]).reset_index()
        if len(monthly) > 1:
            axes[1].plot(monthly["date"], monthly["mean"], color=PALETTE[0], linewidth=2)
            axes[1].fill_between(
                monthly["date"],
                monthly["mean"] - monthly["std"],
                monthly["mean"] + monthly["std"],
                alpha=0.2, color=PALETTE[0]
            )
            axes[1].set_xlabel("Date")
            axes[1].set_ylabel(f"Mean {lab_name}")
            axes[1].set_title(f"{lab_name} Monthly Trend (Mean ± Std)")
            axes[1].tick_params(axis="x", rotation=45)
        else:
            axes[1].text(0.5, 0.5, "Insufficient temporal data", ha="center",
                         va="center", transform=axes[1].transAxes)
    else:
        axes[1].text(0.5, 0.5, "No temporal data available", ha="center",
                     va="center", transform=axes[1].transAxes)

    fig.tight_layout()
    return save_plot(fig, f"lab_trend_{lab_name.lower()}.png", output_dir)


def run_eda(feature_df: pd.DataFrame, cleaned_data: dict, output_dir: str = "outputs/plots"):
    """Run all EDA visualizations."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    plots = {}
    plots["age_dist"] = plot_age_distribution(feature_df, output_dir)
    plots["gender_dist"] = plot_gender_distribution(feature_df, output_dir)
    plots["los_mortality"] = plot_los_vs_mortality(feature_df, output_dir)
    plots["los_age"] = plot_los_by_age_group(feature_df, output_dir)
    plots["top_diag"] = plot_top_diagnoses(cleaned_data["diagnoses"], output_dir=output_dir)
    plots["glucose_trend"] = plot_lab_trends(
        cleaned_data["labevents"], item_id=50931, lab_name="Glucose", output_dir=output_dir
    )
    plots["creatinine_trend"] = plot_lab_trends(
        cleaned_data["labevents"], item_id=50912, lab_name="Creatinine", output_dir=output_dir
    )

    print(f"\n[INFO] Generated {sum(1 for v in plots.values() if v)} EDA plots")
    return plots


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Evaluate a trained model on test data.
    Returns a dict of metrics.
    """
    y_pred = model.predict(X_test)

    # Get probabilities (handle pipelines)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = y_pred.astype(float)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "avg_precision": average_precision_score(y_test, y_prob),
    }

    print(f"\n{'─' * 40}")
    print(f"  {model_name}")
    print(f"{'─' * 40}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1 Score:    {metrics['f1']:.4f}")
    print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
    print(f"  Avg Prec:    {metrics['avg_precision']:.4f}")

    return metrics


def evaluate_all_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """Evaluate all models and return a comparison DataFrame."""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    results = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        results.append(metrics)

    return pd.DataFrame(results).set_index("model")


def plot_roc_curves(models: dict, X_test, y_test, output_dir: str = "outputs/plots"):
    """Plot ROC curves for all models on the same axes."""
    fig, ax = plt.subplots(figsize=(8, 7))

    for i, (name, model) in enumerate(models.items()):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_prob = model.predict(X_test).astype(float)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], linewidth=2,
                label=f"{name} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Mortality Prediction")
    ax.legend(loc="lower right")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    fig.tight_layout()
    return save_plot(fig, "roc_curves.png", output_dir)


def plot_confusion_matrices(models: dict, X_test, y_test, output_dir: str = "outputs/plots"):
    """Plot confusion matrices for all models."""
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for i, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=["Survived", "Died"], yticklabels=["Survived", "Died"])
        axes[i].set_title(name)
        axes[i].set_ylabel("Actual")
        axes[i].set_xlabel("Predicted")

    fig.suptitle("Confusion Matrices", fontsize=14, y=1.02)
    fig.tight_layout()
    return save_plot(fig, "confusion_matrices.png", output_dir)


def plot_feature_importance(models: dict, feature_names: list,
                            n_top: int = 20, output_dir: str = "outputs/plots"):
    """Plot feature importance for tree-based models and LR coefficients."""
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 8))
    if len(models) == 1:
        axes = [axes]

    for i, (name, model) in enumerate(models.items()):
        importances = None
        labels = feature_names

        # Extract importances
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        elif hasattr(model, "named_steps"):
            clf = model.named_steps.get("clf")
            if clf is not None:
                if hasattr(clf, "feature_importances_"):
                    importances = clf.feature_importances_
                elif hasattr(clf, "coef_"):
                    importances = np.abs(clf.coef_[0])

        if importances is not None and len(importances) == len(labels):
            # Sort and take top N
            idx = np.argsort(importances)[-n_top:]
            axes[i].barh(
                [labels[j] for j in idx],
                importances[idx],
                color=PALETTE[i % len(PALETTE)],
                edgecolor="white"
            )
            axes[i].set_title(f"{name}\nFeature Importance (Top {n_top})")
            axes[i].set_xlabel("Importance")
        else:
            axes[i].text(0.5, 0.5, "N/A", ha="center", va="center",
                         transform=axes[i].transAxes)
            axes[i].set_title(name)

    fig.tight_layout()
    return save_plot(fig, "feature_importance.png", output_dir)


def plot_metrics_comparison(results_df: pd.DataFrame, output_dir: str = "outputs/plots"):
    """Plot bar chart comparing model metrics side by side."""
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    available_metrics = [m for m in metrics_to_plot if m in results_df.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(available_metrics))
    width = 0.8 / len(results_df)

    for i, (model_name, row) in enumerate(results_df.iterrows()):
        offset = (i - len(results_df) / 2 + 0.5) * width
        values = [row[m] for m in available_metrics]
        bars = ax.bar(x + offset, values, width, label=model_name,
                      color=PALETTE[i % len(PALETTE)], edgecolor="white")
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper().replace("_", "-") for m in available_metrics])
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_ylim(0, 1.1)
    ax.legend()
    fig.tight_layout()
    return save_plot(fig, "metrics_comparison.png", output_dir)
