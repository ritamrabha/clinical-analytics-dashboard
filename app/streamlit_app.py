"""
streamlit_app.py - Interactive Dashboard for MIMIC-IV Clinical Analysis
=======================================================================
Displays EDA visualizations, model predictions, and performance metrics.
Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import glob
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MIMIC-IV Clinical Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1B4965;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5FA8D3;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1B4965 0%, #2E86AB 100%);
        border-radius: 12px;
        padding: 1.2rem;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.85;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)

PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B", "#44BBA4"]


# ─── Helper Functions ────────────────────────────────────────────────────────

@st.cache_data
def load_feature_matrix(path: str) -> pd.DataFrame:
    """Load the precomputed feature matrix."""
    return pd.read_csv(path, low_memory=False)


@st.cache_resource
def load_saved_models(models_dir: str) -> dict:
    """Load all saved models from disk."""
    models = {}
    for fpath in glob.glob(os.path.join(models_dir, "*.joblib")):
        fname = os.path.basename(fpath).replace(".joblib", "")
        if fname not in ("feature_names", "training_meta"):
            name = fname.replace("_", " ").title()
            models[name] = joblib.load(fpath)
    return models


@st.cache_data
def load_feature_names(models_dir: str) -> list:
    """Load saved feature names."""
    path = os.path.join(models_dir, "feature_names.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return []


def find_output_dir():
    """Find the outputs directory relative to project root."""
    candidates = [
        os.path.join(PROJECT_ROOT, "outputs"),
        "outputs",
        os.path.join("..", "outputs"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)
    return os.path.join(PROJECT_ROOT, "outputs")


# ─── Sidebar ─────────────────────────────────────────────────────────────────

def sidebar():
    st.sidebar.markdown("## 🏥 MIMIC-IV Dashboard")
    st.sidebar.markdown("---")

    output_dir = find_output_dir()
    plots_dir = os.path.join(output_dir, "plots")
    models_dir = os.path.join(output_dir, "models")

    # Data source
    st.sidebar.markdown("### Data Source")
    data_option = st.sidebar.radio(
        "Choose data source:",
        ["Use preloaded data", "Upload CSV file"],
        index=0,
    )

    feature_df = None
    feature_path = os.path.join(output_dir, "feature_matrix.csv")

    if data_option == "Use preloaded data":
        if os.path.exists(feature_path):
            feature_df = load_feature_matrix(feature_path)
            st.sidebar.success(f"Loaded {len(feature_df):,} records")
        else:
            st.sidebar.warning("No preloaded data found. Run `python main.py` first.")
    else:
        uploaded = st.sidebar.file_uploader("Upload feature_matrix.csv", type=["csv"])
        if uploaded is not None:
            feature_df = pd.read_csv(uploaded, low_memory=False)
            st.sidebar.success(f"Uploaded {len(feature_df):,} records")

    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio(
        "Go to:",
        ["📊 Overview", "📈 EDA Visualizations", "🤖 Model Performance", "🔮 Predict Mortality"],
        index=0,
    )

    return page, feature_df, output_dir, plots_dir, models_dir


# ─── Pages ───────────────────────────────────────────────────────────────────

def page_overview(df: pd.DataFrame):
    """Overview page with key statistics."""
    st.markdown('<div class="main-header">MIMIC-IV Clinical Data Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">In-Hospital Mortality Prediction Dashboard</div>',
                unsafe_allow_html=True)

    if df is None:
        st.info("Load data to view the overview.")
        return

    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Admissions", f"{len(df):,}")
    with col2:
        n_patients = df["subject_id"].nunique() if "subject_id" in df.columns else "—"
        st.metric("Unique Patients", f"{n_patients:,}" if isinstance(n_patients, int) else n_patients)
    with col3:
        mort_rate = df["hospital_expire_flag"].mean() * 100 if "hospital_expire_flag" in df.columns else 0
        st.metric("Mortality Rate", f"{mort_rate:.1f}%")
    with col4:
        avg_los = df["los_days"].mean() if "los_days" in df.columns else 0
        st.metric("Avg LOS (days)", f"{avg_los:.1f}")
    with col5:
        avg_age = df["age_at_admission"].mean() if "age_at_admission" in df.columns else 0
        st.metric("Avg Age", f"{avg_age:.0f}")

    st.markdown("---")

    # Quick summary
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Admission Types")
        if "admission_type" in df.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            counts = df["admission_type"].value_counts().head(8)
            ax.barh(counts.index, counts.values, color=PALETTE[0], edgecolor="white")
            ax.set_xlabel("Count")
            ax.invert_yaxis()
            st.pyplot(fig)
            plt.close(fig)

    with col2:
        st.subheader("Insurance Distribution")
        if "insurance" in df.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            counts = df["insurance"].value_counts()
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                   colors=PALETTE, startangle=140)
            ax.set_title("")
            st.pyplot(fig)
            plt.close(fig)

    # Data preview
    st.markdown("---")
    with st.expander("📋 Data Preview (first 100 rows)"):
        display_cols = [c for c in df.columns if not c.startswith("diag_")][:20]
        st.dataframe(df[display_cols].head(100), use_container_width=True)


def page_eda(df: pd.DataFrame, plots_dir: str):
    """EDA Visualizations page."""
    st.markdown('<div class="main-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    if df is None:
        st.info("Load data to view visualizations.")
        return

    # Check for pre-generated plots
    plot_files = glob.glob(os.path.join(plots_dir, "*.png")) if os.path.isdir(plots_dir) else []

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎂 Age", "⚤ Gender", "🏥 LOS", "📋 Diagnoses", "🧪 Lab Values"
    ])

    with tab1:
        st.subheader("Age Distribution")
        if "age_at_admission" in df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].hist(df["age_at_admission"].dropna(), bins=40, color=PALETTE[0],
                         edgecolor="white", alpha=0.85)
            axes[0].axvline(df["age_at_admission"].median(), color=PALETTE[1],
                            linestyle="--", label=f'Median: {df["age_at_admission"].median():.0f}')
            axes[0].set_xlabel("Age")
            axes[0].set_ylabel("Count")
            axes[0].set_title("Age at Admission")
            axes[0].legend()

            if "hospital_expire_flag" in df.columns:
                age_mort = df.groupby(pd.cut(df["age_at_admission"], bins=10))["hospital_expire_flag"].mean() * 100
                axes[1].bar(range(len(age_mort)), age_mort.values, color=PALETTE[1], edgecolor="white")
                axes[1].set_xticks(range(len(age_mort)))
                axes[1].set_xticklabels([str(x) for x in age_mort.index], rotation=45, fontsize=8)
                axes[1].set_ylabel("Mortality Rate (%)")
                axes[1].set_title("Mortality Rate by Age Bin")

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with tab2:
        st.subheader("Gender Distribution")
        if "gender" in df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            counts = df["gender"].value_counts()
            axes[0].bar(counts.index, counts.values, color=[PALETTE[0], PALETTE[1]], edgecolor="white")
            axes[0].set_title("Gender Counts")
            axes[0].set_ylabel("Count")

            if "hospital_expire_flag" in df.columns:
                mort = df.groupby("gender")["hospital_expire_flag"].mean() * 100
                axes[1].bar(mort.index, mort.values, color=[PALETTE[0], PALETTE[1]], edgecolor="white")
                axes[1].set_title("Mortality Rate by Gender")
                axes[1].set_ylabel("Mortality Rate (%)")

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with tab3:
        st.subheader("Length of Stay Analysis")
        if "los_days" in df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            if "hospital_expire_flag" in df.columns:
                for flag, color, label in [(0, PALETTE[0], "Survived"), (1, PALETTE[1], "Died")]:
                    subset = df[df["hospital_expire_flag"] == flag]["los_days"].dropna()
                    subset = subset[subset <= subset.quantile(0.95)]
                    axes[0].hist(subset, bins=50, color=color, alpha=0.5, density=True,
                                 label=label, edgecolor="white")
                axes[0].set_xlabel("LOS (days)")
                axes[0].set_ylabel("Density")
                axes[0].set_title("LOS by Mortality")
                axes[0].legend()

            if "age_group" in df.columns:
                age_los = df.groupby("age_group")["los_days"].mean()
                axes[1].bar(range(len(age_los)), age_los.values, color=PALETTE[0], edgecolor="white")
                axes[1].set_xticks(range(len(age_los)))
                axes[1].set_xticklabels(age_los.index)
                axes[1].set_ylabel("Mean LOS (days)")
                axes[1].set_title("Mean LOS by Age Group")

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with tab4:
        st.subheader("Top Diagnoses")
        diag_cols = [c for c in df.columns if c.startswith("diag_") and c != "diag_total_count"]
        if diag_cols:
            diag_sums = df[diag_cols].sum().sort_values(ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(diag_sums)), diag_sums.values, color=PALETTE[0], edgecolor="white")
            ax.set_yticks(range(len(diag_sums)))
            ax.set_yticklabels([c.replace("diag_", "") for c in diag_sums.index])
            ax.invert_yaxis()
            ax.set_xlabel("Total Count")
            ax.set_title("Top Diagnostic Categories")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            # Show pre-generated plot
            img_path = os.path.join(plots_dir, "top_diagnoses.png")
            if os.path.exists(img_path):
                st.image(img_path)
            else:
                st.info("No diagnosis data available.")

    with tab5:
        st.subheader("Lab Value Distributions")
        lab_cols = [c for c in df.columns if c.endswith("_mean") and "lab" not in c]
        if lab_cols:
            selected_lab = st.selectbox("Select lab value:", lab_cols)
            fig, ax = plt.subplots(figsize=(10, 5))
            vals = df[selected_lab].dropna()
            lo, hi = vals.quantile(0.02), vals.quantile(0.98)
            vals_clipped = vals[(vals >= lo) & (vals <= hi)]
            ax.hist(vals_clipped, bins=50, color=PALETTE[0], edgecolor="white", alpha=0.85)
            ax.axvline(vals_clipped.median(), color=PALETTE[2], linestyle="--",
                       label=f"Median: {vals_clipped.median():.1f}")
            ax.set_xlabel(selected_lab.replace("_", " ").title())
            ax.set_ylabel("Frequency")
            ax.set_title(f"Distribution: {selected_lab.replace('_', ' ').title()}")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            # Show pre-generated plots
            for fname in ["lab_trend_glucose.png", "lab_trend_creatinine.png"]:
                img_path = os.path.join(plots_dir, fname)
                if os.path.exists(img_path):
                    st.image(img_path)


def page_model_performance(df: pd.DataFrame, models_dir: str, plots_dir: str):
    """Model performance and evaluation page."""
    st.markdown('<div class="main-header">Model Performance</div>', unsafe_allow_html=True)

    # Load models
    models = load_saved_models(models_dir)
    feature_names = load_feature_names(models_dir)

    if not models:
        st.warning("No trained models found. Run `python main.py` first.")
        return

    if df is None or not feature_names:
        st.info("Load data and models to view performance metrics.")
        return

    st.success(f"Loaded {len(models)} models: {', '.join(models.keys())}")

    # Prepare test data
    available_features = [f for f in feature_names if f in df.columns]
    X = df[available_features].fillna(0).replace([np.inf, -np.inf], 0)
    y = df["hospital_expire_flag"].fillna(0).astype(int) if "hospital_expire_flag" in df.columns else None

    if y is None:
        st.warning("Target column 'hospital_expire_flag' not found.")
        return

    # Metrics table
    st.subheader("📊 Performance Metrics")
    results = []
    for name, model in models.items():
        try:
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]
            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y, y_pred),
                "Precision": precision_score(y, y_pred, zero_division=0),
                "Recall": recall_score(y, y_pred, zero_division=0),
                "F1 Score": f1_score(y, y_pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y, y_prob),
            })
        except Exception as e:
            st.warning(f"Error evaluating {name}: {e}")

    if results:
        results_df = pd.DataFrame(results).set_index("Model")
        st.dataframe(results_df.style.format("{:.4f}").highlight_max(axis=0, color="#c8e6c9"),
                      use_container_width=True)

    # Plots
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ROC Curves")
        roc_path = os.path.join(plots_dir, "roc_curves.png")
        if os.path.exists(roc_path):
            st.image(roc_path)
        else:
            # Generate on the fly
            fig, ax = plt.subplots(figsize=(7, 6))
            for i, (name, model) in enumerate(models.items()):
                try:
                    y_prob = model.predict_proba(X)[:, 1]
                    fpr, tpr, _ = roc_curve(y, y_prob)
                    auc_val = roc_auc_score(y, y_prob)
                    ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], linewidth=2,
                            label=f"{name} (AUC={auc_val:.3f})")
                except Exception:
                    pass
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_title("ROC Curves")
            ax.legend(loc="lower right", fontsize=9)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with col2:
        st.subheader("Confusion Matrices")
        cm_path = os.path.join(plots_dir, "confusion_matrices.png")
        if os.path.exists(cm_path):
            st.image(cm_path)

    # Feature importance
    st.markdown("---")
    st.subheader("🔑 Feature Importance")
    fi_path = os.path.join(plots_dir, "feature_importance.png")
    if os.path.exists(fi_path):
        st.image(fi_path)
    else:
        selected_model_name = st.selectbox("Select model:", list(models.keys()))
        model = models[selected_model_name]
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "named_steps"):
            clf = model.named_steps.get("clf")
            if clf and hasattr(clf, "coef_"):
                importances = np.abs(clf.coef_[0])

        if importances is not None and len(importances) == len(available_features):
            idx = np.argsort(importances)[-20:]
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.barh([available_features[j] for j in idx], importances[idx],
                    color=PALETTE[0], edgecolor="white")
            ax.set_title(f"{selected_model_name} — Top 20 Features")
            ax.set_xlabel("Importance")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


def page_predict(df: pd.DataFrame, models_dir: str):
    """Interactive prediction page."""
    st.markdown('<div class="main-header">Predict In-Hospital Mortality</div>', unsafe_allow_html=True)

    models = load_saved_models(models_dir)
    feature_names = load_feature_names(models_dir)

    if not models:
        st.warning("No trained models found. Run `python main.py` first.")
        return

    st.markdown("Enter patient parameters below to predict mortality risk:")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age at Admission", 18, 91, 65)
        gender = st.selectbox("Gender", ["Male", "Female"])
        los = st.number_input("Length of Stay (days)", min_value=0.0, max_value=365.0, value=5.0, step=0.5)

    with col2:
        glucose = st.number_input("Glucose (mean)", min_value=0.0, max_value=1000.0, value=120.0)
        creatinine = st.number_input("Creatinine (mean)", min_value=0.0, max_value=30.0, value=1.0)
        hemoglobin = st.number_input("Hemoglobin (mean)", min_value=0.0, max_value=25.0, value=12.0)

    with col3:
        sodium = st.number_input("Sodium (mean)", min_value=100.0, max_value=180.0, value=140.0)
        potassium = st.number_input("Potassium (mean)", min_value=1.0, max_value=10.0, value=4.0)
        wbc = st.number_input("WBC (mean)", min_value=0.0, max_value=100.0, value=8.0)

    # Build input vector matching feature_names
    input_data = {
        "age_at_admission": age,
        "gender_numeric": 1 if gender == "Male" else 0,
        "los_days": los,
        "glucose_mean": glucose,
        "creatinine_mean": creatinine,
        "hemoglobin_mean": hemoglobin,
        "sodium_mean": sodium,
        "potassium_mean": potassium,
        "wbc_mean": wbc,
    }

    # Fill all expected features with 0 if not provided
    full_input = {f: input_data.get(f, 0.0) for f in feature_names}
    input_df = pd.DataFrame([full_input])

    selected_model_name = st.selectbox("Select model for prediction:", list(models.keys()))

    if st.button("🔮 Predict", type="primary"):
        model = models[selected_model_name]
        try:
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0]

            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                if pred == 1:
                    st.error(f"⚠️ **HIGH RISK** — Predicted Mortality")
                else:
                    st.success(f"✅ **LOW RISK** — Predicted Survival")

            with col2:
                st.metric("Survival Probability", f"{prob[0]:.1%}")
                st.metric("Mortality Probability", f"{prob[1]:.1%}")

            # Risk gauge
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.barh(0, prob[0], color="#44BBA4", height=0.5, label="Survival")
            ax.barh(0, prob[1], left=prob[0], color="#C73E1D", height=0.5, label="Mortality")
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel("Probability")
            ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.4))
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ─── Main App ────────────────────────────────────────────────────────────────

def main():
    page, feature_df, output_dir, plots_dir, models_dir = sidebar()

    if page == "📊 Overview":
        page_overview(feature_df)
    elif page == "📈 EDA Visualizations":
        page_eda(feature_df, plots_dir)
    elif page == "🤖 Model Performance":
        page_model_performance(feature_df, models_dir, plots_dir)
    elif page == "🔮 Predict Mortality":
        page_predict(feature_df, models_dir)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
        "MIMIC-IV Clinical Data Analysis & Prediction System • Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
