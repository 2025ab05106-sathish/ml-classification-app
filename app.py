"""
Online Shoppers Purchasing Intention — Classification App
=========================================================
Interactive Streamlit dashboard for predicting whether an online
shopping session will result in a purchase (Revenue).
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, matthews_corrcoef
)

# ─────────────────────────────────────────────────────────────────────────────
# Page Config & Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🛒 Shopper Intent Classifier",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1a237e, #0d47a1, #00897b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.5rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a237e 0%, #00897b 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 0.85rem;
        opacity: 0.9;
        font-weight: 400;
    }
    .metric-card h2 {
        margin: 0.3rem 0 0 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .section-divider {
        height: 3px;
        background: linear-gradient(90deg, transparent, #00897b, transparent);
        border: none;
        margin: 2rem 0;
    }
    .footer-text {
        text-align: center;
        color: #888;
        font-size: 0.85rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Data Loading
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

PKL_NAME_MAP = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost"
}

# Encoding maps (same as training)
MONTH_MAP = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6,
             "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
VISITOR_MAP = {"Returning_Visitor": 0, "New_Visitor": 1, "Other": 2}


@st.cache_data
def load_results():
    with open(os.path.join(MODEL_DIR, "results.json"), "r") as f:
        return json.load(f)


@st.cache_resource
def load_model(pkl_name):
    return joblib.load(os.path.join(MODEL_DIR, f"{pkl_name}.pkl"))


@st.cache_resource
def load_scaler():
    return joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))


def preprocess_uploaded(data, feature_names):
    """Preprocess uploaded CSV to match training format."""
    df = data.copy()

    # Encode Month if it's a string
    if "Month" in df.columns and df["Month"].dtype == object:
        df["Month"] = df["Month"].map(MONTH_MAP).fillna(0).astype(int)

    # Encode VisitorType if it's a string
    if "VisitorType" in df.columns and df["VisitorType"].dtype == object:
        df["VisitorType"] = df["VisitorType"].map(VISITOR_MAP).fillna(2).astype(int)

    # Encode Weekend if boolean
    if "Weekend" in df.columns:
        df["Weekend"] = df["Weekend"].astype(int)

    # Encode Revenue if present
    if "Revenue" in df.columns:
        df["Revenue"] = df["Revenue"].map({True: 1, False: 0, "TRUE": 1, "FALSE": 0, 1: 1, 0: 0}).fillna(0).astype(int)

    return df


results = load_results()
model_names = list(results["results"].keys())
feature_names = results["feature_names"]
scaler = load_scaler()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Control Panel")
    st.markdown("---")

    selected_model = st.selectbox(
        "🤖 Select Classification Model",
        model_names,
        index=model_names.index("Random Forest") if "Random Forest" in model_names else 0
    )

    st.markdown("---")
    st.markdown("##### 📊 Dataset Info")
    st.markdown("""
    - **Source:** UCI Repository
    - **Samples:** 12,330
    - **Features:** 17
    - **Classes:** Purchase / No Purchase
    - **Imbalance:** 84.5% No, 15.5% Yes
    """)

    st.markdown("---")
    st.markdown("##### 🧠 Models")
    for name in model_names:
        icon = "✅" if name == selected_model else "⬜"
        st.markdown(f"{icon} {name}")

# Load selected model
pkl_file = PKL_NAME_MAP.get(selected_model)
model = load_model(pkl_file)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">🛒 Online Shopper Purchase Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict whether an online shopping session will result in a purchase — Compare 6 ML classifiers across 6 metrics</p>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📤 Upload & Predict", "📈 Model Performance", "🔍 Compare All Models"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Upload & Predict
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📤 Upload Session Data for Prediction")
    st.markdown("Upload a CSV file with online shopping session features. The model will predict if each session leads to a **Purchase** or **No Purchase**.")

    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload a CSV with session features (Administrative, Informational, ProductRelated, etc.)"
        )

    with col_info:
        with st.expander("ℹ️ Expected CSV Format", expanded=False):
            st.markdown("""
            Your CSV should have these columns:
            - `Administrative`, `Administrative_Duration`
            - `Informational`, `Informational_Duration`
            - `ProductRelated`, `ProductRelated_Duration`
            - `BounceRates`, `ExitRates`, `PageValues`
            - `SpecialDay`, `Month`, `OperatingSystems`
            - `Browser`, `Region`, `TrafficType`
            - `VisitorType`, `Weekend`

            *`Revenue` column is optional (used to show accuracy).*
            """)

    if uploaded is not None:
        data = pd.read_csv(uploaded)
        data = preprocess_uploaded(data, feature_names)

        has_target = "Revenue" in data.columns

        st.markdown("#### 🔎 Data Preview")
        st.dataframe(data.head(10), use_container_width=True)
        st.caption(f"Showing first {min(10, len(data))} of {len(data)} rows")

        # Check missing features
        missing = [col for col in feature_names if col not in data.columns]
        if missing:
            st.error(f"❌ Missing required columns: **{', '.join(missing)}**")
        else:
            X_new = data[feature_names]
            X_scaled = scaler.transform(X_new)

            preds = model.predict(X_scaled)
            data["Prediction"] = ["🟢 Purchase" if p == 1 else "🔴 No Purchase" for p in preds]

            purchase_count = sum(preds == 1)
            no_purchase_count = sum(preds == 0)

            st.markdown("#### 🎯 Prediction Results")
            col_p, col_np, col_t = st.columns(3)
            col_p.metric("🟢 Purchase", purchase_count)
            col_np.metric("🔴 No Purchase", no_purchase_count)
            col_t.metric("📊 Total Sessions", len(preds))

            st.dataframe(data, use_container_width=True)

            csv_out = data.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv_out,
                file_name="shopper_predictions.csv",
                mime="text/csv"
            )

            if has_target:
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                st.markdown("#### 📊 Performance on Uploaded Data")

                y_true = data["Revenue"]
                y_pred = preds

                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1_val = f1_score(y_true, y_pred, zero_division=0)
                mcc_val = matthews_corrcoef(y_true, y_pred)

                m1, m2, m3, m4, m5 = st.columns(5)
                for col, label, val in zip(
                    [m1, m2, m3, m4, m5],
                    ["Accuracy", "Precision", "Recall", "F1 Score", "MCC"],
                    [acc, prec, rec, f1_val, mcc_val]
                ):
                    col.markdown(f"""
                    <div class="metric-card">
                        <h3>{label}</h3>
                        <h2>{val:.4f}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("")
                cm_fig, cm_ax = plt.subplots(figsize=(6, 5))
                cm = confusion_matrix(y_true, y_pred)
                sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", ax=cm_ax,
                            xticklabels=["No Purchase", "Purchase"],
                            yticklabels=["No Purchase", "Purchase"],
                            annot_kws={"size": 16})
                cm_ax.set_xlabel("Predicted Label", fontsize=12)
                cm_ax.set_ylabel("True Label", fontsize=12)
                cm_ax.set_title(f"Confusion Matrix — {selected_model}", fontsize=14, fontweight="bold")
                st.pyplot(cm_fig)

    else:
        st.info("👆 Upload a CSV file to get started. You can use the Online Shoppers dataset from UCI repository.")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"### 📈 Performance Dashboard — {selected_model}")
    st.markdown("Evaluation on held-out test set (20% split, stratified)")

    test_data = pd.read_csv(os.path.join(MODEL_DIR, "test_data.csv"))
    X_test = test_data[feature_names]
    y_test = test_data["Revenue"]

    X_scaled = scaler.transform(X_test)
    preds = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1_val = f1_score(y_test, preds, zero_division=0)
    mcc_val = matthews_corrcoef(y_test, preds)
    auc_val = roc_auc_score(y_test, proba) if proba is not None else 0.0

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    metric_data = [
        ("Accuracy", acc), ("AUC-ROC", auc_val), ("Precision", prec),
        ("Recall", rec), ("F1 Score", f1_val), ("MCC", mcc_val)
    ]
    for col, (label, val) in zip([c1, c2, c3, c4, c5, c6], metric_data):
        col.markdown(f"""
        <div class="metric-card">
            <h3>{label}</h3>
            <h2>{val:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### 🔲 Confusion Matrix")
        cm = confusion_matrix(y_test, preds)
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", ax=ax1,
                    xticklabels=["No Purchase", "Purchase"],
                    yticklabels=["No Purchase", "Purchase"],
                    annot_kws={"size": 18}, linewidths=2, linecolor="white")
        ax1.set_xlabel("Predicted Label", fontsize=12)
        ax1.set_ylabel("True Label", fontsize=12)
        ax1.set_title(f"{selected_model}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig1)

    with chart_col2:
        st.markdown("#### 📊 Metrics Radar View")
        labels_radar = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
        values_radar = [acc, auc_val, prec, rec, f1_val, max(0, mcc_val)]

        angles = np.linspace(0, 2 * np.pi, len(labels_radar), endpoint=False).tolist()
        values_radar += values_radar[:1]
        angles += angles[:1]

        fig_radar, ax_radar = plt.subplots(figsize=(6, 5), subplot_kw=dict(polar=True))
        ax_radar.fill(angles, values_radar, color="#00897b", alpha=0.25)
        ax_radar.plot(angles, values_radar, color="#00897b", linewidth=2, marker="o", markersize=6)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(labels_radar, fontsize=11)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title(f"{selected_model}", fontsize=14, fontweight="bold", pad=20)
        ax_radar.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_radar)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Compare All Models
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🔍 Multi-Model Comparison")
    st.markdown("Side-by-side comparison of all 6 classifiers on the test dataset.")

    rows = []
    for name, metrics in results["results"].items():
        rows.append({
            "Model": name,
            "Accuracy": metrics["Accuracy"],
            "AUC": metrics["AUC"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1"],
            "MCC": metrics["MCC"]
        })
    comparison_df = pd.DataFrame(rows)

    st.markdown("#### 📋 Results Table")
    styled_df = comparison_df.set_index("Model").style.highlight_max(
        axis=0, color="#c8e6c9"
    ).highlight_min(axis=0, color="#ffcdd2").format("{:.4f}")
    st.dataframe(styled_df, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("#### 📊 Performance Comparison Charts")
    metric_options = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    selected_metrics = st.multiselect(
        "Select metrics to compare:",
        metric_options,
        default=["Accuracy", "AUC", "F1"]
    )

    if selected_metrics:
        fig_bar, axes = plt.subplots(1, len(selected_metrics), figsize=(5 * len(selected_metrics), 5))
        if len(selected_metrics) == 1:
            axes = [axes]

        colors = ["#1a237e", "#0d47a1", "#0277bd", "#00897b", "#2e7d32", "#558b2f"]

        for ax, metric in zip(axes, selected_metrics):
            values = comparison_df[metric].values
            bars = ax.barh(comparison_df["Model"], values, color=colors, edgecolor="white", height=0.6)
            ax.set_xlabel(metric, fontsize=12)
            ax.set_title(metric, fontsize=14, fontweight="bold")
            ax.set_xlim(0, 1)
            ax.invert_yaxis()

            for bar, val in zip(bars, values):
                ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{val:.4f}", va="center", fontsize=10)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig_bar)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("#### 🔲 Confusion Matrices — All Models")
    cm_cols = st.columns(3)
    for i, name in enumerate(model_names):
        cm_data = np.array(results["confusion_matrices"][name])
        with cm_cols[i % 3]:
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(cm_data, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                        xticklabels=["No", "Yes"], yticklabels=["No", "Yes"],
                        annot_kws={"size": 14}, linewidths=1.5, linecolor="white")
            ax_cm.set_title(name, fontsize=12, fontweight="bold")
            ax_cm.set_xlabel("Predicted", fontsize=10)
            ax_cm.set_ylabel("Actual", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig_cm)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-text">
    🛒 <strong>Online Shopper Purchase Prediction</strong> — ML Assignment 2 | BITS Pilani WILP<br>
    Built with Streamlit • scikit-learn • XGBoost
</div>
""", unsafe_allow_html=True)
