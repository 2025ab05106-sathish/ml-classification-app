"""
Training script for Online Shoppers Purchasing Intention classification.
Trains 6 ML models, evaluates with 6 metrics, saves everything.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix)
import joblib
import json
import os

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_and_preprocess():
    """Load the Online Shoppers dataset and preprocess it."""
    df = pd.read_csv(os.path.join(SAVE_DIR, "online_shoppers_intention.csv"))

    # Convert target to int (True/False -> 1/0)
    df["Revenue"] = df["Revenue"].astype(int)

    # Encode categorical columns
    # Month -> numeric
    month_map = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6,
                 "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
    df["Month"] = df["Month"].map(month_map).fillna(0).astype(int)

    # VisitorType -> numeric
    visitor_map = {"Returning_Visitor": 0, "New_Visitor": 1, "Other": 2}
    df["VisitorType"] = df["VisitorType"].map(visitor_map).fillna(2).astype(int)

    # Weekend -> int
    df["Weekend"] = df["Weekend"].astype(int)

    # Features and target
    feature_cols = [c for c in df.columns if c != "Revenue"]
    X = df[feature_cols]
    y = df["Revenue"]

    return X, y, feature_cols


def get_models():
    """Return dict of model name -> (model object, pkl filename)."""
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            "logistic_regression"
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            "decision_tree"
        ),
        "KNN": (
            KNeighborsClassifier(n_neighbors=5),
            "knn"
        ),
        "Naive Bayes": (
            GaussianNB(),
            "naive_bayes"
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=100, random_state=42),
            "random_forest"
        ),
        "XGBoost": (
            XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss"),
            "xgboost"
        )
    }
    return models


def calc_metrics(y_true, y_pred, y_proba=None):
    """Calculate all 6 evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    if y_proba is not None:
        auc = roc_auc_score(y_true, y_proba)
    else:
        auc = 0.0

    return {
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4)
    }


def main():
    print("Loading and preprocessing data...")
    X, y, feature_cols = load_and_preprocess()
    print(f"Total samples: {len(X)}, Features: {len(feature_cols)}")
    print(f"Class distribution: Purchase={sum(y==1)}, No Purchase={sum(y==0)}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.pkl"))

    models = get_models()
    all_metrics = {}
    all_cm = {}

    print("\nTraining models...\n")

    for name, (model, pkl_name) in models.items():
        print(f"  Training {name}...", end=" ")
        model.fit(X_train_sc, y_train)

        y_pred = model.predict(X_test_sc)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_sc)[:, 1]

        metrics = calc_metrics(y_test, y_pred, y_proba)
        all_metrics[name] = metrics
        all_cm[name] = confusion_matrix(y_test, y_pred).tolist()

        joblib.dump(model, os.path.join(SAVE_DIR, f"{pkl_name}.pkl"))
        print(f"Accuracy: {metrics['Accuracy']}, AUC: {metrics['AUC']}")

    # Save test data
    test_df = X_test.copy()
    test_df["Revenue"] = y_test.values
    test_df.to_csv(os.path.join(SAVE_DIR, "test_data.csv"), index=False)

    # Save results
    results = {
        "results": all_metrics,
        "confusion_matrices": all_cm,
        "feature_names": feature_cols
    }
    with open(os.path.join(SAVE_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nDone! All models saved.")
    print("\nResults summary:")
    print("-" * 80)
    print(f"{'Model':<22} {'Acc':>8} {'AUC':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'MCC':>8}")
    print("-" * 80)
    for name, m in all_metrics.items():
        print(f"{name:<22} {m['Accuracy']:>8} {m['AUC']:>8} {m['Precision']:>8} {m['Recall']:>8} {m['F1']:>8} {m['MCC']:>8}")


if __name__ == "__main__":
    main()
