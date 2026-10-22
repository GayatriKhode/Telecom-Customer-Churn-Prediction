"""
Telecom Customer Churn Prediction — Feature Engineering
Run from the project root:
    python feature_engineering.py --data data/telecom_churn.csv
Optional baseline metrics JSON:
    python feature_engineering.py --data data/telecom_churn.csv --baseline output/baseline_metrics.json
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

TARGET_CANDIDATES = ["Churn", "churn", "Exited", "Attrition"]
ID_COLUMNS = ["customerID", "CustomerID", "customer_id", "CustomerId"]

def find_target(df):
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        "Could not find the churn target column. Expected one of: "
        + ", ".join(TARGET_CANDIDATES)
    )

def to_binary_target(series):
    """Convert common Yes/No or 1/0 target formats into 0/1."""
    if pd.api.types.is_numeric_dtype(series):
        vals = set(series.dropna().unique())
        if vals.issubset({0, 1}):
            return series.astype("Int64").astype(int)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1, "no": 0, "true": 1, "false": 0,
        "1": 1, "0": 0, "churn": 1, "stay": 0, "left": 1, "retained": 0
    }
    converted = normalized.map(mapping)
    if converted.isna().any():
        bad = sorted(normalized[converted.isna()].unique().tolist())
        raise ValueError(f"Unsupported target values: {bad}. Please map them to Yes/No or 1/0.")
    return converted.astype(int)

def engineer_features(df):
    df = df.copy()
    # Standard Telco Customer Churn column names.
    if {"tenure", "MonthlyCharges"}.issubset(df.columns):
        df["TenureMonthlyCharges"] = (
            pd.to_numeric(df["tenure"], errors="coerce")
            * pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        )
    if "TotalCharges" in df.columns:
        total = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # log1p requires non-negative values; invalid/negative values become missing.
        total = total.where(total >= 0)
        df["LogTotalCharges"] = np.log1p(total)
        df["TotalCharges"] = total
    return df

def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipe, categorical_cols))
    if not transformers:
        raise ValueError("No usable feature columns were found.")
    return ColumnTransformer(transformers=transformers, remainder="drop")

def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    result = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, pred, zero_division=0)),
    }
    if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
        result["roc_auc"] = float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    else:
        result["roc_auc"] = None
    result["confusion_matrix"] = confusion_matrix(y_test, pred).tolist()
    result["classification_report"] = classification_report(
        y_test, pred, zero_division=0, output_dict=True
    )
    return result

def main():
    parser = argparse.ArgumentParser(description="Engineer features and evaluate churn prediction.")
    parser.add_argument("--data", default="data/telecom_churn.csv", help="Path to input CSV")
    parser.add_argument("--baseline", default=None, help="Optional JSON file containing baseline metrics")
    parser.add_argument("--output-dir", default="output", help="Folder for output files")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"Dataset not found: {args.data}\n"
            "Place your telecom churn CSV at that path, or pass --data with its actual location."
        )
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.data)
    if df.empty:
        raise ValueError("The CSV file contains no rows.")

    target_col = find_target(df)
    y = to_binary_target(df[target_col])
    X = df.drop(columns=[target_col] + [c for c in ID_COLUMNS if c in df.columns])
    X = engineer_features(X)

    # Remove columns that are entirely empty; imputers cannot learn from them.
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        raise ValueError("No feature columns remain after removing target and ID columns.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    pipeline = Pipeline([
        ("preprocess", build_preprocessor(X_train)),
        ("model", LogisticRegression(max_iter=2000, class_weight=None, random_state=args.random_state))
    ])
    pipeline.fit(X_train, y_train)
    metrics = evaluate(pipeline, X_test, y_test)

    # Save artifacts and metrics.
    metrics_path = os.path.join(args.output_dir, "feature_engineered_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = pd.DataFrame([{
        "model": "Logistic Regression (engineered features + scaling)",
        **{k: metrics[k] for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]}
    }])
    comparison_path = os.path.join(args.output_dir, "model_comparison.csv")
    if args.baseline and os.path.exists(args.baseline):
        with open(args.baseline, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        baseline_row = {"model": "Baseline (Day 12)"}
        for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            baseline_row[key] = baseline.get(key)
        report = pd.concat([pd.DataFrame([baseline_row]), report], ignore_index=True)
    report.to_csv(comparison_path, index=False)

    # Export test predictions without including personally identifying customer IDs.
    pred = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    pd.DataFrame({
        "actual_churn": y_test.to_numpy(),
        "predicted_churn": pred,
        "churn_probability": probabilities
    }).to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)

    with open(os.path.join(args.output_dir, "conclusion.txt"), "w", encoding="utf-8") as f:
        f.write("TELECOM CUSTOMER CHURN — FEATURE ENGINEERING RESULTS\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Rows loaded: {len(df)}\n")
        f.write(f"Target column: {target_col}\n")
        f.write(f"Features used before preprocessing: {X.shape[1]}\n")
        f.write(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}\n\n")
        f.write("Evaluation metrics (held-out test set):\n")
        for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            val = metrics.get(key)
            f.write(f"- {key}: {val:.4f}\n" if val is not None else f"- {key}: unavailable\n")
        f.write("\nConfusion matrix [[TN, FP], [FN, TP]]:\n")
        f.write(str(metrics["confusion_matrix"]) + "\n\n")
        if args.baseline and os.path.exists(args.baseline):
            f.write("Baseline comparison saved to output/model_comparison.csv.\n")
        else:
            f.write(
                "No baseline metrics were supplied. To compare with Day 12, save its metrics as "
                "output/baseline_metrics.json and rerun with --baseline output/baseline_metrics.json.\n"
            )
        f.write("\nNote: Results depend on the dataset and train/test split. Do not compare scores "
                "from different splits as if they were identical experiments.\n")

    print("Model training and evaluation completed.")
    print(f"Metrics: {metrics_path}")
    print(f"Comparison: {comparison_path}")
    print(f"Predictions: {os.path.join(args.output_dir, 'test_predictions.csv')}")
    print(f"Conclusion: {os.path.join(args.output_dir, 'conclusion.txt')}")
    print("\nTest metrics:")
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        val = metrics.get(key)
        print(f"  {key}: {val:.4f}" if val is not None else f"  {key}: unavailable")

if __name__ == "__main__":
    main()
