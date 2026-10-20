"""
Telecom Customer Churn Prediction
Uses a telecom-style sample CSV included with this project.
The included dataset is synthetic and intended for learning/practice.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    classification_report, ConfusionMatrixDisplay
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "telecom_churn.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    # 1. Import the churn CSV into pandas
    df = pd.read_csv(DATA_PATH)
    print("Dataset loaded successfully.")
    print(f"Rows and columns: {df.shape}")
    print("\nFirst five rows:")
    print(df.head())
    print("\nMissing values by column:")
    print(df.isnull().sum())

    # Remove identifier because it does not help predict churn.
    df = df.drop(columns=["customerID"], errors="ignore")
    target = "Churn"
    X = df.drop(columns=[target])
    y = df[target].map({"No": 0, "Yes": 1})

    # 2. Preprocess categorical variables with one-hot encoding.
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols)
    ])

    # 3. Split data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    # 4. Train a LogisticRegression classifier.
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # 5. Evaluate using accuracy, precision, recall, and confusion matrix.
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)

    print("\n--- Model Evaluation ---")
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=["No Churn", "Churn"], zero_division=0))

    metrics = pd.DataFrame([{
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }])
    metrics.to_csv(OUTPUT_DIR / "evaluation_metrics.csv", index=False)

    display = ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=["No Churn", "Churn"], cmap="Blues"
    )
    display.ax_.set_title("Telecom Customer Churn - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.show()
    print(f"\nSaved metrics and confusion matrix in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
