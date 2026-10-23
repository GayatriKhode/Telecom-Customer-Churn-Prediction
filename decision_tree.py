
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from pathlib import Path

# Load dataset
df = pd.read_csv("telecom_churn.csv")

# Find the target column
target = next(
    (c for c in df.columns if c.strip().lower() in ["churn", "exited"]),
    None
)

if target is None:
    raise ValueError(f"Churn column not found. Columns: {list(df.columns)}")

# Remove customer ID
df = df.drop(
    columns=[c for c in df.columns
             if c.strip().lower() in ["customerid", "customer_id"]],
    errors="ignore"
)

# Prepare features and target
y = df[target]
X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)

if y.dtype == "object":
    y = y.astype(str).str.strip().map(
        {"Yes": 1, "No": 0, "1": 1, "0": 0}
    )

if y.isna().any():
    raise ValueError("Check the values in your churn column.")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

print("Decision Tree Accuracy:", f"{model.score(X_test, y_test):.2%}")

# Save tree diagram
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

plt.figure(figsize=(22, 12))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No Churn", "Churn"],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.tight_layout()
plt.savefig(output_dir / "decision_tree.png", dpi=200)
plt.close()

print("First split feature:", X.columns[model.tree_.feature[0]])
print("Saved: outputs/decision_tree.png")

