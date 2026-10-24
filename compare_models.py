
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load the existing dataset
df = pd.read_csv("telecom_churn.csv")

# Remove rows where the target is missing
target = "Churn"
if target not in df.columns:
    raise ValueError(
        f"Column '{target}' not found. Available columns: {list(df.columns)}"
    )

df = df.dropna(subset=[target]).copy()
y = df.pop(target).astype(str).str.strip()

# Convert target labels to numeric classes
# Convert Churn labels to numeric values
y = y.astype(str).str.strip().str.lower()

y = y.map({
    "yes": 1,
    "no": 0,
    "1": 1,
    "0": 0,
    "true": 1,
    "false": 0
})

if y.isna().any():
    raise ValueError("Unexpected values found in the Churn column.")

y = y.astype(int)



# Remove customer ID columns if present
X = df.drop(
    columns=["customerID", "CustomerID", "customer_id"],
    errors="ignore"
)

# Encode text columns and fill missing values
X = pd.get_dummies(X, dummy_na=True)
X = X.replace([float("inf"), float("-inf")], float("nan"))
X = X.fillna(X.median(numeric_only=True)).fillna(0)

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced")
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, model in models.items():
    scores = cross_val_score(
        model, X, y, cv=cv, scoring="accuracy"
    )
    results.append({
        "Model": name,
        "Mean Accuracy": scores.mean(),
        "Standard Deviation": scores.std()
    })
    print(f"\n{name}")
    print(f"Mean accuracy: {scores.mean():.4f}")
    print(f"Standard deviation: {scores.std():.4f}")

# Save results
import os
os.makedirs("outputs", exist_ok=True)
results_df = pd.DataFrame(results)
results_df.to_csv("outputs/model_comparison.csv", index=False)

# Create comparison chart
plt.figure(figsize=(8, 5))
plt.bar(
    results_df["Model"],
    results_df["Mean Accuracy"],
    yerr=results_df["Standard Deviation"],
    capsize=5
)
plt.ylabel("Cross-validated accuracy")
plt.title("Logistic Regression vs Random Forest")
plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150)
plt.close()

# Write conclusion
best = results_df.loc[results_df["Mean Accuracy"].idxmax()]
stable = results_df.loc[results_df["Standard Deviation"].idxmin()]

with open("outputs/conclusion.txt", "w", encoding="utf-8") as file:
    file.write(
        f"Higher mean accuracy: {best['Model']} "
        f"({best['Mean Accuracy']:.4f}).\n"
        f"More stable model: {stable['Model']} "
        f"(standard deviation: {stable['Standard Deviation']:.4f}).\n"
        "A lower standard deviation means accuracy varied less across folds.\n"
    )

print("\nComparison completed!")
print("Check the outputs folder for model_comparison.csv,")
print("model_comparison.png, and conclusion.txt.")
