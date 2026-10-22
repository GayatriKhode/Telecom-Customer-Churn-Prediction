# Telecom Customer Churn Prediction — Feature Engineering

## Objective
Improve a Logistic Regression churn model by adding engineered features, scaling numeric values, retraining, evaluating metrics, and comparing the result with a baseline model.

## Features added
- `TenureMonthlyCharges` = `tenure` × `MonthlyCharges` (when both columns exist)
- `LogTotalCharges` = `log(1 + TotalCharges)` for non-negative numeric values (when `TotalCharges` exists)
- Numeric columns are imputed with the median and standardized with `StandardScaler`.
- Categorical columns are imputed with the most frequent value and one-hot encoded.
- Customer ID columns are excluded from model training when recognized.

## Project structure
```text
telecom_churn_feature_engineering/
├── data/
│   └── telecom_churn.csv       # Add your dataset here; usually don't commit private/raw data
├── output/                     # Generated metrics and predictions
├── models/                     # Reserved for saved model artifacts
├── feature_engineering.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup in VS Code (Windows PowerShell)
Open the project folder in VS Code, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, you can skip activation and run:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Add your dataset
Place your CSV in `data/` and name it `telecom_churn.csv`, or use the actual file path in the command. The dataset should contain a churn target column named `Churn` or `churn` (also supports `Exited` or `Attrition`). Common target values such as `Yes/No` or `1/0` are supported.

**Important:** The script does not include a dataset. Use the dataset you already have, and check that the target column is present.

## Run
```powershell
python feature_engineering.py --data "data/telecom_churn.csv"
```

If your CSV is inside a nested folder, update the path, for example:
```powershell
python feature_engineering.py --data "data/telecom/telecom_churn.csv"
```

## Outputs
After a successful run, `output/` contains:
- `feature_engineered_metrics.json` — metrics and confusion matrix
- `model_comparison.csv` — improved model metrics, plus baseline metrics if supplied
- `test_predictions.csv` — actual label, predicted label, and churn probability for each test row
- `conclusion.txt` — readable results summary

## Compare with your Day 12 baseline
If you have baseline metrics, create `output/baseline_metrics.json` with this structure (replace the example values with your real Day 12 results):

```json
{
  "accuracy": 0.80,
  "precision": 0.60,
  "recall": 0.55,
  "f1_score": 0.57,
  "roc_auc": 0.72
}
```

Then run:
```powershell
python feature_engineering.py --data "data/telecom_churn.csv" --baseline "output/baseline_metrics.json"
```

Do not use the example scores as real results. If you don't have Day 12 metrics, run the model without `--baseline`; the comparison CSV will contain the new model's results only.

## GitHub notes
- Do not commit passwords, tokens, or sensitive customer data.
- The raw CSV is excluded by `.gitignore` by default. If your instructor requires a sample dataset, use a public or anonymized dataset and follow its license.
- Generated output files are ignored by default; you can selectively add non-sensitive results if you want to show them in your repository.
