# Telecom Customer Churn Prediction

## Aim
Build a Logistic Regression machine-learning model to predict whether a telecom customer may leave the company (churn).

## Project steps
1. Load the CSV dataset with pandas.
2. Preprocess categorical features using one-hot encoding.
3. Split the data into training and testing sets.
4. Train a Logistic Regression classifier.
5. Evaluate accuracy, precision, recall, and plot a confusion matrix.

## Files
- `telecom_churn.csv` — telecom-style sample dataset.
- `churn_prediction.py` — complete model implementation.
- `requirements.txt` — Python libraries required.
- `outputs/` — evaluation files created after running the script.

**Dataset note:** The included CSV is synthetic (generated for practice), not real customer information. If your course/platform supplies an official churn CSV, replace this file with that dataset and keep the column names compatible with the script.

## Run in VS Code (Windows)
1. Extract the ZIP file.
2. Open the extracted `Telecom_Customer_Churn_Prediction` folder in VS Code.
3. Open Terminal → New Terminal.
4. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
5. Run the model:
   ```bash
   python churn_prediction.py
   ```
6. When it finishes, check the `outputs` folder for `evaluation_metrics.csv` and `confusion_matrix.png`.

If `python` does not work on your computer, try `py` instead:
```bash
py -m pip install -r requirements.txt
py churn_prediction.py
```
