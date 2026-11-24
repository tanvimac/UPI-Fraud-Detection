"""Version 2 model training script.

- Uses the processed Version 2 dataset (`upi_transactions_features_v2.csv`).
- Identical feature set to the original model.
- 80/20 stratified split, random_state=42.
- IsolationForest with the same parameters as the original.
- Saves the model to `models/isolation_forest_v2.pkl`.
- Writes an evaluation report to `reports/model_evaluation_v2.txt`.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
    roc_auc_score,
)
import joblib

# ---------------------------------------------------------------------------
# 1️⃣ Load Version 2 processed data
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "processed" / "upi_transactions_features_v2.csv"
print(f"Loading processed V2 data from: {data_path}")

df = pd.read_csv(data_path)

# ---------------------------------------------------------------------------
# 2️⃣ Verify dataset integrity
# ---------------------------------------------------------------------------
assert len(df) == 10_000, f"Expected 10,000 rows, got {len(df)}"
fraud_counts = df["is_fraud"].value_counts().to_dict()
assert fraud_counts.get(1, 0) / len(df) == 0.02, "Fraud ratio not 2%"
assert not df.isnull().any().any(), "Missing values detected"
assert df["transaction_id"].is_unique, "Duplicate transaction IDs found"

# ---------------------------------------------------------------------------
# 3️⃣ Feature preparation
# ---------------------------------------------------------------------------
feature_cols = [
    "amount",
    "transactions_last_1hr",
    "transaction_hour",
    "transaction_day",
    "is_weekend",
    "is_night",
    "log_amount",
]
X = df[feature_cols]
y = df["is_fraud"]

# Ensure no infinite values in numeric columns
assert not np.isinf(X.select_dtypes(include=[np.float64, np.float32])).any().any(), "Infinite values in features"

# ---------------------------------------------------------------------------
# 4️⃣ Train‑test split (80/20 stratified)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
)
# Sanity checks for split
assert X_train.shape[0] == 8000, f"Unexpected training size {X_train.shape[0]}"
assert X_test.shape[0] == 2000, f"Unexpected test size {X_test.shape[0]}"
assert y_train.sum() == 160, f"Unexpected fraud count in training {y_train.sum()}"
assert y_test.sum() == 40, f"Unexpected fraud count in test {y_test.sum()}"

print(f"Training rows: {X_train.shape[0]}, Test rows: {X_test.shape[0]}")

# ---------------------------------------------------------------------------
# 5️⃣ Train Isolation Forest on training data
# ---------------------------------------------------------------------------
model = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=RANDOM_SEED,
)
model.fit(X_train)

# ---------------------------------------------------------------------------
# 6️⃣ Predictions on test set
# ---------------------------------------------------------------------------
raw_pred = model.predict(X_test)
# Map -1 → 1 (fraud), 1 → 0 (normal)
y_pred = np.where(raw_pred == -1, 1, 0)

# Sanity checks after predictions
assert len(y_pred) == len(y_test), f"Prediction length {len(y_pred)} differs from true label length {len(y_test)}"

# ---------------------------------------------------------------------------
# 7️⃣ Evaluation metrics
# ---------------------------------------------------------------------------
conf_mat = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
class_report = classification_report(y_test, y_pred, zero_division=0)
# ROC‑AUC using negative decision function (higher score = more anomalous)
try:
    scores = -model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, scores)
except Exception:
    roc_auc = None

# Final sanity check: confusion matrix totals must equal test size
assert conf_mat.sum() == len(y_test), f"Confusion matrix total {conf_mat.sum()} does not match test size {len(y_test)}"

# ---------------------------------------------------------------------------
# 8️⃣ Save model and evaluation report
# ---------------------------------------------------------------------------
models_dir = project_root / "models"
models_dir.mkdir(parents=True, exist_ok=True)
model_path = models_dir / "isolation_forest_v2.pkl"
joblib.dump(model, model_path)
print(f"Version 2 model saved to: {model_path}")

reports_dir = project_root / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)
report_path = reports_dir / "model_evaluation_v2.txt"
with open(report_path, "w") as f:
    f.write("Isolation Forest Model Evaluation – Version 2\n")
    f.write("=" * 45 + "\n\n")
    f.write(f"Dataset rows: {len(df)} (fraud {fraud_counts.get(1,0)} / {fraud_counts.get(0,0)})\n")
    f.write("Features used: " + ", ".join(feature_cols) + "\n\n")
    f.write("Confusion Matrix (true vs pred):\n")
    f.write(str(conf_mat) + "\n\n")
    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1-score : {f1:.4f}\n")
    if roc_auc is not None:
        f.write(f"ROC‑AUC  : {roc_auc:.4f}\n")
    else:
        f.write("ROC‑AUC  : not computed\n")
    f.write("\nClassification Report:\n")
    f.write(class_report)
    f.write("\nFalse Positives (normal predicted as fraud): " + str(conf_mat[0,1]) + "\n")
    f.write("False Negatives (fraud predicted as normal): " + str(conf_mat[1,0]) + "\n")
    f.write("\nNote: Fraud scenario labels were not stored in the dataset; per‑scenario analysis is not available.\n")

print(f"Evaluation report written to {report_path}")
