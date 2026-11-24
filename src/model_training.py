# src/model_training.py
"""Model training script for the synthetic UPI fraud detection dataset.

Goal: Train an Isolation Forest to flag unusual transactions (anomaly detection).
The script follows a clear, beginner‑friendly flow:

1️⃣ Load the processed CSV (generated in Phase 5).
2️⃣ Select only numeric features that are safe for the model.
3️⃣ Split the data into a train and a test set (80 % / 20 %).
4️⃣ Train ``sklearn.ensemble.IsolationForest``.
5️⃣ Convert the model's ``-1/1`` predictions to ``0/1`` labels.
6️⃣ Evaluate the predictions against the true ``is_fraud`` column using
   common classification metrics.
7️⃣ Save the trained model (pickle) and a plain‑text evaluation report.

All paths are built with ``pathlib`` so the code works on any machine –
no hard‑coded absolute paths are used.
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
)
import joblib  # joblib is part of sklearn's dependencies and already installed

# ---------------------------------------------------------------------------
# 1️⃣ Load the processed dataset
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "processed" / "upi_transactions_features.csv"
print(f"Loading processed data from: {data_path}")

df = pd.read_csv(data_path)

# ---------------------------------------------------------------------------
# 2️⃣ Choose numeric features for the model (exclude identifiers / target)
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
# The true label – used **only** for evaluation later
y_true = df["is_fraud"]

print("Features used for training:")
for col in feature_cols:
    print(f" - {col}")

# ---------------------------------------------------------------------------
# 3️⃣ Train‑test split (80 % train, 20 % test) – fixed random seed for reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y_true, test_size=0.20, random_state=RANDOM_SEED, stratify=y_true
)
print(f"Training rows: {X_train.shape[0]}, Test rows: {X_test.shape[0]}")

# ---------------------------------------------------------------------------
# 4️⃣ Isolation Forest – an unsupervised anomaly detector
# ---------------------------------------------------------------------------
# ``contamination`` tells the model what proportion of the data we expect to be
# outliers. Because the synthetic data contains ~2 % fraud, we set it to 0.02.
# ``random_state`` ensures reproducible trees.
model = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=RANDOM_SEED,
)
model.fit(X_train)

# ---------------------------------------------------------------------------
# 5️⃣ Generate predictions for the **test** set only
# ---------------------------------------------------------------------------
# IsolationForest returns 1 for inliers (normal) and -1 for outliers (anomalies).
raw_pred = model.predict(X_test)
# Convert to binary labels matching our ``is_fraud`` column:
#   0 → normal, 1 → suspicious (anomaly)
# We invert the sign: -1 (anomaly) -> 1, 1 (normal) -> 0
y_pred = np.where(raw_pred == -1, 1, 0)

# ---------------------------------------------------------------------------
# 6️⃣ Evaluation – compare predictions to the ground‑truth ``is_fraud`` label
# ---------------------------------------------------------------------------
conf_mat = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, zero_division=0)

print("\n=== Model Evaluation on Test Set ===")
print("Confusion matrix (rows = true, cols = predicted):")
print(conf_mat)
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1‑score : {f1:.4f}")
print(f"Accuracy : {accuracy:.4f}")
print("\nClassification report:")
print(class_report)
print(f"Number of transactions flagged as suspicious: {y_pred.sum()}")

# ---------------------------------------------------------------------------
# 7️⃣ Save model and evaluation report
# ---------------------------------------------------------------------------
models_dir = project_root / "models"
models_dir.mkdir(parents=True, exist_ok=True)
model_path = models_dir / "isolation_forest.pkl"
joblib.dump(model, model_path)
print(f"Trained model saved to: {model_path}")

reports_dir = project_root / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)
report_path = reports_dir / "model_evaluation.txt"
with open(report_path, "w") as f:
    f.write("Isolation Forest model evaluation (test set)\n")
    f.write("=" * 45 + "\n\n")
    f.write("Confusion matrix (rows = true, cols = predicted):\n")
    f.write(str(conf_mat) + "\n\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1‑score : {f1:.4f}\n")
    f.write(f"Accuracy : {accuracy:.4f}\n\n")
    f.write("Classification report:\n")
    f.write(class_report)
    f.write("\nNumber of transactions flagged as suspicious: " + str(y_pred.sum()))
print(f"Evaluation report written to: {report_path}")

# ---------------------------------------------------------------------------
# 8️⃣ Reminder about synthetic nature of data
# ---------------------------------------------------------------------------
print("\nNOTE: This model is trained on synthetic data. While Isolation Forest can\n"
      "identify outliers in the simulated dataset, performance does NOT reflect\n"
      "real‑world fraud detection capabilities. Use this as a learning example\n"
      "only.")
""" End of script """
