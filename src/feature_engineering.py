# src/feature_engineering.py
"""Feature engineering for the synthetic UPI transaction dataset.

This script loads the raw CSV generated in Phase 3, adds a handful of
simple, beginner‑friendly features, saves the enriched data to the
*processed* folder, and prints a quick validation summary.

All paths are built with ``pathlib`` so the code works on any machine –
no hard‑coded absolute paths.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# 1️⃣ Load the raw dataset (relative to the project root)
# ---------------------------------------------------------------------------
# The script lives in ``src``; the raw CSV is two levels up: ``../data/raw``
project_root = Path(__file__).resolve().parent.parent
raw_path = project_root / "data" / "raw" / "upi_transactions.csv"
print(f"Loading data from: {raw_path}")

df = pd.read_csv(raw_path)

# ---------------------------------------------------------------------------
# 2️⃣ Create new features
# ---------------------------------------------------------------------------
# Convert the timestamp column to a proper datetime object.
# ``errors='coerce'`` will turn any unparsable value into NaT (later caught).
# Convert the timestamp column to a proper datetime object.
# Use ``errors='coerce'`` to handle both formats (with/without milliseconds).
# After coercion we explicitly fill any remaining NaT values by parsing the simpler format.
transaction_dt = pd.to_datetime(df["transaction_time"], errors="coerce")
# If any timestamps are still NaT (unlikely), parse them with a strict format.
if transaction_dt.isna().any():
    fallback_mask = transaction_dt.isna()
    transaction_dt.loc[fallback_mask] = pd.to_datetime(
        df.loc[fallback_mask, "transaction_time"], format="%Y-%m-%dT%H:%M:%S"
    )

# a) Hour of the day (0‑23)
# b) Day of week (Monday=0 … Sunday=6)
# c) Weekend flag (Saturday=5, Sunday=6)
# d) Night flag – transactions between 00:00 and 04:59 are considered night.
# e) Log‑transformed amount – natural log, safe because amounts are > 0.

df["transaction_hour"] = transaction_dt.dt.hour
df["transaction_day"] = transaction_dt.dt.dayofweek
df["is_weekend"] = df["transaction_day"].apply(lambda d: 1 if d >= 5 else 0)
df["is_night"] = df["transaction_hour"].apply(lambda h: 1 if 0 <= h <= 4 else 0)
df["log_amount"] = np.log(df["amount"])  # np.log returns a float array

# ---------------------------------------------------------------------------
# 3️⃣ Save the enriched dataframe to the processed folder
# ---------------------------------------------------------------------------
processed_dir = project_root / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)
output_path = processed_dir / "upi_transactions_features.csv"
df.to_csv(output_path, index=False)
print(f"Saved processed data to: {output_path}")

# ---------------------------------------------------------------------------
# 4️⃣ Validation checks
# ---------------------------------------------------------------------------
print("\n=== Validation ===")
# Row count
row_count = len(df)
print(f"Rows: {row_count}")
# Column check – original columns plus the five new ones
expected_new = {"transaction_hour", "transaction_day", "is_weekend",
                "is_night", "log_amount"}
missing_new = expected_new - set(df.columns)
print(f"Missing new features: {missing_new if missing_new else 'None'}")
# Original columns list (from the raw CSV)
original_cols = ["transaction_id", "user_id", "amount", "transaction_time",
                 "transaction_type", "merchant_category", "device_type",
                 "location", "transactions_last_1hr", "is_fraud"]
missing_original = set(original_cols) - set(df.columns)
print(f"Missing original columns: {missing_original if missing_original else 'None'}")
# Missing values
missing_vals = df.isnull().sum()
print("Missing values per column (should all be 0):")
print(missing_vals.to_dict())
# Infinite values (e.g., from log of 0 – not possible here, but we check)
inf_mask = np.isinf(df.select_dtypes(include=[np.float64, np.float32]))
has_inf = inf_mask.any().any()
print(f"Infinite values present: {has_inf}")
# Unique transaction IDs
duplicate_ids = df["transaction_id"].duplicated().sum()
print(f"Duplicate transaction_id count: {duplicate_ids}")
# Fraud label counts – should match the original 200 fraud, 9800 normal
fraud_counts = df["is_fraud"].value_counts().sort_index()
print("Fraud label distribution (0=normal, 1=fraud):")
print(fraud_counts.to_dict())

# ---------------------------------------------------------------------------
# 5️⃣ Show a quick preview of the first five rows
# ---------------------------------------------------------------------------
print("\nFirst five rows of the processed dataset:")
print(df.head().to_string(index=False))
""" End of script """
