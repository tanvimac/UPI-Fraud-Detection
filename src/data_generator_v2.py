# src/data_generator_v2.py
"""Version 2 synthetic UPI transaction data generator.

Creates a more realistic dataset while keeping the same schema:
- 10 000 rows
- ~2 % fraud (200 rows)
- Overlapping amount distributions
- Mixed time, frequency, device and merchant patterns

The script is deterministic (fixed seed) and writes two CSV files:
- data/raw/upi_transactions_v2.csv (raw transactions)
- data/processed/upi_transactions_features_v2.csv (engineered features)

It deliberately leaves the original `upi_transactions.csv` and its
processed version untouched.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration (same size, same fraud ratio, reproducible seed)
# ---------------------------------------------------------------------------
NUM_ROWS = 10_000
FRAUD_RATIO = 0.02  # 2 % fraud → 200 rows
SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Helper categorical data (same as original generator)
# ---------------------------------------------------------------------------
user_ids = np.arange(1, 2001)  # 2000 distinct users
merchant_categories = [
    "Food",
    "Travel",
    "Shopping",
    "Entertainment",
    "Bills",
    "Health",
    "Education",
    "Gifts",
    "Other",
]
device_types = ["Mobile", "Web", "POS", "ATM"]
locations = ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad"]

# ---------------------------------------------------------------------------
# 1️⃣ Generate base (mostly normal) transactions
# ---------------------------------------------------------------------------
users = np.random.choice(user_ids, size=NUM_ROWS)
merchant = np.random.choice(merchant_categories, size=NUM_ROWS)
device = np.random.choice(device_types, size=NUM_ROWS)
loc = np.random.choice(locations, size=NUM_ROWS)

# Transaction type – debit is far more common than credit
trans_type = np.random.choice(["debit", "credit"], size=NUM_ROWS, p=[0.9, 0.1])

# Amount – log‑normal distribution (most <5 000 INR)
normal_amount = np.random.lognormal(mean=5, sigma=1.2, size=NUM_ROWS)
normal_amount = np.clip(normal_amount, 10, 10_000)  # keep within realistic bounds

# Transaction time – uniform over last 30 days
now = datetime.now()
start_date = now - timedelta(days=30)
random_seconds = np.random.randint(0, int((now - start_date).total_seconds()), size=NUM_ROWS)
trans_time = [(start_date + timedelta(seconds=int(s))).isoformat() for s in random_seconds]

# Transactions in the last hour – Poisson λ=1 (normal behaviour)
transactions_last_1hr = np.random.poisson(lam=1, size=NUM_ROWS)

# ---------------------------------------------------------------------------
# 2️⃣ Select fraud rows and apply varied fraud scenarios
# ---------------------------------------------------------------------------
num_fraud = int(NUM_ROWS * FRAUD_RATIO)
fraud_indices = np.random.choice(NUM_ROWS, size=num_fraud, replace=False)

# Define fraud scenarios (weights sum to 1)
#   0 – High‑amount & high‑frequency (classic)
#   1 – Moderate amount, mixed time
#   2 – Night‑time with normal amount
#   3 – Rare device/merchant combo with normal amount
fraud_scenarios = np.random.choice([0, 1, 2, 3], size=num_fraud, p=[0.4, 0.3, 0.2, 0.1])

for idx, scenario in zip(fraud_indices, fraud_scenarios):
    # Base values (may be overwritten by scenario)
    amount = normal_amount[idx]
    recent = transactions_last_1hr[idx]
    ts = datetime.fromisoformat(trans_time[idx])
    dev = device[idx]
    merch = merchant[idx]

    if scenario == 0:
        # Classic high‑amount & high‑frequency fraud
        amount = np.random.uniform(50_000, 200_000)  # very high amount
        recent = np.random.randint(5, 21)  # high recent count
        # Late‑night time (0‑4 am)
        hour = np.random.randint(0, 5)
        ts = ts.replace(hour=hour, minute=np.random.randint(0, 60), second=np.random.randint(0, 60))
        # Rare device / merchant
        dev = np.random.choice(["POS", "ATM"])
        merch = np.random.choice(["Gifts", "Other"])
    elif scenario == 1:
        # Moderate amount fraud – overlap with normal distribution
        amount = np.random.uniform(5_000, 30_000)
        recent = np.random.randint(2, 6)  # modestly high recent count
        # Random hour (could be day or night)
        hour = np.random.randint(0, 24)
        ts = ts.replace(hour=hour, minute=np.random.randint(0, 60), second=np.random.randint(0, 60))
        # Device / merchant may be common
        dev = np.random.choice(device_types)
        merch = np.random.choice(merchant_categories)
    elif scenario == 2:
        # Night‑time fraud with otherwise normal amount
        amount = np.random.uniform(1_000, 12_000)  # overlaps normal range
        recent = np.random.randint(0, 4)  # could be low
        hour = np.random.randint(0, 5)  # late night
        ts = ts.replace(hour=hour, minute=np.random.randint(0, 60), second=np.random.randint(0, 60))
        # Device / merchant can be common
        dev = np.random.choice(device_types)
        merch = np.random.choice(merchant_categories)
    else:  # scenario == 3
        # Rare device/merchant combo, amount within normal range
        amount = np.random.uniform(1_000, 15_000)
        recent = np.random.randint(0, 3)
        # Keep original timestamp
        # Choose rare device & merchant
        dev = np.random.choice(["POS", "ATM"])
        merch = np.random.choice(["Gifts", "Other"])

    # Apply the generated values back to the arrays
    normal_amount[idx] = amount
    transactions_last_1hr[idx] = recent
    trans_time[idx] = ts.isoformat()
    device[idx] = dev
    merchant[idx] = merch

# ---------------------------------------------------------------------------
# 3️⃣ Inject a few anomalous normal rows (noise) – 5 % of non‑fraud rows
# ---------------------------------------------------------------------------
num_normal = NUM_ROWS - num_fraud
noise_count = int(0.05 * num_normal)  # 5 % of normal rows
noise_indices = np.random.choice([i for i in range(NUM_ROWS) if i not in fraud_indices],
                                 size=noise_count, replace=False)
for idx in noise_indices:
    # Randomly decide which anomaly to insert
    choice = np.random.choice(["high_amount", "night_time"])
    if choice == "high_amount":
        normal_amount[idx] = np.random.uniform(10_000, 30_000)  # overlap high tail
    else:  # night_time
        ts = datetime.fromisoformat(trans_time[idx])
        hour = np.random.randint(0, 5)
        ts = ts.replace(hour=hour, minute=np.random.randint(0, 60), second=np.random.randint(0, 60))
        trans_time[idx] = ts.isoformat()

# ---------------------------------------------------------------------------
# 4️⃣ Assemble DataFrame and write raw CSV (Version 2)
# ---------------------------------------------------------------------------
transaction_id = [f"TX{str(i).zfill(7)}" for i in range(1, NUM_ROWS + 1)]

df = pd.DataFrame({
    "transaction_id": transaction_id,
    "user_id": users,
    "amount": np.round(normal_amount, 2),
    "transaction_time": trans_time,
    "transaction_type": trans_type,
    "merchant_category": merchant,
    "device_type": device,
    "location": loc,
    "transactions_last_1hr": transactions_last_1hr,
    "is_fraud": 0,
})
# Mark fraud rows
df.loc[fraud_indices, "is_fraud"] = 1

# Ensure no duplicate IDs (shouldn't happen) and no missing values
assert df["transaction_id"].is_unique, "Duplicate transaction IDs detected"
assert not df.isnull().any().any(), "Missing values found"

raw_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "upi_transactions_v2.csv"
raw_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(raw_path, index=False)
print(f"Version 2 raw dataset written to {raw_path}")

# ---------------------------------------------------------------------------
# 5️⃣ Run the existing feature engineering on the new raw file
# ---------------------------------------------------------------------------
# The original feature_engineering script expects the raw path at
#   ../../data/raw/upi_transactions.csv .  We'll temporarily copy the v2 file
#   to that location, run the script, then move the processed output.
# This avoids modifying the original script.

# Temporary copy
temp_raw = Path(__file__).resolve().parent.parent / "data" / "raw" / "upi_transactions.csv"
backup_raw = None
if temp_raw.exists():
    backup_raw = temp_raw.with_name("upi_transactions_backup_before_v2.csv")
    temp_raw.rename(backup_raw)
raw_path.rename(temp_raw)  # place v2 as the expected file

# Execute feature engineering script
os.system(".venv/bin/python src/feature_engineering.py")

# Move processed file to version‑2 name
processed_src = Path(__file__).resolve().parent.parent / "data" / "processed" / "upi_transactions_features.csv"
processed_v2 = Path(__file__).resolve().parent.parent / "data" / "processed" / "upi_transactions_features_v2.csv"
processed_src.rename(processed_v2)

# Restore original raw file if we backed it up
if backup_raw:
    backup_raw.rename(temp_raw)
else:
    # Remove the temporary raw file (original was absent)
    temp_raw.unlink()

print(f"Version 2 processed dataset written to {processed_v2}")
