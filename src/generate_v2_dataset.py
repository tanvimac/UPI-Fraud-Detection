# src/generate_v2_dataset.py
"""Generate Version 2 synthetic UPI dataset.

Creates:
- data/raw/upi_transactions_v2.csv   (raw data)
- data/processed/upi_transactions_features_v2.csv   (engineered features)

The script leaves the original Version 1 files untouched.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration (same size and fraud ratio as original)
# ---------------------------------------------------------------------------
NUM_ROWS = 10_000
FRAUD_RATIO = 0.02  # 2 % fraud → 200 rows
SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Categorical helpers (same as original generator)
# ---------------------------------------------------------------------------
user_ids = np.arange(1, 2001)
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
# 1️⃣ Base (mostly normal) transactions
# ---------------------------------------------------------------------------
users = np.random.choice(user_ids, size=NUM_ROWS)
merchant = np.random.choice(merchant_categories, size=NUM_ROWS)
device = np.random.choice(device_types, size=NUM_ROWS)
loc = np.random.choice(locations, size=NUM_ROWS)

trans_type = np.random.choice(["debit", "credit"], size=NUM_ROWS, p=[0.9, 0.1])

# Amount – log‑normal (most <5 000 INR), clipped between 10 and 10 000
normal_amount = np.random.lognormal(mean=5, sigma=1.2, size=NUM_ROWS)
normal_amount = np.clip(normal_amount, 10, 10_000)

# Timestamp – uniform over last 30 days
now = datetime.now()
start_date = now - timedelta(days=30)
rand_secs = np.random.randint(0, int((now - start_date).total_seconds()), size=NUM_ROWS)
trans_time = [(start_date + timedelta(seconds=int(s))).isoformat() for s in rand_secs]

# Recent transaction count – Poisson λ=1
transactions_last_1hr = np.random.poisson(lam=1, size=NUM_ROWS)

# ---------------------------------------------------------------------------
# 2️⃣ Fraud injection with varied scenarios
# ---------------------------------------------------------------------------
num_fraud = int(NUM_ROWS * FRAUD_RATIO)
fraud_idx = np.random.choice(NUM_ROWS, size=num_fraud, replace=False)

# Scenarios (weights sum to 1)
# 0: classic high‑amount & high‑frequency
# 1: moderate amount with modest frequency
# 2: night‑time with normal amount
# 3: rare device/merchant combo, normal amount
fraud_scen = np.random.choice([0, 1, 2, 3], size=num_fraud, p=[0.4, 0.3, 0.2, 0.1])

for idx, scen in zip(fraud_idx, fraud_scen):
    # Base values (will be overwritten as needed)
    amt = normal_amount[idx]
    recent = transactions_last_1hr[idx]
    ts = datetime.fromisoformat(trans_time[idx])
    dev = device[idx]
    merch = merchant[idx]

    if scen == 0:
        amt = np.random.uniform(50_000, 200_000)
        recent = np.random.randint(5, 21)
        hour = np.random.randint(0, 5)  # late night
        ts = ts.replace(hour=hour, minute=np.random.randint(0, 60), second=np.random.randint(0, 60))
        dev = np.random.choice(["POS", "ATM"])
        merch = np.random.choice(["Gifts", "Other"])  # rare categories
    elif scen == 1:
        amt = np.random.uniform(5_000, 30_000)  # overlaps normal tail
        recent = np.random.randint(2, 6)
        hour = np.random.randint(0, 24)
        ts = ts.replace(hour=hour, minute=np.random.randint(0, 60), second=np.random.randint(0, 60))
        # device/merchant can be common
        dev = np.random.choice(device_types)
        merch = np.random.choice(merchant_categories)
    elif scen == 2:
        amt = np.random.uniform(1_000, 12_000)  # overlaps normal
        recent = np.random.randint(0, 4)
        hour = np.random.randint(0, 5)  # night
        ts = ts.replace(hour=hour, minute=np.random.randint(0, 60), second=np.random.randint(0, 60))
        dev = np.random.choice(device_types)
        merch = np.random.choice(merchant_categories)
    else:  # scen == 3
        amt = np.random.uniform(1_000, 15_000)
        recent = np.random.randint(0, 3)
        dev = np.random.choice(["POS", "ATM"])
        merch = np.random.choice(["Gifts", "Other"])
        # keep original timestamp
    # Write back
    normal_amount[idx] = amt
    transactions_last_1hr[idx] = recent
    trans_time[idx] = ts.isoformat()
    device[idx] = dev
    merchant[idx] = merch

# ---------------------------------------------------------------------------
# 3️⃣ Add anomalous normal rows (5 % of non‑fraud rows)
# ---------------------------------------------------------------------------
non_fraud_idx = [i for i in range(NUM_ROWS) if i not in fraud_idx]
noise_cnt = int(0.05 * len(non_fraud_idx))
noise_idx = np.random.choice(non_fraud_idx, size=noise_cnt, replace=False)
for idx in noise_idx:
    choice = np.random.choice(["high_amount", "night_time"])
    if choice == "high_amount":
        normal_amount[idx] = np.random.uniform(10_000, 30_000)
    else:
        ts = datetime.fromisoformat(trans_time[idx])
        hour = np.random.randint(0, 5)
        ts = ts.replace(hour=hour, minute=np.random.randint(0, 60), second=np.random.randint(0, 60))
        trans_time[idx] = ts.isoformat()

# ---------------------------------------------------------------------------
# 4️⃣ Assemble raw DataFrame and write
# ---------------------------------------------------------------------------
transaction_id = [f"TX{str(i).zfill(7)}" for i in range(1, NUM_ROWS + 1)]
raw_df = pd.DataFrame({
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
raw_df.loc[fraud_idx, "is_fraud"] = 1

# Validate
assert raw_df["transaction_id"].is_unique, "Duplicate IDs"
assert not raw_df.isnull().any().any(), "Missing values"

raw_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "upi_transactions_v2.csv"
raw_path.parent.mkdir(parents=True, exist_ok=True)
raw_df.to_csv(raw_path, index=False)
print(f"Raw V2 dataset written to {raw_path}")

# ---------------------------------------------------------------------------
# 5️⃣ Feature engineering (same logic as src/feature_engineering.py)
# ---------------------------------------------------------------------------
# Convert timestamp
transaction_dt = pd.to_datetime(raw_df["transaction_time"], errors="coerce")
# Hour, day, flags, log amount
raw_df["transaction_hour"] = transaction_dt.dt.hour
raw_df["transaction_day"] = transaction_dt.dt.dayofweek
raw_df["is_weekend"] = raw_df["transaction_day"].apply(lambda d: 1 if d >= 5 else 0)
raw_df["is_night"] = raw_df["transaction_hour"].apply(lambda h: 1 if 0 <= h <= 4 else 0)
raw_df["log_amount"] = np.log(raw_df["amount"])

# Save processed version
proc_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "upi_transactions_features_v2.csv"
proc_path.parent.mkdir(parents=True, exist_ok=True)
raw_df.to_csv(proc_path, index=False)
print(f"Processed V2 dataset written to {proc_path}")
