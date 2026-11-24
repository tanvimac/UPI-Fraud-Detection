# src/data_generator.py
"""Synthetic UPI Transaction Data Generator

This script creates a mock dataset of 10,000 UPI transactions.
It uses only NumPy and pandas (both already installed in the virtual
environment) and a fixed random seed so the output is reproducible.

The generated CSV is saved to ``data/raw/upi_transactions.csv``.

Columns produced:
- transaction_id: unique identifier (string)
- user_id: simulated user identifier
- amount: transaction amount in INR (float)
- transaction_time: timestamp (ISO‑format string)
- transaction_type: "debit" or "credit"
- merchant_category: category of the merchant (e.g. "Food", "Travel")
- device_type: device used for the transaction
- location: city name (synthetic)
- transactions_last_1hr: number of transactions the user made in the
  previous hour (int)
- is_fraud: 0 = normal, 1 = simulated fraudulent transaction

Fraudulent rows are made to look suspicious by:
* Very high amounts (>> normal range)
* Very high recent transaction count
* Odd transaction times (late night)
* Less common device / merchant combos
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Configuration (change here if you want a different size or fraud rate)
# ---------------------------------------------------------------------------
NUM_ROWS = 10_000
FRAUD_RATIO = 0.02  # 2% of rows will be marked as fraud
SEED = 42

np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Helper data – realistic‑looking categorical values
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
# A small list of synthetic city names
locations = ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad"]

# ---------------------------------------------------------------------------
# 1️⃣ Generate basic (mostly normal) transactions
# ---------------------------------------------------------------------------
# Randomly choose user, merchant, device, location
users = np.random.choice(user_ids, size=NUM_ROWS)
merchant = np.random.choice(merchant_categories, size=NUM_ROWS)
device = np.random.choice(device_types, size=NUM_ROWS)
loc = np.random.choice(locations, size=NUM_ROWS)

# Transaction type – debit is far more common than credit
trans_type = np.random.choice(["debit", "credit"], size=NUM_ROWS, p=[0.9, 0.1])

# Amount – normal amounts follow a log‑normal distribution (most < 5000 INR)
normal_amount = np.random.lognormal(mean=5, sigma=1.2, size=NUM_ROWS)
# Clip extreme outliers that would look like fraud (we will add those later)
normal_amount = np.clip(normal_amount, 10, 10_000)

# Transaction time – uniform over the last 30 days
now = datetime.now()
start_date = now - timedelta(days=30)
random_seconds = np.random.randint(0, int((now - start_date).total_seconds()), size=NUM_ROWS)
trans_time = [ (start_date + timedelta(seconds=int(s))).isoformat() for s in random_seconds ]

# Transactions in the last hour – Poisson with λ=1 for normal behaviour
transactions_last_1hr = np.random.poisson(lam=1, size=NUM_ROWS)

# ---------------------------------------------------------------------------
# 2️⃣ Inject fraudulent records (replace a subset of rows)
# ---------------------------------------------------------------------------
num_fraud = int(NUM_ROWS * FRAUD_RATIO)
fraud_indices = np.random.choice(NUM_ROWS, size=num_fraud, replace=False)

# Fraudulent amounts – much higher, up to 200k INR
fraud_amount = np.random.uniform(50_000, 200_000, size=num_fraud)
normal_amount[fraud_indices] = fraud_amount

# Fraudulent recent transaction count – unusually high (5‑20)
fraud_recent = np.random.randint(5, 21, size=num_fraud)
transactions_last_1hr[fraud_indices] = fraud_recent

# Fraudulent times – late night (0‑4 am) within the same 30‑day window
late_seconds = np.random.randint(0, 4 * 3600, size=num_fraud)
for idx, sec in zip(fraud_indices, late_seconds):
    # Keep same day as originally generated but replace hour component
    original = datetime.fromisoformat(trans_time[idx])
    new_dt = original.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(seconds=int(sec))
    trans_time[idx] = new_dt.isoformat()

# Fraudulent device – less common devices (e.g., POS for consumer transaction)
fraud_device_choices = ["POS", "ATM"]
device[fraud_indices] = np.random.choice(fraud_device_choices, size=num_fraud)

# Fraudulent merchant category – pick rare ones
rare_categories = ["Gifts", "Other"]
merchant[fraud_indices] = np.random.choice(rare_categories, size=num_fraud)

# ---------------------------------------------------------------------------
# 3️⃣ Assemble DataFrame
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

# ---------------------------------------------------------------------------
# 4️⃣ Save to CSV (raw data folder)
# ---------------------------------------------------------------------------
import os
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "upi_transactions.csv"))
df.to_csv(output_path, index=False)
print(f"Generated {NUM_ROWS} rows → {output_path}")
print(f"Fraudulent rows: {num_fraud}")
"""
End of script.
"""
