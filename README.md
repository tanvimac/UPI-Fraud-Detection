# UPI Fraud Detection & Real‑Time Transaction Monitoring System

## Project Overview
A compact, open‑source demonstration of a fintech fraud detection pipeline built with **Streamlit**, **FastAPI**, and **scikit‑learn**. The application loads a synthetic UPI transaction dataset, visualises key metrics, and provides a real‑time prediction API powered by an Isolation Forest model.

---

## Problem Statement & Objectives
- **Problem**: Detect potentially fraudulent UPI transactions in near‑real‑time.
- **Objectives**:
  1. Generate a synthetic transaction dataset (no real user data).
  2. Engineer simple temporal and statistical features.
  3. Train an Isolation Forest model (Version 2) on the engineered features.
  4. Expose a lightweight FastAPI service for on‑demand predictions.
  5. Visualise data, model insights, and live predictions via a Streamlit UI.

---

## Key Features
- **Dashboard Overview** – KPI cards, fraud‑vs‑normal distribution donut, recent transactions table.
- **Transaction Monitoring** – Interactive filters, pagination, and a **Clear filters** button.
- **Test a Transaction** – Form to submit a single transaction and receive a prediction.
- **Fraud Analytics** – Multiple static charts (amount distribution, fraud by merchant, device, hour, etc.).
- **Backend API** – `/health` health‑check and `/predict` endpoint returning a human‑readable label.
- **Automated Test Suite** – 9 passing pytest cases covering API health, prediction schema, validation errors, and model loading.

---

## Technology Stack
| Layer | Technology |
|-------|------------|
| UI | **Streamlit** (v0.x) – interactive web app, Plotly for charts |
| API | **FastAPI** – lightweight HTTP service |
| ML | **scikit‑learn** – Isolation Forest |
| Data processing | **pandas**, **NumPy** |
| Environment | Python 3.13, virtual environment (`.venv`) |
| Testing | **pytest** |
| Visualization (static) | Plotly, Matplotlib/Seaborn (used in EDA notebooks) |

---

## System Architecture & Data Flow
```
Synthetic Data (data/raw/upi_transactions_v2.csv)
        |
        v
Feature Engineering (src/feature_engineering.py)
        |
        v
Processed Data (data/processed/upi_transactions_features_v2.csv)
        |
        |---► Frontend (frontend/app.py)
        |        • Loads processed CSV (cached)
        |        • Shows charts, tables, filters
        |        • Sends transaction JSON to FastAPI /predict
        |
        v
Isolation Forest Model (models/isolation_forest_v2.pkl)
        |
        v
Backend (backend/main.py)
        • Loads model at startup
        • /predict builds same feature set, runs model, returns label
        • /health returns status
```
The **frontend** never touches the raw data; it only reads the processed CSV. When a user tests a transaction, the frontend posts a JSON payload to the **backend**, which replicates the feature engineering steps, runs the Isolation Forest, and returns a prediction.

---

## Repository Structure
```
UPI-Fraud-Detection/
├─ backend/                 # FastAPI application
│   ├─ main.py              # API routes & model loading
├─ data/                    # Dataset files
│   ├─ raw/                 # Synthetic raw CSVs
│   │   └─ upi_transactions_v2.csv
│   └─ processed/           # Feature‑engineered CSVs
│       ├─ upi_transactions_features_v2.csv
├─ frontend/                # Streamlit UI
│   ├─ app.py               # Main Streamlit script (single‑page app)
├─ models/                  # Serialized ML models
│   ├─ isolation_forest.pkl
│   └─ isolation_forest_v2.pkl
├─ reports/                 # Evaluation & EDA artefacts
│   ├─ model_evaluation_v2.txt
│   ├─ model_evaluation.txt
│   └─ eda/ (charts)
├─ src/                     # Core scripts
│   ├─ data_generator.py
│   ├─ data_generator_v2.py
│   ├─ feature_engineering.py
│   ├─ model_training_v2.py
│   └─ eda.py
├─ tests/                   # pytest suite
│   └─ test_api.py
├─ requirements.txt
└─ README.md                # (this file)
```

---

## Dataset Description
The synthetic dataset mimics typical UPI transaction logs.

### Raw columns (`data/raw/upi_transactions_v2.csv`)
| Column | Description |
|--------|-------------|
| `transaction_id` | Unique identifier (string) |
| `user_id` | Synthetic user identifier |
| `amount` | Transaction amount (INR) |
| `transaction_time` | ISO‑8601 timestamp |
| `transaction_type` | `debit` or `credit` |
| `merchant_category` | Category string (e.g., Food, Travel) |
| `device_type` | Device used (mobile, web, POS, ATM) |
| `location` | City name |
| `transactions_last_1hr` | Number of prior transactions by the user in the last hour |
| `is_fraud` | Ground‑truth label (0 = normal, 1 = fraud) |

### Processed columns (`data/processed/upi_transactions_features_v2.csv`)
All raw columns are retained **plus** the engineered features used for modelling:
- `transaction_hour` (0‑23)
- `transaction_day` (0‑6, Monday=0)
- `is_weekend` (1 if Saturday/Sunday, else 0)
- `is_night` (1 if hour 0‑4, else 0)
- `log_amount` (natural log of `amount`)

> **Synthetic data disclaimer** – The data is artificially generated and does not contain any real user information.

---

## Exploratory Data Analysis & Feature Engineering
- Basic distribution plots (amount, fraud by merchant/device/hour) are stored in `reports/eda/` as PNG images.
- Feature engineering is performed in `src/feature_engineering.py` and reproduced in the backend (`backend/main.py`):
  1. Parse `transaction_time` → hour, weekday.
  2. Derive `is_weekend` and `is_night` flags.
  3. Compute `log_amount` to reduce skew.
- The engineered feature set is **exactly** the list used during model training:
  ```
  amount, transactions_last_1hr, transaction_hour, transaction_day,
  is_weekend, is_night, log_amount
  ```

---

## Isolation Forest (Version 2) Training Methodology
Implemented in `src/model_training_v2.py`:
1. Load `data/processed/upi_transactions_features_v2.csv`.
2. Verify dataset integrity (10 000 rows, 2 % fraud, no missing values, unique transaction IDs).
3. Split with stratified 80/20 split (`random_state=42`).
4. Train `IsolationForest` with:
   - `n_estimators=100`
   - `contamination=0.02`
   - `random_state=42`
5. Map model output `-1 → fraud (1)` and `1 → normal (0)`.
6. Evaluate on the held‑out test set and write a human‑readable report.
7. Serialize the model to `models/isolation_forest_v2.pkl` using `joblib.dump`.

---

## Model Evaluation (Version 2)
Extracted from `reports/model_evaluation_v2.txt`:
```
Isolation Forest Model Evaluation – Version 2
=============================================

Dataset rows: 10000 (fraud 200 / 9800)
Features used: amount, transactions_last_1hr, transaction_hour, transaction_day, is_weekend, is_night, log_amount

Confusion Matrix (true vs pred):
[[1940   20]
 [  16   24]]

Accuracy : 0.9820
Precision: 0.5455
Recall   : 0.6000
F1-score : 0.5714
ROC‑AUC  : 0.9866
```
### Interpretation
- **True Negatives (1940)** – Normal transactions correctly identified.
- **False Positives (20)** – Normal transactions flagged as fraud.
- **False Negatives (16)** – Fraudulent transactions missed (predicted as normal).
- **True Positives (24)** – Fraudulent transactions correctly identified.
- **Precision (0.5455)** indicates that when the model flags fraud, ~55 % are true frauds.
- **Recall (0.6000)** shows the model catches 60 % of all fraud cases.
- **ROC‑AUC (0.9866)** reflects strong separability between normal and fraudulent patterns, despite the modest precision/recall that stem from the highly imbalanced data.

---

## FastAPI Endpoints
### 1️⃣ Health Check
```http
GET /health HTTP/1.1
Host: 127.0.0.1:8000
```
**Response** (`200 OK`):
```json
{ "status": "ok", "message": "FastAPI backend is running" }
```

### 2️⃣ Prediction
```http
POST /predict HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "amount": 123.45,
  "transaction_time": "2023-01-01T12:34:56",
  "transaction_type": "debit",
  "merchant_category": "Food",
  "device_type": "mobile",
  "location": "NY",
  "transactions_last_1hr": 2
}
```
**Response** (`200 OK`):
```json
{
  "prediction": "Normal",      // or "Suspicious"
  "is_suspicious": false,       // true when prediction is "Suspicious"
  "message": "Transaction looks normal"
}
```
The endpoint returns a **400** error with a descriptive message if validation fails (e.g., negative amount, malformed timestamp).

---

## Installation (macOS)
```bash
# 1. Clone the repository (if not already)
git clone <repo‑url>
cd UPI-Fraud-Detection

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```
All required packages are listed in `requirements.txt` (Streamlit, FastAPI, scikit‑learn, pandas, numpy, httpx, pytest, etc.).

---

## Running the Application
### Backend (FastAPI)
```bash
# From the project root, with the virtual environment active
.venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
The server prints a startup message and is ready at `http://127.0.0.1:8000`.

### Frontend (Streamlit UI)
```bash
.venv/bin/streamlit run frontend/app.py --server.port 8502
```
Open a browser at `http://localhost:8502` to view the dashboard, monitoring page, and transaction tester.

---

## Running Automated Tests
```bash
# Ensure the virtual environment is active
source .venv/bin/activate

# Execute the pytest suite
.venv/bin/python -m pytest -q
```
All **9 tests pass**:
```
.........                                                                [100%]
9 passed, 2 warnings in 2.5s
```
The tests validate health‑check, prediction schema, input validation, and that the backend loads the V2 model file.

---

## Screenshots
*Replace the placeholders with actual PNG files generated from the running application.*

| Page | Screenshot |
|------|------------|
| Dashboard Overview | `![Dashboard](path/to/dashboard.png)` |
| Transaction Monitoring | `![Monitoring](path/to/monitoring.png)` |
| Test a Transaction | `![Test Transaction](path/to/test_transaction.png)` |
| Fraud Analytics | `![Analytics](path/to/analytics.png)` |

---

## Limitations & Future Improvements
- **Synthetic data** – Real‑world performance cannot be inferred.
- **Model** – Isolation Forest provides coarse anomaly detection; supervised models could improve precision/recall.
- **Feature set** – Limited to simple temporal and amount‑based features; richer behavioural signals would be beneficial.
- **Scalability** – The current FastAPI service loads the model in memory and handles a single worker; production deployment would need proper concurrency, monitoring, and security.
- **UI** – The Streamlit UI is functional but could benefit from pagination optimisation and richer visualisation components.

---

## Disclaimer
The dataset and model are **synthetic demonstrations** designed for educational purposes only. They are **not** intended for use in any real banking or financial‑service production environment.

---

*Repository generated on 2026‑11‑24 using Antigravity.*
