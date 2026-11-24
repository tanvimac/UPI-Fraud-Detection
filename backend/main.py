# backend/main.py
"""FastAPI backend for the synthetic UPI fraud detection project.

The API loads the IsolationForest model trained in Phase 6 once at startup and
exposes two endpoints:

* ``GET /health`` – simple health check.
* ``POST /predict`` – receives transaction details, builds the same feature set
  that was used during training, runs the model and returns a human‑readable
  prediction.

All code is written to be beginner‑friendly and suitable for interview
discussions.
"""

from pathlib import Path
from typing import Literal
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# App and CORS configuration (allow all origins for local development)
# ---------------------------------------------------------------------------
app = FastAPI(title="UPI Fraud Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production you would restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load the trained Isolation Forest model (once at start‑up)
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
model_path = project_root / "models" / "isolation_forest_v2.pkl"
if not model_path.is_file():
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = joblib.load(model_path)

# ---------------------------------------------------------------------------
# Pydantic schema for incoming transaction data
# ---------------------------------------------------------------------------
class TransactionInput(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in INR")
    transaction_time: str = Field(
        ..., description="ISO‑8601 timestamp, e.g. 2026-11-18T14:23:05"
    )
    transaction_type: Literal["debit", "credit"] = Field(..., description="debit or credit")
    merchant_category: str = Field(..., description="Category of merchant (Food, Travel, …)")
    device_type: str = Field(..., description="Device used (Mobile, Web, POS, ATM)")
    location: str = Field(..., description="City name")
    transactions_last_1hr: int = Field(
        ..., ge=0, description="Number of transactions the user made in the previous hour"
    )

    @validator("transaction_time")
    def validate_timestamp(cls, v: str) -> str:
        # Attempt to parse; raise a clear error if the format is invalid.
        try:
            datetime.fromisoformat(v)
        except Exception as e:
            raise ValueError(
                "transaction_time must be a valid ISO‑8601 datetime string"
            ) from e
        return v

# ---------------------------------------------------------------------------
# Helper: build the feature vector exactly as during training
# ---------------------------------------------------------------------------
def build_features(data: TransactionInput) -> pd.DataFrame:
    # Parse timestamp
    dt = datetime.fromisoformat(data.transaction_time)
    transaction_hour = dt.hour
    transaction_day = dt.weekday()  # Monday=0 … Sunday=6
    is_weekend = 1 if transaction_day >= 5 else 0
    is_night = 1 if 0 <= transaction_hour <= 4 else 0
    log_amount = np.log(data.amount)

    features = {
        "amount": data.amount,
        "transactions_last_1hr": data.transactions_last_1hr,
        "transaction_hour": transaction_hour,
        "transaction_day": transaction_day,
        "is_weekend": is_weekend,
        "is_night": is_night,
        "log_amount": log_amount,
    }
    # Return as a single‑row DataFrame to match the model's expected input shape
    return pd.DataFrame([features])

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "FastAPI backend is running"}

@app.post("/predict")
def predict(transaction: TransactionInput):
    # Build feature DataFrame
    try:
        X = build_features(transaction)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Model predicts 1 for normal, -1 for anomaly
    raw_pred = model.predict(X)[0]
    is_suspicious = bool(raw_pred == -1)
    prediction_label = "Suspicious" if is_suspicious else "Normal"

    return {
        "prediction": prediction_label,
        "is_suspicious": is_suspicious,
        "message": (
            "Transaction appears anomalous based on the Isolation Forest model"
            if is_suspicious
            else "Transaction looks normal"
        ),
    }

# ---------------------------------------------------------------------------
# Note for interviewers:
# The feature engineering mirrors exactly what was done in src/feature_engineering.py
# (hour, day, weekend flag, night flag, log of amount). No target variable is used.
# ---------------------------------------------------------------------------
