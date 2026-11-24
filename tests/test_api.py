# tests/test_api.py
"""Automated tests for the FastAPI backend of the UPI Fraud Detection project.
These tests use FastAPI's built‑in TestClient so they do not require a separately
running server.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path

# Import the app instance from the backend module
from backend.main import app, model_path

client = TestClient(app)

# ----- Helper payloads -----
VALID_PAYLOAD = {
    "amount": 100.0,
    "transaction_time": "2023-01-01T12:00:00",
    "transaction_type": "debit",
    "merchant_category": "Food",
    "device_type": "mobile",
    "location": "NY",
    "transactions_last_1hr": 1,
}

LARGE_PAYLOAD = {
    "amount": 50000.0,
    "transaction_time": "2023-01-01T12:00:00",
    "transaction_type": "debit",
    "merchant_category": "Travel",
    "device_type": "mobile",
    "location": "NY",
    "transactions_last_1hr": 20,
}

# ----- Tests -----

def test_health_check():
    """GET /health should return HTTP 200 and a JSON with status and message."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert data.get("status") == "ok"
    assert "message" in data


def test_predict_valid():
    """POST /predict with a valid payload should return a prediction structure."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    # Expected keys
    for key in ["prediction", "is_suspicious", "message"]:
        assert key in data
    # prediction should be either "Normal" or "Suspicious"
    assert data["prediction"] in {"Normal", "Suspicious"}
    # is_suspicious must be a boolean
    assert isinstance(data["is_suspicious"], bool)


def test_predict_large_amount():
    """POST /predict with a large amount & high frequency works and returns same schema."""
    response = client.post("/predict", json=LARGE_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    for key in ["prediction", "is_suspicious", "message"]:
        assert key in data
    assert data["prediction"] in {"Normal", "Suspicious"}

# ----- Validation error tests (expect HTTP 422) -----

def test_predict_negative_amount():
    payload = VALID_PAYLOAD.copy()
    payload["amount"] = -10.0
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_zero_amount():
    payload = VALID_PAYLOAD.copy()
    payload["amount"] = 0.0
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_amount():
    payload = VALID_PAYLOAD.copy()
    payload.pop("amount")
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_timestamp():
    payload = VALID_PAYLOAD.copy()
    payload["transaction_time"] = "not-a-date"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_transactions_last_1hr():
    payload = VALID_PAYLOAD.copy()
    payload["transactions_last_1hr"] = -5
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_backend_loads_v2_model():
    """Confirm that the backend is configured to load the V2 model file."""
    # model_path is imported from backend.main; ensure it points to the V2 pickle
    assert isinstance(model_path, Path)
    assert model_path.name == "isolation_forest_v2.pkl"
    # The file should exist
    assert model_path.is_file()

"""Automated tests for the FastAPI backend of the UPI Fraud Detection project.
These tests use FastAPI's built‑in TestClient so they do not require a separately
running server.
"""

import pytest
from fastapi.testclient import TestClient

# Import the app instance from the backend module
from backend.main import app

client = TestClient(app)

# ----- Helper payloads -----
VALID_PAYLOAD = {
    "amount": 100.0,
    "transaction_time": "2023-01-01T12:00:00",
    "transaction_type": "debit",
    "merchant_category": "Food",
    "device_type": "mobile",
    "location": "NY",
    "transactions_last_1hr": 1,
}

LARGE_PAYLOAD = {
    "amount": 50000.0,
    "transaction_time": "2023-01-01T12:00:00",
    "transaction_type": "debit",
    "merchant_category": "Travel",
    "device_type": "mobile",
    "location": "NY",
    "transactions_last_1hr": 20,
}

# ----- Tests -----

def test_health_check():
    """GET /health should return HTTP 200 and a JSON with status and message."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert data.get("status") == "ok"
    assert "message" in data


def test_predict_valid():
    """POST /predict with a valid payload should return a prediction structure."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    # Expected keys
    for key in ["prediction", "is_suspicious", "message"]:
        assert key in data
    # prediction should be either "Normal" or "Suspicious"
    assert data["prediction"] in {"Normal", "Suspicious"}
    # is_suspicious must be a boolean
    assert isinstance(data["is_suspicious"], bool)


def test_predict_large_amount():
    """POST /predict with a large amount & high frequency works and returns same schema."""
    response = client.post("/predict", json=LARGE_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    for key in ["prediction", "is_suspicious", "message"]:
        assert key in data
    assert data["prediction"] in {"Normal", "Suspicious"}

# ----- Validation error tests (expect HTTP 422) -----

def test_predict_negative_amount():
    payload = VALID_PAYLOAD.copy()
    payload["amount"] = -10.0
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_zero_amount():
    payload = VALID_PAYLOAD.copy()
    payload["amount"] = 0.0
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_amount():
    payload = VALID_PAYLOAD.copy()
    payload.pop("amount")
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_timestamp():
    payload = VALID_PAYLOAD.copy()
    payload["transaction_time"] = "not-a-date"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_transactions_last_1hr():
    payload = VALID_PAYLOAD.copy()
    payload["transactions_last_1hr"] = -5
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
