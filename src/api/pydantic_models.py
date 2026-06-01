"""
pydantic_models.py
------------------
Request and response schemas for the /predict endpoint.
Task 6 implementation.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Optional


FEATURE_ORDER = [
    "Amount",
    "Value",
    "PricingStrategy",
    "FraudResult",
    "txn_hour",
    "txn_day",
    "txn_month",
    "txn_year",
    "txn_day_of_week",
    "total_transaction_amount",
    "avg_transaction_amount",
    "transaction_count",
    "std_transaction_amount",
    "total_value",
    "avg_value",
]


class PredictRequest(BaseModel):
    """Input features for a single customer scoring request."""

    model_config = ConfigDict(protected_namespaces=())

    customer_id: str = Field(..., example="CustomerId_4406")
    Amount: float = Field(..., example=1000.0)
    Value: float = Field(..., example=1000.0)
    PricingStrategy: int = Field(..., example=2)
    FraudResult: int = Field(..., ge=0, le=1, example=0)
    txn_hour: int = Field(..., ge=0, le=23, example=14)
    txn_day: int = Field(..., ge=1, le=31, example=15)
    txn_month: int = Field(..., ge=1, le=12, example=11)
    txn_year: int = Field(..., example=2018)
    txn_day_of_week: int = Field(..., ge=0, le=6, example=2)
    total_transaction_amount: float = Field(..., example=10.5)
    avg_transaction_amount: float = Field(..., example=8.2)
    transaction_count: int = Field(..., example=15)
    std_transaction_amount: float = Field(..., example=2.1)
    total_value: float = Field(..., example=10.5)
    avg_value: float = Field(..., example=8.2)

    def to_features(self) -> dict:
        """Return features as ordered dict matching training feature order."""
        data = self.model_dump()
        data.pop("customer_id", None)
        return {k: data[k] for k in FEATURE_ORDER}


class PredictResponse(BaseModel):
    """Risk probability score returned by the API."""

    model_config = ConfigDict(protected_namespaces=())

    customer_id: str
    risk_probability: float = Field(..., ge=0.0, le=1.0, example=0.73)
    risk_label: str = Field(..., example="high_risk")


class HealthResponse(BaseModel):
    """Health check response."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., example="ok")
    model_loaded: bool = Field(..., example=True)
    model_uri: Optional[str] = Field(None, example="/app/models/xgboost_model.pkl")
