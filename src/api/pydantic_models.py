"""
pydantic_models.py
------------------
Request and response schemas for the /predict endpoint.
Task 6 implementation.
"""

from pydantic import BaseModel, Field
from typing import Optional


class PredictRequest(BaseModel):
    """
    Input features for a single customer scoring request.
    All fields correspond to engineered features from data_processing.py.
    """
    customer_id: str = Field(..., example="CustomerId_4406")

    # Transaction value features (log-transformed)
    total_transaction_amount: float = Field(..., example=10.5)
    avg_transaction_amount: float = Field(..., example=8.2)
    transaction_count: int = Field(..., example=15)
    std_transaction_amount: float = Field(..., example=2.1)
    total_value: float = Field(..., example=10.5)
    avg_value: float = Field(..., example=8.2)

    # Raw amount and value
    Amount: float = Field(..., example=1000.0)
    Value: float = Field(..., example=1000.0)

    # Temporal features
    txn_hour: int = Field(..., ge=0, le=23, example=14)
    txn_day: int = Field(..., ge=1, le=31, example=15)
    txn_month: int = Field(..., ge=1, le=12, example=11)
    txn_year: int = Field(..., example=2018)
    txn_day_of_week: int = Field(..., ge=0, le=6, example=2)

    # Other features
    PricingStrategy: int = Field(..., example=2)
    FraudResult: int = Field(..., ge=0, le=1, example=0)

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CustomerId_4406",
                "total_transaction_amount": 10.5,
                "avg_transaction_amount": 8.2,
                "transaction_count": 15,
                "std_transaction_amount": 2.1,
                "total_value": 10.5,
                "avg_value": 8.2,
                "Amount": 1000.0,
                "Value": 1000.0,
                "txn_hour": 14,
                "txn_day": 15,
                "txn_month": 11,
                "txn_year": 2018,
                "txn_day_of_week": 2,
                "PricingStrategy": 2,
                "FraudResult": 0,
            }
        }

    def to_features(self) -> dict:
        """Convert request to a feature dict for model inference."""
        data = self.model_dump()
        data.pop("customer_id", None)
        return data


class PredictResponse(BaseModel):
    """Risk probability score returned by the API."""
    customer_id: str
    risk_probability: float = Field(..., ge=0.0, le=1.0, example=0.73)
    risk_label: str = Field(..., example="high_risk")

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CustomerId_4406",
                "risk_probability": 0.73,
                "risk_label": "high_risk",
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., example="ok")
    model_loaded: bool = Field(..., example=True)
    model_uri: Optional[str] = Field(None, example="models:/LogisticRegression/latest")
