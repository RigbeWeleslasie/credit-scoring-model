"""
main.py
-------
FastAPI application exposing the /predict endpoint.
Loads the best model from the MLflow registry on startup.

Task 6 implementation.
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import PredictRequest, PredictResponse, HealthResponse
from src.predict import load_model, predict_single

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_URI = os.getenv("MODEL_URI", "models:/XGBoost/latest")

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    global _model
    logger.info(f"Loading model from: {MODEL_URI}")
    try:
        _model = load_model(MODEL_URI)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        _model = None
    yield
    _model = None
    logger.info("Model released.")


app = FastAPI(
    title="Credit Risk Scoring API",
    description=(
        "Returns a credit risk probability score for a given customer profile. "
        "Built for Bati Bank's buy-now-pay-later service."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    """Health check endpoint."""
    model_loaded = _model is not None
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_uri=MODEL_URI,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Scoring"])
def predict(request: PredictRequest):
    """
    Accept customer features and return a risk probability score.

    - **risk_probability**: Float between 0 and 1. Higher = more risky.
    - **risk_label**: 'high_risk' if probability >= 0.5, else 'low_risk'.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check /health for status.",
        )

    try:
        features = request.to_features()
        result = predict_single(_model, features)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Prediction failed: {str(e)}",
        )

    return PredictResponse(
        customer_id=request.customer_id,
        risk_probability=result["risk_probability"],
        risk_label=result["risk_label"],
    )
