"""
predict.py
----------
Inference utilities: loads the best model from MLflow registry and
produces a risk probability score for a given input record.

Task 6 implementation.
"""

import logging
import pandas as pd
import mlflow.pyfunc

logger = logging.getLogger(__name__)


def load_model(model_uri: str):
    """
    Load a model from the MLflow registry or local path.

    Args:
        model_uri: MLflow model URI e.g. 'models:/LogisticRegression/latest'

    Returns:
        Loaded MLflow PyFunc model

    Raises:
        ValueError: If model_uri is empty or None
        RuntimeError: If model cannot be loaded
    """
    if not model_uri:
        raise ValueError("model_uri must be a non-empty string.")

    logger.info(f"Loading model from: {model_uri}")
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from '{model_uri}': {e}") from e


def predict(model, input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Run inference and return risk probability scores.

    Args:
        model:      Loaded MLflow PyFunc model
        input_data: DataFrame with model feature columns

    Returns:
        DataFrame with columns: risk_probability, risk_label

    Raises:
        ValueError: If input_data is empty or None
        RuntimeError: If inference fails
    """
    if input_data is None or input_data.empty:
        raise ValueError("input_data must be a non-empty DataFrame.")

    logger.info(f"Running inference on {len(input_data):,} records.")

    try:
        probabilities = model.predict(input_data)
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}") from e

    if hasattr(probabilities, "shape") and len(probabilities.shape) > 1:
        risk_prob = probabilities[:, 1]
    else:
        risk_prob = probabilities

    results = pd.DataFrame({
        "risk_probability": risk_prob.round(4),
        "risk_label": ["high_risk" if p >= 0.5 else "low_risk" for p in risk_prob],
    })

    logger.info(
        f"Predictions: {(results['risk_label'] == 'high_risk').sum()} high_risk, "
        f"{(results['risk_label'] == 'low_risk').sum()} low_risk"
    )
    return results


def predict_single(model, features: dict) -> dict:
    """
    Predict risk for a single customer record.

    Args:
        model:    Loaded MLflow PyFunc model
        features: Dictionary of feature name -> value

    Returns:
        Dictionary with risk_probability and risk_label

    Raises:
        ValueError: If features dict is empty
    """
    if not features:
        raise ValueError("features dict must not be empty.")

    input_df = pd.DataFrame([features])
    result = predict(model, input_df)
    return {
        "risk_probability": float(result["risk_probability"].iloc[0]),
        "risk_label": result["risk_label"].iloc[0],
    }
