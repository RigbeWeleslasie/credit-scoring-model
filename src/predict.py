"""
predict.py
----------
Inference utilities: loads the registered MLflow model and
produces a risk probability score for a given input record.

Implemented in Task 6.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def load_model(model_uri: str):
    """Load a model from the MLflow registry."""
    logger.info(f"Loading model from {model_uri}")
    raise NotImplementedError("Implement in Task 6")


def predict(model, input_data: pd.DataFrame) -> list:
    """
    Run inference and return risk probability scores.
    To be implemented in Task 6.
    """
    raise NotImplementedError("Implement in Task 6")
