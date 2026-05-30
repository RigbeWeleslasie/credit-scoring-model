"""
data_processing.py
------------------
Feature engineering pipeline: transforms raw Xente transaction data
into a model-ready dataset, including RFM aggregation, temporal feature
extraction, encoding, scaling, and proxy target variable construction.

Implemented in Task 3 & Task 4.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def load_data(path: str) -> pd.DataFrame:
    """Load raw transaction data from CSV."""
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows x {df.shape[1]} columns")
    return df


def build_pipeline():
    """
    Build and return the full sklearn preprocessing pipeline.
    To be implemented in Task 3.
    """
    raise NotImplementedError("Implement in Task 3")


def compute_rfm(df: pd.DataFrame, snapshot_date: str = None) -> pd.DataFrame:
    """
    Compute Recency, Frequency, and Monetary values per CustomerId.
    To be implemented in Task 4.
    """
    raise NotImplementedError("Implement in Task 4")


def assign_risk_labels(rfm: pd.DataFrame, n_clusters: int = 3,
                       random_state: int = 42) -> pd.DataFrame:
    """
    K-Means cluster customers on RFM and label the high-risk segment.
    To be implemented in Task 4.
    """
    raise NotImplementedError("Implement in Task 4")