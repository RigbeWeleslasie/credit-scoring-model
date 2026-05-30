"""
data_processing.py
------------------
Feature engineering pipeline: transforms raw Xente transaction data
into a model-ready dataset, including aggregate features, temporal
feature extraction, encoding, scaling, and missing value handling.

Task 3 & Task 4 implementation.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Load raw transaction data from CSV."""
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows x {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Custom Transformers
# ---------------------------------------------------------------------------

class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts temporal features from TransactionStartTime:
    hour, day, month, year, day_of_week.
    """

    def __init__(self, time_col: str = "TransactionStartTime"):
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        dt = pd.to_datetime(X[self.time_col], utc=True)
        X["txn_hour"] = dt.dt.hour
        X["txn_day"] = dt.dt.day
        X["txn_month"] = dt.dt.month
        X["txn_year"] = dt.dt.year
        X["txn_day_of_week"] = dt.dt.dayofweek
        X = X.drop(columns=[self.time_col])
        logger.info("Temporal features extracted.")
        return X


class AggregateFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Builds per-customer aggregate features and merges them back
    onto the transaction-level DataFrame.
    """

    def __init__(
        self,
        customer_col: str = "CustomerId",
        amount_col: str = "Amount",
        value_col: str = "Value",
    ):
        self.customer_col = customer_col
        self.amount_col = amount_col
        self.value_col = value_col
        self.agg_df_ = None

    def fit(self, X, y=None):
        self.agg_df_ = (
            X.groupby(self.customer_col)
            .agg(
                total_transaction_amount=(self.amount_col, "sum"),
                avg_transaction_amount=(self.amount_col, "mean"),
                transaction_count=(self.amount_col, "count"),
                std_transaction_amount=(self.amount_col, "std"),
                total_value=(self.value_col, "sum"),
                avg_value=(self.value_col, "mean"),
            )
            .reset_index()
        )
        return self

    def transform(self, X):
        X = X.copy()
        X = X.merge(self.agg_df_, on=self.customer_col, how="left")
        X["std_transaction_amount"] = X["std_transaction_amount"].fillna(0)
        logger.info("Aggregate features merged.")
        return X


class DropIdentifierColumns(BaseEstimator, TransformerMixin):
    """
    Drops high-cardinality identifier columns that carry no
    predictive signal for the model.
    """

    def __init__(self):
        self.cols_to_drop = [
            "TransactionId",
            "BatchId",
            "AccountId",
            "SubscriptionId",
            "CustomerId",
            "CountryCode",
            "CurrencyCode",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        existing = [c for c in self.cols_to_drop if c in X.columns]
        X = X.drop(columns=existing)
        logger.info(f"Dropped identifier columns: {existing}")
        return X


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Applies log1p transformation to skewed numerical columns.
    """

    def __init__(self, cols: list = None):
        self.cols = cols or [
            "Value",
            "total_value",
            "avg_value",
            "total_transaction_amount",
            "avg_transaction_amount",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            if col in X.columns:
                X[col] = np.log1p(X[col].clip(lower=0))
        logger.info(f"Log1p transformation applied to: {self.cols}")
        return X


# ---------------------------------------------------------------------------
# Pipeline Builders
# ---------------------------------------------------------------------------

def build_feature_pipeline() -> Pipeline:
    """
    Build and return a sklearn Pipeline that transforms raw transaction
    data into a model-ready DataFrame.
    """
    pipeline = Pipeline(steps=[
        ("temporal", TemporalFeatureExtractor()),
        ("aggregates", AggregateFeatureBuilder()),
        ("drop_ids", DropIdentifierColumns()),
        ("log_transform", LogTransformer()),
    ])
    return pipeline


def build_preprocessing_pipeline(
    categorical_cols: list,
    numerical_cols: list,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that imputes and scales numerical features
    and imputes and one-hot encodes categorical features.
    """
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])

    return preprocessor


# ---------------------------------------------------------------------------
# Full Processing Entry Point
# ---------------------------------------------------------------------------

def process_raw_data(input_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Load raw data, apply the feature engineering pipeline, and optionally
    save the processed DataFrame to a parquet file.
    """
    df = load_data(input_path)
    pipeline = build_feature_pipeline()
    df_processed = pipeline.fit_transform(df)
    logger.info(f"Processed shape: {df_processed.shape}")
    if output_path:
        df_processed.to_parquet(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
    return df_processed


# ---------------------------------------------------------------------------
# Helpers for column selection
# ---------------------------------------------------------------------------

def get_categorical_cols(df: pd.DataFrame) -> list:
    """Return list of categorical column names."""
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def get_numerical_cols(df: pd.DataFrame) -> list:
    """Return list of numerical column names."""
    return df.select_dtypes(include=["number"]).columns.tolist()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw data CSV")
    parser.add_argument("--output", required=True, help="Path to save processed parquet")
    args = parser.parse_args()

    process_raw_data(args.input, args.output)