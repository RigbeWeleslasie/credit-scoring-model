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
from sklearn.cluster import KMeans

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
# RFM Computation (Task 4)
# ---------------------------------------------------------------------------

def compute_rfm(
    df: pd.DataFrame,
    customer_col: str = "CustomerId",
    date_col: str = "TransactionStartTime",
    amount_col: str = "Amount",
    snapshot_date: str = None,
) -> pd.DataFrame:
    """
    Compute Recency, Frequency, and Monetary (RFM) values per customer.

    Args:
        df:            Raw transaction DataFrame
        customer_col:  Column identifying the customer
        date_col:      Column with transaction timestamps
        amount_col:    Column with transaction amounts
        snapshot_date: Reference date for recency calculation.
                       Defaults to one day after the last transaction.

    Returns:
        DataFrame with columns [customer_col, Recency, Frequency, Monetary]
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True)

    if snapshot_date is None:
        snapshot = df[date_col].max() + pd.Timedelta(days=1)
    else:
        snapshot = pd.Timestamp(snapshot_date, tz="UTC")

    logger.info(f"RFM snapshot date: {snapshot}")

    # Use only positive amounts (debits) for monetary value
    debits = df[df[amount_col] > 0]

    rfm = (
        df.groupby(customer_col)
        .agg(
            Recency=(date_col, lambda x: (snapshot - x.max()).days),
            Frequency=(date_col, "count"),
        )
        .reset_index()
    )

    monetary = (
        debits.groupby(customer_col)[amount_col]
        .sum()
        .reset_index()
        .rename(columns={amount_col: "Monetary"})
    )

    rfm = rfm.merge(monetary, on=customer_col, how="left")
    rfm["Monetary"] = rfm["Monetary"].fillna(0)

    logger.info(f"RFM computed for {len(rfm):,} customers.")
    return rfm


# ---------------------------------------------------------------------------
# Risk Label Assignment via K-Means (Task 4)
# ---------------------------------------------------------------------------

def assign_risk_labels(
    rfm: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Segment customers into n_clusters using K-Means on scaled RFM features,
    then label the high-risk cluster as is_high_risk = 1.

    The high-risk cluster is identified as the one with:
      - Highest Recency   (least recent)
      - Lowest Frequency  (fewest transactions)
      - Lowest Monetary   (lowest spend)

    Args:
        rfm:          DataFrame with Recency, Frequency, Monetary columns
        n_clusters:   Number of K-Means clusters (default 3)
        random_state: Random seed for reproducibility

    Returns:
        rfm DataFrame with cluster and is_high_risk columns added
    """
    rfm = rfm.copy()

    # Scale RFM features before clustering
    scaler = StandardScaler()
    rfm_features = ["Recency", "Frequency", "Monetary"]
    rfm_scaled = scaler.fit_transform(rfm[rfm_features])

    # Fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    # Identify high-risk cluster: highest recency, lowest frequency & monetary
    cluster_summary = rfm.groupby("cluster")[rfm_features].mean()
    logger.info(f"Cluster summary:\n{cluster_summary}")

    # Score each cluster: high recency = bad, low frequency = bad, low monetary = bad
    # Normalize each dimension to [0,1] then compute risk score
    norm = (cluster_summary - cluster_summary.min()) / (
        cluster_summary.max() - cluster_summary.min() + 1e-9
    )
    # High risk = high recency + low frequency + low monetary
    risk_score = norm["Recency"] - norm["Frequency"] - norm["Monetary"]
    high_risk_cluster = int(risk_score.idxmax())

    logger.info(f"High-risk cluster identified: {high_risk_cluster}")
    logger.info(f"Risk scores per cluster:\n{risk_score}")

    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    n_high = rfm["is_high_risk"].sum()
    logger.info(
        f"High-risk customers: {n_high:,} ({n_high / len(rfm):.1%})"
    )

    return rfm


# ---------------------------------------------------------------------------
# Full Processing Entry Point
# ---------------------------------------------------------------------------

def process_raw_data(
    input_path: str,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Load raw data, apply the feature engineering pipeline, compute RFM,
    assign risk labels, and merge is_high_risk into the processed dataset.

    Args:
        input_path:  Path to raw data CSV
        output_path: Optional path to save processed data as parquet

    Returns:
        Processed DataFrame with is_high_risk target column
    """
    df = load_data(input_path)

    # Step 1: Compute RFM and assign risk labels (before dropping CustomerId)
    rfm = compute_rfm(df)
    rfm = assign_risk_labels(rfm)
    risk_map = rfm.set_index("CustomerId")["is_high_risk"].to_dict()

    # Step 2: Apply feature engineering pipeline
    pipeline = build_feature_pipeline()
    df_processed = pipeline.fit_transform(df)

    # Step 3: Re-attach CustomerId temporarily to merge risk labels
    df_processed["CustomerId"] = df["CustomerId"].values
    df_processed["is_high_risk"] = df_processed["CustomerId"].map(risk_map)
    df_processed = df_processed.drop(columns=["CustomerId"])

    logger.info(f"Processed shape: {df_processed.shape}")
    logger.info(
        f"is_high_risk distribution:\n"
        f"{df_processed['is_high_risk'].value_counts(normalize=True)}"
    )

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
