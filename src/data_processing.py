"""
data_processing.py
------------------
Feature engineering pipeline: transforms raw Xente transaction data
into a model-ready dataset, including aggregate features, temporal
feature extraction, WoE/IV encoding, scaling, and missing value handling.

Also implements RFM-based proxy target variable construction with
configurable risk thresholds for different portfolio risk appetites.

Task 3, 4, and 7 implementation.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Risk Threshold Configuration
# ---------------------------------------------------------------------------

# Parameterized risk thresholds for different portfolio risk appetites.
# These map risk_probability output to loan origination decisions.
# Adjust RISK_PROFILE to change the bank's risk tolerance.
#
# Conservative: tighter thresholds, fewer approvals, lower default exposure
# Moderate:     balanced approach, standard BNPL product
# Aggressive:   broader approvals, higher volume, higher default tolerance
#
# Each profile defines:
#   low_risk_threshold:    p below this → full credit, long duration
#   medium_risk_threshold: p below this → reduced credit, standard duration
#   high_risk_threshold:   p below this → minimal credit, short duration
#   decline_threshold:     p above this → decline
#   credit_limits:         dict of multipliers for requested amount
#   loan_durations_days:   dict of loan durations per risk band

RISK_PROFILES = {
    "conservative": {
        "description": "Tight thresholds for a new BNPL product with limited loss tolerance",
        "low_risk_threshold":    0.20,
        "medium_risk_threshold": 0.40,
        "high_risk_threshold":   0.60,
        "decline_threshold":     0.60,
        "credit_limits": {
            "low":    1.00,
            "medium": 0.50,
            "high":   0.20,
            "decline": 0.00,
        },
        "loan_durations_days": {
            "low":    60,
            "medium": 21,
            "high":   7,
            "decline": 0,
        },
    },
    "moderate": {
        "description": "Balanced thresholds for a mature BNPL product",
        "low_risk_threshold":    0.30,
        "medium_risk_threshold": 0.60,
        "high_risk_threshold":   0.80,
        "decline_threshold":     0.80,
        "credit_limits": {
            "low":    1.00,
            "medium": 0.75,
            "high":   0.25,
            "decline": 0.00,
        },
        "loan_durations_days": {
            "low":    90,
            "medium": 30,
            "high":   14,
            "decline": 0,
        },
    },
    "aggressive": {
        "description": "Broad approvals for maximum market penetration",
        "low_risk_threshold":    0.40,
        "medium_risk_threshold": 0.70,
        "high_risk_threshold":   0.90,
        "decline_threshold":     0.90,
        "credit_limits": {
            "low":    1.00,
            "medium": 0.80,
            "high":   0.50,
            "decline": 0.00,
        },
        "loan_durations_days": {
            "low":    90,
            "medium": 45,
            "high":   21,
            "decline": 0,
        },
    },
}

# Default profile used at inference time
DEFAULT_RISK_PROFILE = "moderate"


def get_risk_decision(
    risk_probability: float,
    profile_name: str = DEFAULT_RISK_PROFILE,
) -> dict:
    """
    Translate a risk probability into a loan origination decision.

    Args:
        risk_probability: Float between 0 and 1 from the model
        profile_name:     One of 'conservative', 'moderate', 'aggressive'

    Returns:
        dict with keys: risk_band, credit_limit_pct, loan_duration_days,
                        decision, profile_used
    """
    if profile_name not in RISK_PROFILES:
        raise ValueError(
            f"Unknown profile '{profile_name}'. "
            f"Choose from: {list(RISK_PROFILES.keys())}"
        )

    profile = RISK_PROFILES[profile_name]
    p = risk_probability

    if p < profile["low_risk_threshold"]:
        band = "low"
        decision = "approve"
    elif p < profile["medium_risk_threshold"]:
        band = "medium"
        decision = "approve_with_conditions"
    elif p < profile["high_risk_threshold"]:
        band = "high"
        decision = "reduced_offer"
    else:
        band = "decline"
        decision = "decline"

    return {
        "risk_band":          band,
        "credit_limit_pct":   profile["credit_limits"][band],
        "loan_duration_days": profile["loan_durations_days"][band],
        "decision":           decision,
        "profile_used":       profile_name,
    }


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Load raw transaction data from CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
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

    Aggregate features computed:
    - total_transaction_amount: sum of all transaction amounts per customer
    - avg_transaction_amount:   mean transaction amount per customer
    - transaction_count:        number of transactions per customer
    - std_transaction_amount:   std dev of amounts (0 for single-tx customers)
    - total_value:              sum of absolute transaction values
    - avg_value:                mean absolute transaction value
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
    Clips values at 0 before transformation to handle negatives safely.
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


class WoEEncoder(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence (WoE) encoder for categorical features.

    WoE transforms each category into:
        WoE_i = ln(Distribution of Events_i / Distribution of Non-Events_i)

    Where:
        Distribution of Events_i    = Count of events in bin i / Total events
        Distribution of Non-Events_i = Count of non-events in bin i / Total non-events

    Business interpretation:
        WoE > 0: category associated with lower risk (more non-events)
        WoE < 0: category associated with higher risk (more events)
        WoE = 0: category has no discriminatory power

    Information Value (IV) measures the total predictive power of a feature:
        IV = sum[(Dist_Events - Dist_NonEvents) * WoE]

    IV interpretation (standard credit scoring guidelines):
        IV < 0.02:  Useless predictor
        IV 0.02-0.1: Weak predictor
        IV 0.1-0.3:  Medium predictor
        IV 0.3-0.5:  Strong predictor
        IV > 0.5:    Suspicious (possible data leakage)

    This implementation is used for the Logistic Regression path.
    For XGBoost, one-hot encoding is used instead.
    """

    def __init__(
        self,
        cat_cols: list = None,
        target_col: str = "is_high_risk",
        min_samples: int = 5,
        smoothing: float = 0.5,
    ):
        self.cat_cols = cat_cols or [
            "ProductCategory",
            "ChannelId",
            "ProviderId",
            "PricingStrategy",
        ]
        self.target_col = target_col
        self.min_samples = min_samples
        self.smoothing = smoothing
        self.woe_maps_ = {}
        self.iv_scores_ = {}

    def fit(self, X, y=None):
        """
        Compute WoE and IV for each categorical column using the target variable.
        """
        if y is None and self.target_col in X.columns:
            y = X[self.target_col]
        elif y is None:
            logger.warning("No target provided to WoEEncoder; skipping fit.")
            return self

        total_events = y.sum() + self.smoothing
        total_nonevents = (1 - y).sum() + self.smoothing

        for col in self.cat_cols:
            if col not in X.columns:
                continue

            woe_map = {}
            iv = 0.0

            for cat in X[col].unique():
                mask = X[col] == cat
                n = mask.sum()
                if n < self.min_samples:
                    woe_map[cat] = 0.0
                    continue

                events = y[mask].sum() + self.smoothing
                nonevents = (1 - y[mask]).sum() + self.smoothing

                dist_events = events / total_events
                dist_nonevents = nonevents / total_nonevents

                woe = np.log(dist_events / dist_nonevents)
                iv += (dist_events - dist_nonevents) * woe
                woe_map[cat] = round(woe, 4)

            self.woe_maps_[col] = woe_map
            self.iv_scores_[col] = round(iv, 4)

        logger.info("WoE encoding fitted.")
        self._log_iv_summary()
        return self

    def transform(self, X):
        """
        Replace each categorical column with its WoE-encoded values.
        Unknown categories get WoE = 0 (neutral).
        """
        X = X.copy()
        for col in self.cat_cols:
            if col not in X.columns or col not in self.woe_maps_:
                continue
            X[f"{col}_woe"] = X[col].map(self.woe_maps_[col]).fillna(0.0)
            X = X.drop(columns=[col])
        logger.info(f"WoE transformation applied to: {list(self.woe_maps_.keys())}")
        return X

    def _log_iv_summary(self):
        """Log IV scores with business interpretation."""
        logger.info("Information Value (IV) Summary:")
        for col, iv in sorted(self.iv_scores_.items(), key=lambda x: -x[1]):
            if iv < 0.02:
                strength = "Useless"
            elif iv < 0.1:
                strength = "Weak"
            elif iv < 0.3:
                strength = "Medium"
            elif iv < 0.5:
                strength = "Strong"
            else:
                strength = "Suspicious (check for leakage)"
            logger.info(f"  {col:25s}: IV={iv:.4f} ({strength})")

    def get_iv_report(self) -> pd.DataFrame:
        """Return a DataFrame summarizing IV scores for all encoded features."""
        rows = []
        for col, iv in self.iv_scores_.items():
            if iv < 0.02:
                strength = "Useless"
            elif iv < 0.1:
                strength = "Weak"
            elif iv < 0.3:
                strength = "Medium"
            elif iv < 0.5:
                strength = "Strong"
            else:
                strength = "Suspicious"
            rows.append({"feature": col, "iv": iv, "strength": strength})
        return pd.DataFrame(rows).sort_values("iv", ascending=False)


# ---------------------------------------------------------------------------
# Pipeline Builders
# ---------------------------------------------------------------------------

def build_feature_pipeline() -> Pipeline:
    """
    Build and return a sklearn Pipeline that transforms raw transaction
    data into a model-ready DataFrame.

    Steps:
      1. Extract temporal features from TransactionStartTime
      2. Build per-customer aggregate features
      3. Drop identifier columns
      4. Apply log1p to skewed columns
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
    Build a ColumnTransformer for model-ready output:
    - Numerical: median imputation + StandardScaler
    - Categorical: most_frequent imputation + OneHotEncoder

    Used for XGBoost path.
    For Logistic Regression, use build_woe_pipeline() instead.
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


def build_woe_pipeline(target_col: str = "is_high_risk") -> Pipeline:
    """
    Build a WoE-based pipeline for the Logistic Regression (scorecard) path.

    Steps:
      1. WoE encode categorical features (requires target at fit time)
      2. Median imputation for remaining numerical features
      3. StandardScaler normalization

    WoE encoding business rationale:
    - Transforms categories into log-odds of default, making coefficients
      directly interpretable as credit risk contributions
    - Monotonizes the relationship between each category and the target,
      satisfying Basel II model documentation requirements
    - IV scores guide feature selection (drop IV < 0.02)
    """
    pipeline = Pipeline(steps=[
        ("woe", WoEEncoder(target_col=target_col)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return pipeline


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

    The high-risk cluster is identified algorithmically as the one with the
    highest risk score = normalized_recency - normalized_frequency - normalized_monetary.

    Clustering notes:
    - StandardScaler applied before K-Means to prevent Monetary (large range)
      from dominating Recency and Frequency (smaller ranges)
    - random_state=42 ensures reproducibility
    - n_clusters=3 chosen to represent low/medium/high risk tiers
      (validated against business intuition: disengaged / moderate / power users)

    Args:
        rfm:          DataFrame with Recency, Frequency, Monetary columns
        n_clusters:   Number of K-Means clusters (default 3)
        random_state: Random seed for reproducibility

    Returns:
        rfm DataFrame with cluster and is_high_risk columns added
    """
    rfm = rfm.copy()

    scaler = StandardScaler()
    rfm_features = ["Recency", "Frequency", "Monetary"]
    rfm_scaled = scaler.fit_transform(rfm[rfm_features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    cluster_summary = rfm.groupby("cluster")[rfm_features].mean()
    logger.info(f"Cluster summary:\n{cluster_summary}")

    norm = (cluster_summary - cluster_summary.min()) / (
        cluster_summary.max() - cluster_summary.min() + 1e-9
    )
    risk_score = norm["Recency"] - norm["Frequency"] - norm["Monetary"]
    high_risk_cluster = int(risk_score.idxmax())

    logger.info(f"High-risk cluster identified: {high_risk_cluster}")
    logger.info(f"Risk scores per cluster:\n{risk_score}")

    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    n_high = rfm["is_high_risk"].sum()
    pct = n_high / len(rfm)
    logger.info(f"High-risk customers: {n_high:,} ({pct:.1%})")

    if pct < 0.10 or pct > 0.60:
        logger.warning(
            f"High-risk proportion ({pct:.1%}) is outside expected range [10%, 60%]. "
            f"Consider adjusting n_clusters or reviewing the data window."
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

    rfm = compute_rfm(df)
    rfm = assign_risk_labels(rfm)
    risk_map = rfm.set_index("CustomerId")["is_high_risk"].to_dict()

    pipeline = build_feature_pipeline()
    df_processed = pipeline.fit_transform(df)

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
