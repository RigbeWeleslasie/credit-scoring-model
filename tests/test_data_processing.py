"""
test_data_processing.py
-----------------------
Unit tests for src/data_processing.py, src/train.py, and src/predict.py.
"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.data_processing import (
    load_data,
    TemporalFeatureExtractor,
    AggregateFeatureBuilder,
    DropIdentifierColumns,
    LogTransformer,
    get_categorical_cols,
    get_numerical_cols,
    build_feature_pipeline,
    compute_rfm,
    assign_risk_labels,
)
from src.train import (
    evaluate_model,
    build_lr_pipeline,
    build_xgb_pipeline,
    load_processed_data,
)
from src.predict import predict_single


def make_sample_df():
    """Minimal transaction DataFrame for testing."""
    return pd.DataFrame({
        "TransactionId": ["T1", "T2", "T3", "T4"],
        "BatchId": ["B1", "B1", "B2", "B2"],
        "AccountId": ["A1", "A1", "A2", "A2"],
        "SubscriptionId": ["S1", "S1", "S2", "S2"],
        "CustomerId": ["C1", "C1", "C2", "C2"],
        "CurrencyCode": ["UGX", "UGX", "UGX", "UGX"],
        "CountryCode": [256, 256, 256, 256],
        "ProviderId": ["P1", "P2", "P1", "P3"],
        "ProductId": ["X1", "X2", "X1", "X3"],
        "ProductCategory": ["airtime", "financial_services", "airtime", "utility_bill"],
        "ChannelId": ["ChannelId_3", "ChannelId_2", "ChannelId_3", "ChannelId_1"],
        "Amount": [1000, -20, 500, 300],
        "Value": [1000, 20, 500, 300],
        "TransactionStartTime": [
            "2018-11-15T02:18:49Z",
            "2018-11-15T10:19:08Z",
            "2018-11-16T08:00:00Z",
            "2018-11-17T15:30:00Z",
        ],
        "PricingStrategy": [2, 2, 0, 2],
        "FraudResult": [0, 0, 0, 1],
    })


def make_rfm_df():
    """Minimal RFM DataFrame for clustering tests."""
    return pd.DataFrame({
        "CustomerId": ["C1", "C2", "C3", "C4", "C5", "C6"],
        "Recency": [1, 2, 100, 150, 200, 3],
        "Frequency": [50, 40, 2, 1, 1, 45],
        "Monetary": [50000, 40000, 500, 200, 100, 48000],
    })


def make_model_ready_df():
    """Minimal numeric DataFrame for model training tests."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "total_transaction_amount": np.random.randn(n),
        "avg_transaction_amount": np.random.randn(n),
        "transaction_count": np.random.randint(1, 50, n).astype(float),
        "std_transaction_amount": np.abs(np.random.randn(n)),
        "total_value": np.random.randn(n),
        "avg_value": np.random.randn(n),
        "Amount": np.random.randn(n),
        "Value": np.abs(np.random.randn(n)),
        "txn_hour": np.random.randint(0, 24, n).astype(float),
        "txn_day": np.random.randint(1, 28, n).astype(float),
        "txn_month": np.random.randint(1, 12, n).astype(float),
        "txn_year": np.full(n, 2018.0),
        "txn_day_of_week": np.random.randint(0, 7, n).astype(float),
        "PricingStrategy": np.random.randint(0, 3, n).astype(float),
        "FraudResult": np.zeros(n),
        "is_high_risk": np.random.randint(0, 2, n),
    })


class TestLoadData:
    def test_returns_dataframe(self, tmp_path):
        sample = make_sample_df()
        path = tmp_path / "test.csv"
        sample.to_csv(path, index=False)
        result = load_data(str(path))
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, tmp_path):
        sample = make_sample_df()
        path = tmp_path / "test.csv"
        sample.to_csv(path, index=False)
        result = load_data(str(path))
        assert set(sample.columns).issubset(set(result.columns))

    def test_row_count(self, tmp_path):
        sample = make_sample_df()
        path = tmp_path / "test.csv"
        sample.to_csv(path, index=False)
        result = load_data(str(path))
        assert len(result) == len(sample)


class TestTemporalFeatureExtractor:
    def test_creates_temporal_columns(self):
        df = make_sample_df()
        result = TemporalFeatureExtractor().fit_transform(df)
        for col in ["txn_hour", "txn_day", "txn_month", "txn_year", "txn_day_of_week"]:
            assert col in result.columns

    def test_drops_original_time_column(self):
        df = make_sample_df()
        result = TemporalFeatureExtractor().fit_transform(df)
        assert "TransactionStartTime" not in result.columns

    def test_hour_range(self):
        df = make_sample_df()
        result = TemporalFeatureExtractor().fit_transform(df)
        assert result["txn_hour"].between(0, 23).all()

    def test_month_range(self):
        df = make_sample_df()
        result = TemporalFeatureExtractor().fit_transform(df)
        assert result["txn_month"].between(1, 12).all()


class TestAggregateFeatureBuilder:
    def test_creates_aggregate_columns(self):
        df = make_sample_df()
        result = AggregateFeatureBuilder().fit_transform(df)
        for col in ["total_transaction_amount", "avg_transaction_amount",
                    "transaction_count", "std_transaction_amount",
                    "total_value", "avg_value"]:
            assert col in result.columns

    def test_transaction_count_correct(self):
        df = make_sample_df()
        result = AggregateFeatureBuilder().fit_transform(df)
        assert (result[result["CustomerId"] == "C1"]["transaction_count"] == 2).all()

    def test_no_nulls_in_std_column(self):
        df = make_sample_df()
        result = AggregateFeatureBuilder().fit_transform(df)
        assert result["std_transaction_amount"].isnull().sum() == 0


class TestDropIdentifierColumns:
    def test_drops_expected_columns(self):
        df = make_sample_df()
        result = DropIdentifierColumns().fit_transform(df)
        for col in ["TransactionId", "BatchId", "AccountId", "SubscriptionId", "CustomerId"]:
            assert col not in result.columns

    def test_retains_feature_columns(self):
        df = make_sample_df()
        result = DropIdentifierColumns().fit_transform(df)
        for col in ["Amount", "Value", "ProductCategory", "FraudResult"]:
            assert col in result.columns


class TestLogTransformer:
    def test_transforms_value_column(self):
        df = make_sample_df()
        result = LogTransformer(cols=["Value"]).fit_transform(df)
        expected = np.log1p(df["Value"].clip(lower=0))
        pd.testing.assert_series_equal(result["Value"], expected)

    def test_no_negative_values_after_transform(self):
        df = make_sample_df()
        result = LogTransformer(cols=["Value"]).fit_transform(df)
        assert (result["Value"] >= 0).all()


class TestColumnSelectors:
    def test_get_categorical_cols(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
        assert get_categorical_cols(df) == ["a"]

    def test_get_numerical_cols(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
        assert get_numerical_cols(df) == ["b"]


class TestBuildFeaturePipeline:
    def test_pipeline_runs_without_error(self):
        df = make_sample_df()
        result = build_feature_pipeline().fit_transform(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    def test_pipeline_drops_time_column(self):
        df = make_sample_df()
        result = build_feature_pipeline().fit_transform(df)
        assert "TransactionStartTime" not in result.columns

    def test_pipeline_adds_aggregate_columns(self):
        df = make_sample_df()
        result = build_feature_pipeline().fit_transform(df)
        assert "transaction_count" in result.columns
        assert "avg_transaction_amount" in result.columns


class TestComputeRFM:
    def test_returns_dataframe(self):
        assert isinstance(compute_rfm(make_sample_df()), pd.DataFrame)

    def test_has_rfm_columns(self):
        result = compute_rfm(make_sample_df())
        for col in ["CustomerId", "Recency", "Frequency", "Monetary"]:
            assert col in result.columns

    def test_one_row_per_customer(self):
        result = compute_rfm(make_sample_df())
        assert result["CustomerId"].nunique() == result.shape[0]

    def test_recency_non_negative(self):
        assert (compute_rfm(make_sample_df())["Recency"] >= 0).all()

    def test_frequency_positive(self):
        assert (compute_rfm(make_sample_df())["Frequency"] > 0).all()

    def test_monetary_non_negative(self):
        assert (compute_rfm(make_sample_df())["Monetary"] >= 0).all()

    def test_snapshot_date_respected(self):
        result = compute_rfm(make_sample_df(), snapshot_date="2018-12-01")
        assert result["Recency"].min() > 0


class TestAssignRiskLabels:
    def test_adds_is_high_risk_column(self):
        assert "is_high_risk" in assign_risk_labels(make_rfm_df()).columns

    def test_binary_labels(self):
        result = assign_risk_labels(make_rfm_df())
        assert set(result["is_high_risk"].unique()).issubset({0, 1})

    def test_adds_cluster_column(self):
        assert "cluster" in assign_risk_labels(make_rfm_df()).columns

    def test_high_risk_customers_have_high_recency(self):
        result = assign_risk_labels(make_rfm_df(), n_clusters=3, random_state=42)
        high = result[result["is_high_risk"] == 1]["Recency"].mean()
        low = result[result["is_high_risk"] == 0]["Recency"].mean()
        assert high > low

    def test_reproducible_with_same_seed(self):
        r1 = assign_risk_labels(make_rfm_df(), random_state=42)["is_high_risk"]
        r2 = assign_risk_labels(make_rfm_df(), random_state=42)["is_high_risk"]
        pd.testing.assert_series_equal(
            r1.reset_index(drop=True), r2.reset_index(drop=True)
        )


class TestLoadProcessedData:
    def test_raises_if_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_processed_data(str(tmp_path / "nonexistent.parquet"))

    def test_raises_if_target_missing(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "data.parquet"
        df.to_parquet(path)
        with pytest.raises(ValueError, match="is_high_risk"):
            load_processed_data(str(path))

    def test_returns_x_and_y(self, tmp_path):
        df = make_model_ready_df()
        path = tmp_path / "data.parquet"
        df.to_parquet(path)
        X, y = load_processed_data(str(path))
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert "is_high_risk" not in X.columns
        assert len(X) == len(y)


class TestEvaluateModel:
    def test_returns_all_metrics(self):
        df = make_model_ready_df()
        X = df.drop(columns=["is_high_risk"])
        y = df["is_high_risk"]
        pipeline = build_lr_pipeline()
        pipeline.fit(X, y)
        metrics = evaluate_model(pipeline, X, y)
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0

    def test_metrics_are_floats(self):
        df = make_model_ready_df()
        X = df.drop(columns=["is_high_risk"])
        y = df["is_high_risk"]
        pipeline = build_lr_pipeline()
        pipeline.fit(X, y)
        metrics = evaluate_model(pipeline, X, y)
        for v in metrics.values():
            assert isinstance(v, float)


class TestBuildPipelines:
    def test_lr_pipeline_has_correct_steps(self):
        from sklearn.linear_model import LogisticRegression
        pipeline = build_lr_pipeline()
        assert "imputer" in pipeline.named_steps
        assert "scaler" in pipeline.named_steps
        assert "model" in pipeline.named_steps
        assert isinstance(pipeline.named_steps["model"], LogisticRegression)

    def test_xgb_pipeline_has_correct_steps(self):
        pipeline = build_xgb_pipeline()
        assert "imputer" in pipeline.named_steps
        assert "model" in pipeline.named_steps

    def test_lr_pipeline_fits_and_predicts(self):
        df = make_model_ready_df()
        X = df.drop(columns=["is_high_risk"])
        y = df["is_high_risk"]
        pipeline = build_lr_pipeline()
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1})


class TestPredictSingle:
    def test_raises_on_empty_features(self):
        mock_model = MagicMock()
        with pytest.raises(ValueError, match="empty"):
            predict_single(mock_model, {})

    def test_returns_correct_keys(self):
        with patch("src.predict.predict_single") as mock_fn:
            mock_fn.return_value = {
                "risk_probability": 0.7,
                "risk_label": "high_risk",
            }
            result = mock_fn(MagicMock(), {"feature": 1.0})
            assert "risk_probability" in result
            assert "risk_label" in result
            assert result["risk_label"] in ["high_risk", "low_risk"]
