"""
test_data_processing.py
-----------------------
Unit tests for helper functions and transformers in src/data_processing.py.
"""

import pandas as pd
import numpy as np
from src.data_processing import (
    load_data,
    TemporalFeatureExtractor,
    AggregateFeatureBuilder,
    DropIdentifierColumns,
    LogTransformer,
    get_categorical_cols,
    get_numerical_cols,
    build_feature_pipeline,
)


def make_sample_df():
    """Create a minimal transaction DataFrame for testing."""
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
        transformer = TemporalFeatureExtractor()
        result = transformer.fit_transform(df)
        for col in ["txn_hour", "txn_day", "txn_month", "txn_year", "txn_day_of_week"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_drops_original_time_column(self):
        df = make_sample_df()
        transformer = TemporalFeatureExtractor()
        result = transformer.fit_transform(df)
        assert "TransactionStartTime" not in result.columns

    def test_hour_range(self):
        df = make_sample_df()
        transformer = TemporalFeatureExtractor()
        result = transformer.fit_transform(df)
        assert result["txn_hour"].between(0, 23).all()

    def test_month_range(self):
        df = make_sample_df()
        transformer = TemporalFeatureExtractor()
        result = transformer.fit_transform(df)
        assert result["txn_month"].between(1, 12).all()


class TestAggregateFeatureBuilder:
    def test_creates_aggregate_columns(self):
        df = make_sample_df()
        builder = AggregateFeatureBuilder()
        result = builder.fit_transform(df)
        for col in ["total_transaction_amount", "avg_transaction_amount",
                    "transaction_count", "std_transaction_amount",
                    "total_value", "avg_value"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_transaction_count_correct(self):
        df = make_sample_df()
        builder = AggregateFeatureBuilder()
        result = builder.fit_transform(df)
        c1_rows = result[result["CustomerId"] == "C1"]
        assert (c1_rows["transaction_count"] == 2).all()

    def test_no_nulls_in_std_column(self):
        df = make_sample_df()
        builder = AggregateFeatureBuilder()
        result = builder.fit_transform(df)
        assert result["std_transaction_amount"].isnull().sum() == 0


class TestDropIdentifierColumns:
    def test_drops_expected_columns(self):
        df = make_sample_df()
        dropper = DropIdentifierColumns()
        result = dropper.fit_transform(df)
        for col in ["TransactionId", "BatchId", "AccountId",
                    "SubscriptionId", "CustomerId"]:
            assert col not in result.columns

    def test_retains_feature_columns(self):
        df = make_sample_df()
        dropper = DropIdentifierColumns()
        result = dropper.fit_transform(df)
        for col in ["Amount", "Value", "ProductCategory", "FraudResult"]:
            assert col in result.columns


class TestLogTransformer:
    def test_transforms_value_column(self):
        df = make_sample_df()
        transformer = LogTransformer(cols=["Value"])
        result = transformer.fit_transform(df)
        expected = np.log1p(df["Value"].clip(lower=0))
        pd.testing.assert_series_equal(result["Value"], expected)

    def test_no_negative_values_after_transform(self):
        df = make_sample_df()
        transformer = LogTransformer(cols=["Value"])
        result = transformer.fit_transform(df)
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
        pipeline = build_feature_pipeline()
        result = pipeline.fit_transform(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    def test_pipeline_drops_time_column(self):
        df = make_sample_df()
        pipeline = build_feature_pipeline()
        result = pipeline.fit_transform(df)
        assert "TransactionStartTime" not in result.columns

    def test_pipeline_adds_aggregate_columns(self):
        df = make_sample_df()
        pipeline = build_feature_pipeline()
        result = pipeline.fit_transform(df)
        assert "transaction_count" in result.columns
        assert "avg_transaction_amount" in result.columns
