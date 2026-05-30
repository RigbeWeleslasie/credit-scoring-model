"""
-----------------------
Unit tests for helper functions in src/data_processing.py.
"""

import pytest
import pandas as pd
from src.data_processing import load_data


def make_sample_df():
    """Create a minimal transaction DataFrame for testing."""
    return pd.DataFrame({
        "TransactionId": ["T1", "T2", "T3"],
        "CustomerId": ["C1", "C1", "C2"],
        "Amount": [1000, -20, 500],
        "Value": [1000, 20, 500],
        "TransactionStartTime": [
            "2018-11-15T02:18:49Z",
            "2018-11-15T02:19:08Z",
            "2018-11-16T10:00:00Z",
        ],
        "ProductCategory": ["airtime", "financial_services", "airtime"],
        "ChannelId": ["ChannelId_3", "ChannelId_2", "ChannelId_3"],
        "FraudResult": [0, 0, 0],
    })


class TestLoadData:
    def test_returns_dataframe(self, tmp_path):
        """load_data should return a pandas DataFrame."""
        sample = make_sample_df()
        path = tmp_path / "test.csv"
        sample.to_csv(path, index=False)
        result = load_data(str(path))
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, tmp_path):
        """Loaded DataFrame should preserve all original columns."""
        sample = make_sample_df()
        path = tmp_path / "test.csv"
        sample.to_csv(path, index=False)
        result = load_data(str(path))
        assert set(sample.columns).issubset(set(result.columns))

    def test_row_count(self, tmp_path):
        """Loaded DataFrame should have the same number of rows as the file."""
        sample = make_sample_df()
        path = tmp_path / "test.csv"
        sample.to_csv(path, index=False)
        result = load_data(str(path))
        assert len(result) == len(sample)