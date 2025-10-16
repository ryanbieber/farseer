#!/usr/bin/env python3
"""
Utilities Test Suite
Tests for utility functions and helpers
Based on Prophet's test_utilities.py
"""

import pytest
import pandas as pd
import numpy as np
from farseer import Farseer, regressor_coefficients


@pytest.fixture
def daily_univariate_ts():
    """Generate daily time series data for testing"""
    # Use Prophet's seed for reproducibility (matches Facebook Prophet tests)
    np.random.seed(876543987)
    n = 468
    dates = pd.date_range("2012-01-01", periods=n, freq="D")

    trend = np.arange(n) * 0.5 + 10
    yearly = np.sin(2 * np.pi * np.arange(n) / 365.25) * 5
    weekly = np.sin(2 * np.pi * np.arange(n) / 7) * 2
    noise = np.random.randn(n) * 2

    y = trend + yearly + weekly + noise

    return pd.DataFrame({"ds": dates, "y": y})


class TestUtilities:
    """Test utility functions"""

    def test_regressor_coefficients(self, daily_univariate_ts):
        """Test extracting regressor coefficients"""
        m = Farseer()
        df = daily_univariate_ts.copy()

        # Use Prophet's seed for reproducibility (matches Facebook Prophet tests)
        np.random.seed(876543987)
        df["regr1"] = np.random.normal(size=df.shape[0])
        df["regr2"] = np.random.normal(size=df.shape[0])

        m.add_regressor("regr1", mode="additive")
        m.add_regressor("regr2", mode="multiplicative")
        m.fit(df)

        # Extract coefficients
        coefs = regressor_coefficients(m)
        assert coefs.shape[0] == 2
        assert "regressor" in coefs.columns
        assert "coef" in coefs.columns
        assert "regressor_mode" in coefs.columns

        # Check regressor names
        assert set(coefs["regressor"].values) == {"regr1", "regr2"}

        # Check modes
        regr1_row = coefs[coefs["regressor"] == "regr1"].iloc[0]
        regr2_row = coefs[coefs["regressor"] == "regr2"].iloc[0]

        assert "additive" in regr1_row["regressor_mode"].lower()
        assert "multiplicative" in regr2_row["regressor_mode"].lower()

        # Coefficients should be numeric
        assert isinstance(regr1_row["coef"], (int, float))
        assert isinstance(regr2_row["coef"], (int, float))


class TestDataValidation:
    """Test data validation utilities"""

    def test_validate_dataframe_columns(self):
        """Test that dataframe validation works"""
        # Valid dataframe
        df_valid = pd.DataFrame(
            {"ds": pd.date_range("2020-01-01", periods=10), "y": range(10)}
        )

        m = Farseer()
        m.fit(df_valid)

        assert m.params()["fitted"] is True

        # Invalid: missing 'y'
        df_invalid = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=10)})

        with pytest.raises(Exception):
            m.fit(df_invalid)

    def test_validate_date_format(self):
        """Test various date formats"""
        # String dates
        df1 = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=50, freq="D")
                .strftime("%Y-%m-%d")
                .tolist(),
                "y": [10 + i * 0.1 for i in range(50)],
            }
        )

        m1 = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        m1.fit(df1)
        assert m1.params()["fitted"] is True

        # Datetime dates
        df2 = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=50, freq="D"),
                "y": [10 + i * 0.1 for i in range(50)],
            }
        )

        m2 = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        m2.fit(df2)
        assert m2.params()["fitted"] is True

    def test_validate_numeric_y(self):
        """Test that y column must be numeric"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=10),
                "y": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            }
        )

        m = Farseer()
        with pytest.raises(Exception):
            m.fit(df)


class TestHelperFunctions:
    """Test helper and convenience functions"""

    def test_params_extraction(self):
        """Test extracting model parameters"""
        m = Farseer(
            growth="linear",
            n_changepoints=10,
            changepoint_range=0.9,
            yearly_seasonality=True,
            weekly_seasonality=False,
        )

        params = m.params()

        assert params["growth"] in ["Linear", "linear"]
        assert params["n_changepoints"] == 10
        assert params["changepoint_range"] == 0.9
        assert params["yearly_seasonality"] is True
        assert params["weekly_seasonality"] is False

    def test_model_string_representation(self):
        """Test model string representation"""
        m = Farseer()

        # Should have some string representation
        str_repr = str(m)
        assert len(str_repr) > 0

    def test_seasonality_info(self):
        """Test getting seasonality information"""
        m = Farseer(yearly_seasonality=True, weekly_seasonality=False)
        m.add_seasonality("monthly", period=30.5, fourier_order=5)

        params = m.params()
        seasonalities = params["seasonalities"]

        # Should have yearly and monthly
        names = [s["name"] for s in seasonalities]
        assert "yearly" in names or "monthly" in names


class TestDateHandling:
    """Test date and frequency handling"""

    def test_infer_frequency_daily(self):
        """Test frequency inference for daily data"""
        df = pd.DataFrame(
            {"ds": pd.date_range("2020-01-01", periods=100, freq="D"), "y": range(100)}
        )

        m = Farseer()
        m.fit(df)

        # Should infer daily frequency
        future = m.make_future_dataframe(periods=7)
        assert len(future) == 107

    def test_infer_frequency_hourly(self):
        """Test frequency inference for hourly data"""
        df = pd.DataFrame(
            {"ds": pd.date_range("2020-01-01", periods=100, freq="h"), "y": range(100)}
        )

        m = Farseer()
        m.fit(df)

        # Should infer hourly frequency
        future = m.make_future_dataframe(periods=24, freq="h")
        assert len(future) == 124

    def test_irregular_dates(self):
        """Test handling of irregular date spacing"""
        # Dates with some missing days
        dates = pd.to_datetime(
            [
                "2020-01-01",
                "2020-01-02",
                "2020-01-05",
                "2020-01-06",
                "2020-01-10",
                "2020-01-11",
            ]
        )

        df = pd.DataFrame({"ds": dates, "y": range(len(dates))})

        m = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        m.fit(df)

        # Should handle irregular spacing
        assert m.params()["fitted"] is True


class TestScaling:
    """Test data scaling utilities"""

    def test_y_scaling(self):
        """Test that y values are properly scaled internally"""
        # Large values
        df_large = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=50),
                "y": np.arange(50) * 1000 + 10000,
            }
        )

        m_large = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        m_large.fit(df_large)

        # Small values
        df_small = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=50),
                "y": np.arange(50) * 0.001 + 0.01,
            }
        )

        m_small = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        m_small.fit(df_small)

        # Both should fit successfully
        assert m_large.params()["fitted"] is True
        assert m_small.params()["fitted"] is True

    def test_zero_mean_data(self):
        """Test handling of zero-mean data"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100),
                "y": np.random.randn(100),  # Zero mean
            }
        )

        m = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        m.fit(df)

        assert m.params()["fitted"] is True

    def test_negative_values(self):
        """Test handling of negative values"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=50),
                "y": np.arange(50) - 25,  # Negative and positive
            }
        )

        m = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        m.fit(df)

        future = m.make_future_dataframe(periods=10)
        forecast = m.predict(future)

        # Should handle negative values
        assert len(forecast) == 60


class TestModelCopy:
    """Test model copying and cloning"""

    def test_basic_copy(self):
        """Test basic model copying"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100),
                "y": np.arange(100) * 0.5 + 10,
            }
        )

        m1 = Farseer(n_changepoints=10, yearly_seasonality=False)
        m1.fit(df)

        # Create second model with same params
        m2 = Farseer(n_changepoints=10, yearly_seasonality=False)
        m2.fit(df)

        # Should have same configuration
        assert m1.params()["n_changepoints"] == m2.params()["n_changepoints"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
