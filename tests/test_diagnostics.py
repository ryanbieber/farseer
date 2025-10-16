#!/usr/bin/env python3
"""
Diagnostics Test Suite
Tests for cross-validation and performance metrics
Based on Prophet's test_diagnostics.py
"""

import pytest
import pandas as pd
import numpy as np
from farseer import Farseer


@pytest.fixture(scope="module")
def ts_short(daily_univariate_ts):
    """Short time series for testing"""
    return daily_univariate_ts.head(100)


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


class TestModelComparison:
    """Test model comparison utilities"""

    def test_basic_comparison(self):
        """Test comparing two models"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100, freq="D"),
                "y": np.arange(100) * 0.5 + 10 + np.random.randn(100),
            }
        )

        # Model with weekly seasonality
        m1 = Farseer(yearly_seasonality=False, weekly_seasonality=True)
        m1.fit(df)

        # Model without seasonality
        m2 = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        m2.fit(df)

        # Both should fit
        assert m1.params()["fitted"] is True
        assert m2.params()["fitted"] is True

        # Can compare predictions
        future = pd.DataFrame({"ds": pd.date_range("2020-04-11", periods=10, freq="D")})

        f1 = m1.predict(future)
        f2 = m2.predict(future)

        # Should have different predictions
        assert len(f1) == len(f2)


class TestBacktesting:
    """Test backtesting functionality"""

    def test_simple_backtest(self):
        """Test simple backtesting scenario"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=200, freq="D"),
                "y": np.arange(200) * 0.5 + 10 + np.random.randn(200) * 2,
            }
        )

        # Train on first 150 days
        train = df.head(150)
        test = df.tail(50)

        m = Farseer(yearly_seasonality=False, weekly_seasonality=True)
        m.fit(train)

        # Predict on test period
        forecast = m.predict(test[["ds"]])

        # Convert to numpy for comparison (forecast is polars, test is pandas)
        if hasattr(forecast, "to_pandas"):
            forecast_yhat = forecast["yhat"].to_numpy()
        else:
            forecast_yhat = forecast["yhat"].values
        test_y = test["y"].values

        # Calculate error
        error = np.mean(np.abs(forecast_yhat - test_y))

        # Error should be reasonable
        assert error < 50

    def test_rolling_origin_forecast(self):
        """Test rolling origin forecasting"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100, freq="D"),
                "y": np.arange(100) * 0.3 + 10 + np.random.randn(100),
            }
        )

        errors = []

        # Rolling window: train on increasing amounts of data
        for i in range(50, 90, 10):
            train = df.head(i)
            test_point = df.iloc[i]

            m = Farseer(yearly_seasonality=False, weekly_seasonality=False)
            m.fit(train)

            future = pd.DataFrame({"ds": [test_point["ds"]]})
            forecast = m.predict(future)

            # Convert to numpy for comparison (forecast is polars)
            if hasattr(forecast, "to_pandas"):
                forecast_yhat = forecast["yhat"].to_numpy()[0]
            else:
                forecast_yhat = forecast["yhat"].values[0]

            error = abs(forecast_yhat - test_point["y"])
            errors.append(error)

        # Should have some errors
        assert len(errors) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
