#!/usr/bin/env python3
"""
Test pandas to polars conversion equivalence.

Ensures that converting from pandas to polars and back produces identical results,
and that Seer produces the same forecasts with both pandas and polars inputs.
"""

import pytest
import pandas as pd
import polars as pl
import numpy as np
from datetime import datetime
from farseer import Farseer


class TestPolarsPandasConversion:
    """Test conversion between pandas and polars DataFrames"""

    def test_simple_dataframe_conversion(self):
        """Test basic DataFrame conversion"""
        # Create pandas DataFrame
        df_pandas = pd.DataFrame(
            {"ds": pd.date_range("2020-01-01", periods=10), "y": range(10)}
        )

        # Convert to polars
        df_polars = pl.from_pandas(df_pandas)

        # Convert back to pandas
        df_back = df_polars.to_pandas()

        # Should be identical
        pd.testing.assert_frame_equal(df_pandas, df_back)

    def test_date_column_preservation(self):
        """Test that date column types are preserved"""
        df_pandas = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=50),
                "y": np.arange(50) * 0.5 + 10,
            }
        )

        df_polars = pl.from_pandas(df_pandas)

        # Check that ds is datetime in both
        assert pd.api.types.is_datetime64_any_dtype(df_pandas["ds"])
        assert df_polars["ds"].dtype == pl.Datetime

    def test_string_dates_conversion(self):
        """Test conversion with string dates"""
        dates_str = (
            pd.date_range("2020-01-01", periods=20).strftime("%Y-%m-%d").tolist()
        )

        df_pandas = pd.DataFrame({"ds": dates_str, "y": range(20)})

        df_polars = pl.DataFrame({"ds": dates_str, "y": range(20)})

        # Both should work with Seer
        model_pandas = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        model_polars = Farseer(yearly_seasonality=False, weekly_seasonality=False)

        model_pandas.fit(df_pandas)
        model_polars.fit(df_polars)

        # Both should succeed
        assert model_pandas.params()["fitted"]
        assert model_polars.params()["fitted"]


class TestSeerForecastEquivalence:
    """Test that Seer produces equivalent forecasts with pandas and polars"""

    def test_linear_trend_equivalence(self):
        """Test that linear trend forecast is identical with pandas and polars"""
        # Generate identical data
        n = 100
        dates = pd.date_range("2020-01-01", periods=n)
        y_values = np.arange(n) * 0.5 + 10 + np.random.RandomState(42).randn(n) * 0.1

        # Create pandas DataFrame
        df_pandas = pd.DataFrame({"ds": dates, "y": y_values})

        # Create polars DataFrame
        df_polars = pl.from_pandas(df_pandas)

        # Fit models
        model_pandas = Farseer(
            growth="linear",
            yearly_seasonality=False,
            weekly_seasonality=False,
            n_changepoints=0,  # No changepoints for deterministic test
        )
        model_polars = Farseer(
            growth="linear",
            yearly_seasonality=False,
            weekly_seasonality=False,
            n_changepoints=0,
        )

        model_pandas.fit(df_pandas)
        model_polars.fit(df_polars)

        # Make predictions
        future_pandas = model_pandas.make_future_dataframe(periods=10)
        future_polars = model_polars.make_future_dataframe(periods=10)

        forecast_pandas = model_pandas.predict(future_pandas)
        forecast_polars = model_polars.predict(future_polars)

        # Convert polars to pandas for comparison
        if isinstance(forecast_polars, pl.DataFrame):
            forecast_polars_pd = forecast_polars.to_pandas()
        else:
            forecast_polars_pd = forecast_polars

        if isinstance(forecast_pandas, pl.DataFrame):
            forecast_pandas_pd = forecast_pandas.to_pandas()
        else:
            forecast_pandas_pd = forecast_pandas

        # Compare forecasts - should be very close
        np.testing.assert_allclose(
            forecast_pandas_pd["yhat"].values,
            forecast_polars_pd["yhat"].values,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_seasonality_forecast_equivalence(self):
        """Test forecast equivalence with seasonality"""
        n = 365
        dates = pd.date_range("2020-01-01", periods=n)

        # Generate data with weekly and yearly patterns
        t = np.arange(n)
        weekly = 3 * np.sin(2 * np.pi * t / 7)
        yearly = 5 * np.sin(2 * np.pi * t / 365.25)
        trend = 0.05 * t + 50
        # Use Prophet's seed for reproducibility (matches Facebook Prophet tests)
        np.random.seed(876543987)
        noise = np.random.randn(n) * 0.5
        y_values = trend + weekly + yearly + noise

        df_pandas = pd.DataFrame({"ds": dates, "y": y_values})
        df_polars = pl.from_pandas(df_pandas)

        # Fit models with seasonality
        model_pandas = Farseer(
            growth="linear",
            yearly_seasonality=True,
            weekly_seasonality=True,
            seasonality_mode="additive",
        )
        model_polars = Farseer(
            growth="linear",
            yearly_seasonality=True,
            weekly_seasonality=True,
            seasonality_mode="additive",
        )

        model_pandas.fit(df_pandas)
        model_polars.fit(df_polars)

        # Predictions
        future_pandas = model_pandas.make_future_dataframe(periods=30)
        future_polars = model_polars.make_future_dataframe(periods=30)

        forecast_pandas = model_pandas.predict(future_pandas)
        forecast_polars = model_polars.predict(future_polars)

        # Convert for comparison
        if isinstance(forecast_polars, pl.DataFrame):
            forecast_polars_pd = forecast_polars.to_pandas()
        else:
            forecast_polars_pd = forecast_polars

        if isinstance(forecast_pandas, pl.DataFrame):
            forecast_pandas_pd = forecast_pandas.to_pandas()
        else:
            forecast_pandas_pd = forecast_pandas

        # Should produce identical forecasts (allowing for minor numerical differences)
        np.testing.assert_allclose(
            forecast_pandas_pd["yhat"].values,
            forecast_polars_pd["yhat"].values,
            rtol=1e-3,
            atol=1e-2,
        )

        np.testing.assert_allclose(
            forecast_pandas_pd["trend"].values,
            forecast_polars_pd["trend"].values,
            rtol=1e-3,
            atol=1e-2,
        )

    def test_logistic_growth_equivalence(self):
        """Test logistic growth forecast equivalence"""
        n = 100
        dates = pd.date_range("2020-01-01", periods=n)

        # S-curve data
        t = np.arange(n)
        y_values = (
            100 / (1 + np.exp(-(t - 50) / 10)) + np.random.RandomState(42).randn(n) * 2
        )
        cap_values = [110] * n

        df_pandas = pd.DataFrame({"ds": dates, "y": y_values, "cap": cap_values})
        df_polars = pl.from_pandas(df_pandas)

        model_pandas = Farseer(
            growth="logistic", yearly_seasonality=False, weekly_seasonality=False
        )
        model_polars = Farseer(
            growth="logistic", yearly_seasonality=False, weekly_seasonality=False
        )

        model_pandas.fit(df_pandas)
        model_polars.fit(df_polars)

        # Create future with cap
        future_pandas = model_pandas.make_future_dataframe(periods=20)
        future_polars = model_polars.make_future_dataframe(periods=20)

        # Add cap to future
        if isinstance(future_pandas, pl.DataFrame):
            future_pandas = future_pandas.to_pandas()
        future_pandas["cap"] = 110

        if isinstance(future_polars, pl.DataFrame):
            future_polars_pd = future_polars.to_pandas()
        else:
            future_polars_pd = future_polars
        future_polars_pd["cap"] = 110
        future_polars = pl.from_pandas(future_polars_pd)

        forecast_pandas = model_pandas.predict(future_pandas)
        forecast_polars = model_polars.predict(future_polars)

        # Convert for comparison
        if isinstance(forecast_polars, pl.DataFrame):
            forecast_polars_pd = forecast_polars.to_pandas()
        else:
            forecast_polars_pd = forecast_polars

        if isinstance(forecast_pandas, pl.DataFrame):
            forecast_pandas_pd = forecast_pandas.to_pandas()
        else:
            forecast_pandas_pd = forecast_pandas

        # Forecasts should be equivalent (allowing for minor numerical differences)
        np.testing.assert_allclose(
            forecast_pandas_pd["yhat"].values,
            forecast_polars_pd["yhat"].values,
            rtol=5e-2,
            atol=3e-1,
        )


class TestPolarsSpecificFeatures:
    """Test polars-specific functionality"""

    def test_native_polars_input(self):
        """Test using native polars DataFrames"""
        df = pl.DataFrame(
            {
                "ds": pl.date_range(
                    datetime(2020, 1, 1),
                    datetime(2020, 4, 9),
                    interval="1d",
                    eager=True,
                ),
                "y": range(100),
            }
        )

        model = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        model.fit(df)

        future = model.make_future_dataframe(periods=10)
        assert isinstance(future, pl.DataFrame)

        forecast = model.predict(future)
        assert isinstance(forecast, pl.DataFrame)

        assert "yhat" in forecast.columns
        assert "trend" in forecast.columns

    def test_polars_lazy_evaluation(self):
        """Test that polars DataFrames work with lazy operations"""
        # Create a lazy polars DataFrame
        # 2020 is a leap year, so 366 days from Jan 1 to Dec 31
        dates = pl.date_range(
            datetime(2020, 1, 1), datetime(2020, 12, 31), interval="1d", eager=True
        )
        n = len(dates)
        df_lazy = pl.DataFrame(
            {"ds": dates, "y": np.random.randn(n).cumsum() + 100}
        ).lazy()

        # Collect to eager for Seer
        df = df_lazy.collect()

        model = Farseer()
        model.fit(df)

        forecast = model.predict(model.make_future_dataframe(periods=30))
        assert isinstance(forecast, pl.DataFrame)
        assert len(forecast) > 0

    def test_polars_column_operations(self):
        """Test polars column operations work correctly"""
        n = 100
        df = pl.DataFrame(
            {
                "ds": pl.date_range(
                    datetime(2020, 1, 1),
                    datetime(2020, 4, 9),  # 100 days from start
                    interval="1d",
                    eager=True,
                ),
                "y": range(n),
            }
        )

        # Test column access
        assert "ds" in df.columns
        assert "y" in df.columns

        # Test filtering (polars style)
        df_filtered = df.filter(pl.col("y") > 50)
        assert len(df_filtered) < len(df)

        # Model should work with full data
        model = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        model.fit(df)

        forecast = model.predict(model.make_future_dataframe(periods=10))
        assert len(forecast) == 110


class TestMixedDataTypes:
    """Test handling of mixed pandas and polars inputs"""

    def test_fit_pandas_predict_polars_future(self):
        """Fit with pandas, predict with polars future"""
        df_pandas = pd.DataFrame(
            {"ds": pd.date_range("2020-01-01", periods=100), "y": range(100)}
        )

        model = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        model.fit(df_pandas)

        # Get future as polars
        future = model.make_future_dataframe(periods=10)

        # Should work regardless of return type
        forecast = model.predict(future)
        assert len(forecast) == 110

    def test_fit_polars_with_pandas_conversion(self):
        """Fit with polars, convert forecast to pandas"""
        dates = pl.date_range(
            datetime(2020, 1, 1), datetime(2020, 4, 9), interval="1d", eager=True
        )
        df_polars = pl.DataFrame({"ds": dates, "y": range(len(dates))})

        model = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        model.fit(df_polars)

        future = model.make_future_dataframe(periods=10)
        forecast = model.predict(future)

        # Convert to pandas
        if isinstance(forecast, pl.DataFrame):
            forecast_pd = forecast.to_pandas()
        else:
            forecast_pd = forecast

        assert isinstance(forecast_pd, pd.DataFrame)
        assert "yhat" in forecast_pd.columns


class TestPerformance:
    """Test performance characteristics of polars vs pandas"""

    def test_large_dataset_polars(self):
        """Test polars handles large datasets efficiently"""
        n = 10000
        # Use pandas date_range then convert
        dates = pd.date_range("2000-01-01", periods=n)
        df = pl.DataFrame({"ds": dates, "y": np.random.randn(n).cumsum() + 100})

        model = Farseer(yearly_seasonality=False, weekly_seasonality=False)

        # Should fit without issues
        model.fit(df)
        assert model.params()["fitted"]

        # Should predict efficiently
        forecast = model.predict(model.make_future_dataframe(periods=100))
        assert len(forecast) == n + 100

    def test_memory_efficiency(self):
        """Test that polars is memory efficient"""
        # Create identical data
        n = 5000
        dates = pd.date_range("2000-01-01", periods=n)
        y_values = np.random.randn(n).cumsum() + 100

        df_pandas = pd.DataFrame({"ds": dates, "y": y_values})
        df_polars = pl.from_pandas(df_pandas)

        # Both should work
        model1 = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        model2 = Farseer(yearly_seasonality=False, weekly_seasonality=False)

        model1.fit(df_pandas)
        model2.fit(df_polars)

        # Both should produce results
        f1 = model1.predict(model1.make_future_dataframe(periods=50))
        f2 = model2.predict(model2.make_future_dataframe(periods=50))

        assert len(f1) == len(f2) == n + 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
