#!/usr/bin/env python3
"""
Test for ds column datetime handling
Ensures the ds column is properly converted to datetime and not NaT
"""

import pandas as pd
import polars as pl
import numpy as np
from farseer import Farseer


class TestDsColumnDatetime:
    """Test that ds column is properly handled as datetime"""

    def test_ds_column_not_nat_pandas_input(self):
        """Test that ds column doesn't become NaT with pandas input"""
        # Create sample data with pandas
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100, freq="D"),
                "y": np.random.randn(100).cumsum() + 100,
            }
        )

        # Fit model
        model = Farseer(
            yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False
        )
        model.fit(df)

        # Make predictions
        future = model.make_future_dataframe(periods=10)
        forecast = model.predict(future)

        # Check that ds column is datetime and has no NaT values
        assert (
            forecast["ds"].dtype == pl.Datetime
        ), f"Expected Datetime, got {forecast['ds'].dtype}"
        assert (
            forecast["ds"].null_count() == 0
        ), f"Found {forecast['ds'].null_count()} null values in ds column"

        # Verify all dates are valid
        assert len(forecast) == 110

    def test_ds_column_not_nat_polars_input(self):
        """Test that ds column doesn't become NaT with polars input"""
        # Create sample data with polars
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pl.DataFrame({"ds": dates, "y": np.random.randn(100).cumsum() + 100})

        # Fit model
        model = Farseer(
            yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False
        )
        model.fit(df)

        # Make predictions
        future = model.make_future_dataframe(periods=10)
        forecast = model.predict(future)

        # Check that ds column is datetime and has no NaT values
        assert (
            forecast["ds"].dtype == pl.Datetime
        ), f"Expected Datetime, got {forecast['ds'].dtype}"
        assert (
            forecast["ds"].null_count() == 0
        ), f"Found {forecast['ds'].null_count()} null values in ds column"

        # Verify all dates are valid
        assert len(forecast) == 110

    def test_ds_column_with_custom_dates(self):
        """Test ds column with custom date range passed to predict"""
        # Create sample data
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100, freq="D"),
                "y": np.random.randn(100).cumsum() + 100,
            }
        )

        # Fit model
        model = Farseer(
            yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False
        )
        model.fit(df)

        # Create custom future dates
        custom_dates = pl.DataFrame(
            {"ds": pd.date_range("2020-04-10", periods=20, freq="D")}
        )

        forecast = model.predict(custom_dates)

        # Check that ds column is datetime and has no NaT values
        assert (
            forecast["ds"].dtype == pl.Datetime
        ), f"Expected Datetime, got {forecast['ds'].dtype}"
        assert (
            forecast["ds"].null_count() == 0
        ), f"Found {forecast['ds'].null_count()} null values in ds column"

        # Verify dates match what we passed in
        assert len(forecast) == 20

    def test_ds_column_to_pandas_conversion(self):
        """Test that ds column can be converted to pandas without NaT"""
        # Create sample data
        df = pl.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=50, freq="D"),
                "y": np.random.randn(50).cumsum() + 100,
            }
        )

        # Fit and predict
        model = Farseer(
            yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False
        )
        model.fit(df)
        forecast = model.predict(model.make_future_dataframe(periods=10))

        # Convert to pandas
        forecast_pd = forecast.to_pandas()

        # Check pandas ds column
        assert pd.api.types.is_datetime64_any_dtype(
            forecast_pd["ds"]
        ), f"Expected datetime64 type, got {forecast_pd['ds'].dtype}"
        assert (
            forecast_pd["ds"].isna().sum() == 0
        ), f"Found {forecast_pd['ds'].isna().sum()} NaT values in pandas ds column"

    def test_ds_column_prophet_comparison(self):
        """Test that ds column format is compatible with Prophet output"""
        # Create sample data
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        df = pd.DataFrame({"ds": dates, "y": np.random.randn(50).cumsum() + 100})

        # Fit Seer model
        model = Farseer(
            yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False
        )
        model.fit(df)

        # Get forecast
        future = model.make_future_dataframe(periods=10)
        forecast = model.predict(future)

        # Convert to pandas for comparison
        forecast_pd = forecast.to_pandas()

        # Check that ds is datetime and matches expected dates
        assert pd.api.types.is_datetime64_any_dtype(forecast_pd["ds"])
        assert len(forecast_pd) == 60

        # Verify first and last dates are correct
        expected_start = pd.Timestamp("2020-01-01")
        expected_end = pd.Timestamp("2020-02-29")  # 60 days from 2020-01-01

        assert (
            forecast_pd["ds"].iloc[0] == expected_start
        ), f"First date mismatch: {forecast_pd['ds'].iloc[0]} vs {expected_start}"
        assert (
            forecast_pd["ds"].iloc[-1] == expected_end
        ), f"Last date mismatch: {forecast_pd['ds'].iloc[-1]} vs {expected_end}"

    def test_ds_column_with_string_dates(self):
        """Test ds column when input has string dates"""
        # Create data with string dates
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=50, freq="D")
                .strftime("%Y-%m-%d")
                .tolist(),
                "y": np.random.randn(50).cumsum() + 100,
            }
        )

        # Fit and predict
        model = Farseer(
            yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False
        )
        model.fit(df)

        forecast = model.predict(model.make_future_dataframe(periods=10))

        # Check that ds column is datetime (not string) and has no NaT
        assert (
            forecast["ds"].dtype == pl.Datetime
        ), f"Expected Datetime, got {forecast['ds'].dtype}"
        assert (
            forecast["ds"].null_count() == 0
        ), f"Found {forecast['ds'].null_count()} null values in ds column"
