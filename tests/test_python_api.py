#!/usr/bin/env python3
"""
Comprehensive test suite for Seer library.
Tests all major features and use cases.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from farseer import Farseer


class TestBasicFunctionality:
    """Test basic model creation, fitting, and prediction"""

    def test_model_creation(self):
        """Test basic model instantiation"""
        model = Farseer()
        assert model is not None

    def test_model_with_parameters(self):
        """Test model creation with custom parameters"""
        model = Farseer(
            growth="linear",
            n_changepoints=10,
            changepoint_range=0.9,
            changepoint_prior_scale=0.1,
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="additive",
            interval_width=0.95,
        )
        params = model.params()
        assert params["n_changepoints"] == 10
        assert params["changepoint_range"] == 0.9
        assert not params["yearly_seasonality"]
        assert params["weekly_seasonality"]

    def test_fit_basic(self):
        """Test basic model fitting"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100, freq="D"),
                "y": np.arange(100) * 0.5 + 10 + np.random.randn(100) * 0.5,
            }
        )

        model = Farseer()
        model.fit(df)

        params = model.params()
        assert params["fitted"]

    def test_predict_basic(self):
        """Test basic prediction"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100, freq="D"),
                "y": np.arange(100) * 0.5 + 10,
            }
        )

        model = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        model.fit(df)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        assert len(forecast) == 130  # 100 history + 30 future
        assert "yhat" in forecast.columns
        assert "yhat_lower" in forecast.columns
        assert "yhat_upper" in forecast.columns
        assert "trend" in forecast.columns

    def test_fit_with_string_dates(self):
        """Test fitting with string dates"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=50, freq="D")
                .strftime("%Y-%m-%d")
                .tolist(),
                "y": [10 + i * 0.5 for i in range(50)],
            }
        )

        model = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        model.fit(df)

        future = model.make_future_dataframe(periods=10)
        forecast = model.predict(future)

        assert len(forecast) == 60


class TestTrendTypes:
    """Test different trend configurations"""

    def test_linear_trend(self):
        """Test linear trend"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=50, freq="D"),
                "y": np.arange(50) * 2 + 10,
            }
        )

        model = Farseer(
            growth="linear", yearly_seasonality=False, weekly_seasonality=False
        )
        model.fit(df)

        future = model.make_future_dataframe(periods=10)
        forecast = model.predict(future)

        # Check that trend is increasing (forecast is polars)
        if hasattr(forecast, "to_pandas"):
            forecast_pd = forecast.to_pandas()
            assert forecast_pd["trend"].iloc[-1] > forecast_pd["trend"].iloc[0]
        else:
            assert forecast["trend"].iloc[-1] > forecast["trend"].iloc[0]

    def test_logistic_trend(self):
        """Test logistic growth"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=50, freq="D"),
                "y": np.arange(50) * 0.5 + 10,
                "cap": [100] * 50,
            }
        )

        model = Farseer(
            growth="logistic", yearly_seasonality=False, weekly_seasonality=False
        )
        model.fit(df)

        future = model.make_future_dataframe(periods=10)
        forecast = model.predict(future)

        # Logistic growth should be bounded
        assert all(forecast["yhat"] < 100)

    def test_flat_trend(self):
        """Test flat trend"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=50, freq="D"),
                "y": [10 + np.random.randn() * 0.1 for _ in range(50)],
            }
        )

        model = Farseer(
            growth="flat", yearly_seasonality=False, weekly_seasonality=False
        )
        model.fit(df)

        future = model.make_future_dataframe(periods=10)
        forecast = model.predict(future)

        # Flat trend should be roughly constant (forecast is polars)
        if hasattr(forecast, "to_pandas"):
            forecast_pd = forecast.to_pandas()
            trend_diff = abs(
                forecast_pd["trend"].iloc[-1] - forecast_pd["trend"].iloc[0]
            )
        else:
            trend_diff = abs(forecast["trend"].iloc[-1] - forecast["trend"].iloc[0])
        assert trend_diff < 1.0


class TestSeasonality:
    """Test seasonality features"""

    def test_yearly_seasonality(self):
        """Test yearly seasonality detection"""
        dates = pd.date_range("2018-01-01", periods=365 * 3, freq="D")
        y = 10 + np.sin(2 * np.pi * np.arange(len(dates)) / 365.25) * 5

        df = pd.DataFrame({"ds": dates, "y": y})

        model = Farseer(
            yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False
        )
        model.fit(df)

        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)

        assert "yearly" in forecast.columns
        assert forecast["yearly"].std() > 0  # Should have variation

    def test_weekly_seasonality(self):
        """Test weekly seasonality detection"""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        y = 10 + np.sin(2 * np.pi * np.arange(len(dates)) / 7) * 2

        df = pd.DataFrame({"ds": dates, "y": y})

        model = Farseer(
            yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False
        )
        model.fit(df)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        assert "weekly" in forecast.columns
        assert forecast["weekly"].std() > 0

    def test_additive_seasonality(self):
        """Test additive seasonality mode"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100, freq="D"),
                "y": np.arange(100) * 0.5
                + 10
                + np.sin(np.arange(100) * 2 * np.pi / 7) * 2,
            }
        )

        model = Farseer(
            seasonality_mode="additive",
            weekly_seasonality=True,
            yearly_seasonality=False,
        )
        model.fit(df)

        params = model.params()
        assert params["seasonality_mode"] == "Additive"

    def test_multiplicative_seasonality(self):
        """Test multiplicative seasonality mode"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100, freq="D"),
                "y": (np.arange(100) * 0.5 + 10)
                * (1 + np.sin(np.arange(100) * 2 * np.pi / 7) * 0.1),
            }
        )

        model = Farseer(
            seasonality_mode="multiplicative",
            weekly_seasonality=True,
            yearly_seasonality=False,
        )
        model.fit(df)

        params = model.params()
        assert params["seasonality_mode"] == "Multiplicative"


class TestCustomSeasonality:
    """Test custom seasonality components"""

    def test_add_custom_seasonality(self):
        """Test adding custom seasonality"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=365, freq="D"),
                "y": np.arange(365) * 0.1 + 10,
            }
        )

        model = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
        model.fit(df)

        params = model.params()
        assert len(params["seasonalities"]) == 1
        assert params["seasonalities"][0]["name"] == "monthly"
        assert params["seasonalities"][0]["period"] == 30.5
        assert params["seasonalities"][0]["fourier_order"] == 5

    def test_custom_seasonality_with_prior_scale(self):
        """Test custom seasonality with prior scale"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100, freq="D"),
                "y": np.arange(100) * 0.1 + 10,
            }
        )

        model = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        model.add_seasonality(
            name="custom", period=15.0, fourier_order=3, prior_scale=5.0
        )
        model.fit(df)

        params = model.params()
        assert params["seasonalities"][0]["prior_scale"] == 5.0

    def test_custom_seasonality_mode(self):
        """Test custom seasonality with specific mode"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100, freq="D"),
                "y": np.arange(100) * 0.1 + 10,
            }
        )

        model = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        model.add_seasonality(
            name="custom", period=15.0, fourier_order=3, mode="multiplicative"
        )
        model.fit(df)

        params = model.params()
        assert params["seasonalities"][0]["mode"] == "Multiplicative"

    def test_multiple_custom_seasonalities(self):
        """Test adding multiple custom seasonalities"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=365, freq="D"),
                "y": np.arange(365) * 0.1 + 10,
            }
        )

        model = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        model.add_seasonality("monthly", period=30.5, fourier_order=5)
        model.add_seasonality("quarterly", period=91.25, fourier_order=3)
        model.fit(df)

        params = model.params()
        assert len(params["seasonalities"]) == 2
        seasonality_names = [s["name"] for s in params["seasonalities"]]
        assert "monthly" in seasonality_names
        assert "quarterly" in seasonality_names


class TestHolidays:
    """Test holiday effects"""

    def test_add_holidays(self):
        """Test adding custom holidays"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=365, freq="D"),
                "y": np.arange(365) * 0.1 + 10,
            }
        )

        model = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        model.add_holidays("christmas", ["2020-12-25"], lower_window=0, upper_window=0)
        model.fit(df)

        params = model.params()
        assert len(params["holidays"]) == 1
        assert params["holidays"][0]["name"] == "christmas"
        assert "2020-12-25" in params["holidays"][0]["dates"]

    def test_holidays_with_windows(self):
        """Test holidays with lower and upper windows"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=365, freq="D"),
                "y": np.arange(365) * 0.1 + 10,
            }
        )

        model = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        model.add_holidays("newyear", ["2020-01-01"], lower_window=-2, upper_window=2)
        model.fit(df)

        params = model.params()
        assert params["holidays"][0]["lower_window"] == -2
        assert params["holidays"][0]["upper_window"] == 2

    def test_multiple_holidays(self):
        """Test adding multiple holiday dates"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2019-01-01", periods=730, freq="D"),
                "y": np.arange(730) * 0.1 + 10,
            }
        )

        model = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        model.add_holidays(
            "christmas", ["2019-12-25", "2020-12-25"], lower_window=-1, upper_window=1
        )
        model.fit(df)

        params = model.params()
        assert len(params["holidays"][0]["dates"]) == 2


class TestFutureDataframe:
    """Test future dataframe generation"""

    def test_make_future_dataframe_daily(self):
        """Test creating future dataframe with daily frequency"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100, freq="D"),
                "y": np.arange(100),
            }
        )

        model = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        model.fit(df)

        future = model.make_future_dataframe(periods=30, freq="D")
        assert len(future) == 130

    def test_make_future_dataframe_without_history(self):
        """Test creating future dataframe without history"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100, freq="D"),
                "y": np.arange(100),
            }
        )

        model = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        model.fit(df)

        future = model.make_future_dataframe(
            periods=30, freq="D", include_history=False
        )
        assert len(future) == 30

    def test_make_future_dataframe_hourly(self):
        """Test creating future dataframe with hourly frequency"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100, freq="h"),
                "y": np.arange(100),
            }
        )

        model = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        model.fit(df)

        future = model.make_future_dataframe(periods=24, freq="h")
        assert len(future) == 124

    def test_make_future_dataframe_weekly(self):
        """Test creating future dataframe with weekly frequency"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100, freq="W"),
                "y": np.arange(100),
            }
        )

        model = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        model.fit(df)

        future = model.make_future_dataframe(periods=10, freq="W")
        assert len(future) == 110

    def test_predict_with_future_dataframe_fills_forecasts(self):
        """Test that predict properly fills in all future dates from make_future_dataframe"""
        # Create training data
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100, freq="D"),
                "y": np.arange(100) * 0.5 + 10 + np.random.randn(100) * 0.1,
            }
        )

        model = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        model.fit(df)

        # Create future dataframe with history
        future_with_history = model.make_future_dataframe(
            periods=30, freq="D", include_history=True
        )
        forecast_with_history = model.predict(future_with_history)

        # Convert to pandas if polars
        if hasattr(forecast_with_history, "to_pandas"):
            forecast_with_history = forecast_with_history.to_pandas()

        # Verify all dates are filled
        assert (
            len(forecast_with_history) == 130
        ), f"Expected 130 rows, got {len(forecast_with_history)}"
        assert forecast_with_history["ds"].notna().all(), "Some dates are missing"
        assert (
            forecast_with_history["yhat"].notna().all()
        ), "Some predictions are missing"

        # Verify the last 30 rows are future predictions
        last_training_date = df["ds"].max()
        future_rows = forecast_with_history[
            forecast_with_history["ds"] > last_training_date
        ]
        assert (
            len(future_rows) == 30
        ), f"Expected 30 future rows, got {len(future_rows)}"

        # Verify all future predictions have values
        assert future_rows["yhat"].notna().all(), "Some future predictions are missing"
        assert (
            future_rows["yhat_lower"].notna().all()
        ), "Some future lower bounds are missing"
        assert (
            future_rows["yhat_upper"].notna().all()
        ), "Some future upper bounds are missing"
        assert future_rows["trend"].notna().all(), "Some future trends are missing"

        # Test without history
        future_only = model.make_future_dataframe(
            periods=30, freq="D", include_history=False
        )
        forecast_only = model.predict(future_only)

        # Convert to pandas if polars
        if hasattr(forecast_only, "to_pandas"):
            forecast_only = forecast_only.to_pandas()

        # Verify all future dates are filled
        assert len(forecast_only) == 30, f"Expected 30 rows, got {len(forecast_only)}"
        assert forecast_only["ds"].notna().all(), "Some dates are missing (future only)"
        assert (
            forecast_only["yhat"].notna().all()
        ), "Some predictions are missing (future only)"

        # Verify dates are consecutive and in the future
        assert (
            forecast_only["ds"] > last_training_date
        ).all(), "Some dates are not in the future"
        date_diffs = forecast_only["ds"].diff()[1:]
        expected_diff = pd.Timedelta(days=1)
        assert (date_diffs == expected_diff).all(), "Dates are not consecutive"

    def test_predict_with_future_dataframe_different_frequencies(self):
        """Test that predict works with different frequencies"""
        # Test hourly
        df_hourly = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=168, freq="h"),  # 1 week
                "y": np.sin(np.arange(168) * 2 * np.pi / 24) + 10,
            }
        )

        model_hourly = Farseer(
            yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=True
        )
        model_hourly.fit(df_hourly)

        future_hourly = model_hourly.make_future_dataframe(periods=24, freq="h")
        forecast_hourly = model_hourly.predict(future_hourly)

        if hasattr(forecast_hourly, "to_pandas"):
            forecast_hourly = forecast_hourly.to_pandas()

        assert len(forecast_hourly) == 192, "Hourly forecast length incorrect"
        assert (
            forecast_hourly["yhat"].notna().all()
        ), "Hourly predictions have missing values"

        # Test monthly
        df_monthly = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=24, freq="MS"),
                "y": np.arange(24) * 2 + 100,
            }
        )

        model_monthly = Farseer(yearly_seasonality=True, weekly_seasonality=False)
        model_monthly.fit(df_monthly)

        future_monthly = model_monthly.make_future_dataframe(periods=12, freq="MS")
        forecast_monthly = model_monthly.predict(future_monthly)

        if hasattr(forecast_monthly, "to_pandas"):
            forecast_monthly = forecast_monthly.to_pandas()

        assert len(forecast_monthly) == 36, "Monthly forecast length incorrect"
        assert (
            forecast_monthly["yhat"].notna().all()
        ), "Monthly predictions have missing values"

        # Verify future months are filled
        last_training_date = df_monthly["ds"].max()
        future_monthly_rows = forecast_monthly[
            forecast_monthly["ds"] > last_training_date
        ]
        assert len(future_monthly_rows) == 12, "Expected 12 future months"


class TestUncertaintyIntervals:
    """Test uncertainty interval configuration"""

    def test_default_interval_width(self):
        """Test default 80% uncertainty interval"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100, freq="D"),
                "y": np.arange(100) * 0.5 + 10 + np.random.randn(100) * 2,
            }
        )

        model = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        model.fit(df)

        params = model.params()
        assert params["interval_width"] == 0.8

    def test_custom_interval_width(self):
        """Test custom uncertainty interval width"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100, freq="D"),
                "y": np.arange(100) * 0.5 + 10,
            }
        )

        model = Farseer(
            interval_width=0.95, yearly_seasonality=False, weekly_seasonality=False
        )
        model.fit(df)

        params = model.params()
        assert params["interval_width"] == 0.95

    def test_uncertainty_bounds(self):
        """Test that uncertainty bounds are reasonable"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100, freq="D"),
                "y": np.arange(100) * 0.5 + 10 + np.random.randn(100) * 1,
            }
        )

        model = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        model.fit(df)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Lower bound should be below yhat, upper above
        assert all(forecast["yhat_lower"] <= forecast["yhat"])
        assert all(forecast["yhat_upper"] >= forecast["yhat"])


class TestModelPersistence:
    """Test model serialization and loading"""

    def test_to_json(self):
        """Test model serialization to JSON"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=50, freq="D"),
                "y": np.arange(50) * 0.5 + 10,
            }
        )

        model = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        model.fit(df)

        json_str = model.to_json()
        assert json_str is not None
        assert len(json_str) > 0
        assert "fitted" in json_str

    def test_from_json(self):
        """Test model deserialization from JSON"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=50, freq="D"),
                "y": np.arange(50) * 0.5 + 10,
            }
        )

        model1 = Farseer(
            n_changepoints=15, yearly_seasonality=False, weekly_seasonality=False
        )
        model1.fit(df)

        json_str = model1.to_json()
        model2 = Farseer.from_json(json_str)

        params2 = model2.params()
        assert params2["n_changepoints"] == 15
        assert params2["fitted"]

    def test_save_and_load(self):
        """Test saving and loading model from file"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=50, freq="D"),
                "y": np.arange(50) * 0.5 + 10,
            }
        )

        model1 = Farseer(
            n_changepoints=20, yearly_seasonality=False, weekly_seasonality=False
        )
        model1.fit(df)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            model1.save(temp_path)
            model2 = Farseer.load(temp_path)

            params2 = model2.params()
            assert params2["n_changepoints"] == 20
            assert params2["fitted"]
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_serialization_preserves_predictions(self):
        """Test that loaded model produces same predictions"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=50, freq="D"),
                "y": np.arange(50) * 0.5 + 10,
            }
        )

        model1 = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        model1.fit(df)

        future = pd.DataFrame({"ds": pd.date_range("2020-02-20", periods=10, freq="D")})
        forecast1 = model1.predict(future)

        # Serialize and deserialize
        json_str = model1.to_json()
        model2 = Farseer.from_json(json_str)
        forecast2 = model2.predict(future)

        # Predictions should be identical (or very close)
        # Convert to numpy (forecasts are polars)
        if hasattr(forecast1, "to_pandas"):
            yhat1 = forecast1["yhat"].to_numpy()
            yhat2 = forecast2["yhat"].to_numpy()
        else:
            yhat1 = forecast1["yhat"].values
            yhat2 = forecast2["yhat"].values

        np.testing.assert_array_almost_equal(yhat1, yhat2, decimal=10)


class TestMethodChaining:
    """Test method chaining interface"""

    def test_chaining_add_seasonality(self):
        """Test method chaining with add_seasonality"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100, freq="D"),
                "y": np.arange(100) * 0.1 + 10,
            }
        )

        model = (
            Farseer(yearly_seasonality=False, weekly_seasonality=False)
            .add_seasonality("monthly", period=30.5, fourier_order=5)
            .add_seasonality("quarterly", period=91.25, fourier_order=3)
        )

        model.fit(df)
        params = model.params()
        assert len(params["seasonalities"]) == 2

    def test_chaining_add_holidays(self):
        """Test method chaining with add_holidays"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=365, freq="D"),
                "y": np.arange(365) * 0.1 + 10,
            }
        )

        model = (
            Farseer(yearly_seasonality=False, weekly_seasonality=False)
            .add_holidays("christmas", ["2020-12-25"])
            .add_holidays("newyear", ["2020-01-01"])
        )

        model.fit(df)
        params = model.params()
        assert len(params["holidays"]) == 2


class TestErrorHandling:
    """Test error handling and validation"""

    def test_missing_columns(self):
        """Test error when required columns are missing"""
        df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=10)})

        model = Farseer()
        with pytest.raises(ValueError, match="must have 'ds' and 'y' columns"):
            model.fit(df)

    def test_predict_before_fit(self):
        """Test error when predicting before fitting"""
        df = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=10, freq="D")})

        model = Farseer()
        with pytest.raises(Exception):  # Should raise RuntimeError from Rust
            model.predict(df)

    def test_make_future_before_fit(self):
        """Test error when making future dataframe before fitting"""
        model = Farseer()
        with pytest.raises(Exception):  # Should raise RuntimeError from Rust
            model.make_future_dataframe(periods=10)

    def test_invalid_growth_type(self):
        """Test error with invalid growth type"""
        with pytest.raises(Exception):  # Should raise ValueError from Rust
            Farseer(growth="invalid")

    def test_invalid_seasonality_mode(self):
        """Test error with invalid seasonality mode"""
        with pytest.raises(Exception):  # Should raise ValueError from Rust
            Farseer(seasonality_mode="invalid")


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_single_data_point(self):
        """Test with single data point - should raise error (Prophet compatibility)"""
        df = pd.DataFrame(
            {"ds": pd.date_range("2020-01-01", periods=1, freq="D"), "y": [10]}
        )

        model = Farseer(
            n_changepoints=0, yearly_seasonality=False, weekly_seasonality=False
        )
        # Should raise ValueError for insufficient data
        with pytest.raises(ValueError, match="less than 2 non-NaN rows"):
            model.fit(df)

    def test_very_short_series(self):
        """Test with very short time series"""
        # Use 50 points minimum for stable Stan optimization
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=50, freq="D"),
                "y": [
                    10 + i * 0.2 + np.sin(i / 3) * 0.5 for i in range(50)
                ],  # Add slight variation
            }
        )

        model = Farseer(
            n_changepoints=1,  # Use minimal changepoints for short series
            yearly_seasonality=False,
            weekly_seasonality=False,
        )
        model.fit(df)

        future = model.make_future_dataframe(periods=5)
        forecast = model.predict(future)
        assert len(forecast) == 55  # 50 + 5

    def test_constant_values(self):
        """Test with constant time series"""
        df = pd.DataFrame(
            {"ds": pd.date_range("2020-01-01", periods=50, freq="D"), "y": [10.0] * 50}
        )

        model = Farseer(yearly_seasonality=False, weekly_seasonality=False)
        model.fit(df)

        future = model.make_future_dataframe(periods=10)
        forecast = model.predict(future)

        # Predictions should be close to constant value
        assert all(abs(forecast["yhat"] - 10.0) < 5.0)


class TestRealWorldScenarios:
    """Test real-world use case scenarios"""

    def test_daily_sales_forecast(self):
        """Test daily sales forecasting scenario"""
        # Simulate daily sales with trend and weekly seasonality
        n_days = 365
        dates = pd.date_range("2020-01-01", periods=n_days, freq="D")

        # Trend + weekly pattern + noise
        trend = np.arange(n_days) * 0.1 + 100
        weekly = np.sin(2 * np.pi * np.arange(n_days) / 7) * 10
        noise = np.random.randn(n_days) * 5
        y = trend + weekly + noise

        df = pd.DataFrame({"ds": dates, "y": y})

        model = Farseer(
            yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False
        )
        model.fit(df)

        # Forecast next month
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        assert len(forecast) == n_days + 30
        assert "weekly" in forecast.columns

    def test_seasonal_product_demand(self):
        """Test seasonal product demand forecasting"""
        # Simulate 3 years of data with strong yearly seasonality
        n_days = 365 * 3
        dates = pd.date_range("2018-01-01", periods=n_days, freq="D")

        # Trend + yearly seasonality + noise
        trend = np.arange(n_days) * 0.05 + 50
        yearly = np.sin(2 * np.pi * np.arange(n_days) / 365.25) * 20
        noise = np.random.randn(n_days) * 3
        y = trend + yearly + noise

        df = pd.DataFrame({"ds": dates, "y": y})

        model = Farseer(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.01,
        )
        model.fit(df)

        # Forecast next year
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)

        assert "yearly" in forecast.columns
        assert forecast["yearly"].std() > 0

    def test_website_traffic_with_holidays(self):
        """Test website traffic forecasting with holiday effects"""
        n_days = 365
        dates = pd.date_range("2020-01-01", periods=n_days, freq="D")

        # Base traffic with weekly pattern
        base = 1000 + np.arange(n_days) * 0.5
        weekly = np.sin(2 * np.pi * np.arange(n_days) / 7) * 100
        noise = np.random.randn(n_days) * 50
        y = base + weekly + noise

        df = pd.DataFrame({"ds": dates, "y": y})

        model = Farseer(yearly_seasonality=False, weekly_seasonality=True)

        # Add holiday effects
        model.add_holidays(
            "blackfriday",
            ["2020-11-27"],
            lower_window=-2,
            upper_window=2,
            prior_scale=20.0,
        )

        model.fit(df)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        assert len(forecast) == n_days + 30

    def test_hourly_energy_consumption(self):
        """Test hourly energy consumption forecasting"""
        n_hours = 24 * 30  # 30 days
        dates = pd.date_range("2020-01-01", periods=n_hours, freq="h")

        # Daily pattern + trend
        hour_of_day = np.arange(n_hours) % 24
        daily_pattern = np.sin(2 * np.pi * hour_of_day / 24) * 50 + 100
        trend = np.arange(n_hours) * 0.01
        noise = np.random.randn(n_hours) * 10
        y = daily_pattern + trend + noise

        df = pd.DataFrame({"ds": dates, "y": y})

        model = Farseer(
            yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=True
        )
        model.fit(df)

        # Forecast next 24 hours
        future = model.make_future_dataframe(periods=24, freq="h")
        forecast = model.predict(future)

        assert len(forecast) == n_hours + 24


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
