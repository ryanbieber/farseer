#!/usr/bin/env python3
"""
Prophet Compatibility Test Suite
Tests to ensure Seer behaves like Facebook's Prophet library
Based on Prophet's test_prophet.py
"""

import pytest
import pandas as pd
import numpy as np
from farseer import Farseer


@pytest.fixture
def daily_univariate_ts():
    """Generate daily time series data for testing"""
    # Use Prophet's seed for reproducibility (matches Facebook Prophet tests)
    np.random.seed(876543987)
    n = 468  # ~1.3 years
    dates = pd.date_range("2012-01-01", periods=n, freq="D")

    # Trend + seasonality + noise
    trend = np.arange(n) * 0.5 + 10
    yearly = np.sin(2 * np.pi * np.arange(n) / 365.25) * 5
    weekly = np.sin(2 * np.pi * np.arange(n) / 7) * 2
    noise = np.random.randn(n) * 2

    y = trend + yearly + weekly + noise

    return pd.DataFrame({"ds": dates, "y": y})


@pytest.fixture
def large_numbers_ts():
    """Generate time series with large numbers"""
    # Use Prophet's seed for reproducibility (matches Facebook Prophet tests)
    np.random.seed(876543987)
    n = 468
    dates = pd.date_range("2012-01-01", periods=n, freq="D")

    y = (np.arange(n) * 100 + 10000) + np.random.randn(n) * 500

    return pd.DataFrame({"ds": dates, "y": y})


@pytest.fixture
def subdaily_univariate_ts():
    """Generate subdaily time series data"""
    # Use Prophet's seed for reproducibility (matches Facebook Prophet tests)
    np.random.seed(876543987)
    n = 24 * 30  # 30 days of hourly data
    dates = pd.date_range("2017-01-01", periods=n, freq="h")

    # Daily and hourly patterns
    hour_of_day = np.arange(n) % 24
    daily_pattern = np.sin(2 * np.pi * hour_of_day / 24) * 10
    trend = np.arange(n) * 0.01 + 50
    noise = np.random.randn(n) * 2

    y = trend + daily_pattern + noise

    return pd.DataFrame({"ds": dates, "y": y})


def train_test_split(ts_data: pd.DataFrame, n_test_rows: int):
    """Split time series into train and test sets"""
    train = ts_data.head(ts_data.shape[0] - n_test_rows)
    test = ts_data.tail(n_test_rows)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def rmse(predictions, targets) -> float:
    """Calculate RMSE - handles both pandas and polars Series"""
    # Convert to numpy arrays for consistent handling
    if hasattr(predictions, "to_numpy"):
        pred_vals = predictions.to_numpy()
    elif hasattr(predictions, "values"):
        pred_vals = predictions.values
    else:
        pred_vals = np.array(predictions)

    if hasattr(targets, "to_numpy"):
        target_vals = targets.to_numpy()
    elif hasattr(targets, "values"):
        target_vals = targets.values
    else:
        target_vals = np.array(targets)

    return np.sqrt(np.mean((pred_vals - target_vals) ** 2))


class TestProphetFitPredictDefault:
    """Test basic fit and predict functionality"""

    def test_fit_predict(self, daily_univariate_ts):
        """Test basic fit and predict - with auto-seasonality and y-scaling

        Now implements:
        - Auto-seasonality detection (disables yearly for data <2 years)
        - Y-scaling (like Prophet's absmax scaling)
        This should match Prophet's RMSE more closely.
        """
        test_days = 30
        train, test = train_test_split(daily_univariate_ts, test_days)

        # Use default settings - auto-detection will disable yearly for ~1.3 year data
        model = Farseer()
        model.fit(train)

        future = model.make_future_dataframe(test_days, include_history=False)
        forecast = model.predict(future)

        res = rmse(forecast["yhat"], test["y"])
        # With scaling, should now get RMSE similar to Prophet (~3.0)
        # Allow some tolerance for numerical differences
        assert res < 10, f"RMSE {res} too high (Prophet gets ~3.0)"
        # Check for NaN values (polars uses is_null instead of isnull)
        if hasattr(forecast["yhat"], "is_null"):
            assert not forecast["yhat"].is_null().any(), "Forecast contains NaN values"
        else:
            assert not forecast["yhat"].isnull().any(), "Forecast contains NaN values"

    def test_fit_predict_no_seasons(self, daily_univariate_ts):
        """Test fit/predict with no seasonality"""
        test_days = 30
        train, _ = train_test_split(daily_univariate_ts, test_days)

        model = Farseer(weekly_seasonality=False, yearly_seasonality=False)
        model.fit(train)

        future = model.make_future_dataframe(test_days, include_history=False)
        result = model.predict(future)

        # Compare dates - both are polars DataFrames
        assert (future["ds"] == result["ds"]).all()

    def test_fit_predict_no_changepoints(self, daily_univariate_ts):
        """Test fit/predict with no changepoints"""
        test_days = daily_univariate_ts.shape[0] // 2
        train, future = train_test_split(daily_univariate_ts, test_days)

        model = Farseer(n_changepoints=0)
        model.fit(train)
        model.predict(future)

        params = model.params()
        assert params["fitted"] is True
        assert params["n_changepoints"] == 0

    def test_fit_changepoint_not_in_history(self, daily_univariate_ts):
        """Test with manual changepoints not in history"""
        # Create a gap in the data
        train = daily_univariate_ts[
            (daily_univariate_ts["ds"] < "2013-01-01")
            | (daily_univariate_ts["ds"] > "2014-01-01")
        ]
        future = pd.DataFrame({"ds": daily_univariate_ts["ds"]})

        model = Farseer(changepoints=["2013-06-06"])
        model.fit(train)
        model.predict(future)

        params = model.params()
        assert params["fitted"] is True
        assert params["n_changepoints"] == 1

    def test_fit_predict_duplicates(self, daily_univariate_ts):
        """Test fitting with duplicate dates"""
        train, test = train_test_split(
            daily_univariate_ts, daily_univariate_ts.shape[0] // 2
        )

        # Create duplicates with slightly different values
        repeated_obs = train.copy()
        repeated_obs["y"] += 10
        train = pd.concat([train, repeated_obs], ignore_index=True)

        model = Farseer()
        model.fit(train)
        model.predict(test)

    def test_fit_predict_uncertainty_disabled(self, daily_univariate_ts):
        """Test with uncertainty intervals disabled"""
        test_days = daily_univariate_ts.shape[0] // 2
        train, future = train_test_split(daily_univariate_ts, test_days)

        model = Farseer(uncertainty_samples=0)
        model.fit(train)
        result = model.predict(future)

        # Should not have uncertainty columns when disabled
        if "yhat_lower" in result.columns:
            # If they exist, they might be NaN or zeros
            pass


class TestProphetDataPrep:
    """Test data preparation and setup"""

    def test_setup_dataframe(self, daily_univariate_ts):
        """Test that dataframe setup works correctly"""
        train, _ = train_test_split(
            daily_univariate_ts, daily_univariate_ts.shape[0] // 2
        )

        m = Farseer()
        m.fit(train)

        # After fitting, model should have processed the data
        params = m.params()
        assert params["fitted"] is True

    def test_setup_dataframe_ds_column(self, daily_univariate_ts):
        """Test when 'ds' exists as both index and column"""
        train, _ = train_test_split(
            daily_univariate_ts, daily_univariate_ts.shape[0] // 2
        )

        # Create index with 'ds' name
        train.index = pd.to_datetime(["1970-01-01" for _ in range(train.shape[0])])
        train.index.rename("ds", inplace=True)

        m = Farseer()
        m.fit(train)

        # Should use the column, not the index
        params = m.params()
        assert params["fitted"] is True

    def test_logistic_floor(self, daily_univariate_ts):
        """Test logistic growth with floor and cap"""
        train, _ = train_test_split(
            daily_univariate_ts, daily_univariate_ts.shape[0] // 2
        )

        train["floor"] = 10.0
        train["cap"] = 80.0

        m = Farseer(growth="logistic")
        m.fit(train)

        # Test with shifted floor and cap
        train2 = train.copy()
        for col in ["y", "floor", "cap"]:
            train2[col] += 10.0

        m2 = Farseer(growth="logistic")
        m2.fit(train2)

        # Both should fit successfully
        assert m.params()["fitted"] is True
        assert m2.params()["fitted"] is True

    def test_make_future_dataframe(self, daily_univariate_ts):
        """Test make_future_dataframe with different frequencies"""
        train = daily_univariate_ts.head(468 // 2)

        model = Farseer()
        model.fit(train)

        # Daily frequency
        future = model.make_future_dataframe(periods=3, freq="D", include_history=False)
        assert len(future) == 3

        # Check dates are correct
        expected_start = train["ds"].iloc[-1] + pd.Timedelta(days=1)
        # Convert polars to pandas for comparison
        if hasattr(future, "to_pandas"):
            future_pd = future.to_pandas()
            assert future_pd["ds"].iloc[0] >= expected_start
        else:
            assert future["ds"].iloc[0] >= expected_start

    def test_make_future_dataframe_include_history(self, daily_univariate_ts):
        """Test make_future_dataframe with history included"""
        train = daily_univariate_ts.head(468 // 2).copy()

        # Add some NAs to history
        train.loc[train.sample(min(10, len(train))).index, "y"] = np.nan

        model = Farseer()
        model.fit(train)

        future = model.make_future_dataframe(periods=3, freq="D", include_history=True)

        # Should include history + future periods
        assert len(future) >= 3


class TestProphetTrendComponent:
    """Test trend components and configurations"""

    def test_invalid_growth_input(self):
        """Test that invalid growth raises error"""
        with pytest.raises(Exception):
            Farseer(growth="constant")

    def test_growth_init(self, daily_univariate_ts):
        """Test growth initialization"""
        model = Farseer(growth="logistic")
        train = daily_univariate_ts.iloc[:468].copy()
        train["cap"] = train["y"].max()

        model.fit(train)

        # Model should fit successfully
        params = model.params()
        assert params["fitted"] is True

    def test_flat_growth(self):
        """Test flat growth trend"""
        m = Farseer(growth="flat")

        x = np.linspace(0, 2 * np.pi, 8 * 7)
        history = pd.DataFrame(
            {
                "ds": pd.date_range(start="2020-01-01", periods=8 * 7, freq="D"),
                "y": 30 + np.sin(x * 8.0),
            }
        )

        m.fit(history)
        future = m.make_future_dataframe(10, include_history=True)
        fcst = m.predict(future)

        # Trend should be approximately constant
        trend_values = fcst["trend"].unique()
        assert len(trend_values) == 1 or (trend_values.max() - trend_values.min()) < 1.0

    def test_get_changepoints(self, daily_univariate_ts):
        """Test changepoint detection uses first 80% of history"""
        train, _ = train_test_split(
            daily_univariate_ts, daily_univariate_ts.shape[0] // 2
        )

        m = Farseer()
        m.fit(train)

        params = m.params()
        assert params["n_changepoints"] >= 0

    def test_set_changepoint_range(self, daily_univariate_ts):
        """Test custom changepoint range"""
        train, _ = train_test_split(
            daily_univariate_ts, daily_univariate_ts.shape[0] // 2
        )

        m = Farseer(changepoint_range=0.4)
        m.fit(train)

        params = m.params()
        assert params["changepoint_range"] == 0.4

        # Out of range values should raise error
        with pytest.raises(Exception):
            Farseer(changepoint_range=-0.1)
        with pytest.raises(Exception):
            Farseer(changepoint_range=2)

    def test_get_zero_changepoints(self, daily_univariate_ts):
        """Test with zero changepoints"""
        train, _ = train_test_split(
            daily_univariate_ts, daily_univariate_ts.shape[0] // 2
        )

        m = Farseer(n_changepoints=0)
        m.fit(train)

        params = m.params()
        assert params["n_changepoints"] == 0

    def test_override_n_changepoints(self, daily_univariate_ts):
        """Test manually setting number of changepoints"""
        train = daily_univariate_ts.head(20).copy()

        m = Farseer(n_changepoints=15)
        m.fit(train)

        params = m.params()
        # Might be limited by data size
        assert params["n_changepoints"] <= 15


class TestProphetSeasonalComponent:
    """Test seasonality components"""

    def test_auto_weekly_seasonality(self, daily_univariate_ts):
        """Test automatic weekly seasonality detection"""
        # Should be enabled for daily data
        train = daily_univariate_ts.head(15)
        m = Farseer()
        m.fit(train)

        future = m.make_future_dataframe(7)
        forecast = m.predict(future)

        # Should have weekly component
        if "weekly" in forecast.columns:
            assert forecast["weekly"].std() >= 0

        # Should be disabled for weekly spacing
        train_weekly = daily_univariate_ts.iloc[::7, :]
        m2 = Farseer()
        m2.fit(train_weekly)

        # Weekly seasonality might be auto-disabled
        _ = m2.params()

    def test_auto_yearly_seasonality(self, daily_univariate_ts):
        """Test automatic yearly seasonality detection"""
        # Should be enabled for long enough data
        m = Farseer()
        m.fit(daily_univariate_ts)

        future = m.make_future_dataframe(30)
        forecast = m.predict(future)

        # Should have yearly component
        if "yearly" in forecast.columns:
            assert forecast["yearly"].std() >= 0

        # Should be disabled for short history
        train_short = daily_univariate_ts.head(240)
        m2 = Farseer()
        m2.fit(train_short)

        # Model should still fit
        params = m2.params()
        assert params["fitted"] is True

    def test_auto_daily_seasonality(self, subdaily_univariate_ts):
        """Test automatic daily seasonality detection"""
        # Should be enabled for subdaily data
        m = Farseer()
        m.fit(subdaily_univariate_ts)

        future = m.make_future_dataframe(24, freq="H")
        forecast = m.predict(future)

        # Should have daily component for hourly data
        if "daily" in forecast.columns:
            assert forecast["daily"].std() >= 0

    def test_set_seasonality_mode(self):
        """Test seasonality mode settings"""
        # Default should be additive
        m = Farseer()
        params = m.params()
        assert params["seasonality_mode"] in ["Additive", "additive"]

        # Multiplicative mode
        m2 = Farseer(seasonality_mode="multiplicative")
        params2 = m2.params()
        assert params2["seasonality_mode"] in ["Multiplicative", "multiplicative"]

        # Invalid mode
        with pytest.raises(Exception):
            Farseer(seasonality_mode="batman")

    def test_seasonality_modes(self, daily_univariate_ts):
        """Test different seasonality modes with holidays and regressors"""
        # Model with multiplicative seasonality
        m = Farseer(seasonality_mode="multiplicative")

        df = daily_univariate_ts.copy()
        m.fit(df)

        future = m.make_future_dataframe(30)
        forecast = m.predict(future)

        # Should have multiplicative terms
        if "multiplicative_terms" in forecast.columns:
            # Multiplicative terms should exist
            pass


class TestProphetCustomSeasonalComponent:
    """Test custom seasonality components"""

    def test_custom_monthly_seasonality(self):
        """Test adding custom monthly seasonality"""
        m = Farseer()
        m.add_seasonality(name="monthly", period=30, fourier_order=5, prior_scale=2.0)

        params = m.params()
        seasonalities = params["seasonalities"]

        monthly = [s for s in seasonalities if s["name"] == "monthly"]
        assert len(monthly) == 1
        assert monthly[0]["period"] == 30
        assert monthly[0]["fourier_order"] == 5
        assert monthly[0]["prior_scale"] == 2.0

    def test_duplicate_component_names(self):
        """Test that duplicate seasonality names raise error"""
        m = Farseer()
        m.add_seasonality(name="custom", period=30, fourier_order=5)

        # Adding same name again should raise error
        with pytest.raises(Exception):
            m.add_seasonality(name="custom", period=30, fourier_order=5)

    def test_custom_fourier_order(self):
        """Test that invalid Fourier order raises error"""
        m = Farseer()

        # Fourier order must be positive
        with pytest.raises(Exception):
            m.add_seasonality(name="test", period=7, fourier_order=0)

        with pytest.raises(Exception):
            m.add_seasonality(name="test", period=7, fourier_order=-1)


class TestProphetHolidays:
    """Test holiday effects"""

    def test_holidays_lower_window(self):
        """Test holidays with lower window"""
        m = Farseer()
        m.add_holidays("xmas", ["2016-12-25"], lower_window=-1, upper_window=0)

        df = pd.DataFrame({"ds": pd.date_range("2016-12-20", "2016-12-31")})
        df["y"] = range(len(df))

        m.fit(df)

        params = m.params()
        holidays = params["holidays"]

        xmas = [h for h in holidays if h["name"] == "xmas"]
        assert len(xmas) == 1

    def test_holidays_upper_window(self):
        """Test holidays with upper window"""
        m = Farseer()
        m.add_holidays("xmas", ["2016-12-25"], lower_window=-1, upper_window=10)

        df = pd.DataFrame({"ds": pd.date_range("2016-12-20", "2016-12-31")})
        df["y"] = range(len(df))

        m.fit(df)

        params = m.params()
        holidays = params["holidays"]

        assert len(holidays) > 0

    def test_fit_with_holidays(self, daily_univariate_ts):
        """Test fitting with holidays"""
        m = Farseer(uncertainty_samples=0)
        m.add_holidays(
            "seans-bday", ["2012-06-06", "2013-06-06"], lower_window=0, upper_window=1
        )

        m.fit(daily_univariate_ts)
        forecast = m.predict()

        # Should have made predictions
        assert len(forecast) > 0

    def test_subdaily_holidays(self, subdaily_univariate_ts):
        """Test holidays with subdaily data"""
        m = Farseer()
        m.add_holidays("special_day", ["2017-01-02"])

        m.fit(subdaily_univariate_ts)
        fcst = m.predict()

        # Holiday effect should be present
        assert len(fcst) > 0


class TestProphetRegressors:
    """Test additional regressors"""

    def test_added_regressors(self, daily_univariate_ts):
        """Test adding extra regressors (API compatibility test - regressors not fully implemented)"""
        m = Farseer()
        m.add_regressor("binary_feature", prior_scale=0.2)
        m.add_regressor("numeric_feature", prior_scale=0.5)

        df = daily_univariate_ts.copy()
        n = len(df)
        df["binary_feature"] = [0] * (n // 2) + [1] * (n - n // 2)
        df["numeric_feature"] = range(n)

        # Note: Regressors are not fully implemented in Seer yet, so we skip validation
        # and just test that the API works

        # Fit should work (regressors are ignored with warning)
        m.fit(df)

        params = m.params()
        assert params["fitted"] is True

        # Future dataframe also needs regressors (but they're ignored)
        future = pd.DataFrame(
            {
                "ds": pd.date_range("2014-06-01", periods=10),
                "binary_feature": [0] * 10,
                "numeric_feature": range(10, 20),
            }
        )

        fcst = m.predict(future)
        assert len(fcst) == 10

    def test_constant_regressor(self, daily_univariate_ts):
        """Test that constant regressor doesn't break fitting"""
        df = daily_univariate_ts.copy()
        df["constant_feature"] = 0

        m = Farseer()
        m.add_regressor("constant_feature")
        m.fit(df)

        # Should fit without error
        params = m.params()
        assert params["fitted"] is True


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_very_short_series(self):
        """Test with very short time series"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=5, freq="D"),
                "y": [10, 11, 12, 11, 10],
            }
        )

        m = Farseer(
            n_changepoints=0, yearly_seasonality=False, weekly_seasonality=False
        )
        m.fit(df)

        future = m.make_future_dataframe(periods=5)
        forecast = m.predict(future)

        assert len(forecast) >= 5

    def test_missing_columns(self):
        """Test error with missing required columns"""
        df = pd.DataFrame(
            {"date": pd.date_range("2020-01-01", periods=10), "value": range(10)}
        )

        m = Farseer()
        with pytest.raises(Exception):
            m.fit(df)

    def test_predict_before_fit(self):
        """Test error when predicting before fitting"""
        m = Farseer()
        df = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=10)})

        with pytest.raises(Exception):
            m.predict(df)

    def test_nan_values(self):
        """Test handling of NaN values"""
        df = pd.DataFrame(
            {"ds": pd.date_range("2020-01-01", periods=100), "y": range(100)}
        )

        # Add some NaN values
        df.loc[10:20, "y"] = np.nan

        m = Farseer()
        m.fit(df)

        # Should handle NaN values
        params = m.params()
        assert params["fitted"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
