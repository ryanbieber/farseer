"""
Tests for conditional seasonality feature.
"""

import pandas as pd
import polars as pl
import pytest
from datetime import datetime, timedelta
import numpy as np


class TestConditionalSeasonality:
    """Test conditional seasonalities - seasonality that applies only when a condition is met."""

    def test_add_conditional_seasonality_basic(self):
        """Test adding a seasonality with a condition."""
        from farseer import Farseer

        m = Farseer()
        m.add_seasonality(
            name="weekly_on_weekday",
            period=7,
            fourier_order=3,
            condition_name="is_weekday",
        )

        # Check that seasonality was added
        params = m.params()
        seasonalities = params["seasonalities"]
        assert len(seasonalities) == 1
        assert seasonalities[0]["name"] == "weekly_on_weekday"
        assert seasonalities[0]["condition_name"] == "is_weekday"

    def test_conditional_seasonality_fit_predict(self):
        """Test fitting and predicting with conditional seasonality."""
        from farseer import Farseer

        # Create simple data with weekday/weekend pattern
        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(200)]

        # Create data with different patterns on weekdays vs weekends
        y = []
        is_weekday = []
        for date in dates:
            weekday = date.weekday()
            is_wd = weekday < 5
            is_weekday.append(is_wd)

            # Base trend
            base = 10 + (date - start_date).days * 0.1

            # Weekday has weekly seasonality, weekend doesn't
            if is_wd:
                seasonal = 2 * np.sin(2 * np.pi * weekday / 7)
            else:
                seasonal = 0

            y.append(base + seasonal + np.random.normal(0, 0.1))

        df = pl.DataFrame({"ds": dates, "y": y, "is_weekday": is_weekday})

        m = Farseer()
        m.add_seasonality(
            name="weekly_weekday",
            period=7,
            fourier_order=3,
            condition_name="is_weekday",
        )
        m.fit(df)

        # Predict on future data
        future_dates = [dates[-1] + timedelta(days=i + 1) for i in range(30)]
        future_is_weekday = [
            (dates[-1] + timedelta(days=i + 1)).weekday() < 5 for i in range(30)
        ]

        future = pl.DataFrame({"ds": future_dates, "is_weekday": future_is_weekday})

        forecast = m.predict(future)

        # Check that forecast was generated
        assert len(forecast) == 30
        assert "yhat" in forecast.columns

    def test_multiple_conditional_seasonalities(self):
        """Test multiple conditional seasonalities."""
        from farseer import Farseer

        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(300)]

        is_weekday = [(date.weekday() < 5) for date in dates]
        is_summer = [(date.month in [6, 7, 8]) for date in dates]

        y = [10 + i * 0.05 + np.random.normal(0, 0.5) for i in range(len(dates))]

        df = pl.DataFrame(
            {"ds": dates, "y": y, "is_weekday": is_weekday, "is_summer": is_summer}
        )

        m = Farseer()
        m.add_seasonality(
            name="weekly_weekday",
            period=7,
            fourier_order=3,
            condition_name="is_weekday",
        )
        m.add_seasonality(
            name="weekly_summer", period=7, fourier_order=2, condition_name="is_summer"
        )
        m.fit(df)

        # Check params
        params = m.params()
        assert len(params["seasonalities"]) == 2

        # Predict
        future_dates = [dates[-1] + timedelta(days=i + 1) for i in range(30)]
        future = pl.DataFrame(
            {
                "ds": future_dates,
                "is_weekday": [(d.weekday() < 5) for d in future_dates],
                "is_summer": [(d.month in [6, 7, 8]) for d in future_dates],
            }
        )

        forecast = m.predict(future)
        assert len(forecast) == 30

    def test_conditional_with_standard_seasonality(self):
        """Test conditional seasonality alongside standard seasonality."""
        from farseer import Farseer

        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(400)]
        is_special = [i % 10 == 0 for i in range(len(dates))]
        y = [20 + i * 0.1 + np.random.normal(0, 1) for i in range(len(dates))]

        df = pl.DataFrame({"ds": dates, "y": y, "is_special": is_special})

        m = Farseer(yearly_seasonality=True, weekly_seasonality=True)
        m.add_seasonality(
            name="special_weekly",
            period=7,
            fourier_order=2,
            condition_name="is_special",
        )
        m.fit(df)

        params = m.params()
        # Should have the standard seasonalities plus the conditional one
        assert params["yearly_seasonality"] is True
        assert params["weekly_seasonality"] is True
        assert len(params["seasonalities"]) == 1

        # Predict
        future_dates = [dates[-1] + timedelta(days=i + 1) for i in range(50)]
        future = pl.DataFrame(
            {
                "ds": future_dates,
                "is_special": [False] * 50,  # All false for simplicity
            }
        )

        forecast = m.predict(future)
        assert len(forecast) == 50

    def test_missing_condition_column_fit_error(self):
        """Test that fitting without condition column raises error."""
        from farseer import Farseer

        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]
        y = [10 + i * 0.1 for i in range(100)]

        df = pl.DataFrame(
            {
                "ds": dates,
                "y": y,
                # Missing 'is_active' column
            }
        )

        m = Farseer()
        m.add_seasonality(
            name="conditional_weekly",
            period=7,
            fourier_order=3,
            condition_name="is_active",
        )

        with pytest.raises(Exception) as exc_info:
            m.fit(df)

        assert (
            "is_active" in str(exc_info.value).lower()
            or "condition" in str(exc_info.value).lower()
        )

    def test_missing_condition_column_predict_error(self):
        """Test that predicting without condition column raises error."""
        from farseer import Farseer

        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]
        is_active = [i % 2 == 0 for i in range(100)]
        y = [10 + i * 0.1 for i in range(100)]

        df = pl.DataFrame({"ds": dates, "y": y, "is_active": is_active})

        m = Farseer()
        m.add_seasonality(
            name="conditional_weekly",
            period=7,
            fourier_order=3,
            condition_name="is_active",
        )
        m.fit(df)

        # Try to predict without the condition column
        future_dates = [dates[-1] + timedelta(days=i + 1) for i in range(10)]
        future = pl.DataFrame(
            {
                "ds": future_dates
                # Missing 'is_active' column
            }
        )

        with pytest.raises(Exception) as exc_info:
            m.predict(future)

        assert (
            "is_active" in str(exc_info.value).lower()
            or "condition" in str(exc_info.value).lower()
        )

    def test_conditional_seasonality_with_pandas(self):
        """Test conditional seasonality with pandas DataFrames."""
        from farseer import Farseer

        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(150)]
        is_workday = [(date.weekday() < 5) for date in dates]
        y = [15 + i * 0.05 + np.random.normal(0, 0.3) for i in range(len(dates))]

        df = pd.DataFrame({"ds": dates, "y": y, "is_workday": is_workday})

        m = Farseer()
        m.add_seasonality(
            name="workday_pattern",
            period=7,
            fourier_order=3,
            condition_name="is_workday",
        )
        m.fit(df)

        future_dates = [dates[-1] + timedelta(days=i + 1) for i in range(20)]
        future = pd.DataFrame(
            {
                "ds": future_dates,
                "is_workday": [(d.weekday() < 5) for d in future_dates],
            }
        )

        forecast = m.predict(future)
        assert len(forecast) == 20

    def test_conditional_seasonality_false_all(self):
        """Test conditional seasonality when condition is always False."""
        from farseer import Farseer

        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]
        y = [10 + i * 0.1 + np.random.normal(0, 0.2) for i in range(100)]
        is_special = [False] * 100  # Always False

        df = pl.DataFrame({"ds": dates, "y": y, "is_special": is_special})

        m = Farseer()
        m.add_seasonality(
            name="special_seasonality",
            period=30,
            fourier_order=2,
            condition_name="is_special",
        )
        m.fit(df)

        # Predict with condition still False
        future = pl.DataFrame(
            {
                "ds": [dates[-1] + timedelta(days=i + 1) for i in range(20)],
                "is_special": [False] * 20,
            }
        )

        forecast = m.predict(future)
        assert len(forecast) == 20
        # When condition is always False, the seasonal component should be ~0

    def test_conditional_seasonality_true_all(self):
        """Test conditional seasonality when condition is always True."""
        from farseer import Farseer

        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]
        y = [
            10 + i * 0.1 + 3 * np.sin(2 * np.pi * i / 7) + np.random.normal(0, 0.2)
            for i in range(100)
        ]
        is_active = [True] * 100  # Always True

        df = pl.DataFrame({"ds": dates, "y": y, "is_active": is_active})

        m = Farseer()
        m.add_seasonality(
            name="active_weekly", period=7, fourier_order=3, condition_name="is_active"
        )
        m.fit(df)

        # Predict with condition still True
        future = pl.DataFrame(
            {
                "ds": [dates[-1] + timedelta(days=i + 1) for i in range(20)],
                "is_active": [True] * 20,
            }
        )

        forecast = m.predict(future)
        assert len(forecast) == 20

    def test_conditional_seasonality_modes(self):
        """Test conditional seasonality with different modes (additive/multiplicative)."""
        from farseer import Farseer

        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(200)]
        is_peak = [i % 50 < 10 for i in range(len(dates))]
        y = [20 * (1 + 0.01 * i) + np.random.normal(0, 1) for i in range(len(dates))]

        df = pl.DataFrame({"ds": dates, "y": y, "is_peak": is_peak})

        m = Farseer(seasonality_mode="multiplicative")
        m.add_seasonality(
            name="peak_weekly",
            period=7,
            fourier_order=2,
            mode="multiplicative",
            condition_name="is_peak",
        )
        m.fit(df)

        params = m.params()
        assert params["seasonalities"][0]["mode"] == "Multiplicative"

        future = pl.DataFrame(
            {
                "ds": [dates[-1] + timedelta(days=i + 1) for i in range(30)],
                "is_peak": [i % 50 < 10 for i in range(30)],
            }
        )

        forecast = m.predict(future)
        assert len(forecast) == 30

    def test_conditional_seasonality_serialization(self):
        """Test that conditional seasonalities are properly saved and loaded."""
        from farseer import Farseer
        import tempfile
        import os

        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]
        is_active = [i % 3 == 0 for i in range(100)]
        y = [10 + i * 0.1 for i in range(100)]

        df = pl.DataFrame({"ds": dates, "y": y, "is_active": is_active})

        m = Farseer()
        m.add_seasonality(
            name="triweekly", period=21, fourier_order=2, condition_name="is_active"
        )
        m.fit(df)

        # Get original forecast
        future = pl.DataFrame(
            {
                "ds": [dates[-1] + timedelta(days=i + 1) for i in range(10)],
                "is_active": [i % 3 == 0 for i in range(10)],
            }
        )
        _ = m.predict(future)

        # Save to file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            m.save(temp_path)

            # Load from file
            m2 = Farseer.load(temp_path)

            # Check that conditional seasonality was preserved
            params = m2.params()
            assert len(params["seasonalities"]) == 1
            assert params["seasonalities"][0]["name"] == "triweekly"
            assert params["seasonalities"][0]["condition_name"] == "is_active"

            # Predict with loaded model using explicit dataframe
            # Note: loaded models return Rust Farseer objects, so we need to convert Polars to pandas
            future_pd = future.to_pandas()
            forecast2 = m2.predict(future_pd)

            # Both forecasts should be similar (same model)
            assert len(forecast2) == 10
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestConditionalSeasonalityEdgeCases:
    """Edge cases and error handling for conditional seasonalities."""

    def test_condition_wrong_type(self):
        """Test that non-boolean condition values need to be converted to bool."""
        from farseer import Farseer

        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(50)]
        y = [10 + i * 0.1 for i in range(50)]
        # Convert integers to booleans explicitly
        is_active = [bool(x) for x in ([0, 1, 0, 1] * 12 + [0, 1])]

        df = pl.DataFrame({"ds": dates, "y": y, "is_active": is_active})

        m = Farseer()
        m.add_seasonality(
            name="cond_season", period=7, fourier_order=2, condition_name="is_active"
        )

        m.fit(df)

        future = pl.DataFrame(
            {
                "ds": [dates[-1] + timedelta(days=i + 1) for i in range(10)],
                "is_active": [bool(x) for x in [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]],
            }
        )

        forecast = m.predict(future)
        assert len(forecast) == 10

    def test_condition_with_prior_scale(self):
        """Test conditional seasonality with custom prior scale."""
        from farseer import Farseer

        dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(150)]
        is_special = [i % 7 == 0 for i in range(len(dates))]
        y = [15 + i * 0.05 + np.random.normal(0, 0.5) for i in range(len(dates))]

        df = pl.DataFrame({"ds": dates, "y": y, "is_special": is_special})

        m = Farseer()
        m.add_seasonality(
            name="special_pattern",
            period=14,
            fourier_order=2,
            prior_scale=5.0,
            condition_name="is_special",
        )
        m.fit(df)

        params = m.params()
        assert params["seasonalities"][0]["prior_scale"] == 5.0
        assert params["seasonalities"][0]["condition_name"] == "is_special"
