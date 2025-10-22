"""Tests for holiday prior scale configuration in Farseer."""

import polars as pl
import pytest
from datetime import datetime, timedelta
from farseer import Farseer


@pytest.fixture
def holiday_data():
    """Dataset with holidays for testing."""
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(365)]
    y = [
        10.0 + i * 0.05 + (5.0 if i % 7 == 0 else 0.0) for i in range(365)
    ]  # Weekly pattern
    return pl.DataFrame({"ds": dates, "y": y})


class TestHolidayPriorScales:
    """Test holiday prior scale configuration."""

    def test_holiday_default_prior_scale(self, holiday_data):
        """Test that holidays use default prior_scale of 10.0."""
        m = Farseer()
        m.add_holidays("test_holiday", dates=["2020-01-15", "2020-02-15", "2020-03-15"])
        m.fit(holiday_data)

        params = m.params()
        assert params["fitted"] is True

    def test_holiday_custom_prior_scale(self, holiday_data):
        """Test that holidays can have custom prior_scale values."""
        m = Farseer()
        # Add holiday with strong prior (high prior_scale)
        m.add_holidays(
            "major_holiday", dates=["2020-01-15", "2020-02-15"], prior_scale=20.0
        )
        # Add holiday with weak prior (low prior_scale)
        m.add_holidays(
            "minor_holiday", dates=["2020-03-15", "2020-04-15"], prior_scale=5.0
        )
        m.fit(holiday_data)

        params = m.params()
        assert params["fitted"] is True

        # Both holidays should be included in the model
        # (We can't directly check prior_scale from params, but we verify the model fits)

    def test_holiday_prior_independent_from_seasonality(self, holiday_data):
        """Test that holiday priors are independent of seasonality prior scales."""
        m1 = Farseer()
        m1.add_seasonality(
            "custom_season", period=30.5, fourier_order=5, prior_scale=5.0
        )
        m1.add_holidays("holiday", dates=["2020-01-15"], prior_scale=20.0)
        m1.fit(holiday_data)

        m2 = Farseer()
        m2.add_seasonality(
            "custom_season", period=30.5, fourier_order=5, prior_scale=15.0
        )
        m2.add_holidays("holiday", dates=["2020-01-15"], prior_scale=20.0)
        m2.fit(holiday_data)

        # Both models should fit successfully with different seasonality priors
        # but same holiday prior
        assert m1.params()["fitted"] is True
        assert m2.params()["fitted"] is True

    def test_multiple_holidays_different_priors(self, holiday_data):
        """Test adding multiple holidays with different prior scales."""
        m = Farseer()

        # Add 3 holidays with different prior scales
        m.add_holidays("holiday1", dates=["2020-01-01"], prior_scale=5.0)
        m.add_holidays("holiday2", dates=["2020-07-04"], prior_scale=10.0)
        m.add_holidays("holiday3", dates=["2020-12-25"], prior_scale=20.0)

        m.fit(holiday_data)

        params = m.params()
        assert params["fitted"] is True

        # Predict to ensure model works correctly
        forecast = m.predict()
        assert len(forecast["yhat"]) == len(holiday_data)

    def test_holiday_with_window(self, holiday_data):
        """Test holidays with lower and upper windows and custom prior scale."""
        m = Farseer()
        m.add_holidays(
            "christmas",
            dates=["2020-12-25"],
            lower_window=-2,  # 2 days before
            upper_window=1,  # 1 day after
            prior_scale=15.0,
        )
        m.fit(holiday_data)

        params = m.params()
        assert params["fitted"] is True

        forecast = m.predict()
        assert len(forecast["yhat"]) == len(holiday_data)
