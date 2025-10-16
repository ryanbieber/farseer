#!/usr/bin/env python3
"""
Test that Seer's output schema exactly matches Prophet's output schema
"""

import pytest
import pandas as pd
import numpy as np
from farseer import Farseer

try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


@pytest.mark.skipif(not PROPHET_AVAILABLE, reason="Prophet not installed")
class TestOutputSchemaMatchesProphet:
    """Verify Seer's predict() output schema matches Prophet exactly"""

    def test_column_order_matches_prophet(self):
        """Test that column order exactly matches Prophet"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100),
                "y": np.random.randn(100).cumsum() + 100,
            }
        )

        # Prophet
        m_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        m_prophet.fit(df)
        forecast_prophet = m_prophet.predict(df)

        # Seer
        m_seer = Farseer(yearly_seasonality=True, weekly_seasonality=True)
        m_seer.fit(df)
        forecast_seer = m_seer.predict(df).to_pandas()

        # Column order must match exactly
        assert (
            list(forecast_prophet.columns) == list(forecast_seer.columns)
        ), f"Column order mismatch:\nProphet: {list(forecast_prophet.columns)}\nSeer: {list(forecast_seer.columns)}"

    def test_all_columns_present(self):
        """Test that all Prophet columns are present in Seer output"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100),
                "y": np.random.randn(100).cumsum() + 100,
            }
        )

        # Prophet
        m_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        m_prophet.fit(df)
        forecast_prophet = m_prophet.predict(df)

        # Seer
        m_seer = Farseer(yearly_seasonality=True, weekly_seasonality=True)
        m_seer.fit(df)
        forecast_seer = m_seer.predict(df).to_pandas()

        # All Prophet columns must be in Seer
        prophet_cols = set(forecast_prophet.columns)
        seer_cols = set(forecast_seer.columns)

        assert (
            prophet_cols == seer_cols
        ), f"Column set mismatch:\nMissing from Seer: {prophet_cols - seer_cols}\nExtra in Seer: {seer_cols - prophet_cols}"

    def test_yearly_weekly_always_present(self):
        """Test that yearly and weekly columns are always present even when disabled"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100),
                "y": np.random.randn(100).cumsum() + 100,
            }
        )

        # Test with seasonality disabled
        m = Farseer(
            yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False
        )
        m.fit(df)
        forecast = m.predict(df).to_pandas()

        # Columns must still be present
        assert "yearly" in forecast.columns
        assert "yearly_lower" in forecast.columns
        assert "yearly_upper" in forecast.columns
        assert "weekly" in forecast.columns
        assert "weekly_lower" in forecast.columns
        assert "weekly_upper" in forecast.columns

        # But they should be all zeros when disabled
        assert (forecast["yearly"] == 0).all()
        assert (forecast["weekly"] == 0).all()

    def test_uncertainty_bounds_for_all_components(self):
        """Test that all components have uncertainty bounds"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100),
                "y": np.random.randn(100).cumsum() + 100,
            }
        )

        m = Farseer(yearly_seasonality=True, weekly_seasonality=True)
        m.fit(df)
        forecast = m.predict(df).to_pandas()

        # Check that all components have lower/upper bounds
        components_with_bounds = [
            "yhat",
            "trend",
            "additive_terms",
            "multiplicative_terms",
            "yearly",
            "weekly",
        ]

        for comp in components_with_bounds:
            assert comp in forecast.columns, f"Missing component: {comp}"
            assert (
                f"{comp}_lower" in forecast.columns
            ), f"Missing lower bound for {comp}"
            assert (
                f"{comp}_upper" in forecast.columns
            ), f"Missing upper bound for {comp}"

    def test_required_columns_present(self):
        """Test that all required Prophet columns are present"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100),
                "y": np.random.randn(100).cumsum() + 100,
            }
        )

        m = Farseer(yearly_seasonality=True, weekly_seasonality=True)
        m.fit(df)
        forecast = m.predict(df).to_pandas()

        required_columns = [
            "ds",
            "trend",
            "yhat_lower",
            "yhat_upper",
            "trend_lower",
            "trend_upper",
            "additive_terms",
            "additive_terms_lower",
            "additive_terms_upper",
            "weekly",
            "weekly_lower",
            "weekly_upper",
            "yearly",
            "yearly_lower",
            "yearly_upper",
            "multiplicative_terms",
            "multiplicative_terms_lower",
            "multiplicative_terms_upper",
            "yhat",
        ]

        for col in required_columns:
            assert col in forecast.columns, f"Missing required column: {col}"

    def test_column_count(self):
        """Test that the number of columns matches Prophet"""
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=100),
                "y": np.random.randn(100).cumsum() + 100,
            }
        )

        # Prophet
        m_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        m_prophet.fit(df)
        forecast_prophet = m_prophet.predict(df)

        # Seer
        m_seer = Farseer(yearly_seasonality=True, weekly_seasonality=True)
        m_seer.fit(df)
        forecast_seer = m_seer.predict(df).to_pandas()

        assert (
            len(forecast_prophet.columns) == len(forecast_seer.columns)
        ), f"Column count mismatch: Prophet has {len(forecast_prophet.columns)}, Seer has {len(forecast_seer.columns)}"
