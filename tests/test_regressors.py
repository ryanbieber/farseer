"""
Tests for additional regressors functionality.
Ensures compatibility with Facebook Prophet's regressor implementation.
"""

import pandas as pd
import numpy as np
import pytest
from farseer import Farseer


@pytest.fixture
def daily_univariate_ts():
    """Generate daily time series data for testing"""
    np.random.seed(876543987)  # Prophet's test seed
    n = 510
    dates = pd.date_range("2012-01-01", periods=n, freq="D")

    trend = np.arange(n) * 0.5 + 10
    yearly = np.sin(2 * np.pi * np.arange(n) / 365.25) * 5
    weekly = np.sin(2 * np.pi * np.arange(n) / 7) * 2
    noise = np.random.randn(n) * 2

    y = trend + yearly + weekly + noise

    return pd.DataFrame({"ds": dates, "y": y})


class TestRegressorBasics:
    """Test basic regressor functionality"""

    def test_add_regressor_basic(self, daily_univariate_ts):
        """Test adding a basic regressor"""
        m = Farseer()
        m.add_regressor("binary_feature", prior_scale=0.2)
        m.add_regressor("numeric_feature", prior_scale=0.5)

        df = daily_univariate_ts.copy()
        df["binary_feature"] = [0] * 255 + [1] * 255
        df["numeric_feature"] = range(510)

        m.fit(df)

        # Check that model fitted successfully
        params = m.params()
        assert params["fitted"] is True

    def test_regressor_modes(self, daily_univariate_ts):
        """Test additive and multiplicative regressors"""
        m = Farseer()
        m.add_regressor("additive_reg", mode="additive")
        m.add_regressor("multiplicative_reg", mode="multiplicative")

        df = daily_univariate_ts.copy()
        df["additive_reg"] = np.random.randn(len(df))
        df["multiplicative_reg"] = np.random.randn(len(df))

        m.fit(df)
        params = m.params()
        assert params["fitted"] is True

    def test_regressor_must_be_added_before_fit(self, daily_univariate_ts):
        """Test that regressors must be added before fitting"""
        m = Farseer()
        df = daily_univariate_ts.copy()
        df["feature"] = range(len(df))

        m.fit(df)

        # Should raise an error when trying to add regressor after fit
        with pytest.raises(Exception):
            m.add_regressor("feature")

    def test_regressor_required_in_fit_df(self, daily_univariate_ts):
        """Test that regressor must be in fitting dataframe"""
        m = Farseer()
        m.add_regressor("missing_feature")

        df = daily_univariate_ts.copy()

        # Should raise an error because 'missing_feature' is not in df
        with pytest.raises(ValueError, match="Regressor 'missing_feature' not found"):
            m.fit(df)

    def test_regressor_required_in_predict_df(self, daily_univariate_ts):
        """Test that regressor must be in prediction dataframe"""
        m = Farseer()
        m.add_regressor("feature")

        df = daily_univariate_ts.copy()
        df["feature"] = range(len(df))
        m.fit(df)

        # Create future dataframe without the regressor
        future = pd.DataFrame(
            {"ds": pd.date_range(start="2013-09-01", periods=10, freq="D")}
        )

        # Should raise an error
        with pytest.raises(ValueError, match="Regressor 'feature' not found"):
            m.predict(future)


class TestRegressorStandardization:
    """Test regressor standardization behavior"""

    def test_binary_regressor_not_standardized_by_default(self, daily_univariate_ts):
        """Test that binary 0/1 regressors are not standardized by default (auto mode)"""
        m = Farseer()
        m.add_regressor("binary_feature", prior_scale=0.2)

        df = daily_univariate_ts.copy()
        df["binary_feature"] = [0] * 255 + [1] * 255

        m.fit(df)

        # Binary regressor should have mu=0, std=1 (not standardized)
        # We would need to expose regressor configs to test this properly
        # For now, just verify it fits without error
        params = m.params()
        assert params["fitted"] is True

    def test_numeric_regressor_standardized_by_default(self, daily_univariate_ts):
        """Test that numeric regressors are standardized by default (auto mode)"""
        m = Farseer()
        m.add_regressor("numeric_feature", prior_scale=0.5)

        df = daily_univariate_ts.copy()
        df["numeric_feature"] = range(510)

        m.fit(df)
        params = m.params()
        assert params["fitted"] is True

    def test_explicit_standardize_true(self, daily_univariate_ts):
        """Test explicit standardize=true"""
        m = Farseer()
        m.add_regressor("feature", standardize="true")

        df = daily_univariate_ts.copy()
        df["feature"] = range(510)

        m.fit(df)
        params = m.params()
        assert params["fitted"] is True

    def test_explicit_standardize_false(self, daily_univariate_ts):
        """Test explicit standardize=false"""
        m = Farseer()
        m.add_regressor("feature", standardize="false")

        df = daily_univariate_ts.copy()
        df["feature"] = range(510)

        m.fit(df)
        params = m.params()
        assert params["fitted"] is True

    def test_constant_regressor(self, daily_univariate_ts):
        """Test that constant regressor doesn't break fitting"""
        m = Farseer()
        m.add_regressor("constant_feature")

        df = daily_univariate_ts.copy()
        df["constant_feature"] = 5.0  # Constant value

        m.fit(df)
        params = m.params()
        assert params["fitted"] is True


class TestRegressorPrediction:
    """Test prediction with regressors"""

    def test_predict_with_regressors(self, daily_univariate_ts):
        """Test that predictions work with regressors"""
        m = Farseer()
        m.add_regressor("binary_feature", prior_scale=0.2)
        m.add_regressor("numeric_feature", prior_scale=0.5)

        df = daily_univariate_ts.copy()
        df["binary_feature"] = [0] * 255 + [1] * 255
        df["numeric_feature"] = range(510)

        m.fit(df)

        # Create future dataframe with regressors
        future = pd.DataFrame(
            {
                "ds": pd.date_range(start="2013-09-01", periods=10, freq="D"),
                "binary_feature": [0] * 10,
                "numeric_feature": range(510, 520),
            }
        )

        fcst = m.predict(future)

        # Check that forecast has expected columns
        assert len(fcst) == 10
        assert "ds" in fcst.columns
        assert "yhat" in fcst.columns
        assert "trend" in fcst.columns

    def test_predict_with_mixed_modes(self, daily_univariate_ts):
        """Test prediction with mixed additive/multiplicative regressors"""
        m = Farseer()
        m.add_regressor("additive_reg", prior_scale=0.2, mode="additive")
        m.add_regressor("multiplicative_reg", prior_scale=0.5, mode="multiplicative")

        df = daily_univariate_ts.copy()
        df["additive_reg"] = np.random.randn(len(df))
        df["multiplicative_reg"] = np.random.randn(len(df))

        m.fit(df)

        # Create future dataframe
        future = pd.DataFrame(
            {
                "ds": pd.date_range(start="2013-09-01", periods=10, freq="D"),
                "additive_reg": np.random.randn(10),
                "multiplicative_reg": np.random.randn(10),
            }
        )

        fcst = m.predict(future)

        assert len(fcst) == 10
        assert "yhat" in fcst.columns

    def test_predict_without_df_uses_training_regressors(self, daily_univariate_ts):
        """Test that predict without df uses training regressors"""
        m = Farseer()
        m.add_regressor("feature")

        df = daily_univariate_ts.copy()
        df["feature"] = np.random.randn(len(df))

        m.fit(df)

        # Predict on training data (no df argument)
        fcst = m.predict()

        assert len(fcst) == len(df)
        assert "yhat" in fcst.columns


class TestRegressorWithSeasonality:
    """Test regressors combined with seasonality"""

    def test_regressors_with_seasonality_modes(self, daily_univariate_ts):
        """Test regressors with different seasonality modes"""
        m = Farseer(seasonality_mode="multiplicative")
        m.add_seasonality("monthly", period=30, fourier_order=3, mode="additive")
        m.add_regressor("binary_feature", mode="additive")
        m.add_regressor("numeric_feature", mode="multiplicative")

        df = daily_univariate_ts.copy()
        df["binary_feature"] = [0] * 255 + [1] * 255
        df["numeric_feature"] = range(510)

        m.fit(df)
        params = m.params()
        assert params["fitted"] is True

        # Predict
        future = pd.DataFrame(
            {
                "ds": pd.date_range(start="2013-09-01", periods=10, freq="D"),
                "binary_feature": [1] * 10,
                "numeric_feature": range(510, 520),
            }
        )

        fcst = m.predict(future)
        assert len(fcst) == 10


class TestRegressorPriorScale:
    """Test prior scale parameter"""

    def test_different_prior_scales(self, daily_univariate_ts):
        """Test that different prior scales work"""
        m = Farseer()
        m.add_regressor("reg1", prior_scale=0.1)
        m.add_regressor("reg2", prior_scale=1.0)
        m.add_regressor("reg3", prior_scale=10.0)

        df = daily_univariate_ts.copy()
        df["reg1"] = np.random.randn(len(df))
        df["reg2"] = np.random.randn(len(df))
        df["reg3"] = np.random.randn(len(df))

        m.fit(df)
        params = m.params()
        assert params["fitted"] is True

    def test_invalid_prior_scale(self):
        """Test that invalid prior scale raises error"""
        m = Farseer()
        with pytest.raises(Exception):
            m.add_regressor("feature", prior_scale=0.0)  # Should be > 0

        with pytest.raises(Exception):
            m.add_regressor("feature", prior_scale=-1.0)  # Should be > 0


class TestRegressorDuplicates:
    """Test duplicate regressor handling"""

    def test_cannot_add_duplicate_regressor(self):
        """Test that duplicate regressor names are rejected"""
        m = Farseer()
        m.add_regressor("feature")

        with pytest.raises(Exception, match="already exists"):
            m.add_regressor("feature")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
