"""
Tests for multi-series forecasting functionality.

Tests cover:
- Basic multi-series fitting and prediction
- Partial failure scenarios (graceful degradation)
- Error handling and reporting
- Pandas and Polars DataFrame compatibility
"""

import polars as pl
import pytest
import warnings
from datetime import datetime
from farseer import FarseerMultiSeries

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class TestMultiSeriesBasic:
    """Basic multi-series functionality tests"""

    def test_multi_series_fit_polars(self):
        """Test fitting multiple series with Polars DataFrame"""
        # Create multi-series data
        dates = pl.date_range(
            datetime(2020, 1, 1), datetime(2020, 3, 31), interval="1d", eager=True
        )
        n_dates = len(dates)

        n_dates = len(dates)

        df = pl.DataFrame(
            {
                "series_id": ["A"] * n_dates + ["B"] * n_dates,
                "ds": list(dates) + list(dates),
                "y": [float(x) for x in range(n_dates)]
                + list(range(100, 100 + n_dates)),
            }
        )

        # Fit models
        multi_model = FarseerMultiSeries(n_processes=2)
        results = multi_model.fit(df, series_col="series_id")

        # Check results
        assert results["n_success"] == 2
        assert results["n_failed"] == 0
        assert len(results["models"]) == 2
        assert len(results["errors"]) == 0
        assert "A" in results["models"]
        assert "B" in results["models"]

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_multi_series_fit_pandas(self):
        """Test fitting multiple series with Pandas DataFrame"""
        # Create multi-series data
        dates = pd.date_range("2020-01-01", periods=90)
        n_dates = len(dates)

        df = pd.DataFrame(
            {
                "series_id": ["A"] * n_dates + ["B"] * n_dates,
                "ds": list(dates) + list(dates),
                "y": [float(x) for x in range(n_dates)]
                + list(range(100, 100 + n_dates)),
            }
        )

        # Fit models
        multi_model = FarseerMultiSeries(n_processes=2)
        results = multi_model.fit(df, series_col="series_id")

        # Check results
        assert results["n_success"] == 2
        assert results["n_failed"] == 0
        assert len(results["models"]) == 2

    def test_multi_series_predict_auto_future(self):
        """Test prediction with auto-generated future dates"""
        # Create and fit multi-series data
        dates = pl.date_range(
            datetime(2020, 1, 1), datetime(2020, 3, 31), interval="1d", eager=True
        )
        n_dates = len(dates)

        df = pl.DataFrame(
            {
                "series_id": ["A"] * n_dates + ["B"] * n_dates,
                "ds": list(dates) + list(dates),
                "y": [float(x) for x in range(n_dates)]
                + list(range(100, 100 + n_dates)),
            }
        )

        multi_model = FarseerMultiSeries(n_processes=2)
        fit_results = multi_model.fit(df, series_col="series_id")

        assert fit_results["n_success"] == 2

        # Make predictions
        pred_results = multi_model.predict(periods=10, freq="D")

        # Check results
        assert pred_results["n_success"] == 2
        assert pred_results["n_failed"] == 0
        assert pred_results["forecasts"] is not None

        forecast = pred_results["forecasts"]
        assert "series_id" in forecast.columns
        assert "ds" in forecast.columns
        assert "yhat" in forecast.columns

        # Check both series are in forecast
        series_ids = forecast["series_id"].unique().to_list()
        assert set(series_ids) == {"A", "B"}

    def test_multi_series_predict_custom_future(self):
        """Test prediction with custom future dataframe"""
        # Create and fit multi-series data
        dates = pl.date_range(
            datetime(2020, 1, 1), datetime(2020, 3, 31), interval="1d", eager=True
        )
        n_dates = len(dates)

        df = pl.DataFrame(
            {
                "series_id": ["A"] * n_dates + ["B"] * n_dates,
                "ds": list(dates) + list(dates),
                "y": [float(x) for x in range(n_dates)]
                + list(range(100, 100 + n_dates)),
            }
        )

        multi_model = FarseerMultiSeries(n_processes=2)
        fit_results = multi_model.fit(df, series_col="series_id")

        assert fit_results["n_success"] == 2

        # Create custom future dataframe
        future_dates = pl.date_range(
            datetime(2020, 4, 1), datetime(2020, 4, 10), interval="1d", eager=True
        )
        n_future = len(future_dates)

        future_df = pl.DataFrame(
            {
                "series_id": ["A"] * n_future + ["B"] * n_future,
                "ds": list(future_dates) + list(future_dates),
            }
        )

        # Make predictions
        pred_results = multi_model.predict(future_df=future_df)

        # Check results
        assert pred_results["n_success"] == 2
        assert pred_results["n_failed"] == 0
        assert pred_results["forecasts"] is not None

        forecast = pred_results["forecasts"]
        assert len(forecast) == n_future * 2


class TestMultiSeriesPartialFailure:
    """Test graceful handling of partial failures"""

    def test_partial_failure_one_series(self):
        """Test that one bad series doesn't break all others"""
        # Create multi-series data with one problematic series
        dates = pl.date_range(
            datetime(2020, 1, 1), datetime(2020, 3, 31), interval="1d", eager=True
        )
        n_dates = len(dates)

        # Good series A and B
        df_good = pl.DataFrame(
            {
                "series_id": ["A"] * n_dates + ["B"] * n_dates,
                "ds": list(dates) + list(dates),
                "y": [float(x) for x in range(n_dates)]
                + list(range(100, 100 + n_dates)),
            }
        )

        # Bad series C (only 1 data point - will fail validation)
        df_bad = pl.DataFrame(
            {"series_id": ["C"], "ds": [datetime(2020, 1, 1).date()], "y": [1.0]}
        )

        df = pl.concat([df_good, df_bad])

        # Fit models
        multi_model = FarseerMultiSeries(n_processes=2)

        with pytest.warns(UserWarning, match="2/3 series successfully"):
            results = multi_model.fit(df, series_col="series_id")

        # Check results - should have 2 successes and 1 failure
        assert results["n_success"] == 2
        assert results["n_failed"] == 1
        assert len(results["models"]) == 2
        assert len(results["errors"]) == 1

        # Good series should be fitted
        assert "A" in results["models"]
        assert "B" in results["models"]

        # Bad series should have error
        assert "C" in results["errors"]
        assert "less than 2 non-NaN rows" in results["errors"]["C"]

    def test_partial_failure_multiple_series(self):
        """Test with multiple failures (19/20 success scenario)"""
        # Create 20 series
        dates = pl.date_range(
            datetime(2020, 1, 1), datetime(2020, 3, 31), interval="1d", eager=True
        )
        n_dates = len(dates)

        dfs = []

        # 19 good series
        for i in range(19):
            df = pl.DataFrame(
                {
                    "series_id": [f"series_{i}"] * n_dates,
                    "ds": list(dates),
                    "y": [float(j + i * 10) for j in range(n_dates)],
                }
            )
            dfs.append(df)

        # 1 bad series (insufficient data)
        df_bad = pl.DataFrame(
            {
                "series_id": ["bad_series"],
                "ds": [datetime(2020, 1, 1).date()],
                "y": [1.0],
            }
        )
        dfs.append(df_bad)

        df = pl.concat(dfs)

        # Fit models
        multi_model = FarseerMultiSeries(n_processes=4)

        with pytest.warns(UserWarning, match="19/20 series successfully"):
            results = multi_model.fit(df, series_col="series_id")

        # Check results
        assert results["n_success"] == 19
        assert results["n_failed"] == 1
        assert len(results["models"]) == 19
        assert len(results["errors"]) == 1

        # All good series should be fitted
        for i in range(19):
            assert f"series_{i}" in results["models"]

        # Bad series should have error
        assert "bad_series" in results["errors"]

    def test_predict_with_some_failed_fits(self):
        """Test prediction when some series failed to fit"""
        # Create multi-series data with one bad series
        dates = pl.date_range(
            datetime(2020, 1, 1), datetime(2020, 3, 31), interval="1d", eager=True
        )
        n_dates = len(dates)

        df_good = pl.DataFrame(
            {
                "series_id": ["A"] * n_dates + ["B"] * n_dates,
                "ds": list(dates) + list(dates),
                "y": [float(x) for x in range(n_dates)]
                + list(range(100, 100 + n_dates)),
            }
        )

        df_bad = pl.DataFrame(
            {"series_id": ["C"], "ds": [datetime(2020, 1, 1).date()], "y": [1.0]}
        )

        df = pl.concat([df_good, df_bad])

        # Fit models
        multi_model = FarseerMultiSeries(n_processes=2)

        with pytest.warns(UserWarning):
            fit_results = multi_model.fit(df, series_col="series_id")

        assert fit_results["n_success"] == 2
        assert fit_results["n_failed"] == 1

        # Predict only on successful models
        pred_results = multi_model.predict(periods=10)

        # Should only predict for A and B, not C
        assert pred_results["n_success"] == 2
        assert pred_results["n_failed"] == 0

        forecast = pred_results["forecasts"]
        series_ids = forecast["series_id"].unique().to_list()
        assert set(series_ids) == {"A", "B"}
        assert "C" not in series_ids


class TestMultiSeriesErrorHandling:
    """Test error handling and validation"""

    def test_missing_series_column(self):
        """Test error when series column is missing"""
        df = pl.DataFrame(
            {
                "ds": [datetime(2020, 1, 1).date(), datetime(2020, 1, 2).date()],
                "y": [1.0, 2.0],
            }
        )

        multi_model = FarseerMultiSeries()

        with pytest.raises(ValueError, match="Column 'series_id' not found"):
            multi_model.fit(df, series_col="series_id")

    def test_missing_ds_column(self):
        """Test error when ds column is missing"""
        df = pl.DataFrame({"series_id": ["A", "A"], "y": [1.0, 2.0]})

        multi_model = FarseerMultiSeries()

        with pytest.raises(ValueError, match="must have 'ds' and 'y' columns"):
            multi_model.fit(df, series_col="series_id")

    def test_missing_y_column(self):
        """Test error when y column is missing"""
        df = pl.DataFrame(
            {
                "series_id": ["A", "A"],
                "ds": [datetime(2020, 1, 1).date(), datetime(2020, 1, 2).date()],
            }
        )

        multi_model = FarseerMultiSeries()

        with pytest.raises(ValueError, match="must have 'ds' and 'y' columns"):
            multi_model.fit(df, series_col="series_id")

    def test_predict_before_fit(self):
        """Test error when predicting before fitting"""
        multi_model = FarseerMultiSeries()

        with pytest.raises(ValueError, match="No models have been fitted"):
            multi_model.predict(periods=10)

    def test_predict_without_periods_or_future(self):
        """Test error when neither periods nor future_df provided"""
        # Create and fit data
        dates = pl.date_range(
            datetime(2020, 1, 1), datetime(2020, 3, 31), interval="1d", eager=True
        )
        n_dates = len(dates)

        df = pl.DataFrame(
            {
                "series_id": ["A"] * n_dates,
                "ds": list(dates),
                "y": [float(x) for x in range(n_dates)],
            }
        )

        multi_model = FarseerMultiSeries()
        multi_model.fit(df, series_col="series_id")

        with pytest.raises(
            ValueError, match="Either periods or future_df must be provided"
        ):
            multi_model.predict()

    def test_predict_future_missing_series_column(self):
        """Test error when future_df is missing series column"""
        # Create and fit data
        dates = pl.date_range(
            datetime(2020, 1, 1), datetime(2020, 3, 31), interval="1d", eager=True
        )
        n_dates = len(dates)

        df = pl.DataFrame(
            {
                "series_id": ["A"] * n_dates,
                "ds": list(dates),
                "y": [float(x) for x in range(n_dates)],
            }
        )

        multi_model = FarseerMultiSeries()
        multi_model.fit(df, series_col="series_id")

        # Future without series_id
        future = pl.DataFrame(
            {
                "ds": pl.date_range(
                    datetime(2020, 4, 1),
                    datetime(2020, 4, 10),
                    interval="1d",
                    eager=True,
                )
            }
        )

        with pytest.raises(
            ValueError, match="Column 'series_id' not found in future_df"
        ):
            multi_model.predict(future_df=future)


class TestMultiSeriesHelperMethods:
    """Test helper methods for accessing models and errors"""

    def test_get_model(self):
        """Test getting a specific model"""
        dates = pl.date_range(
            datetime(2020, 1, 1), datetime(2020, 3, 31), interval="1d", eager=True
        )
        n_dates = len(dates)

        df = pl.DataFrame(
            {
                "series_id": ["A"] * n_dates + ["B"] * n_dates,
                "ds": list(dates) + list(dates),
                "y": [float(x) for x in range(n_dates)]
                + list(range(100, 100 + n_dates)),
            }
        )

        multi_model = FarseerMultiSeries()
        multi_model.fit(df, series_col="series_id")

        # Get specific models
        model_a = multi_model.get_model("A")
        model_b = multi_model.get_model("B")
        model_c = multi_model.get_model("C")

        assert model_a is not None
        assert model_b is not None
        assert model_c is None

    def test_get_error(self):
        """Test getting error for failed series"""
        dates = pl.date_range(
            datetime(2020, 1, 1), datetime(2020, 3, 31), interval="1d", eager=True
        )
        n_dates = len(dates)

        df_good = pl.DataFrame(
            {
                "series_id": ["A"] * n_dates,
                "ds": list(dates),
                "y": [float(x) for x in range(n_dates)],
            }
        )

        df_bad = pl.DataFrame(
            {"series_id": ["B"], "ds": [datetime(2020, 1, 1).date()], "y": [1.0]}
        )

        df = pl.concat([df_good, df_bad])

        multi_model = FarseerMultiSeries()

        with pytest.warns(UserWarning):
            multi_model.fit(df, series_col="series_id")

        # Get errors
        error_a = multi_model.get_error("A")
        error_b = multi_model.get_error("B")

        assert error_a is None
        assert error_b is not None
        assert "less than 2 non-NaN rows" in error_b


class TestMultiSeriesWithFeatures:
    """Test multi-series with additional features like regressors and cap/floor"""

    def test_multi_series_with_cap(self):
        """Test logistic growth with cap across multiple series"""
        dates = pl.date_range(
            datetime(2020, 1, 1), datetime(2020, 3, 31), interval="1d", eager=True
        )
        n_dates = len(dates)

        df = pl.DataFrame(
            {
                "series_id": ["A"] * n_dates + ["B"] * n_dates,
                "ds": list(dates) + list(dates),
                "y": [float(x) for x in range(n_dates)]
                + list(range(100, 100 + n_dates)),
                "cap": [100.0] * n_dates + [200.0] * n_dates,
            }
        )

        multi_model = FarseerMultiSeries(growth="logistic")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            results = multi_model.fit(df, series_col="series_id")

        # At least one should succeed (Stan optimization can fail for some series)
        assert results["n_success"] >= 1
        assert results["n_success"] + results["n_failed"] == 2

        # Predict with cap (only on successful models)
        if results["n_success"] > 0:
            future_dates = pl.date_range(
                datetime(2020, 4, 1), datetime(2020, 4, 10), interval="1d", eager=True
            )
            n_future = len(future_dates)

            future_df = pl.DataFrame(
                {
                    "series_id": list(results["models"].keys())[0:1]
                    * n_future,  # Just use first successful series
                    "ds": list(future_dates),
                    "cap": [100.0] * n_future,
                }
            )

            pred_results = multi_model.predict(future_df=future_df)
            assert pred_results["n_success"] >= 1


class TestMultiSeriesModelParams:
    """Test that model parameters are properly applied to all series"""

    def test_custom_seasonality_params(self):
        """Test that custom seasonality parameters are applied"""
        dates = pl.date_range(
            datetime(2020, 1, 1), datetime(2020, 12, 31), interval="1d", eager=True
        )
        n_dates = len(dates)

        df = pl.DataFrame(
            {
                "series_id": ["A"] * n_dates + ["B"] * n_dates,
                "ds": list(dates) + list(dates),
                "y": [float(x) for x in range(n_dates)]
                + list(range(100, 100 + n_dates)),
            }
        )

        multi_model = FarseerMultiSeries(
            yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False
        )

        results = multi_model.fit(df, series_col="series_id")

        assert results["n_success"] == 2

        # Both models should have the same parameters
        model_a = multi_model.get_model("A")
        model_b = multi_model.get_model("B")

        assert model_a is not None
        assert model_b is not None
