#!/usr/bin/env python3
"""
Quick integration test for multi-series functionality.
Tests basic fit and predict with graceful error handling.
"""

import polars as pl
import warnings
from datetime import datetime, timedelta
from farseer import FarseerMultiSeries


def test_basic_multi_series():
    """Test basic multi-series functionality"""
    print("Testing basic multi-series functionality...")

    # Create simple test data
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]

    # Two good series
    df = pl.DataFrame(
        {
            "series_id": ["A"] * 100 + ["B"] * 100,
            "ds": dates + dates,
            "y": [float(x) for x in range(100)] + [float(x + 50) for x in range(100)],
        }
    )

    # Fit
    multi_model = FarseerMultiSeries(n_processes=2)
    results = multi_model.fit(df, series_col="series_id")

    assert (
        results["n_success"] == 2
    ), f"Expected 2 successes, got {results['n_success']}"
    assert results["n_failed"] == 0, f"Expected 0 failures, got {results['n_failed']}"
    assert len(results["models"]) == 2
    print("✅ Fit successful for 2 series")

    # Predict
    pred_results = multi_model.predict(periods=10, freq="D")

    assert pred_results["n_success"] == 2
    assert pred_results["forecasts"] is not None
    assert "series_id" in pred_results["forecasts"].columns
    print("✅ Predict successful for 2 series")

    # Access individual model
    model_a = multi_model.get_model("A")
    assert model_a is not None
    print("✅ Individual model access works")


def test_graceful_failure():
    """Test graceful handling of partial failures"""
    print("\nTesting graceful failure handling...")

    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]

    # One good series, one bad series (insufficient data)
    df_good = pl.DataFrame(
        {
            "series_id": ["good"] * 100,
            "ds": dates,
            "y": [float(x) for x in range(100)],  # Convert to float for consistency
        }
    )

    df_bad = pl.DataFrame(
        {"series_id": ["bad"], "ds": [datetime(2020, 1, 1)], "y": [1.0]}
    )

    df = pl.concat([df_good, df_bad])

    # Fit with expected warning
    multi_model = FarseerMultiSeries(n_processes=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        results = multi_model.fit(df, series_col="series_id")

    assert results["n_success"] == 1, f"Expected 1 success, got {results['n_success']}"
    assert results["n_failed"] == 1, f"Expected 1 failure, got {results['n_failed']}"
    assert "good" in results["models"]
    assert "bad" in results["errors"]
    print("✅ Graceful failure: 1 success, 1 failure correctly handled")

    # Check error message
    error_msg = multi_model.get_error("bad")
    assert error_msg is not None
    assert "less than 2" in error_msg.lower()
    print(f"✅ Error message captured: {error_msg[:50]}...")

    # Predict only on successful model
    pred_results = multi_model.predict(periods=10)
    assert pred_results["n_success"] == 1
    forecasts = pred_results["forecasts"]
    series_ids = forecasts["series_id"].unique().to_list()
    assert series_ids == ["good"]
    print("✅ Predict only on successful series")


if __name__ == "__main__":
    print("=" * 70)
    print("Multi-Series Integration Test")
    print("=" * 70)

    try:
        test_basic_multi_series()
        test_graceful_failure()

        print("\n" + "=" * 70)
        print("✅ All integration tests passed!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
