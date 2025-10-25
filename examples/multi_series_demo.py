"""
Multi-Series Forecasting Example with Farseer

This example demonstrates how to use FarseerMultiSeries to fit and predict
multiple time series in parallel using multiprocessing to avoid Python's GIL.

Key features shown:
1. Fitting multiple series in parallel
2. Handling partial failures gracefully
3. Making predictions for all successful series
4. Accessing individual models and errors
"""

import polars as pl
from datetime import datetime, timedelta
from farseer import FarseerMultiSeries
import random

# Set random seed for reproducibility
random.seed(42)


def create_sample_data(n_series=20, n_days=180, include_bad_series=True):
    """
    Create sample multi-series data.

    Parameters
    ----------
    n_series : int
        Number of series to generate
    n_days : int
        Number of days per series
    include_bad_series : bool
        If True, include one series with insufficient data that will fail

    Returns
    -------
    df : polars.DataFrame
        Multi-series dataframe with series_id, ds, and y columns
    """
    print(f"Creating sample data with {n_series} series, {n_days} days each...")

    dfs = []

    # Create good series
    for i in range(n_series - 1 if include_bad_series else n_series):
        # Generate dates
        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(days=d) for d in range(n_days)]

        # Generate synthetic data with trend and seasonality
        trend = [10 + i * 5 + 0.1 * d for d in range(n_days)]
        seasonality = [
            3
            * pl.DataFrame({"x": [d]}).select(
                pl.col("x").map_elements(
                    lambda x: random.uniform(-1, 1), return_dtype=pl.Float64
                )
            )["x"][0]
            for d in range(n_days)
        ]

        y = [t + s + random.gauss(0, 2) for t, s in zip(trend, seasonality)]

        df = pl.DataFrame(
            {"series_id": [f"store_{i:02d}"] * n_days, "ds": dates, "y": y}
        )
        dfs.append(df)

    # Add one problematic series if requested
    if include_bad_series:
        print("  Including 1 problematic series with insufficient data...")
        df_bad = pl.DataFrame(
            {"series_id": ["store_bad"], "ds": [datetime(2020, 1, 1)], "y": [10.0]}
        )
        dfs.append(df_bad)

    return pl.concat(dfs)


def main():
    """Main demonstration"""

    print("=" * 80)
    print("Multi-Series Forecasting with Farseer")
    print("=" * 80)
    print()

    # =========================================================================
    # 1. Create sample data
    # =========================================================================
    n_series = 20
    df = create_sample_data(n_series=n_series, n_days=180, include_bad_series=True)

    print(f"\nDataset shape: {df.shape}")
    print(f"Number of unique series: {df['series_id'].n_unique()}")
    print("\nFirst few rows:")
    print(df.head(10))

    # =========================================================================
    # 2. Fit models in parallel
    # =========================================================================
    print("\n" + "=" * 80)
    print("Fitting models in parallel...")
    print("=" * 80)

    # Create multi-series forecaster
    # n_processes=4 means 4 parallel processes will be used
    multi_model = FarseerMultiSeries(
        n_processes=4,
        growth="linear",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
    )

    # Fit all series
    # This will automatically handle failures gracefully
    fit_results = multi_model.fit(df, series_col="series_id")

    # Display results
    print("\nFitting Results:")
    print(f"  Successful: {fit_results['n_success']} series")
    print(f"  Failed: {fit_results['n_failed']} series")

    # =========================================================================
    # 3. Inspect failures
    # =========================================================================
    if fit_results["errors"]:
        print("\n" + "=" * 80)
        print("Failed Series Details:")
        print("=" * 80)

        for series_id, error_msg in fit_results["errors"].items():
            print(f"\nSeries ID: {series_id}")
            print(f"  Error: {error_msg}")

    # =========================================================================
    # 4. Make predictions for all successful series
    # =========================================================================
    print("\n" + "=" * 80)
    print("Making predictions for successful series...")
    print("=" * 80)

    # Predict 30 days into the future
    pred_results = multi_model.predict(periods=30, freq="D", include_history=False)

    print("\nPrediction Results:")
    print(f"  Successful: {pred_results['n_success']} series")
    print(f"  Failed: {pred_results['n_failed']} series")

    if pred_results["forecasts"] is not None:
        forecasts = pred_results["forecasts"]
        print(f"\nForecast shape: {forecasts.shape}")
        print(f"Forecast columns: {forecasts.columns}")
        print("\nFirst few forecast rows:")
        print(
            forecasts.select(
                ["series_id", "ds", "yhat", "yhat_lower", "yhat_upper"]
            ).head(10)
        )

    # =========================================================================
    # 5. Access individual models
    # =========================================================================
    print("\n" + "=" * 80)
    print("Accessing Individual Models:")
    print("=" * 80)

    # Get a specific model
    store_00_model = multi_model.get_model("store_00")

    if store_00_model is not None:
        print("\nSuccessfully retrieved model for 'store_00'")

        # You can use this model like a normal Farseer model
        # For example, make custom predictions
        custom_future = store_00_model.make_future_dataframe(periods=60, freq="D")
        custom_forecast = store_00_model.predict(custom_future)

        print(f"Custom forecast for store_00 shape: {custom_forecast.shape}")

    # =========================================================================
    # 6. Using custom future dataframe
    # =========================================================================
    print("\n" + "=" * 80)
    print("Making Predictions with Custom Future DataFrame:")
    print("=" * 80)

    # Create custom future dates (e.g., only specific dates)
    future_dates = [datetime(2020, 7, 1) + timedelta(days=d) for d in range(0, 30, 3)]

    # Create future dataframe for all successful series
    future_dfs = []
    for series_id in fit_results["models"].keys():
        future_df = pl.DataFrame(
            {"series_id": [series_id] * len(future_dates), "ds": future_dates}
        )
        future_dfs.append(future_df)

    custom_future_df = pl.concat(future_dfs)

    print(f"\nCustom future dataframe shape: {custom_future_df.shape}")

    # Make predictions
    custom_pred_results = multi_model.predict(future_df=custom_future_df)

    print("\nCustom Prediction Results:")
    print(f"  Successful: {custom_pred_results['n_success']} series")

    if custom_pred_results["forecasts"] is not None:
        print(f"  Forecast shape: {custom_pred_results['forecasts'].shape}")

    # =========================================================================
    # 7. Summary statistics across all series
    # =========================================================================
    print("\n" + "=" * 80)
    print("Summary Statistics Across All Series:")
    print("=" * 80)

    if pred_results["forecasts"] is not None:
        forecasts = pred_results["forecasts"]

        # Calculate statistics by series
        summary = (
            forecasts.group_by("series_id")
            .agg(
                [
                    pl.col("yhat").mean().alias("mean_forecast"),
                    pl.col("yhat").min().alias("min_forecast"),
                    pl.col("yhat").max().alias("max_forecast"),
                    pl.col("yhat_upper").mean().alias("mean_upper"),
                    pl.col("yhat_lower").mean().alias("mean_lower"),
                ]
            )
            .sort("series_id")
        )

        print("\nForecast Summary by Series:")
        print(summary)

    # =========================================================================
    # 8. Export forecasts
    # =========================================================================
    print("\n" + "=" * 80)
    print("Exporting Results:")
    print("=" * 80)

    if pred_results["forecasts"] is not None:
        # Save to CSV
        output_path = "multi_series_forecast.csv"
        pred_results["forecasts"].write_csv(output_path)
        print(f"\nForecasts saved to: {output_path}")

        # Also save error log if there were failures
        if fit_results["errors"] or pred_results["errors"]:
            error_log_path = "multi_series_errors.txt"
            with open(error_log_path, "w") as f:
                f.write("Multi-Series Forecasting Error Log\n")
                f.write("=" * 80 + "\n\n")

                if fit_results["errors"]:
                    f.write("Fitting Errors:\n")
                    f.write("-" * 80 + "\n")
                    for series_id, error in fit_results["errors"].items():
                        f.write(f"\nSeries: {series_id}\n")
                        f.write(f"Error: {error}\n")

                if pred_results["errors"]:
                    f.write("\n\nPrediction Errors:\n")
                    f.write("-" * 80 + "\n")
                    for series_id, error in pred_results["errors"].items():
                        f.write(f"\nSeries: {series_id}\n")
                        f.write(f"Error: {error}\n")

            print(f"Error log saved to: {error_log_path}")

    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. Multi-series fitting happens in parallel using multiprocessing")
    print("2. Failures are handled gracefully - bad series don't break others")
    print("3. You can access individual models for custom analysis")
    print("4. Results include detailed error messages for failed series")
    print("5. Both auto-generated and custom future dataframes are supported")


if __name__ == "__main__":
    main()
