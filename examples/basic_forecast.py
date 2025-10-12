#!/usr/bin/env python3
"""
Basic Forecasting Example

Demonstrates the most common use case: fitting a time series model
with automatic trend and seasonality detection, then making predictions.

This is equivalent to Prophet's quickstart example.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from seer import Seer

# Set random seed for reproducibility
np.random.seed(42)

def generate_sample_data(periods=365, freq='D'):
    """
    Generate synthetic time series data with trend and seasonality.
    
    Parameters
    ----------
    periods : int
        Number of data points
    freq : str
        Frequency string ('D' for daily, 'H' for hourly, etc.)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'ds' (date) and 'y' (value) columns
    """
    # Create date range
    dates = pd.date_range('2020-01-01', periods=periods, freq=freq)
    
    # Generate components
    # 1. Trend: linear growth
    trend = np.linspace(10, 20, periods)
    
    # 2. Yearly seasonality (if enough data)
    if periods >= 14:
        yearly = 5 * np.sin(2 * np.pi * np.arange(periods) / 365.25)
    else:
        yearly = np.zeros(periods)
    
    # 3. Weekly seasonality
    weekly = 2 * np.sin(2 * np.pi * np.arange(periods) / 7)
    
    # 4. Random noise
    noise = np.random.normal(0, 0.5, periods)
    
    # Combine components
    y = trend + yearly + weekly + noise
    
    return pd.DataFrame({'ds': dates, 'y': y})


def main():
    print("=" * 60)
    print("Seer: Basic Forecasting Example")
    print("=" * 60)
    print()
    
    # Step 1: Generate sample data
    print("Step 1: Generating sample data...")
    df = generate_sample_data(periods=365, freq='D')
    print(f"  Created {len(df)} daily observations")
    print(f"  Date range: {df['ds'].min()} to {df['ds'].max()}")
    print(f"  Value range: {df['y'].min():.2f} to {df['y'].max():.2f}")
    print()
    
    # Step 2: Create and configure model
    print("Step 2: Creating Seer model...")
    model = Seer(
        growth='linear',              # Linear trend
        n_changepoints=25,            # 25 potential changepoints
        changepoint_range=0.8,        # Changepoints in first 80% of data
        changepoint_prior_scale=0.05, # Regularization for changepoints
        yearly_seasonality=True,      # Enable yearly seasonality
        weekly_seasonality=True,      # Enable weekly seasonality
        daily_seasonality=False,      # Disable daily (not needed for daily data)
        seasonality_mode='additive',  # Additive seasonality
        interval_width=0.80,          # 80% confidence intervals
    )
    print("  Model configured with linear trend and automatic seasonality")
    print()
    
    # Step 3: Fit the model
    print("Step 3: Fitting model to historical data...")
    model.fit(df)
    print("  ✓ Model fitted successfully")
    print()
    
    # Step 4: Make future dataframe
    print("Step 4: Creating future dataframe...")
    future = model.make_future_dataframe(periods=90, freq='D', include_history=True)
    print(f"  Created dataframe with {len(future)} dates ({len(df)} history + 90 future)")
    print()
    
    # Step 5: Generate predictions
    print("Step 5: Generating forecast...")
    forecast = model.predict(future)
    print("  ✓ Forecast generated")
    print()
    
    # Step 6: Display results
    print("Step 6: Forecast Summary")
    print("-" * 60)
    print("\nLast 5 historical predictions:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[-95:-90].to_string(index=False))
    print("\nFirst 10 future predictions:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10).to_string(index=False))
    print()
    
    # Step 7: Examine components
    print("Step 7: Forecast Components")
    print("-" * 60)
    last_point = forecast.iloc[-1]
    print(f"  Trend:         {last_point['trend']:.2f}")
    if 'yearly' in forecast.columns:
        print(f"  Yearly:        {last_point.get('yearly', 0):.2f}")
    if 'weekly' in forecast.columns:
        print(f"  Weekly:        {last_point.get('weekly', 0):.2f}")
    print(f"  Prediction:    {last_point['yhat']:.2f}")
    print(f"  Interval:      [{last_point['yhat_lower']:.2f}, {last_point['yhat_upper']:.2f}]")
    print()
    
    # Step 8: Model persistence
    print("Step 8: Model Serialization")
    print("-" * 60)
    # Save model to JSON
    model_json = model.to_json()
    print(f"  Model serialized to JSON ({len(model_json)} bytes)")
    
    # Load model from JSON
    model2 = Seer.from_json(model_json)
    print("  ✓ Model loaded from JSON")
    
    # Verify predictions match
    forecast2 = model2.predict(future)
    match = np.allclose(forecast['yhat'].values, forecast2['yhat'].values)
    print(f"  Predictions match: {match}")
    print()
    
    # Step 9: Model parameters
    print("Step 9: Model Parameters")
    print("-" * 60)
    params = model.params()
    print(f"  Growth type:        {params.get('growth', 'unknown')}")
    print(f"  Changepoints:       {params.get('n_changepoints', 0)}")
    print(f"  Yearly seasonality: {params.get('yearly_seasonality', False)}")
    print(f"  Weekly seasonality: {params.get('weekly_seasonality', False)}")
    print(f"  Fitted:             {params.get('fitted', False)}")
    print()
    
    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  - Try plotting: model.plot(forecast)")
    print("  - Explore components: model.plot_components(forecast)")
    print("  - See advanced_features.py for more capabilities")
    print()


if __name__ == '__main__':
    main()
