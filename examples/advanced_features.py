#!/usr/bin/env python3
"""
Advanced Features Example

Demonstrates advanced Seer capabilities:
- Custom seasonality patterns
- Holiday effects
- Different trend types (logistic growth)
- Multiplicative seasonality
- Model tuning and configuration

Equivalent to Prophet's advanced examples.
"""

import pandas as pd
import numpy as np
from seer import Seer

np.random.seed(42)


def generate_logistic_data(periods=730):
    """
    Generate time series with logistic (saturating) growth.
    Useful for modeling growth with a natural limit (capacity).
    """
    dates = pd.date_range('2020-01-01', periods=periods, freq='D')
    
    # Logistic growth parameters
    capacity = 100.0  # Maximum achievable value
    k = 0.01          # Growth rate
    m = 0             # Offset
    
    # Time variable (scaled)
    t = np.arange(periods) / periods
    
    # Logistic trend: capacity / (1 + exp(-k*(t-m)))
    trend = capacity / (1 + np.exp(-k * (t - m) * 100))
    
    # Add seasonality
    yearly = 10 * np.sin(2 * np.pi * np.arange(periods) / 365.25)
    weekly = 3 * np.sin(2 * np.pi * np.arange(periods) / 7)
    
    # Add noise
    noise = np.random.normal(0, 2, periods)
    
    # Combine
    y = trend + yearly + weekly + noise
    
    # Create dataframe with capacity column
    df = pd.DataFrame({
        'ds': dates,
        'y': y,
        'cap': capacity  # Required for logistic growth
    })
    
    return df


def main():
    print("=" * 70)
    print("Seer: Advanced Features Example")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # Example 1: Logistic Growth with Capacity
    # -------------------------------------------------------------------------
    print("Example 1: Logistic Growth (Saturating Trend)")
    print("-" * 70)
    
    df_logistic = generate_logistic_data(periods=730)
    print(f"Generated {len(df_logistic)} observations with logistic growth")
    
    # Create model with logistic growth
    model_logistic = Seer(
        growth='logistic',            # Logistic (saturating) trend
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.1   # More flexible changepoints
    )
    
    print("Fitting logistic growth model...")
    model_logistic.fit(df_logistic)
    
    # Make predictions (must include 'cap' in future dataframe)
    future = model_logistic.make_future_dataframe(periods=180, freq='D')
    future['cap'] = 100.0  # Set capacity for future predictions
    
    forecast = model_logistic.predict(future)
    print(f"✓ Forecast generated for {len(forecast)} periods")
    print(f"  Final prediction: {forecast['yhat'].iloc[-1]:.2f} (approaching capacity of 100)")
    print()
    
    # -------------------------------------------------------------------------
    # Example 2: Custom Seasonality
    # -------------------------------------------------------------------------
    print("Example 2: Custom Seasonality Patterns")
    print("-" * 70)
    
    # Generate data with monthly pattern
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    t = np.arange(1000)
    
    # Trend + monthly seasonality + noise
    y = (10 + 0.01 * t +                              # Linear trend
         3 * np.sin(2 * np.pi * t / 30.0) +           # Monthly (30 days)
         2 * np.sin(2 * np.pi * t / 91.25) +          # Quarterly (~91 days)
         np.random.normal(0, 0.5, 1000))              # Noise
    
    df_custom = pd.DataFrame({'ds': dates, 'y': y})
    
    # Create model and add custom seasonalities
    model_custom = Seer(
        yearly_seasonality=False,    # Disable default yearly
        weekly_seasonality=False,    # Disable default weekly
        daily_seasonality=False
    )
    
    # Add custom seasonality patterns
    model_custom.add_seasonality(
        name='monthly',
        period=30.0,        # 30-day period
        fourier_order=5     # Flexibility of pattern (higher = more flexible)
    )
    
    model_custom.add_seasonality(
        name='quarterly',
        period=91.25,       # ~3 months
        fourier_order=8
    )
    
    print("Added custom seasonalities: monthly (30 days), quarterly (91.25 days)")
    print("Fitting model...")
    model_custom.fit(df_custom)
    
    future = model_custom.make_future_dataframe(periods=90)
    forecast = model_custom.predict(future)
    print(f"✓ Forecast with custom seasonality patterns")
    print()
    
    # -------------------------------------------------------------------------
    # Example 3: Holiday Effects
    # -------------------------------------------------------------------------
    print("Example 3: Holiday Effects")
    print("-" * 70)
    
    # Generate data with holiday spikes
    dates = pd.date_range('2019-01-01', periods=730, freq='D')
    t = np.arange(730)
    
    # Base trend + seasonality
    y = 50 + 0.02 * t + 5 * np.sin(2 * np.pi * t / 365.25)
    
    # Add holiday effects (Christmas spikes)
    for i, date in enumerate(dates):
        # Christmas effect (Dec 25)
        if date.month == 12 and date.day in [23, 24, 25, 26, 27]:
            y[i] += 20
        # New Year effect (Jan 1)
        if date.month == 1 and date.day in [1, 2]:
            y[i] += 10
    
    y += np.random.normal(0, 1, 730)
    df_holidays = pd.DataFrame({'ds': dates, 'y': y})
    
    # Create model with holiday effects
    model_holidays = Seer(
        yearly_seasonality=True,
        weekly_seasonality=False
    )
    
    # Add Christmas holiday
    model_holidays.add_holiday(
        name='christmas',
        dates=['2019-12-25', '2020-12-25'],
        lower_window=-2,   # Include 2 days before
        upper_window=2     # Include 2 days after
    )
    
    # Add New Year holiday
    model_holidays.add_holiday(
        name='new_year',
        dates=['2020-01-01', '2021-01-01'],
        lower_window=0,
        upper_window=1
    )
    
    print("Added holidays: Christmas (±2 days), New Year (+1 day)")
    print("Fitting model...")
    model_holidays.fit(df_holidays)
    
    future = model_holidays.make_future_dataframe(periods=180)
    forecast = model_holidays.predict(future)
    print(f"✓ Forecast with holiday effects modeled")
    print()
    
    # -------------------------------------------------------------------------
    # Example 4: Multiplicative Seasonality
    # -------------------------------------------------------------------------
    print("Example 4: Multiplicative Seasonality")
    print("-" * 70)
    
    # Generate data where seasonality scales with trend
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    t = np.arange(500)
    
    # Exponential trend
    trend = 10 * np.exp(0.003 * t)
    
    # Multiplicative seasonal pattern (scales with trend)
    seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * t / 365.25)
    
    y = trend * seasonal_factor + np.random.normal(0, 1, 500)
    df_mult = pd.DataFrame({'ds': dates, 'y': y})
    
    # Create model with multiplicative seasonality
    model_mult = Seer(
        seasonality_mode='multiplicative',  # Seasonal effects scale with trend
        yearly_seasonality=True,
        weekly_seasonality=False
    )
    
    print("Using multiplicative seasonality (scales with trend level)")
    print("Fitting model...")
    model_mult.fit(df_mult)
    
    future = model_mult.make_future_dataframe(periods=100)
    forecast = model_mult.predict(future)
    print(f"✓ Forecast with multiplicative seasonality")
    print(f"  Start value: {df_mult['y'].iloc[0]:.2f}")
    print(f"  End value: {df_mult['y'].iloc[-1]:.2f}")
    print(f"  Predicted future: {forecast['yhat'].iloc[-1]:.2f}")
    print()
    
    # -------------------------------------------------------------------------
    # Example 5: Tuning Changepoint Detection
    # -------------------------------------------------------------------------
    print("Example 5: Changepoint Tuning")
    print("-" * 70)
    
    # Generate data with clear trend changes
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    t = np.arange(365)
    
    # Piecewise linear trend with a changepoint at day 200
    y = np.where(t < 200, 10 + 0.05 * t, 10 + 0.05 * 200 + 0.15 * (t - 200))
    y += np.random.normal(0, 1, 365)
    df_change = pd.DataFrame({'ds': dates, 'y': y})
    
    # Model with flexible changepoints
    model_flexible = Seer(
        n_changepoints=30,              # More potential changepoints
        changepoint_prior_scale=0.5,    # Less regularization (more flexible)
        yearly_seasonality=False,
        weekly_seasonality=False
    )
    
    print("Fitting model with flexible changepoint detection...")
    model_flexible.fit(df_change)
    
    # Model with rigid changepoints
    model_rigid = Seer(
        n_changepoints=5,                # Fewer changepoints
        changepoint_prior_scale=0.001,   # Strong regularization (less flexible)
        yearly_seasonality=False,
        weekly_seasonality=False
    )
    
    print("Fitting model with rigid changepoint detection...")
    model_rigid.fit(df_change)
    
    future = model_flexible.make_future_dataframe(periods=60)
    forecast_flex = model_flexible.predict(future)
    forecast_rigid = model_rigid.predict(future)
    
    print(f"✓ Flexible model trend change detected near day 200")
    print(f"✓ Rigid model fits smoother trend")
    print()
    
    print("=" * 70)
    print("Advanced examples completed!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("  1. Use logistic growth for saturating trends (requires 'cap' column)")
    print("  2. Add custom seasonality for any periodic pattern")
    print("  3. Model holidays with flexible windows")
    print("  4. Use multiplicative mode when seasonality scales with trend")
    print("  5. Tune changepoints for better trend detection")
    print()


if __name__ == '__main__':
    main()
