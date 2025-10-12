#!/usr/bin/env python3
"""
Multiple Frequencies Example

Demonstrates forecasting at different time frequencies:
- Hourly data
- Daily data
- Weekly data
- Monthly data
- Yearly data

Shows how to handle different data granularities and make
appropriate predictions for each frequency.
"""

import pandas as pd
import numpy as np
from seer import Seer

np.random.seed(42)


def generate_hourly_data(periods=168):
    """Generate hourly data (168 hours = 1 week)"""
    dates = pd.date_range('2023-01-01', periods=periods, freq='h')
    t = np.arange(periods)
    
    # Hourly pattern: daily cycle + trend
    daily_cycle = 5 * np.sin(2 * np.pi * t / 24)  # 24-hour cycle
    trend = 0.01 * t
    noise = np.random.normal(0, 0.5, periods)
    
    y = 50 + trend + daily_cycle + noise
    return pd.DataFrame({'ds': dates, 'y': y})


def generate_daily_data(periods=365):
    """Generate daily data (1 year)"""
    dates = pd.date_range('2023-01-01', periods=periods, freq='D')
    t = np.arange(periods)
    
    # Daily pattern: weekly + yearly
    weekly = 3 * np.sin(2 * np.pi * t / 7)
    yearly = 8 * np.sin(2 * np.pi * t / 365.25)
    trend = 0.05 * t
    noise = np.random.normal(0, 1, periods)
    
    y = 100 + trend + weekly + yearly + noise
    return pd.DataFrame({'ds': dates, 'y': y})


def generate_weekly_data(periods=104):
    """Generate weekly data (2 years = 104 weeks)"""
    dates = pd.date_range('2020-01-01', periods=periods, freq='W')
    t = np.arange(periods)
    
    # Weekly pattern: yearly (52 weeks)
    yearly = 15 * np.sin(2 * np.pi * t / 52)
    trend = 0.2 * t
    noise = np.random.normal(0, 2, periods)
    
    y = 200 + trend + yearly + noise
    return pd.DataFrame({'ds': dates, 'y': y})


def generate_monthly_data(periods=60):
    """Generate monthly data (5 years = 60 months)"""
    dates = pd.date_range('2019-01-01', periods=periods, freq='MS')
    t = np.arange(periods)
    
    # Monthly pattern: yearly (12 months)
    yearly = 20 * np.sin(2 * np.pi * t / 12)
    trend = 0.5 * t
    noise = np.random.normal(0, 3, periods)
    
    y = 500 + trend + yearly + noise
    return pd.DataFrame({'ds': dates, 'y': y})


def main():
    print("=" * 70)
    print("Seer: Multiple Frequencies Example")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # Example 1: Hourly Data
    # -------------------------------------------------------------------------
    print("Example 1: Hourly Data")
    print("-" * 70)
    
    df_hourly = generate_hourly_data(periods=168)  # 1 week
    print(f"Generated {len(df_hourly)} hourly observations")
    print(f"Date range: {df_hourly['ds'].min()} to {df_hourly['ds'].max()}")
    
    # For hourly data, disable yearly/weekly seasonality
    # Add daily seasonality instead
    model_hourly = Seer(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=True,  # 24-hour pattern
        n_changepoints=10
    )
    
    print("Fitting hourly model with daily seasonality...")
    model_hourly.fit(df_hourly)
    
    # Forecast next 24 hours
    future = model_hourly.make_future_dataframe(periods=24, freq='H')
    forecast = model_hourly.predict(future)
    
    print(f"✓ Forecast next 24 hours")
    print(f"  Current value: {df_hourly['y'].iloc[-1]:.2f}")
    print(f"  24h prediction: {forecast['yhat'].iloc[-1]:.2f}")
    print()
    
    # -------------------------------------------------------------------------
    # Example 2: Daily Data
    # -------------------------------------------------------------------------
    print("Example 2: Daily Data")
    print("-" * 70)
    
    df_daily = generate_daily_data(periods=365)
    print(f"Generated {len(df_daily)} daily observations")
    print(f"Date range: {df_daily['ds'].min()} to {df_daily['ds'].max()}")
    
    # Standard daily forecasting with yearly and weekly patterns
    model_daily = Seer(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    
    print("Fitting daily model...")
    model_daily.fit(df_daily)
    
    # Forecast next 30 days
    future = model_daily.make_future_dataframe(periods=30, freq='D')
    forecast = model_daily.predict(future)
    
    print(f"✓ Forecast next 30 days")
    print(f"  Last historical: {df_daily['y'].iloc[-1]:.2f}")
    print(f"  30-day forecast: {forecast['yhat'].iloc[-1]:.2f}")
    print()
    
    # -------------------------------------------------------------------------
    # Example 3: Weekly Data
    # -------------------------------------------------------------------------
    print("Example 3: Weekly Data")
    print("-" * 70)
    
    df_weekly = generate_weekly_data(periods=104)  # 2 years
    print(f"Generated {len(df_weekly)} weekly observations")
    print(f"Date range: {df_weekly['ds'].min()} to {df_weekly['ds'].max()}")
    
    # For weekly data, use yearly seasonality (52 weeks)
    model_weekly = Seer(
        yearly_seasonality=True,   # 52-week pattern
        weekly_seasonality=False,  # Not applicable for weekly data
        daily_seasonality=False
    )
    
    print("Fitting weekly model...")
    model_weekly.fit(df_weekly)
    
    # Forecast next 12 weeks (3 months)
    future = model_weekly.make_future_dataframe(periods=12, freq='W')
    forecast = model_weekly.predict(future)
    
    print(f"✓ Forecast next 12 weeks")
    print(f"  Last week: {df_weekly['y'].iloc[-1]:.2f}")
    print(f"  12 weeks out: {forecast['yhat'].iloc[-1]:.2f}")
    print()
    
    # -------------------------------------------------------------------------
    # Example 4: Monthly Data
    # -------------------------------------------------------------------------
    print("Example 4: Monthly Data")
    print("-" * 70)
    
    df_monthly = generate_monthly_data(periods=60)  # 5 years
    print(f"Generated {len(df_monthly)} monthly observations")
    print(f"Date range: {df_monthly['ds'].min()} to {df_monthly['ds'].max()}")
    
    # For monthly data, yearly pattern = 12 months
    # Can also add custom seasonality
    model_monthly = Seer(
        yearly_seasonality=True,   # 12-month pattern
        weekly_seasonality=False,
        daily_seasonality=False
    )
    
    print("Fitting monthly model...")
    model_monthly.fit(df_monthly)
    
    # Forecast next 6 months
    future = model_monthly.make_future_dataframe(periods=6, freq='M')
    forecast = model_monthly.predict(future)
    
    print(f"✓ Forecast next 6 months")
    print(f"  Last month: {df_monthly['y'].iloc[-1]:.2f}")
    print(f"  6 months out: {forecast['yhat'].iloc[-1]:.2f}")
    print()
    
    # -------------------------------------------------------------------------
    # Example 5: Custom Frequency
    # -------------------------------------------------------------------------
    print("Example 5: Custom Frequency (Business Days)")
    print("-" * 70)
    
    # Business days only (Monday-Friday)
    dates = pd.bdate_range('2023-01-01', periods=252, freq='B')  # ~1 year
    t = np.arange(252)
    
    # Business day pattern
    y = 100 + 0.1 * t + 5 * np.sin(2 * np.pi * t / 5) + np.random.normal(0, 1, 252)
    df_business = pd.DataFrame({'ds': dates, 'y': y})
    
    print(f"Generated {len(df_business)} business day observations")
    print(f"Date range: {df_business['ds'].min()} to {df_business['ds'].max()}")
    
    # Weekly pattern for business days (5 days per week)
    model_business = Seer(
        yearly_seasonality=True,
        weekly_seasonality=False,  # Not standard weekly
        daily_seasonality=False
    )
    
    # Add custom 5-day (work week) seasonality
    model_business.add_seasonality(
        name='workweek',
        period=5.0,
        fourier_order=3
    )
    
    print("Fitting business day model with custom work-week seasonality...")
    model_business.fit(df_business)
    
    # Forecast next 20 business days
    future_dates = pd.bdate_range(
        start=df_business['ds'].max() + pd.Timedelta(days=1),
        periods=20,
        freq='B'
    )
    future = pd.DataFrame({'ds': future_dates})
    forecast = model_business.predict(future)
    
    print(f"✓ Forecast next 20 business days")
    print(f"  Last day: {df_business['y'].iloc[-1]:.2f}")
    print(f"  20 days out: {forecast['yhat'].iloc[-1]:.2f}")
    print()
    
    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    print("Summary: Frequency Recommendations")
    print("-" * 70)
    print()
    print(f"{'Frequency':<15} {'Seasonality':<25} {'Forecast Horizon':<20}")
    print("-" * 70)
    print(f"{'Hourly':<15} {'Daily (24h)':<25} {'Hours to days':<20}")
    print(f"{'Daily':<15} {'Weekly, Yearly':<25} {'Days to months':<20}")
    print(f"{'Weekly':<15} {'Yearly (52w)':<25} {'Weeks to quarters':<20}")
    print(f"{'Monthly':<15} {'Yearly (12m)':<25} {'Months to years':<20}")
    print(f"{'Business Days':<15} {'Custom (5d), Yearly':<25} {'Days to months':<20}")
    print()
    
    # -------------------------------------------------------------------------
    # Frequency codes reference
    # -------------------------------------------------------------------------
    print("Pandas Frequency Codes Reference")
    print("-" * 70)
    print()
    print("Common frequencies for make_future_dataframe():")
    print()
    print("  'H'   - Hourly")
    print("  'D'   - Daily (calendar day)")
    print("  'B'   - Business day (Monday-Friday)")
    print("  'W'   - Weekly (Sunday)")
    print("  'W-MON' - Weekly (Monday)")
    print("  'MS'  - Month start")
    print("  'M'   - Month end")
    print("  'Q'   - Quarter end")
    print("  'Y'   - Year end")
    print()
    print("Usage:")
    print("  future = model.make_future_dataframe(periods=30, freq='D')")
    print("  future = model.make_future_dataframe(periods=24, freq='H')")
    print("  future = model.make_future_dataframe(periods=12, freq='W')")
    print()
    
    # -------------------------------------------------------------------------
    # Tips
    # -------------------------------------------------------------------------
    print("Tips for Different Frequencies")
    print("-" * 70)
    print()
    print("1. Hourly Data:")
    print("   • Enable daily_seasonality for 24-hour patterns")
    print("   • Consider sub-daily patterns (morning/evening peaks)")
    print("   • May need more data to capture weekly patterns")
    print()
    print("2. Daily Data:")
    print("   • Standard weekly and yearly seasonality work well")
    print("   • Consider holidays and special events")
    print("   • At least 2-3 cycles (weeks/years) recommended")
    print()
    print("3. Weekly Data:")
    print("   • Use yearly_seasonality for 52-week patterns")
    print("   • Need at least 2 years for reliable yearly patterns")
    print("   • Weekly pattern not applicable (you're already at weekly level)")
    print()
    print("4. Monthly Data:")
    print("   • Yearly pattern captures 12-month seasonality")
    print("   • Need 3+ years for stable estimates")
    print("   • Consider custom quarterly patterns")
    print()
    print("5. Business Days:")
    print("   • Add custom seasonality for work-week patterns")
    print("   • Handle holidays separately")
    print("   • Mind the gap between Friday and Monday")
    print()
    
    print("=" * 70)
    print("Multiple frequencies example completed!")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()
