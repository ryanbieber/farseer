#!/usr/bin/env python3
"""
M7 Integration Tests: Multiple Frequency Support
Tests for hourly, weekly, monthly, and yearly frequencies in make_future_dataframe
"""

import pandas as pd
import numpy as np
from seer import Seer


def test_hourly_frequency():
    """Test hourly frequency support"""
    print("Testing hourly frequency...")
    
    # Create hourly data
    df = pd.DataFrame({
        'ds': pd.date_range('2020-01-01', periods=72, freq='H'),
        'y': np.linspace(10, 20, 72) + np.random.normal(0, 0.1, 72)
    })
    df['ds'] = df['ds'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    m = Seer(
        growth='linear',
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    m.fit(df)
    
    # Make hourly predictions
    future = m.make_future_dataframe(periods=24, freq='H')
    forecast = m.predict(future)
    
    assert len(forecast) == 96  # 72 history + 24 future
    assert 'yhat' in forecast.columns
    print("✅ Hourly frequency test passed")


def test_daily_frequency():
    """Test daily frequency (already tested, but confirm still works)"""
    print("Testing daily frequency...")
    
    df = pd.DataFrame({
        'ds': pd.date_range('2020-01-01', periods=100, freq='D'),
        'y': np.linspace(10, 20, 100) + np.random.normal(0, 0.5, 100)
    })
    
    m = Seer(growth='linear')
    m.fit(df)
    
    future = m.make_future_dataframe(periods=30, freq='D')
    forecast = m.predict(future)
    
    assert len(forecast) == 130  # 100 history + 30 future
    print("✅ Daily frequency test passed")


def test_weekly_frequency():
    """Test weekly frequency support"""
    print("Testing weekly frequency...")
    
    # Create weekly data
    df = pd.DataFrame({
        'ds': pd.date_range('2020-01-01', periods=52, freq='W'),
        'y': np.linspace(10, 20, 52) + np.random.normal(0, 0.5, 52)
    })
    
    m = Seer(
        growth='linear',
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    m.fit(df)
    
    # Make weekly predictions
    future = m.make_future_dataframe(periods=12, freq='W')
    forecast = m.predict(future)
    
    assert len(forecast) == 64  # 52 history + 12 future
    assert 'yhat' in forecast.columns
    print("✅ Weekly frequency test passed")


def test_monthly_frequency():
    """Test monthly frequency support (approximate - 30 days)"""
    print("Testing monthly frequency...")
    
    # Create monthly-ish data (30 day intervals)
    dates = pd.date_range('2020-01-01', periods=24, freq='30D')
    df = pd.DataFrame({
        'ds': dates,
        'y': np.linspace(10, 20, 24) + np.random.normal(0, 0.5, 24)
    })
    
    m = Seer(
        growth='linear',
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    m.fit(df)
    
    # Make monthly predictions
    future = m.make_future_dataframe(periods=6, freq='M')
    forecast = m.predict(future)
    
    assert len(forecast) == 30  # 24 history + 6 future
    assert 'yhat' in forecast.columns
    print("✅ Monthly frequency test passed")


def test_yearly_frequency():
    """Test yearly frequency support (approximate - 365 days)"""
    print("Testing yearly frequency...")
    
    # Create yearly data
    dates = pd.date_range('2010-01-01', periods=10, freq='365D')
    df = pd.DataFrame({
        'ds': dates,
        'y': np.linspace(10, 20, 10) + np.random.normal(0, 0.5, 10)
    })
    
    m = Seer(
        growth='linear',
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    m.fit(df)
    
    # Make yearly predictions
    future = m.make_future_dataframe(periods=3, freq='Y')
    forecast = m.predict(future)
    
    assert len(forecast) == 13  # 10 history + 3 future
    assert 'yhat' in forecast.columns
    print("✅ Yearly frequency test passed")


def test_mixed_frequency_prediction():
    """Test that we can fit on one frequency and predict on another"""
    print("Testing mixed frequency prediction...")
    
    # Fit on daily data
    df = pd.DataFrame({
        'ds': pd.date_range('2020-01-01', periods=100, freq='D'),
        'y': np.linspace(10, 20, 100) + np.random.normal(0, 0.5, 100)
    })
    
    m = Seer(growth='linear', yearly_seasonality=False, weekly_seasonality=False)
    m.fit(df)
    
    # Predict on weekly frequency
    future = m.make_future_dataframe(periods=4, freq='W', include_history=False)
    forecast = m.predict(future)
    
    assert len(forecast) == 4  # Just the future
    assert 'yhat' in forecast.columns
    print("✅ Mixed frequency prediction test passed")


def test_fallback_to_daily():
    """Test that unknown frequencies fallback to daily"""
    print("Testing fallback to daily for unknown frequency...")
    
    df = pd.DataFrame({
        'ds': pd.date_range('2020-01-01', periods=50, freq='D'),
        'y': np.linspace(10, 20, 50) + np.random.normal(0, 0.5, 50)
    })
    
    m = Seer(growth='linear', yearly_seasonality=False, weekly_seasonality=False)
    m.fit(df)
    
    # Use an unknown frequency - should fallback to daily
    future = m.make_future_dataframe(periods=10, freq='UNKNOWN')
    forecast = m.predict(future)
    
    assert len(forecast) == 60  # 50 history + 10 future (daily)
    print("✅ Fallback to daily test passed")


if __name__ == '__main__':
    print("=" * 60)
    print("M7 INTEGRATION TESTS: Multiple Frequency Support")
    print("=" * 60)
    
    test_hourly_frequency()
    test_daily_frequency()
    test_weekly_frequency()
    test_monthly_frequency()
    test_yearly_frequency()
    test_mixed_frequency_prediction()
    test_fallback_to_daily()
    
    print("\n" + "=" * 60)
    print("✅ All M7 frequency tests passed!")
    print("=" * 60)
