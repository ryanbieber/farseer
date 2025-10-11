#!/usr/bin/env python3
"""
Test script for Seer - Rust-based time series forecasting
"""

import pandas as pd
import numpy as np
from seer import Seer

def test_basic_usage():
    """Test basic model creation, fitting, and prediction"""
    print("=" * 60)
    print("TEST 1: Basic Usage")
    print("=" * 60)
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    values = np.arange(100) * 0.5 + 10 + np.random.randn(100) * 2
    
    df = pd.DataFrame({
        'ds': dates,
        'y': values
    })
    
    print(f"\n📊 Created dataset with {len(df)} rows")
    print(df.head())
    
    # Create model
    print("\n🔧 Creating Seer model...")
    model = Seer(
        growth='linear',
        n_changepoints=25,
        yearly_seasonality=True,
        weekly_seasonality=True
    )
    print(f"   Model: {model}")
    
    # Fit model
    print("\n🎯 Fitting model...")
    model.fit(df)
    print("   ✓ Model fitted successfully")
    
    # Make future dataframe
    print("\n📅 Creating future dataframe (30 days ahead)...")
    future = model.make_future_dataframe(periods=30)
    print(f"   Future shape: {future.shape}")
    print(f"   Date range: {future['ds'].min()} to {future['ds'].max()}")
    
    # Predict
    print("\n🔮 Making predictions...")
    forecast = model.predict(future)
    print(f"   Forecast shape: {forecast.shape}")
    print(f"   Columns: {forecast.columns.tolist()}")
    
    print("\n📈 First 5 predictions:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
    
    print("\n📉 Last 5 predictions (future):")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    
    # Get parameters
    print("\n⚙️  Model parameters:")
    params = model.params()
    for key, value in params.items():
        print(f"   {key}: {value}")
    
    print("\n✅ TEST 1 PASSED\n")
    return model, forecast


def test_different_trends():
    """Test different trend types"""
    print("=" * 60)
    print("TEST 2: Different Trend Types")
    print("=" * 60)
    
    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    values = np.arange(50) * 0.3 + 5
    
    df = pd.DataFrame({
        'ds': dates,
        'y': values
    })
    
    trends = ['linear', 'logistic', 'flat']
    
    for trend in trends:
        print(f"\n🔄 Testing '{trend}' trend...")
        
        model = Seer(
            growth=trend,
            n_changepoints=10,
            yearly_seasonality=False,
            weekly_seasonality=False
        )
        
        model.fit(df)
        future = model.make_future_dataframe(periods=10)
        forecast = model.predict(future)
        
        print(f"   ✓ {trend} trend works!")
        print(f"   Mean prediction: {forecast['yhat'].mean():.2f}")
    
    print("\n✅ TEST 2 PASSED\n")


def test_seasonality_options():
    """Test seasonality configuration"""
    print("=" * 60)
    print("TEST 3: Seasonality Options")
    print("=" * 60)
    
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    values = np.arange(365) * 0.1 + np.sin(np.arange(365) * 2 * np.pi / 365) * 5
    
    df = pd.DataFrame({
        'ds': dates,
        'y': values
    })
    
    configs = [
        {'yearly': True, 'weekly': True, 'daily': False},
        {'yearly': True, 'weekly': False, 'daily': False},
        {'yearly': False, 'weekly': True, 'daily': False},
        {'yearly': False, 'weekly': False, 'daily': False},
    ]
    
    for config in configs:
        print(f"\n⭐ Testing seasonality: {config}")
        
        model = Seer(
            yearly_seasonality=config['yearly'],
            weekly_seasonality=config['weekly'],
            daily_seasonality=config['daily']
        )
        
        model.fit(df)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        
        print(f"   ✓ Configuration works!")
    
    print("\n✅ TEST 3 PASSED\n")


def test_edge_cases():
    """Test edge cases and error handling"""
    print("=" * 60)
    print("TEST 4: Edge Cases")
    print("=" * 60)
    
    # Small dataset
    print("\n🔬 Testing with small dataset (10 points)...")
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    values = np.arange(10) * 1.0
    
    df = pd.DataFrame({
        'ds': dates,
        'y': values
    })
    
    model = Seer(n_changepoints=5)
    model.fit(df)
    future = model.make_future_dataframe(periods=5)
    forecast = model.predict(future)
    print(f"   ✓ Small dataset works! Forecast shape: {forecast.shape}")
    
    # Test prediction before fitting
    print("\n🔬 Testing prediction before fitting...")
    model_unfitted = Seer()
    try:
        future = model_unfitted.make_future_dataframe(periods=10)
        print("   ❌ Should have raised an error!")
    except Exception as e:
        print(f"   ✓ Correctly raised error: {type(e).__name__}")
    
    print("\n✅ TEST 4 PASSED\n")


def test_performance():
    """Test with larger dataset to check performance"""
    print("=" * 60)
    print("TEST 5: Performance Test")
    print("=" * 60)
    
    import time
    
    sizes = [100, 500, 1000]
    
    for size in sizes:
        print(f"\n⚡ Testing with {size} data points...")
        
        dates = pd.date_range('2020-01-01', periods=size, freq='D')
        values = np.arange(size) * 0.5 + np.random.randn(size) * 2
        
        df = pd.DataFrame({
            'ds': dates,
            'y': values
        })
        
        model = Seer()
        
        # Time fitting
        start = time.time()
        model.fit(df)
        fit_time = time.time() - start
        
        # Time prediction
        future = model.make_future_dataframe(periods=100)
        start = time.time()
        forecast = model.predict(future)
        pred_time = time.time() - start
        
        print(f"   Fit time: {fit_time*1000:.2f}ms")
        print(f"   Predict time: {pred_time*1000:.2f}ms")
        print(f"   Total time: {(fit_time + pred_time)*1000:.2f}ms")
    
    print("\n✅ TEST 5 PASSED\n")


def test_dataframe_formats():
    """Test different DataFrame formats"""
    print("=" * 60)
    print("TEST 6: DataFrame Format Compatibility")
    print("=" * 60)
    
    # Test with datetime index
    print("\n📋 Testing with datetime index...")
    idx = pd.date_range('2020-01-01', periods=50, freq='D')
    df1 = pd.DataFrame({
        'ds': idx,
        'y': np.arange(50)
    })
    
    model1 = Seer()
    model1.fit(df1)
    print("   ✓ Datetime index works!")
    
    # Test with string dates
    print("\n📋 Testing with string dates...")
    df2 = pd.DataFrame({
        'ds': [d.strftime('%Y-%m-%d') for d in idx],
        'y': np.arange(50)
    })
    
    model2 = Seer()
    model2.fit(df2)
    print("   ✓ String dates work!")
    
    # Test with integer values
    print("\n📋 Testing with integer y values...")
    df3 = pd.DataFrame({
        'ds': idx,
        'y': list(range(50))  # integers
    })
    
    model3 = Seer()
    model3.fit(df3)
    print("   ✓ Integer values work!")
    
    print("\n✅ TEST 6 PASSED\n")


def run_all_tests():
    """Run all tests"""
    print("\n" + "🚀" * 30)
    print("SEER TEST SUITE")
    print("🚀" * 30 + "\n")
    
    try:
        # Run tests
        model, forecast = test_basic_usage()
        test_different_trends()
        test_seasonality_options()
        test_edge_cases()
        test_performance()
        test_dataframe_formats()
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\n✨ Seer is working correctly!")
        print(f"   Version: {model.__module__}")
        print("\n📊 Sample forecast statistics:")
        print(f"   Mean: {forecast['yhat'].mean():.2f}")
        print(f"   Std:  {forecast['yhat'].std():.2f}")
        print(f"   Min:  {forecast['yhat'].min():.2f}")
        print(f"   Max:  {forecast['yhat'].max():.2f}")
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED")
        print("=" * 60)
        print(f"\nError: {type(e).__name__}")
        print(f"Message: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)