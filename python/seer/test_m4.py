#!/usr/bin/env python3
"""
Test script for M4 features: Holidays
"""

import pandas as pd
import numpy as np
from seer import Seer

def test_custom_holidays():
    """Test adding custom holidays"""
    print("\n" + "="*60)
    print("TEST: Custom Holidays")
    print("="*60)
    
    # Create data with spikes on holidays
    np.random.seed(42)
    dates = pd.date_range('2020-12-01', periods=31, freq='D')
    
    # Base trend with spikes on Christmas (Dec 25) and New Year (Jan 1)
    y = []
    for i, date in enumerate(dates):
        base = 100 + i * 0.5
        if date.month == 12 and date.day == 25:  # Christmas
            y.append(base + 50)
        elif date.month == 1 and date.day == 1:  # New Year
            y.append(base + 40)
        else:
            y.append(base + np.random.randn() * 2)
    
    df = pd.DataFrame({'ds': dates, 'y': y})
    
    # Create model and add holidays
    m = Seer(yearly_seasonality=False, weekly_seasonality=False)
    
    # Add Christmas
    m.add_holidays(
        name='christmas',
        dates=['2020-12-25'],
        lower_window=None,
        upper_window=None,
        prior_scale=None,
        mode=None
    )
    
    print(f"✓ Added Christmas holiday")
    
    # Fit and predict
    m.fit(df)
    forecast = m.predict(df)
    
    print(f"✓ Forecast shape: {forecast.shape}")
    print(f"✓ Columns: {list(forecast.columns)}")
    
    # Check that Christmas day has elevated prediction
    christmas_idx = 24  # Dec 25 is index 24 (0-based from Dec 1)
    christmas_pred = forecast.iloc[christmas_idx]['yhat']
    avg_pred = forecast['yhat'].mean()
    
    print(f"✓ Christmas prediction: {christmas_pred:.2f}")
    print(f"✓ Average prediction: {avg_pred:.2f}")
    
    # Christmas should be elevated
    if christmas_pred > avg_pred + 10:
        print("✅ PASSED: Christmas prediction is elevated")
        return True
    else:
        print(f"❌ FAILED: Christmas not elevated enough: {christmas_pred:.2f} vs avg {avg_pred:.2f}")
        return False


def test_holiday_windows():
    """Test holidays with windows"""
    print("\n" + "="*60)
    print("TEST: Holiday Windows")
    print("="*60)
    
    # Create data where Dec 24-26 are all elevated (Christmas ±1 day)
    dates = pd.date_range('2020-12-20', periods=10, freq='D')
    y = []
    for i, date in enumerate(dates):
        base = 100.0
        # Elevated for Dec 24, 25, 26
        if 24 <= date.day <= 26:
            y.append(base + 30)
        else:
            y.append(base)
    
    df = pd.DataFrame({'ds': dates, 'y': y})
    
    m = Seer(yearly_seasonality=False, weekly_seasonality=False)
    
    # Add Christmas with ±1 day window
    m.add_holidays(
        name='christmas',
        dates=['2020-12-25'],
        lower_window=-1,  # Day before
        upper_window=1,   # Day after
        prior_scale=None,
        mode=None
    )
    
    print(f"✓ Added Christmas with ±1 day window")
    
    m.fit(df)
    forecast = m.predict(df)
    
    # Days 4, 5, 6 should be elevated (Dec 24, 25, 26)
    window_preds = forecast.iloc[4:7]['yhat'].values
    non_window_preds = forecast.iloc[:4]['yhat'].values
    
    window_avg = window_preds.mean()
    non_window_avg = non_window_preds.mean()
    
    print(f"✓ Window days average: {window_avg:.2f}")
    print(f"✓ Non-window days average: {non_window_avg:.2f}")
    
    if window_avg > non_window_avg + 5:
        print("✅ PASSED: Holiday window effect captured")
        return True
    else:
        print(f"❌ FAILED: Window not elevated: {window_avg:.2f} vs {non_window_avg:.2f}")
        return False


def test_multiple_holidays():
    """Test multiple holidays"""
    print("\n" + "="*60)
    print("TEST: Multiple Holidays")
    print("="*60)
    
    # Create data with multiple holiday spikes
    dates = pd.date_range('2020-01-01', periods=60, freq='D')
    y = []
    for i, date in enumerate(dates):
        base = 50 + i * 0.1
        # New Year (Jan 1), MLK Day (Jan 20), Valentine's (Feb 14)
        if (date.month == 1 and date.day == 1) or \
           (date.month == 1 and date.day == 20) or \
           (date.month == 2 and date.day == 14):
            y.append(base + 25)
        else:
            y.append(base + np.random.randn() * 1)
    
    df = pd.DataFrame({'ds': dates, 'y': y})
    
    m = Seer(yearly_seasonality=False, weekly_seasonality=False)
    
    # Add all three holidays
    m.add_holidays('new_year', ['2020-01-01'])
    m.add_holidays('mlk_day', ['2020-01-20'])
    m.add_holidays('valentines', ['2020-02-14'])
    
    print(f"✓ Added 3 holidays")
    
    m.fit(df)
    forecast = m.predict(df)
    
    # Check all three holidays are elevated
    new_year_idx = 0
    mlk_idx = 19
    valentines_idx = 44
    
    new_year_pred = forecast.iloc[new_year_idx]['yhat']
    mlk_pred = forecast.iloc[mlk_idx]['yhat']
    val_pred = forecast.iloc[valentines_idx]['yhat']
    avg_pred = forecast['yhat'].mean()
    
    print(f"✓ New Year prediction: {new_year_pred:.2f}")
    print(f"✓ MLK Day prediction: {mlk_pred:.2f}")
    print(f"✓ Valentine's prediction: {val_pred:.2f}")
    print(f"✓ Average prediction: {avg_pred:.2f}")
    
    all_elevated = (new_year_pred > avg_pred and 
                   mlk_pred > avg_pred and 
                   val_pred > avg_pred)
    
    if all_elevated:
        print("✅ PASSED: All holidays captured")
        return True
    else:
        print("❌ FAILED: Not all holidays elevated")
        return False


def test_future_predictions_with_holidays():
    """Test that holidays work in future predictions"""
    print("\n" + "="*60)
    print("TEST: Future Predictions with Holidays")
    print("="*60)
    
    # Train on data with Christmas 2020
    dates = pd.date_range('2020-12-01', periods=31, freq='D')
    y = []
    for i, date in enumerate(dates):
        base = 100 + i * 0.5
        if date.month == 12 and date.day == 25:
            y.append(base + 40)
        else:
            y.append(base)
    
    df = pd.DataFrame({'ds': dates, 'y': y})
    
    m = Seer(yearly_seasonality=False, weekly_seasonality=False)
    
    # Add both Christmas 2020 and 2021
    m.add_holidays('christmas', ['2020-12-25', '2021-12-25'])
    
    m.fit(df)
    
    # Predict into 2021
    future = m.make_future_dataframe(periods=365, include_history=False)
    forecast = m.predict(future)
    
    # Find Christmas 2021 in the forecast
    forecast['date'] = pd.to_datetime(forecast['ds'])
    christmas_2021 = forecast[(forecast['date'].dt.month == 12) & 
                              (forecast['date'].dt.day == 25)]
    
    if len(christmas_2021) > 0:
        christmas_pred = christmas_2021.iloc[0]['yhat']
        avg_pred = forecast['yhat'].mean()
        
        print(f"✓ Christmas 2021 prediction: {christmas_pred:.2f}")
        print(f"✓ Average prediction: {avg_pred:.2f}")
        
        if christmas_pred > avg_pred + 10:
            print("✅ PASSED: Future holiday effect works")
            return True
        else:
            print(f"❌ FAILED: Future Christmas not elevated: {christmas_pred:.2f} vs {avg_pred:.2f}")
            return False
    else:
        print("❌ FAILED: Could not find Christmas 2021 in forecast")
        return False


def main():
    print("\n" + "🎄"*30)
    print("M4 HOLIDAY FEATURE TEST SUITE")
    print("🎄"*30)
    
    results = []
    
    results.append(("Custom Holidays", test_custom_holidays()))
    results.append(("Holiday Windows", test_holiday_windows()))
    results.append(("Multiple Holidays", test_multiple_holidays()))
    results.append(("Future Predictions", test_future_predictions_with_holidays()))
    
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL M4 TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
