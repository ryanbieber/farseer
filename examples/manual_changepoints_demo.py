#!/usr/bin/env python3
"""
Demo: Manual Changepoints in Seer
Shows how to manually specify changepoint locations for trend changes.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from farseer import Farseer

# Set random seed for reproducibility
np.random.seed(42)

# Create sample data with known changepoints
n = 365 * 3  # 3 years of daily data
dates = pd.date_range("2020-01-01", periods=n, freq="D")

# Create trend with changepoints at specific dates
# - 2020-01-01 to 2020-12-31: slope = 0.5
# - 2021-01-01 to 2021-12-31: slope = 1.5 (policy change!)
# - 2022-01-01 to 2022-12-31: slope = 0.3 (market saturation)

y = []
base = 100
for i, date in enumerate(dates):
    if date < pd.Timestamp("2021-01-01"):
        # First year: moderate growth
        slope = 0.5
        y_val = base + slope * i
    elif date < pd.Timestamp("2022-01-01"):
        # Second year: rapid growth (365 days in)
        days_since_cp1 = (date - pd.Timestamp("2021-01-01")).days
        slope = 1.5
        y_val = base + 0.5 * 365 + slope * days_since_cp1
    else:
        # Third year: slow growth (730 days in)
        days_since_cp2 = (date - pd.Timestamp("2022-01-01")).days
        slope = 0.3
        y_val = base + 0.5 * 365 + 1.5 * 365 + slope * days_since_cp2

    # Add some seasonality and noise
    yearly_season = 10 * np.sin(2 * np.pi * i / 365.25)
    noise = np.random.randn() * 5
    y.append(y_val + yearly_season + noise)

# Create DataFrame
df = pd.DataFrame({"ds": dates, "y": y})

print("=" * 80)
print("DEMO: Manual Changepoints in Seer")
print("=" * 80)
print("\nData: 3 years with known trend changes:")
print("  - 2020: Moderate growth (slope = 0.5)")
print("  - 2021: Rapid growth (slope = 1.5) [Policy change]")
print("  - 2022: Slow growth (slope = 0.3) [Market saturation]")

# Split into train/test
train_size = int(n * 0.85)
train = df[:train_size].copy()
test = df[train_size:].copy()

print(
    f"\nTrain: {len(train)} days ({train['ds'].min().date()} to {train['ds'].max().date()})"
)
print(
    f"Test:  {len(test)} days ({test['ds'].min().date()} to {test['ds'].max().date()})"
)

# Model 1: Automatic changepoints (default)
print("\n" + "=" * 80)
print("MODEL 1: Automatic Changepoint Detection")
print("=" * 80)

m1 = Farseer(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)

print("\nConfiguration:")
print("  - n_changepoints: 25 (automatic)")
print("  - changepoint_range: 0.8 (first 80% of history)")

m1.fit(train)
forecast1 = m1.predict(test)

mae1 = mean_absolute_error(test["y"], forecast1["yhat"])
rmse1 = np.sqrt(mean_squared_error(test["y"], forecast1["yhat"]))

print("\nTest Set Performance:")
print(f"  MAE:  {mae1:.2f}")
print(f"  RMSE: {rmse1:.2f}")

# Model 2: Manual changepoints at known dates
print("\n" + "=" * 80)
print("MODEL 2: Manual Changepoints (Known Dates)")
print("=" * 80)

# Specify changepoints at the exact dates we know trend changed
manual_changepoints = ["2021-01-01", "2022-01-01"]

m2 = Farseer(
    changepoints=manual_changepoints,
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
)

print("\nConfiguration:")
print(f"  - Manual changepoints: {manual_changepoints}")
print(f"  - n_changepoints: {len(manual_changepoints)}")

m2.fit(train)
forecast2 = m2.predict(test)

mae2 = mean_absolute_error(test["y"], forecast2["yhat"])
rmse2 = np.sqrt(mean_squared_error(test["y"], forecast2["yhat"]))

print("\nTest Set Performance:")
print(f"  MAE:  {mae2:.2f}")
print(f"  RMSE: {rmse2:.2f}")

# Model 3: Single manual changepoint
print("\n" + "=" * 80)
print("MODEL 3: Single Manual Changepoint")
print("=" * 80)

m3 = Farseer(
    changepoints=["2021-01-01"],  # Only specify the major change
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
)

print("\nConfiguration:")
print("  - Manual changepoints: ['2021-01-01']")
print("  - n_changepoints: 1")

m3.fit(train)
forecast3 = m3.predict(test)

mae3 = mean_absolute_error(test["y"], forecast3["yhat"])
rmse3 = np.sqrt(mean_squared_error(test["y"], forecast3["yhat"]))

print("\nTest Set Performance:")
print(f"  MAE:  {mae3:.2f}")
print(f"  RMSE: {rmse3:.2f}")

# Model 4: No changepoints
print("\n" + "=" * 80)
print("MODEL 4: No Changepoints (Linear Trend)")
print("=" * 80)

m4 = Farseer(
    n_changepoints=0,
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
)

print("\nConfiguration:")
print("  - n_changepoints: 0 (pure linear trend)")

m4.fit(train)
forecast4 = m4.predict(test)

mae4 = mean_absolute_error(test["y"], forecast4["yhat"])
rmse4 = np.sqrt(mean_squared_error(test["y"], forecast4["yhat"]))

print("\nTest Set Performance:")
print(f"  MAE:  {mae4:.2f}")
print(f"  RMSE: {rmse4:.2f}")

# Summary comparison
print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON")
print("=" * 80)

results = pd.DataFrame(
    {
        "Model": [
            "Automatic (25 changepoints)",
            "Manual (2 changepoints)",
            "Single changepoint",
            "No changepoints",
        ],
        "MAE": [mae1, mae2, mae3, mae4],
        "RMSE": [rmse1, rmse2, rmse3, rmse4],
    }
)

print("\n" + results.to_string(index=False))

print("\n" + "=" * 80)
print("INSIGHTS")
print("=" * 80)
print("""
Manual changepoints allow you to:
  ✓ Incorporate domain knowledge about when trend changes occurred
  ✓ Specify exact dates of policy changes, market events, etc.
  ✓ Reduce overfitting by limiting changepoints to known events
  ✓ Improve forecast accuracy when you know when changes happened
  ✓ Handle data with gaps or irregular sampling

Best practices:
  • Use manual changepoints when you have domain knowledge
  • Start with automatic detection to explore patterns
  • Combine both: augment automatic grid with known dates
  • Monitor model performance to validate changepoint choices
""")

print("=" * 80)
print("SUCCESS! Manual changepoints are working.")
print("=" * 80)

# Show params to verify changepoints were set
print("\nModel 2 Parameters (Manual Changepoints):")
params = m2.params()
print(f"  - n_changepoints: {params['n_changepoints']}")
print(f"  - changepoints: {params['changepoints']}")
print(f"  - specified_changepoints: {params['specified_changepoints']}")
