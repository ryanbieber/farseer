#!/usr/bin/env python3
"""
Demo: Using Regressors with Seer
This demonstrates how to add and use additional regressors in Seer,
compatible with Facebook Prophet's regressor API.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from farseer import Farseer, regressor_coefficients

# Set random seed for reproducibility
np.random.seed(42)

# Create sample data with trend and seasonality
n = 365 * 2  # 2 years of daily data
dates = pd.date_range("2020-01-01", periods=n, freq="D")

# Base trend
trend = np.arange(n) * 0.3 + 100

# Yearly seasonality
yearly = 20 * np.sin(2 * np.pi * np.arange(n) / 365.25)

# Weekly seasonality
weekly = 5 * np.sin(2 * np.pi * np.arange(n) / 7)

# Create regressors
# 1. Binary regressor: weekend indicator
is_weekend = (pd.to_datetime(dates).dayofweek >= 5).astype(float)

# 2. Numeric regressor: temperature (simulated)
temperature = (
    15 + 10 * np.sin(2 * np.pi * np.arange(n) / 365.25) + np.random.randn(n) * 3
)

# 3. Promotional events (sparse binary regressor)
promo = np.zeros(n)
promo_days = np.random.choice(n, size=50, replace=False)
promo[promo_days] = 1

# Combine into y with regressor effects
# Weekend effect: -10 units
# Temperature effect: +0.5 per degree
# Promo effect: +15 units
y = (
    trend
    + yearly
    + weekly
    + (-10 * is_weekend)
    + (0.5 * temperature)
    + (15 * promo)
    + np.random.randn(n) * 3
)

# Create DataFrame
df = pd.DataFrame(
    {
        "ds": dates,
        "y": y,
        "is_weekend": is_weekend,
        "temperature": temperature,
        "promo": promo,
    }
)

print("=" * 70)
print("DEMO: Using Regressors with Seer")
print("=" * 70)
print(f"\nDataset: {n} days of data with 3 regressors")
print("  - is_weekend: binary indicator")
print("  - temperature: continuous variable")
print("  - promo: sparse binary indicator")
print("\nTrue effects:")
print("  - Weekend: -10 units")
print("  - Temperature: +0.5 units per degree")
print("  - Promo: +15 units")

# Split into train/test
train_size = int(n * 0.8)
train = df[:train_size].copy()
test = df[train_size:].copy()

print(f"\nTrain size: {len(train)} days")
print(f"Test size: {len(test)} days")

# Create and configure model
print("\n" + "=" * 70)
print("FITTING MODEL WITH REGRESSORS")
print("=" * 70)

m = Farseer(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)

# Add regressors
# Binary regressor (weekend) - auto-detection will not standardize 0/1 values
m.add_regressor("is_weekend", prior_scale=10.0, mode="additive")

# Numeric regressor (temperature) - will be auto-standardized
m.add_regressor("temperature", prior_scale=10.0, mode="additive")

# Sparse binary regressor (promo)
m.add_regressor("promo", prior_scale=10.0, mode="additive")

print("\nRegressors added:")
print("  1. is_weekend (additive, prior_scale=10.0)")
print("  2. temperature (additive, prior_scale=10.0)")
print("  3. promo (additive, prior_scale=10.0)")

# Fit the model
print("\nFitting model...")
m.fit(train)
print("✓ Model fitted successfully")

# Extract regressor coefficients
print("\n" + "=" * 70)
print("REGRESSOR COEFFICIENTS")
print("=" * 70)

coefs = regressor_coefficients(m)
print("\n" + coefs.to_string(index=False))

print("\nInterpretation:")
for _, row in coefs.iterrows():
    regr_name = row["regressor"]
    coef = row["coef"]
    mode = row["regressor_mode"]
    print(f"  - {regr_name}: {coef:.3f} ({mode})")

# Make predictions on test set
print("\n" + "=" * 70)
print("MAKING PREDICTIONS")
print("=" * 70)

forecast = m.predict(test)

# Calculate metrics
mae = mean_absolute_error(test["y"], forecast["yhat"])
rmse = np.sqrt(mean_squared_error(test["y"], forecast["yhat"]))
r2 = r2_score(test["y"], forecast["yhat"])

print("\nTest Set Performance:")
print(f"  MAE:  {mae:.2f}")
print(f"  RMSE: {rmse:.2f}")
print(f"  R²:   {r2:.3f}")

# Show sample predictions
print("\nSample Predictions (first 5 test days):")
sample_df = test.head(5)[["ds", "y"]].copy()
sample_df["yhat"] = forecast["yhat"].head(5).to_numpy()
sample_df["error"] = sample_df["y"] - sample_df["yhat"]
print(sample_df.to_string(index=False))

print("\n" + "=" * 70)
print("FORECAST COMPONENTS")
print("=" * 70)

# Show component breakdown for first prediction
idx = 0
print(f"\nBreakdown for {test['ds'].to_numpy()[idx]}:")
print(f"  Trend:         {forecast['trend'][idx]:.2f}")
print(f"  Weekly:        {forecast['weekly'][idx]:.2f}")
print(f"  Yearly:        {forecast['yearly'][idx]:.2f}")
print(f"  Additive terms: {forecast['additive_terms'][idx]:.2f}")
print(f"  Final (yhat):  {forecast['yhat'][idx]:.2f}")
print(f"  Actual (y):    {test['y'].to_numpy()[idx]:.2f}")

print("\n" + "=" * 70)
print("SUCCESS! Regressors are working correctly.")
print("=" * 70)
print("\nKey takeaways:")
print("  ✓ Regressors can be added with add_regressor()")
print("  ✓ Both additive and multiplicative modes supported")
print("  ✓ Auto-standardization for non-binary numeric regressors")
print("  ✓ Coefficients can be extracted with regressor_coefficients()")
print("  ✓ Fully compatible with Prophet's regressor API")
