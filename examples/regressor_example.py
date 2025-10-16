"""
Example demonstrating regressor functionality in Seer.
This shows how to add additional features (regressors) to improve forecasts.
"""

import pandas as pd
import numpy as np

from farseer import Farseer

# Create sample data
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=100, freq="D")

# Generate synthetic data with:
# - Base trend
# - Weekly seasonality
# - Temperature effect (regressor)
# - Marketing spend effect (regressor)
trend = np.linspace(100, 150, 100)
weekly = 10 * np.sin(2 * np.pi * np.arange(100) / 7)
temperature = 20 + 5 * np.sin(2 * np.pi * np.arange(100) / 30)  # Monthly variation
marketing = np.random.choice([0, 1], size=100, p=[0.7, 0.3])  # Binary: campaign on/off
temperature_effect = 2 * (temperature - 20)
marketing_effect = 15 * marketing
noise = np.random.normal(0, 3, 100)

y = trend + weekly + temperature_effect + marketing_effect + noise

# Create dataframe
df = pd.DataFrame(
    {"ds": dates, "y": y, "temperature": temperature, "marketing": marketing}
)

print("Sample data:")
print(df.head(10))
print("\n" + "=" * 60 + "\n")

# Create and configure model
m = Farseer()

# Add regressors
# Temperature is continuous -> will be standardized automatically
m.add_regressor("temperature", prior_scale=0.5)

# Marketing is binary (0/1) -> won't be standardized by default
m.add_regressor("marketing", prior_scale=1.0)

print("Model configured with regressors")
print(f"Regressors: {m.get_regressor_names()}")
print("\n" + "=" * 60 + "\n")

# Fit the model
print("Fitting model...")
m.fit(df)
print("Model fitted successfully!")
print("\n" + "=" * 60 + "\n")

# Make predictions
# Create future dataframe with regressor values
future = pd.DataFrame(
    {
        "ds": pd.date_range("2020-04-10", periods=30, freq="D"),
        "temperature": 20 + 5 * np.sin(2 * np.pi * np.arange(30) / 30),
        "marketing": [1] * 15 + [0] * 15,  # Campaign for first 15 days
    }
)

print("Future data with regressors:")
print(future.head(10))
print("\n" + "=" * 60 + "\n")

# Generate forecast
forecast = m.predict(future)

print("Forecast results:")
print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].head(10))
print("\n" + "=" * 60 + "\n")

# Compare predictions with and without marketing campaign
print("Impact of marketing campaign:")
print(f"Average forecast (with marketing): {forecast['yhat'][:15].mean():.2f}")
print(f"Average forecast (without marketing): {forecast['yhat'][15:].mean():.2f}")
print(
    f"Estimated marketing impact: {forecast['yhat'][:15].mean() - forecast['yhat'][15:].mean():.2f}"
)

print("\n✓ Regressors working correctly!")
