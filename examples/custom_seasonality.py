"""
Example: Custom Seasonality with Seer

This example demonstrates M3 features:
- Adding custom seasonality components
- Using additive and multiplicative modes
- Mixing different seasonal patterns
"""

import pandas as pd
import numpy as np
from seer import Seer

# Create sample data with monthly pattern
print("Creating sample dataset with monthly pattern...")
dates = pd.date_range('2020-01-01', periods=365, freq='D')
np.random.seed(42)

# Simulate data with trend + monthly seasonality
trend = 100 + 0.5 * np.arange(365)
monthly_pattern = 10 * np.sin(np.arange(365) / 30.5 * 2 * np.pi)
noise = np.random.randn(365) * 5
y = trend + monthly_pattern + noise

df = pd.DataFrame({'ds': dates, 'y': y})
print(f"✓ Created {len(df)} rows of data")
print(f"  Date range: {df['ds'].min()} to {df['ds'].max()}")
print(f"  Value range: {df['y'].min():.1f} to {df['y'].max():.1f}\n")

# Example 1: Custom Monthly Seasonality (Additive)
print("="*60)
print("Example 1: Custom Monthly Seasonality (Additive)")
print("="*60)

m1 = Seer(
    yearly_seasonality=False,  # Disable default yearly
    weekly_seasonality=False   # Disable default weekly
)

# Add monthly seasonality with 5 Fourier terms
m1.add_seasonality('monthly', period=30.5, fourier_order=5)

print("Configuration:")
print(f"  - Disabled yearly and weekly seasonality")
print(f"  - Added custom 'monthly' seasonality (period=30.5 days, order=5)")
print(f"\nFitting model...")

m1.fit(df)

# Make predictions
future1 = m1.make_future_dataframe(periods=90, include_history=False)
forecast1 = m1.predict(future1)

print(f"✓ Generated {len(forecast1)} predictions")
print(f"\nFuture forecast sample:")
print(forecast1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
print()

# Example 2: Multiplicative Seasonality
print("="*60)
print("Example 2: Multiplicative Seasonality")
print("="*60)

# Create data with multiplicative pattern
y_mult = trend * (1 + 0.2 * np.sin(np.arange(365) / 7 * 2 * np.pi)) + noise
df_mult = pd.DataFrame({'ds': dates, 'y': y_mult})

m2 = Seer(yearly_seasonality=False, daily_seasonality=False)

# Add weekly seasonality as multiplicative
m2.add_seasonality('weekly', period=7.0, fourier_order=3, mode='multiplicative')

print("Configuration:")
print(f"  - Weekly seasonality in MULTIPLICATIVE mode")
print(f"  - Seasonality scales with trend level")
print(f"\nFitting model...")

m2.fit(df_mult)

future2 = m2.make_future_dataframe(periods=30, include_history=False)
forecast2 = m2.predict(future2)

print(f"✓ Generated {len(forecast2)} predictions")
print(f"\nFuture forecast sample:")
print(forecast2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
print()

# Example 3: Mixed Additive and Multiplicative
print("="*60)
print("Example 3: Mixed Additive and Multiplicative")
print("="*60)

# Create data with both patterns
weekly_mult_factor = 1 + 0.15 * np.cos(np.arange(365) / 7 * 2 * np.pi)
monthly_add = 8 * np.sin(np.arange(365) / 30 * 2 * np.pi)
y_mixed = trend * weekly_mult_factor + monthly_add + noise

df_mixed = pd.DataFrame({'ds': dates, 'y': y_mixed})

m3 = Seer(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)

# Add both types
m3.add_seasonality('monthly_add', period=30.0, fourier_order=5, mode='additive')
m3.add_seasonality('weekly_mult', period=7.0, fourier_order=3, mode='multiplicative')

print("Configuration:")
print(f"  - Monthly seasonality: ADDITIVE (absolute variation)")
print(f"  - Weekly seasonality: MULTIPLICATIVE (proportional variation)")
print(f"\nFitting model...")

m3.fit(df_mixed)

future3 = m3.make_future_dataframe(periods=60, include_history=False)
forecast3 = m3.predict(future3)

print(f"✓ Generated {len(forecast3)} predictions")
print(f"\nFuture forecast sample:")
print(forecast3[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
print()

# Example 4: Custom Prior Scale
print("="*60)
print("Example 4: Custom Seasonality with Prior Scale")
print("="*60)

m4 = Seer(yearly_seasonality=False, weekly_seasonality=False)

# Add seasonality with custom prior scale for regularization
m4.add_seasonality(
    'quarterly', 
    period=91.25,  # ~3 months
    fourier_order=4,
    prior_scale=0.1  # Lower value = more regularization
)

print("Configuration:")
print(f"  - Quarterly seasonality (91.25 days)")
print(f"  - Prior scale = 0.1 (for future Stan integration)")
print(f"\nFitting model...")

m4.fit(df)

future4 = m4.make_future_dataframe(periods=180, include_history=False)
forecast4 = m4.predict(future4)

print(f"✓ Generated {len(forecast4)} predictions")
print(f"\nFuture forecast sample:")
print(forecast4[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
print()

print("="*60)
print("Summary")
print("="*60)
print("""
M3 Features Demonstrated:
✓ Custom seasonality with add_seasonality()
✓ Additive mode (default): seasonality added to trend
✓ Multiplicative mode: seasonality scales with trend
✓ Mixed modes: combine additive and multiplicative components
✓ Prior scale parameter (stored for future Stan integration)
✓ Flexible Fourier order configuration
✓ Arbitrary periods (monthly, quarterly, custom)

Note: Current implementation uses OLS for fitting. Full Bayesian
fitting with Stan will be added in a future milestone.
""")
