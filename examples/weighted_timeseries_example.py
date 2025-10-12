#!/usr/bin/env python3
"""
Example demonstrating weighted time series forecasting with Seer.

This example shows how to:
1. Create time series data with custom weights
2. Fit a model using weighted observations
3. Compare weighted vs unweighted results
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import Seer (adjust import based on your installation)
try:
    from seer import Seer
except ImportError:
    print("Error: Could not import seer. Make sure it's installed.")
    print("Run: pip install -e . (from the project root)")
    exit(1)


def generate_data_with_noise_variance():
    """
    Generate synthetic data where noise variance changes over time.
    Early data has low noise, later data has high noise.
    """
    np.random.seed(42)
    
    # Generate dates
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(365)]
    
    # True underlying trend + seasonality
    t = np.arange(365)
    trend = 100 + 0.5 * t
    seasonality = 20 * np.sin(2 * np.pi * t / 365.25)
    true_signal = trend + seasonality
    
    # Add noise with increasing variance over time
    # Early observations (first 6 months): low noise (sigma = 2.0)
    # Later observations (last 6 months): high noise (sigma = 10.0)
    noise_sigma = np.linspace(2.0, 10.0, 365)
    noise = np.random.normal(0, noise_sigma)
    y = true_signal + noise
    
    # Create weights inversely proportional to variance
    # weight ∝ 1/σ²
    weights = 1.0 / (noise_sigma ** 2)
    # Normalize so mean weight = 1.0 (optional, for interpretability)
    weights = weights / weights.mean()
    
    df = pd.DataFrame({
        'ds': dates,
        'y': y,
        'weight': weights
    })
    
    return df, true_signal


def generate_data_with_outliers():
    """
    Generate data with some outlier observations that should be down-weighted.
    """
    np.random.seed(123)
    
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(365)]
    
    t = np.arange(365)
    trend = 50 + 0.3 * t
    seasonality = 15 * np.sin(2 * np.pi * t / 365.25)
    y = trend + seasonality + np.random.normal(0, 3, 365)
    
    # Introduce outliers at random positions
    outlier_indices = np.random.choice(365, size=20, replace=False)
    y[outlier_indices] += np.random.choice([-30, 30], size=20)
    
    # Create weights: low weight for outliers, normal weight for others
    weights = np.ones(365)
    weights[outlier_indices] = 0.1  # Down-weight outliers
    
    df = pd.DataFrame({
        'ds': dates,
        'y': y,
        'weight': weights
    })
    
    return df


def generate_data_with_confidence_scores():
    """
    Generate data where each observation has an associated confidence score.
    For example, data from different sources with known quality.
    """
    np.random.seed(456)
    
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(365)]
    
    t = np.arange(365)
    y = 100 + 0.2 * t + 10 * np.sin(2 * np.pi * t / 365.25)
    y += np.random.normal(0, 5, 365)
    
    # Simulate different data sources with different quality levels
    # High quality sensor (first 100 days): confidence = 0.95
    # Medium quality sensor (next 150 days): confidence = 0.70
    # Low quality manual readings (last 115 days): confidence = 0.40
    confidence = np.concatenate([
        np.full(100, 0.95),
        np.full(150, 0.70),
        np.full(115, 0.40)
    ])
    
    # Convert confidence to weights
    # Simple approach: use confidence directly as weight
    weights = confidence
    
    df = pd.DataFrame({
        'ds': dates,
        'y': y,
        'weight': weights
    })
    
    return df


def example_1_varying_noise():
    """
    Example 1: Time series with varying noise levels
    """
    print("=" * 70)
    print("Example 1: Time Series with Varying Noise Levels")
    print("=" * 70)
    print()
    
    df, true_signal = generate_data_with_noise_variance()
    
    print("Dataset info:")
    print(f"  - Total observations: {len(df)}")
    print(f"  - Weight range: [{df['weight'].min():.3f}, {df['weight'].max():.3f}]")
    print(f"  - Mean weight: {df['weight'].mean():.3f}")
    print()
    
    # Fit model WITHOUT weights (standard approach)
    print("Fitting model WITHOUT weights...")
    model_unweighted = Seer(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    model_unweighted.fit(df[['ds', 'y']])
    print("  ✓ Model fitted")
    
    # Fit model WITH weights
    print("Fitting model WITH weights...")
    model_weighted = Seer(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    model_weighted.fit(df[['ds', 'y', 'weight']])
    print("  ✓ Model fitted")
    print()
    
    # Make predictions
    future = pd.DataFrame({'ds': df['ds']})
    
    forecast_unweighted = model_unweighted.predict(future)
    forecast_weighted = model_weighted.predict(future)
    
    # Calculate errors
    error_unweighted = np.mean((forecast_unweighted['yhat'] - true_signal) ** 2)
    error_weighted = np.mean((forecast_weighted['yhat'] - true_signal) ** 2)
    
    print("Results:")
    print(f"  Unweighted model MSE: {error_unweighted:.2f}")
    print(f"  Weighted model MSE:   {error_weighted:.2f}")
    print(f"  Improvement:          {((error_unweighted - error_weighted) / error_unweighted * 100):.1f}%")
    print()
    print("Interpretation:")
    print("  The weighted model should perform better because it gives more")
    print("  importance to early observations with low noise.")
    print()


def example_2_outliers():
    """
    Example 2: Data with outliers
    """
    print("=" * 70)
    print("Example 2: Data with Outliers")
    print("=" * 70)
    print()
    
    df = generate_data_with_outliers()
    
    # Count outliers (weight < 1.0)
    n_outliers = (df['weight'] < 1.0).sum()
    
    print("Dataset info:")
    print(f"  - Total observations: {len(df)}")
    print(f"  - Number of outliers (down-weighted): {n_outliers}")
    print(f"  - Outlier weight: {df['weight'].min():.2f}")
    print(f"  - Normal weight: 1.00")
    print()
    
    # Fit without weights
    print("Fitting model WITHOUT weights (outliers fully weighted)...")
    model_unweighted = Seer()
    model_unweighted.fit(df[['ds', 'y']])
    print("  ✓ Model fitted")
    
    # Fit with weights
    print("Fitting model WITH weights (outliers down-weighted)...")
    model_weighted = Seer()
    model_weighted.fit(df[['ds', 'y', 'weight']])
    print("  ✓ Model fitted")
    print()
    
    print("Results:")
    print("  The weighted model should be more robust to outliers.")
    print("  Check forecast plots to see the difference.")
    print()


def example_3_confidence_scores():
    """
    Example 3: Data with confidence scores from different sources
    """
    print("=" * 70)
    print("Example 3: Data with Confidence Scores")
    print("=" * 70)
    print()
    
    df = generate_data_with_confidence_scores()
    
    print("Dataset info:")
    print("  Data sources:")
    print("    - High quality (days 0-99):    weight = 0.95")
    print("    - Medium quality (days 100-249): weight = 0.70")
    print("    - Low quality (days 250-364):   weight = 0.40")
    print()
    
    # Fit with weights
    print("Fitting model WITH confidence-based weights...")
    model = Seer(
        yearly_seasonality=True,
        weekly_seasonality=False
    )
    model.fit(df[['ds', 'y', 'weight']])
    print("  ✓ Model fitted")
    print()
    
    print("Results:")
    print("  The model will trust high-quality data more when fitting parameters.")
    print()


def example_4_weight_calculation_patterns():
    """
    Example 4: Different weight calculation patterns
    """
    print("=" * 70)
    print("Example 4: Weight Calculation Patterns")
    print("=" * 70)
    print()
    
    # Example noise standard deviations
    sigma = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    
    print("Pattern 1: Inverse variance weighting")
    print("  If noise std dev σ is known, use weight = 1/σ²")
    print()
    weights_1 = 1.0 / (sigma ** 2)
    print(f"  σ:      {sigma}")
    print(f"  weight: {weights_1}")
    print()
    
    print("Pattern 2: Normalized inverse variance")
    print("  Normalize so mean weight = 1.0")
    print()
    weights_2 = weights_1 / weights_1.mean()
    print(f"  σ:      {sigma}")
    print(f"  weight: {weights_2}")
    print()
    
    print("Pattern 3: From confidence scores")
    print("  Convert confidence [0,1] to weights")
    print()
    confidence = np.array([0.5, 0.7, 0.8, 0.9, 0.95])
    weights_3 = confidence
    print(f"  confidence: {confidence}")
    print(f"  weight:     {weights_3}")
    print()
    
    print("Pattern 4: Exponential time decay")
    print("  Weight recent data more heavily")
    print()
    days_old = np.array([0, 30, 60, 90, 120])
    weights_4 = np.exp(-days_old / 60)  # Half-life = 60 days
    print(f"  days old: {days_old}")
    print(f"  weight:   {weights_4}")
    print()


def main():
    """Run all examples"""
    print()
    print("WEIGHTED TIME SERIES EXAMPLES")
    print()
    
    try:
        # Run examples that don't require actual fitting first
        example_4_weight_calculation_patterns()
        
        # Now run examples that require fitting
        print("Running examples with model fitting...")
        print()
        
        try:
            example_1_varying_noise()
            example_2_outliers()
            example_3_confidence_scores()
        except Exception as e:
            print(f"Note: Model fitting examples skipped due to error: {e}")
            print("This is expected if Stan/BridgeStan is not installed.")
            print()
        
        print("=" * 70)
        print("Examples completed!")
        print("=" * 70)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
