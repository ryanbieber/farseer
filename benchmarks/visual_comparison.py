#!/usr/bin/env python3
"""
Side-by-side comparison of Prophet and Seer on the same dataset.
Generates a visual comparison showing predictions from both models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("WARNING: Prophet not available")

try:
    from seer import Seer
    SEER_AVAILABLE = True
except ImportError:
    SEER_AVAILABLE = False
    print("WARNING: Seer not available")


def generate_example_data(n_days=730):
    """Generate synthetic daily data with trend and seasonality."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    # Components
    trend = np.arange(n_days) * 0.5 + 100
    yearly = np.sin(2 * np.pi * np.arange(n_days) / 365.25) * 10
    weekly = np.sin(2 * np.pi * np.arange(n_days) / 7) * 5
    noise = np.random.randn(n_days) * 3
    
    y = trend + yearly + weekly + noise
    
    return pd.DataFrame({'ds': dates, 'y': y})


def compare_models():
    """Run both models and compare results."""
    if not (PROPHET_AVAILABLE and SEER_AVAILABLE):
        print("Error: Both Prophet and Seer must be available")
        return
    
    # Generate data
    print("Generating data...")
    df = generate_example_data(730)  # 2 years
    n_train = 365 * 2 - 90  # Leave 90 days for testing
    train = df.iloc[:n_train]
    test = df.iloc[n_train:]
    
    # Prophet
    print("\nTraining Prophet...")
    import time
    start = time.time()
    m_prophet = Prophet()
    m_prophet.fit(train)
    prophet_train_time = time.time() - start
    
    print("Forecasting with Prophet...")
    start = time.time()
    future_prophet = m_prophet.make_future_dataframe(90, include_history=False)
    forecast_prophet = m_prophet.predict(future_prophet)
    prophet_predict_time = time.time() - start
    
    # Seer
    print("\nTraining Seer...")
    start = time.time()
    m_seer = Seer()
    m_seer.fit(train)
    seer_train_time = time.time() - start
    
    print("Forecasting with Seer...")
    start = time.time()
    future_seer = m_seer.make_future_dataframe(90, include_history=False)
    forecast_seer = m_seer.predict(future_seer)
    seer_predict_time = time.time() - start
    
    # Calculate metrics
    y_true = test['y'].values
    y_pred_prophet = forecast_prophet['yhat'].values
    y_pred_seer = forecast_seer['yhat'].values
    
    rmse_prophet = np.sqrt(np.mean((y_true - y_pred_prophet) ** 2))
    rmse_seer = np.sqrt(np.mean((y_true - y_pred_seer) ** 2))
    
    # Print results
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print(f"\n{'Metric':<20} {'Prophet':<15} {'Seer':<15} {'Ratio':<10}")
    print("-"*60)
    print(f"{'Train Time':<20} {prophet_train_time:.4f}s{'':<8} {seer_train_time:.4f}s{'':<8} {seer_train_time/prophet_train_time:.2f}x")
    print(f"{'Predict Time':<20} {prophet_predict_time:.4f}s{'':<8} {seer_predict_time:.4f}s{'':<8} {prophet_predict_time/seer_predict_time:.2f}x")
    print(f"{'Total Time':<20} {prophet_train_time+prophet_predict_time:.4f}s{'':<8} {seer_train_time+seer_predict_time:.4f}s{'':<8} {(seer_train_time+seer_predict_time)/(prophet_train_time+prophet_predict_time):.2f}x")
    print("-"*60)
    print(f"{'RMSE':<20} {rmse_prophet:.4f}{'':<10} {rmse_seer:.4f}{'':<10} {((rmse_seer-rmse_prophet)/rmse_prophet*100):+.2f}%")
    print("="*60)
    
    # Create visualization
    print("\nCreating visualization...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Full comparison
    ax1 = axes[0]
    ax1.plot(train['ds'], train['y'], 'k.', alpha=0.3, label='Training Data', markersize=3)
    ax1.plot(test['ds'], test['y'], 'ko', label='Test Data (Actual)', markersize=5)
    ax1.plot(forecast_prophet['ds'], forecast_prophet['yhat'], 'b-', 
             label=f'Prophet (RMSE: {rmse_prophet:.2f})', linewidth=2)
    ax1.plot(forecast_seer['ds'], forecast_seer['yhat'], 'r--', 
             label=f'Seer (RMSE: {rmse_seer:.2f})', linewidth=2)
    
    # Add uncertainty intervals
    ax1.fill_between(forecast_prophet['ds'], 
                     forecast_prophet['yhat_lower'], 
                     forecast_prophet['yhat_upper'],
                     alpha=0.2, color='blue', label='Prophet 95% CI')
    
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('Prophet vs Seer: Full Forecast Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Zoom on test period
    ax2 = axes[1]
    ax2.plot(test['ds'], test['y'], 'ko', label='Actual', markersize=6, zorder=3)
    ax2.plot(forecast_prophet['ds'], forecast_prophet['yhat'], 'b-', 
             label=f'Prophet (RMSE: {rmse_prophet:.2f})', linewidth=2.5)
    ax2.plot(forecast_seer['ds'], forecast_seer['yhat'], 'r--', 
             label=f'Seer (RMSE: {rmse_seer:.2f})', linewidth=2.5)
    
    # Error bars
    prophet_errors = np.abs(y_true - y_pred_prophet)
    seer_errors = np.abs(y_true - y_pred_seer)
    
    ax2.fill_between(forecast_prophet['ds'], 
                     forecast_prophet['yhat_lower'], 
                     forecast_prophet['yhat_upper'],
                     alpha=0.2, color='blue', label='Prophet 95% CI')
    
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('Test Period Detail (90 days)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'benchmarks/comparison_plot.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Show plot
    try:
        plt.show()
    except:
        print("(Display not available, plot saved to file)")
    
    # Component comparison
    print("\n" + "="*60)
    print("COMPONENT COMPARISON")
    print("="*60)
    
    # Extract components for a single point (mid-test period)
    mid_idx = len(forecast_prophet) // 2
    
    print(f"\nSample Point: {forecast_prophet['ds'].iloc[mid_idx]}")
    print(f"\n{'Component':<20} {'Prophet':<15} {'Seer':<15} {'Diff %':<10}")
    print("-"*60)
    
    components = ['trend', 'yearly', 'weekly']
    for comp in components:
        if comp in forecast_prophet.columns and comp in forecast_seer.columns:
            p_val = forecast_prophet[comp].iloc[mid_idx]
            s_val = forecast_seer[comp].iloc[mid_idx]
            diff = ((s_val - p_val) / p_val * 100) if p_val != 0 else 0
            print(f"{comp.capitalize():<20} {p_val:<15.4f} {s_val:<15.4f} {diff:+.2f}%")
    
    print(f"{'Yhat':<20} {forecast_prophet['yhat'].iloc[mid_idx]:<15.4f} {forecast_seer['yhat'].iloc[mid_idx]:<15.4f} {((forecast_seer['yhat'].iloc[mid_idx]-forecast_prophet['yhat'].iloc[mid_idx])/forecast_prophet['yhat'].iloc[mid_idx]*100):+.2f}%")
    print("="*60)
    
    return {
        'prophet': {
            'forecast': forecast_prophet,
            'train_time': prophet_train_time,
            'predict_time': prophet_predict_time,
            'rmse': rmse_prophet
        },
        'seer': {
            'forecast': forecast_seer,
            'train_time': seer_train_time,
            'predict_time': seer_predict_time,
            'rmse': rmse_seer
        }
    }


if __name__ == '__main__':
    compare_models()
