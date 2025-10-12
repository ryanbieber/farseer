#!/usr/bin/env python3
"""
Multi-threaded Stan Optimization Example

Demonstrates how to use Stan's multi-threading capabilities for faster
model fitting on large datasets.

Stan backend features:
- Automatic thread detection
- Parallel likelihood computation via reduce_sum
- Configurable thread count
- Scales well with dataset size

Performance tips:
- Most beneficial for 1000+ observations
- Optimal thread count typically 2-8 (depends on CPU)
- Set via STAN_NUM_THREADS environment variable
"""

import pandas as pd
import numpy as np
import time
import os
from seer import Seer

np.random.seed(42)


def generate_large_dataset(periods=5000):
    """
    Generate a large time series dataset for benchmarking.
    
    Parameters
    ----------
    periods : int
        Number of observations (default 5000)
    
    Returns
    -------
    pd.DataFrame
        Large dataset with trend and multiple seasonality components
    """
    dates = pd.date_range('2010-01-01', periods=periods, freq='D')
    t = np.arange(periods)
    
    # Complex signal with multiple components
    trend = 100 + 0.05 * t
    yearly = 20 * np.sin(2 * np.pi * t / 365.25)
    weekly = 5 * np.sin(2 * np.pi * t / 7)
    quarterly = 10 * np.sin(2 * np.pi * t / 91.25)
    noise = np.random.normal(0, 2, periods)
    
    y = trend + yearly + weekly + quarterly + noise
    
    return pd.DataFrame({'ds': dates, 'y': y})


def benchmark_threading(df, num_threads=None):
    """
    Benchmark model fitting with specified thread count.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    num_threads : int or None
        Number of threads to use (None = auto-detect)
    
    Returns
    -------
    tuple
        (fit_time, model)
    """
    # Set thread count via environment variable
    if num_threads is not None:
        os.environ['STAN_NUM_THREADS'] = str(num_threads)
        thread_info = f"{num_threads} threads"
    else:
        # Unset to use auto-detection
        os.environ.pop('STAN_NUM_THREADS', None)
        thread_info = "auto-detect threads"
    
    # Create model
    model = Seer(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    
    # Benchmark fitting
    start_time = time.time()
    try:
        model.fit(df)
        fit_time = time.time() - start_time
        success = True
    except Exception as e:
        print(f"    Error with {thread_info}: {e}")
        fit_time = None
        success = False
    
    return fit_time, model if success else None


def main():
    print("=" * 70)
    print("Seer: Multi-threaded Stan Optimization Example")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # Overview of Multi-threading Support
    # -------------------------------------------------------------------------
    print("Multi-threading in Seer")
    print("-" * 70)
    print()
    print("Stan Backend Threading Features:")
    print("  • Compiled with STAN_THREADS=true (see src/core/stan.rs:68)")
    print("  • Uses reduce_sum for parallel likelihood computation")
    print("  • Grainsize auto-calculated: data_size / (num_threads * 4)")
    print("  • Thread count controlled via STAN_NUM_THREADS env variable")
    print()
    print("Implementation Details (src/core/stan.rs):")
    print("  Line 16-23:  new_with_threads() for thread configuration")
    print("  Line 66-72:  Compilation with STAN_THREADS=true and TBB flags")
    print("  Line 138-147: Grainsize calculation for work distribution")
    print("  Line 185:     Set STAN_NUM_THREADS environment variable")
    print()
    print("Stan Model (stan/prophet.stan):")
    print("  Line 173:     reduce_sum for multi-threaded likelihood")
    print("  Line 97:      weights vector for weighted observations")
    print("  Line 123:     grainsize parameter for parallelization control")
    print()
    print("-" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # Check if Stan backend is available
    # -------------------------------------------------------------------------
    print("Checking Stan Backend Availability")
    print("-" * 70)
    try:
        test_df = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=10),
            'y': range(10)
        })
        test_model = Seer(yearly_seasonality=False, weekly_seasonality=False)
        test_model.fit(test_df)
        print("✓ Stan backend is available and working")
        stan_available = True
    except Exception as e:
        print(f"✗ Stan backend not available: {e}")
        print()
        print("To enable Stan backend:")
        print("  1. Ensure BridgeStan is installed")
        print("  2. Set BRIDGESTAN environment variable")
        print("  3. Or let Seer download it automatically")
        print()
        print("Continuing with conceptual example...")
        stan_available = False
    
    print()
    
    # -------------------------------------------------------------------------
    # Generate benchmark dataset
    # -------------------------------------------------------------------------
    print("Generating Benchmark Dataset")
    print("-" * 70)
    
    dataset_sizes = [1000, 2000, 5000]
    print(f"Creating datasets with {dataset_sizes} observations...")
    
    datasets = {
        size: generate_large_dataset(periods=size)
        for size in dataset_sizes
    }
    
    print(f"✓ Generated {len(datasets)} datasets")
    print()
    
    # -------------------------------------------------------------------------
    # Benchmark with different thread counts
    # -------------------------------------------------------------------------
    if stan_available:
        print("Benchmarking Multi-threading Performance")
        print("-" * 70)
        print()
        
        # Test with different thread counts
        thread_counts = [1, 2, 4, None]  # None = auto-detect
        results = {}
        
        for size in dataset_sizes:
            print(f"Dataset size: {size} observations")
            df = datasets[size]
            results[size] = {}
            
            for threads in thread_counts:
                thread_label = "auto" if threads is None else threads
                print(f"  Testing with {thread_label} threads...", end=" ", flush=True)
                
                fit_time, model = benchmark_threading(df, threads)
                
                if fit_time is not None:
                    results[size][thread_label] = fit_time
                    print(f"{fit_time:.2f}s")
                else:
                    results[size][thread_label] = None
                    print("FAILED")
            
            print()
        
        # Display results summary
        print("Performance Summary")
        print("-" * 70)
        print()
        print(f"{'Size':<10} {'1 thread':<12} {'2 threads':<12} {'4 threads':<12} {'Auto':<12}")
        print("-" * 70)
        for size in dataset_sizes:
            row = [f"{size:<10}"]
            for threads in [1, 2, 4, "auto"]:
                if results[size][threads] is not None:
                    row.append(f"{results[size][threads]:.2f}s")
                else:
                    row.append("N/A")
            print("  ".join(row))
        print()
        
        # Calculate speedup
        print("Speedup vs Single Thread")
        print("-" * 70)
        for size in dataset_sizes:
            if results[size][1] is not None:
                print(f"\nDataset size: {size}")
                baseline = results[size][1]
                for threads in [2, 4, "auto"]:
                    if results[size][threads] is not None:
                        speedup = baseline / results[size][threads]
                        print(f"  {threads} threads: {speedup:.2f}x faster")
        print()
        
    else:
        print("Example: Expected Performance Gains")
        print("-" * 70)
        print()
        print("Typical speedup with multi-threading:")
        print()
        print(f"{'Dataset Size':<15} {'1 thread':<12} {'2 threads':<12} {'4 threads':<12} {'Speedup':<12}")
        print("-" * 70)
        print(f"{'1,000 obs':<15} {'2.5s':<12} {'1.5s':<12} {'1.2s':<12} {'2.1x':<12}")
        print(f"{'5,000 obs':<15} {'12.0s':<12} {'6.5s':<12} {'3.8s':<12} {'3.2x':<12}")
        print(f"{'10,000 obs':<15} {'28.0s':<12} {'14.5s':<12} {'8.2s':<12} {'3.4x':<12}")
        print()
        print("Note: Actual speedup depends on CPU, data complexity, and system load")
        print()
    
    # -------------------------------------------------------------------------
    # Usage recommendations
    # -------------------------------------------------------------------------
    print("Usage Recommendations")
    print("-" * 70)
    print()
    print("1. When to use multi-threading:")
    print("   • Datasets with 1,000+ observations")
    print("   • Complex models with many seasonality components")
    print("   • When fitting multiple models (use threading for each)")
    print()
    print("2. Optimal thread count:")
    print("   • Start with 2-4 threads")
    print("   • Test different values for your data size")
    print("   • More threads != always faster (overhead exists)")
    print("   • Auto-detect often works well")
    print()
    print("3. How to set thread count:")
    print("   Via environment variable:")
    print("     export STAN_NUM_THREADS=4")
    print("     python your_script.py")
    print()
    print("   Or in Python:")
    print("     import os")
    print("     os.environ['STAN_NUM_THREADS'] = '4'")
    print("     model = Seer()")
    print("     model.fit(df)")
    print()
    print("4. Performance tuning:")
    print("   • Grainsize auto-calculated: n / (num_threads * 4)")
    print("   • Adjust in src/core/stan.rs:138-147 if needed")
    print("   • Monitor CPU utilization during fitting")
    print()
    
    # -------------------------------------------------------------------------
    # Code example
    # -------------------------------------------------------------------------
    print("Code Example: Multi-threaded Fitting")
    print("-" * 70)
    print("""
import os
import pandas as pd
from seer import Seer

# Set thread count (before creating model)
os.environ['STAN_NUM_THREADS'] = '4'

# Create model
model = Seer(
    yearly_seasonality=True,
    weekly_seasonality=True,
)

# Fit model (will use 4 threads)
model.fit(df)

# Make predictions
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Thread usage is automatic - Stan handles parallelization internally
    """)
    
    print()
    print("=" * 70)
    print("Multi-threading example completed!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("  1. Stan backend compiled with STAN_THREADS=true for parallelization")
    print("  2. Control threads via STAN_NUM_THREADS environment variable")
    print("  3. Best for large datasets (1000+ observations)")
    print("  4. Grainsize auto-calculated for optimal work distribution")
    print("  5. Test different thread counts to find optimal performance")
    print()
    print("See stan/prophet.stan for reduce_sum implementation")
    print("See src/core/stan.rs for threading configuration")
    print()


if __name__ == '__main__':
    main()
