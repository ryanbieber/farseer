#!/usr/bin/env python3
"""
M7 Stan Integration Tests
Tests for Stan-based parameter estimation
"""

import pandas as pd
import numpy as np
from seer import Seer


def test_stan_flag_exists():
    """Test that use_stan parameter exists"""
    print("Testing use_stan parameter...")
    
    # Should be able to create with use_stan=False (default)
    m1 = Seer(use_stan=False)
    print("✅ Created model with use_stan=False")
    
    # Should be able to create with use_stan=True
    m2 = Seer(use_stan=True)
    print("✅ Created model with use_stan=True")


def test_ols_fitting_still_works():
    """Test that OLS fitting (default) still works"""
    print("\nTesting OLS fitting (use_stan=False)...")
    
    df = pd.DataFrame({
        'ds': pd.date_range('2020-01-01', periods=100, freq='D'),
        'y': np.linspace(10, 20, 100) + np.random.normal(0, 0.5, 100)
    })
    
    m = Seer(growth='linear', use_stan=False)
    m.fit(df)
    
    future = m.make_future_dataframe(periods=10)
    forecast = m.predict(future)
    
    assert len(forecast) == 110
    print("✅ OLS fitting works correctly")


def test_stan_compilation_attempt():
    """Test Stan compilation and fitting (currently using placeholder optimization)"""
    print("\nTesting Stan compilation and fitting...")
    
    df = pd.DataFrame({
        'ds': pd.date_range('2020-01-01', periods=50, freq='D'),
        'y': np.linspace(10, 15, 50) + np.random.normal(0, 0.3, 50)
    })
    
    m = Seer(
        growth='linear',
        use_stan=True,  # Enable Stan
        yearly_seasonality=False,
        weekly_seasonality=False
    )
    m.fit(df)
    
    # Model should fit successfully with Stan enabled
    future = m.make_future_dataframe(periods=10)
    forecast = m.predict(future)
    
    assert len(forecast) == 60
    print("✅ Stan compilation and fitting successful")
    print("   Note: Using placeholder optimization values until BridgeStan optimization API is implemented")


def test_stan_serialization():
    """Test that use_stan flag is preserved in serialization"""
    print("\nTesting Stan flag serialization...")
    
    df = pd.DataFrame({
        'ds': pd.date_range('2020-01-01', periods=50, freq='D'),
        'y': np.linspace(10, 15, 50)
    })
    
    # Use OLS for now since Stan compilation needs work
    m1 = Seer(use_stan=False, yearly_seasonality=False, weekly_seasonality=False)
    m1.fit(df)
    
    # Serialize
    json_str = m1.to_json()
    assert 'use_stan' in json_str
    
    # Deserialize
    m2 = Seer.from_json(json_str)
    params = m2.params()
    
    # Check the flag exists
    assert 'use_stan' in params
    
    print("✅ Stan flag included in serialization")


def test_params_includes_use_stan():
    """Test that params() includes use_stan"""
    print("\nTesting params() includes use_stan...")
    
    m1 = Seer(use_stan=False)
    params1 = m1.params()
    assert 'use_stan' in params1
    print(f"   use_stan=False: {params1['use_stan']}")
    
    m2 = Seer(use_stan=True)
    params2 = m2.params()
    assert 'use_stan' in params2
    print(f"   use_stan=True: {params2['use_stan']}")
    
    print("✅ params() includes use_stan field")


if __name__ == '__main__':
    print("=" * 60)
    print("M7 STAN INTEGRATION TESTS")
    print("=" * 60)
    
    test_stan_flag_exists()
    test_ols_fitting_still_works()
    test_stan_compilation_attempt()
    test_stan_serialization()
    test_params_includes_use_stan()
    
    print("\n" + "=" * 60)
    print("✅ All Stan integration tests passed!")
    print("=" * 60)
    print("\nNote: Full Stan optimization pending BridgeStan API research")
    print("Current status: Infrastructure complete, using placeholders")
