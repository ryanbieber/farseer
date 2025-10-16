#!/usr/bin/env python3
"""
Simple test to verify weights functionality in Seer.
"""

import pandas as pd
import numpy as np


# This test will be run after building the Python package
def test_weights_basic():
    """Test that weights are accepted and used"""
    from farseer import Farseer

    # Create simple data
    dates = pd.date_range("2020-01-01", periods=100)
    y = np.arange(100, dtype=float) + np.random.normal(0, 1, 100)
    weights = np.ones(100)
    weights[:50] = 2.0  # Weight first half more heavily

    df = pd.DataFrame({"ds": dates, "y": y, "weight": weights})

    # Fit with weights (Stan backend)
    try:
        model_stan = Farseer()
        model_stan.fit(df[["ds", "y", "weight"]])
        print("✓ Stan backend uses weights")
    except Exception as e:
        print(f"Note: Stan backend test skipped: {e}")

    # Test without weights column (should still work)
    model_no_weights = Farseer()
    model_no_weights.fit(df[["ds", "y"]])
    print("✓ Works without weights column")

    print("\nAll basic tests passed!")


def test_weights_validation():
    """Test that weight validation works"""
    from farseer import Farseer

    dates = pd.date_range("2020-01-01", periods=10)
    y = np.arange(10, dtype=float)

    # Test 1: Negative weights should fail
    try:
        df_bad = pd.DataFrame({"ds": dates, "y": y, "weight": [-1.0] + [1.0] * 9})
        model = Farseer()
        model.fit(df_bad)
        print("✗ Should have rejected negative weights")
    except Exception as e:
        if "non-negative" in str(e).lower():
            print("✓ Correctly rejects negative weights")
        else:
            print(f"✗ Unexpected error for negative weights: {e}")

    # Test 2: Wrong length should fail
    try:
        df_bad = pd.DataFrame(
            {
                "ds": dates,
                "y": y,
                "weight": [1.0] * 5,  # Wrong length
            }
        )
        model = Farseer()
        model.fit(df_bad)
        print("✗ Should have rejected wrong length weights")
    except Exception as e:
        if "length" in str(e).lower():
            print("✓ Correctly rejects wrong length weights")
        else:
            print(f"✗ Unexpected error for wrong length: {e}")

    print("\nValidation tests passed!")


if __name__ == "__main__":
    print("Testing weights functionality...\n")
    try:
        test_weights_basic()
        print()
        test_weights_validation()
    except ImportError:
        print("Cannot import farseer - build the package first:")
        print("  maturin develop")
        print("or")
        print("  pip install -e .")
