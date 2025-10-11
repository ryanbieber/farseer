"""
Tests for M5: Serialization and API Polish
"""
import json
import pandas as pd
from seer import Seer


def test_params_completeness():
    """Test that params() returns all model configuration"""
    df = pd.DataFrame({
        'ds': pd.date_range('2020-01-01', periods=50, freq='D'),
        'y': [10.0 + 0.1 * i for i in range(50)]
    })
    
    m = Seer()
    m.fit(df)
    
    params = m.params()
    
    # Check all major fields are present
    assert "version" in params
    assert "fitted" in params
    assert "trend" in params
    assert "n_changepoints" in params
    assert "seasonality_mode" in params
    assert "k" in params
    assert "m" in params
    assert "beta" in params
    assert "season_blocks" in params
    assert "sigma_obs" in params
    assert params["fitted"] is True
    
    print("✓ params() returns complete model state")


def test_json_serialization_round_trip():
    """Test that model can be serialized to JSON and deserialized"""
    df = pd.DataFrame({
        'ds': pd.date_range('2020-01-01', periods=30, freq='D'),
        'y': [100.0 + i for i in range(30)]
    })
    
    # Create and fit model
    m1 = Seer(n_changepoints=5)
    m1.fit(df)
    
    # Serialize
    json_str = m1.to_json()
    assert len(json_str) > 0
    
    # Verify it's valid JSON
    data = json.loads(json_str)
    assert "version" in data
    assert "trend" in data
    
    # Deserialize
    m2 = Seer.from_json(json_str)
    
    # Check configuration matches
    params1 = m1.params()
    params2 = m2.params()
    assert params2["trend"] == params1["trend"]
    assert params2["n_changepoints"] == params1["n_changepoints"]
    assert params2["fitted"] == params1["fitted"]
    
    print("✓ JSON serialization round-trip preserves model")


def test_serialization_with_custom_components():
    """Test that custom seasonalities and holidays survive serialization"""
    df = pd.DataFrame({
        'ds': pd.date_range('2020-01-01', periods=30, freq='D'),
        'y': [100.0 + i for i in range(30)]
    })
    
    # Create model with custom components
    m1 = Seer()
    m1.add_seasonality("monthly", 30.0, 5)
    m1.add_holidays("test_holiday", ["2020-01-15"])
    m1.fit(df)
    
    # Serialize and deserialize
    json_str = m1.to_json()
    m2 = Seer.from_json(json_str)
    
    # Check custom components are preserved
    params = m2.params()
    assert len(params["seasonalities"]) == 1
    assert params["seasonalities"][0]["name"] == "monthly"
    
    assert len(params["holidays"]) == 1
    assert params["holidays"][0]["name"] == "test_holiday"
    
    print("✓ Custom components preserved through serialization")


def test_set_seasonality_mode():
    """Test setting seasonality mode"""
    m = Seer(seasonality_mode="multiplicative")
    params = m.params()
    # Enum value is capitalized in output
    assert params["seasonality_mode"] == "Multiplicative"
    
    # Test invalid mode raises error
    try:
        m = Seer(seasonality_mode="invalid")
        assert False, "Should have raised error for invalid mode"
    except Exception:
        pass  # Expected
    
    print("✓ Seasonality mode can be configured")


def test_set_interval_width():
    """Test setting interval width"""
    m = Seer(interval_width=0.95)
    params = m.params()
    assert params["interval_width"] == 0.95
    
    print("✓ Interval width can be configured")


def test_save_load_workflow():
    """Test realistic save/load workflow"""
    df = pd.DataFrame({
        'ds': pd.date_range('2020-01-01', periods=60, freq='D'),
        'y': [50.0 + 0.5 * i for i in range(60)]
    })
    
    # Train and save model
    m1 = Seer()
    m1.fit(df)
    
    # Save to file
    json_str = m1.to_json()
    with open("/tmp/seer_model.json", "w") as f:
        f.write(json_str)
    
    # Load from file
    with open("/tmp/seer_model.json", "r") as f:
        loaded_json = f.read()
    
    m2 = Seer.from_json(loaded_json)
    
    # Verify loaded model has same parameters
    params1 = m1.params()
    params2 = m2.params()
    
    assert params2["fitted"] == True
    assert abs(params2["k"] - params1["k"]) < 1e-6
    assert abs(params2["m"] - params1["m"]) < 1e-6
    
    print("✓ Save/load workflow works correctly")


if __name__ == "__main__":
    print("Running M5 tests: Serialization and API Polish\n")
    
    test_params_completeness()
    test_json_serialization_round_trip()
    test_serialization_with_custom_components()
    test_set_seasonality_mode()
    test_set_interval_width()
    test_save_load_workflow()
    
    print("\n✅ All M5 tests passed!")
