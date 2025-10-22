"""Tests for y-scaling behavior in Farseer."""

import polars as pl
import pytest
from datetime import datetime, timedelta
from farseer import Farseer


@pytest.fixture
def simple_data():
    """Simple dataset for testing."""
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(100)]
    y = [10.0 + i * 0.5 for i in range(100)]
    return pl.DataFrame({"ds": dates, "y": y})


class TestYScaling:
    """Test y-scaling behavior (absmax scaling)."""

    def test_y_scale_without_floor(self, simple_data):
        """Test that y_scale = max(abs(y)) when floor is not present."""
        m = Farseer()
        m.fit(simple_data)

        params = m.params()
        assert params["fitted"] is True
        assert params["y_scale"] > 0.0

        # y_scale should be max(abs(y))
        expected_y_scale = simple_data["y"].abs().max()
        assert params["y_scale"] == pytest.approx(expected_y_scale, rel=1e-6)

    def test_y_scale_with_floor(self):
        """Test that y_scale = max(abs(y - floor)) when floor is present."""
        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(100)]
        y = [20.0 + i * 0.5 for i in range(100)]
        floor = [10.0] * 100
        cap = [100.0] * 100

        df = pl.DataFrame({"ds": dates, "y": y, "floor": floor, "cap": cap})

        m = Farseer(growth="logistic")
        m.fit(df)

        params = m.params()
        assert params["fitted"] is True
        assert params["logistic_floor"] is True

        # y_scale should be max(abs(y - floor))
        expected_y_scale = max(abs(yi - fi) for yi, fi in zip(y, floor))
        assert params["y_scale"] == pytest.approx(expected_y_scale, rel=1e-6)

    def test_y_scale_minimum_value(self):
        """Test that y_scale has a minimum value of 1.0."""
        # Create data where all y values are very small
        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(100)]
        y = [0.001] * 100  # All values very close to zero

        df = pl.DataFrame({"ds": dates, "y": y})

        m = Farseer()
        m.fit(df)

        params = m.params()
        # y_scale should be at least 1.0 (minimum to avoid division issues)
        assert params["y_scale"] >= 1.0

    def test_prediction_unscaling_without_floor(self, simple_data):
        """Test that predictions are properly unscaled (yhat = yhat_scaled * y_scale)."""
        m = Farseer()
        m.fit(simple_data)

        # Predict on training data
        forecast = m.predict()

        # Check that predictions are in the same scale as input data
        assert (
            forecast["yhat"].min() >= simple_data["y"].min() - 20
        )  # Allow some margin
        assert forecast["yhat"].max() <= simple_data["y"].max() + 20

    def test_prediction_unscaling_with_floor(self):
        """Test that predictions include floor (yhat = yhat_scaled * y_scale + floor)."""
        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(100)]
        y = [20.0 + i * 0.5 for i in range(100)]
        floor = [10.0] * 100
        cap = [100.0] * 100

        df = pl.DataFrame({"ds": dates, "y": y, "floor": floor, "cap": cap})

        m = Farseer(growth="logistic")
        m.fit(df)

        # Predict on training data
        forecast = m.predict()

        # All predictions should be >= floor
        assert (forecast["yhat"] >= 10.0).all()
        # All predictions should be <= cap
        assert (forecast["yhat"] <= 100.0).all()
