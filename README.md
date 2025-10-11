# Seer

A high-performance time series forecasting library built in Rust with Python bindings. Seer aims to provide Prophet-like forecasting capabilities with the speed and reliability of Rust.

[![Tests](https://img.shields.io/badge/tests-37%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.8+-blue)]()
[![Rust](https://img.shields.io/badge/rust-2021-orange)]()

## Features

### ✅ Implemented (v0.2.0 - M7 Partial)

- **Multiple Trend Models**: Linear, logistic (with capacity), and flat trends
- **Automatic Seasonality Detection**: Yearly, weekly, and daily patterns
- **Custom Seasonalities**: Add seasonality with any period and Fourier order
- **Holiday Effects**: Model special events with customizable windows
- **Additive & Multiplicative Modes**: Per-component seasonality modes
- **Uncertainty Intervals**: Configurable prediction intervals
- **Changepoint Detection**: Automatic trend change detection
- **Model Serialization**: Save and load trained models as JSON
- **Multiple Frequencies**: Hourly, daily, weekly, monthly, and yearly data support ⭐ **NEW**
- **BridgeStan Foundation**: Infrastructure for Stan-based Bayesian fitting ⭐ **NEW**
- **High Performance**: Rust core with Python bindings via PyO3

## Quick Start

### Installation

```bash
# From PyPI (when published)
pip install seer

# Development install
git clone https://github.com/yourusername/seer
cd seer
maturin develop --release
```

### Basic Usage

```python
import pandas as pd
import numpy as np
from seer import Seer

# Prepare data
df = pd.DataFrame({
    'ds': pd.date_range('2020-01-01', periods=100, freq='D'),
    'y': np.linspace(10, 20, 100) + np.random.normal(0, 0.5, 100)
})

# Create and fit model
m = Seer(
    growth='linear',
    yearly_seasonality=True,
    weekly_seasonality=True,
    n_changepoints=25
)
m.fit(df)

# Make predictions
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)

print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
```

### Advanced Features

#### Custom Seasonality

```python
# Add monthly seasonality
m = Seer()
m.add_seasonality('monthly', period=30.0, fourier_order=5)
m.fit(df)
```

#### Multiple Frequencies ⭐ NEW

```python
# Hourly data
df_hourly = pd.DataFrame({
    'ds': pd.date_range('2020-01-01', periods=168, freq='h'),
    'y': np.random.randn(168).cumsum()
})
m = Seer(yearly_seasonality=False, weekly_seasonality=False)
m.fit(df_hourly)
future = m.make_future_dataframe(periods=24, freq='H')  # 24 hours ahead
forecast = m.predict(future)

# Weekly data
future = m.make_future_dataframe(periods=12, freq='W')  # 12 weeks ahead

# Monthly data (30-day intervals)
future = m.make_future_dataframe(periods=6, freq='M')   # 6 months ahead

# Yearly data (365-day intervals)
future = m.make_future_dataframe(periods=3, freq='Y')   # 3 years ahead
```

#### Holidays

```python
# Add holiday effects
m = Seer()
m.add_holidays('new_year', ['2020-01-01', '2021-01-01'])
m.fit(df)
```

#### Logistic Growth

```python
# Model with capacity constraint
df['cap'] = 100  # Set capacity
m = Seer(growth='logistic')
m.fit(df)

future = m.make_future_dataframe(periods=30)
future['cap'] = 100
forecast = m.predict(future)
```

#### Model Persistence

```python
# Save model
json_str = m.to_json()
with open('model.json', 'w') as f:
    f.write(json_str)

# Load model
with open('model.json', 'r') as f:
    json_str = f.read()
m_loaded = Seer.from_json(json_str)
```

## API Reference

### Seer

```python
Seer(
    growth='linear',              # 'linear', 'logistic', or 'flat'
    n_changepoints=25,            # Number of potential changepoints
    changepoint_range=0.8,        # Proportion of history for changepoints
    changepoint_prior_scale=0.05, # Changepoint flexibility
    yearly_seasonality=True,      # Auto yearly seasonality
    weekly_seasonality=True,      # Auto weekly seasonality
    daily_seasonality=False,      # Auto daily seasonality
    seasonality_mode='additive',  # 'additive' or 'multiplicative'
    interval_width=0.8            # Width of uncertainty intervals
)
```

### Methods

- `fit(df)` - Fit model to historical data
- `predict(df)` - Generate forecasts
- `make_future_dataframe(periods, freq='D', include_history=True)` - Create future dates
- `add_seasonality(name, period, fourier_order, prior_scale=None, mode=None)` - Add custom seasonality
- `add_holidays(name, dates, lower_window=0, upper_window=0, prior_scale=None, mode=None)` - Add holiday effects
- `params()` - Get complete model state (63 fields)
- `to_json()` - Serialize model to JSON string
- `from_json(json_str)` - Deserialize model from JSON (static method)

## Architecture

Seer is built with a layered architecture:

```
┌─────────────────────────────────┐
│   Python API (seer.Seer)        │  ← High-level interface
├─────────────────────────────────┤
│   PyO3 Bindings (src/lib.rs)    │  ← Python ↔ Rust bridge
├─────────────────────────────────┤
│   Rust Core (src/core/)         │  ← Fast computation
│   - model.rs  (fit/predict)     │
│   - trend.rs  (frequencies)     │  ← NEW: H/D/W/M/Y support
│   - seasonality.rs (Fourier)    │
│   - data.rs   (structures)      │
│   - stan.rs   (BridgeStan)      │  ← NEW: Bayesian backend
└─────────────────────────────────┘
```

## Development Status

### Completed Milestones

- ✅ **M1**: Core forecasting (linear trend, seasonality, changepoints)
- ✅ **M2**: Multiple trend types (logistic, flat) and uncertainty intervals
- ✅ **M3**: Custom seasonalities and additive/multiplicative modes
- ✅ **M4**: Holiday effects with windows
- ✅ **M5**: Model serialization and configuration API
- ✅ **M6**: Comprehensive testing (50 Rust tests, 11 Python integration tests)
- 🚧 **M7** (Partial): Multiple frequencies ✅ + Stan infrastructure 🏗️

### Test Coverage

- **37 Rust tests** (30 module + 6 frequency + 1 ignored Stan) - ALL PASSING ✅
- **18 Python integration tests** (11 M6 + 7 M7 frequencies) - ALL PASSING ✅
- Tests validate: data structures, mathematical functions, API workflows, frequencies, Prophet-like behaviors

### Current Limitations

- **OLS-based fitting**: Uses ordinary least squares instead of Stan optimization
  - Works well for linear trends and basic seasonality
  - Logistic growth parameter estimation is simplified (full Stan integration in progress)
- **Simple uncertainty intervals**: Gaussian-based rather than full posterior sampling
- **No floor parameter**: Logistic growth supports `cap` but not `floor` yet
- **Approximate monthly/yearly**: Monthly uses 30-day periods, yearly uses 365-day periods

### In Progress (M7)

- ✅ **Multiple frequencies**: Hourly, daily, weekly, monthly, yearly support - COMPLETE
- 🏗️ **Stan/BridgeStan integration**: Infrastructure ready, optimization API research needed
- ⏳ **Posterior predictive sampling**: Planned after Stan integration

### Planned Features (M8+)

- Full Bayesian parameter estimation via Stan
- Posterior sampling for uncertainty quantification
- Cross-validation utilities
- Plotting helpers
- Performance benchmarks (OLS vs Stan)
- Documentation examples
- Floor parameter for logistic growth

## Differences from Prophet

Seer aims for Prophet API compatibility while leveraging Rust performance:

**Similarities:**
- Same DataFrame-based API (`ds`, `y`, `cap` columns)
- Similar forecasting components (trend, seasonality, holidays)
- Comparable results for linear trends and basic seasonality
- JSON model serialization

**Differences:**
- **Rust core** → faster execution
- **Multiple frequencies** → H/D/W/M/Y support vs Prophet's automatic detection
- **OLS fitting** vs Prophet's Stan optimization (for now)
- **Simplified logistic growth** parameter estimation (Stan integration in progress)
- **No plotting** (yet)

## Performance

Seer's Rust core provides significant performance benefits:

- Fast model fitting (OLS-based)
- Efficient Fourier feature computation
- Low-overhead Python bindings via PyO3
- Memory-efficient data structures

*(Formal benchmarks coming in M7+)*

## Contributing

Contributions are welcome! Areas of interest:

- Stan/BridgeStan integration
- Additional frequency support (hourly, monthly)
- Plotting utilities
- Documentation improvements
- Bug reports and feature requests

## Development

### Building from Source

```bash
# Install Rust and Python dependencies
cargo build

# Build Python package (development mode)
maturin develop --release

# Run tests
cargo test                                    # Rust tests (50 tests)
python python/seer/test_m6_integration.py     # Python integration tests
```

### Project Structure

```
seer/
├── src/
│   ├── lib.rs              # Python bindings (PyO3)
│   └── core/
│       ├── model.rs        # Core forecasting model
│       ├── trend.rs        # Trend functions
│       ├── seasonality.rs  # Fourier seasonality
│       └── data.rs         # Data structures
├── python/seer/            # Python tests
├── tests/                  # Rust integration tests
├── stan/                   # Stan model (not yet integrated)
├── Cargo.toml             # Rust dependencies
└── pyproject.toml         # Python package config
```

## License

[Add your license here]

## References

- [Prophet](https://facebook.github.io/prophet/) - Original forecasting library by Meta
- [PyO3](https://pyo3.rs/) - Rust bindings for Python
- [maturin](https://www.maturin.rs/) - Build and publish Rust Python packages

## Citation

If you use Seer in your research, please cite:

```
[Add citation information]
```

---

**Status**: Active Development (v0.2.0 - M7 Partial)  
**Last Updated**: October 11, 2025  
**Latest**: ✅ Multiple frequency support (H/D/W/M/Y) | 🏗️ Stan integration infrastructure
