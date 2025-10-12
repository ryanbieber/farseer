# Seer Python Package

Fast Bayesian time series forecasting powered by Rust.

## Installation

```bash
pip install seer
```

Or build from source:

```bash
git clone https://github.com/ryanbieber/seer
cd seer
pip install -e .
```

## Quick Start

```python
import pandas as pd
from seer import Seer

# Prepare your data
df = pd.DataFrame({
    'ds': pd.date_range('2020-01-01', periods=365),
    'y': [your_time_series_data]
})

# Create and fit model
model = Seer()
model.fit(df)

# Make predictions
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Visualize
model.plot(forecast, history=df)
```

## Features

### ✨ Core Capabilities
- **Fast**: Rust-powered computation
- **Flexible**: Multiple trend types and seasonality modes
- **Accurate**: Bayesian approach with uncertainty quantification
- **Easy to use**: Scikit-learn-like interface

### 📊 Trend Models
- **Linear**: Constant growth rate
- **Logistic**: Saturating growth with capacity
- **Flat**: No trend (constant baseline)

### 🔄 Seasonality
- **Built-in**: Yearly, weekly, and daily patterns
- **Custom**: Add any periodic pattern (monthly, quarterly, etc.)
- **Modes**: Additive or multiplicative

### 🎉 Holiday Effects
- **Custom holidays**: Specify exact dates
- **Windows**: Model days before/after holidays
- **Country holidays**: Built-in support for major countries

### 🔬 Advanced Features
- **Changepoints**: Automatic trend change detection
- **Uncertainty intervals**: Configurable confidence levels
- **Model persistence**: Save/load models as JSON
- **Stan integration**: Bayesian inference with BridgeStan

## API Reference

### Model Initialization

```python
model = Seer(
    growth='linear',              # 'linear', 'logistic', or 'flat'
    n_changepoints=25,            # Number of potential changepoints
    changepoint_range=0.8,        # Proportion of history for changepoints
    changepoint_prior_scale=0.05, # Changepoint regularization
    yearly_seasonality=True,      # Enable yearly seasonality
    weekly_seasonality=True,      # Enable weekly seasonality
    daily_seasonality=False,      # Enable daily seasonality
    seasonality_mode='additive',  # 'additive' or 'multiplicative'
    interval_width=0.8,           # Uncertainty interval width (0-1)
)
```

### Core Methods

#### `fit(df)`
Fit the model to historical data.

```python
model.fit(df)  # df must have 'ds' and 'y' columns
```

#### `predict(df)`
Generate predictions for given dates.

```python
forecast = model.predict(future)
# Returns DataFrame with: ds, yhat, yhat_lower, yhat_upper, trend, yearly, weekly
```

#### `make_future_dataframe(periods, freq='D', include_history=True)`
Create a dataframe for future predictions.

```python
future = model.make_future_dataframe(
    periods=30,           # Number of periods ahead
    freq='D',             # Frequency: 'D', 'H', 'W', 'M', 'Y'
    include_history=True  # Include historical dates
)
```

### Customization Methods

#### `add_seasonality(name, period, fourier_order, prior_scale=None, mode=None)`
Add custom seasonality component.

```python
model.add_seasonality(
    name='monthly',
    period=30.5,        # Period in days
    fourier_order=5,    # Number of Fourier terms
    prior_scale=10.0,   # Regularization (optional)
    mode='additive'     # Mode (optional)
)
```

#### `add_holidays(name, dates, lower_window=None, upper_window=None, prior_scale=None, mode=None)`
Add custom holiday effects.

```python
model.add_holidays(
    name='christmas',
    dates=['2020-12-25', '2021-12-25', '2022-12-25'],
    lower_window=-2,    # Days before
    upper_window=2,     # Days after
    prior_scale=10.0,   # Regularization
    mode='additive'     # Mode
)
```

#### `add_country_holidays(country_name)`
Add country-specific holidays.

```python
model.add_country_holidays('US')
```

### Persistence Methods

#### `save(path)`
Save model to JSON file.

```python
model.save('my_model.json')
```

#### `load(path)` (class method)
Load model from JSON file.

```python
model = Seer.load('my_model.json')
```

#### `to_json()`
Serialize model to JSON string.

```python
json_str = model.to_json()
```

#### `from_json(json_str)` (class method)
Deserialize model from JSON string.

```python
model = Seer.from_json(json_str)
```

### Visualization Methods

#### `plot(forecast, ax=None, history=None)`
Plot forecast with uncertainty intervals.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
model.plot(forecast, ax=ax, history=df)
plt.show()
```

#### `plot_components(forecast, figsize=(10, 8))`
Plot individual forecast components.

```python
fig = model.plot_components(forecast)
plt.show()
```

### Utility Methods

#### `params()`
Get model parameters as dictionary.

```python
params = model.params()
print(params['n_changepoints'])  # 25
print(params['fitted'])          # True
```

## Examples

### Example 1: Daily Sales Forecasting

```python
import pandas as pd
import numpy as np
from seer import Seer

# Generate sample daily sales data
dates = pd.date_range('2020-01-01', periods=365, freq='D')
trend = np.arange(365) * 0.1 + 100
weekly = np.sin(2 * np.pi * np.arange(365) / 7) * 10
noise = np.random.randn(365) * 5
sales = trend + weekly + noise

df = pd.DataFrame({'ds': dates, 'y': sales})

# Create model with weekly seasonality
model = Seer(weekly_seasonality=True, yearly_seasonality=False)
model.fit(df)

# Forecast next 30 days
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Plot results
model.plot(forecast, history=df)
```

### Example 2: Seasonal Product Demand

```python
# 3 years of data with strong yearly seasonality
dates = pd.date_range('2018-01-01', periods=365*3, freq='D')
trend = np.arange(len(dates)) * 0.05 + 50
yearly = np.sin(2 * np.pi * np.arange(len(dates)) / 365.25) * 20
demand = trend + yearly + np.random.randn(len(dates)) * 3

df = pd.DataFrame({'ds': dates, 'y': demand})

# Model with yearly seasonality
model = Seer(yearly_seasonality=True, weekly_seasonality=False)
model.fit(df)

# Forecast next year
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# View components
model.plot_components(forecast)
```

### Example 3: Website Traffic with Holidays

```python
# Daily website traffic
dates = pd.date_range('2020-01-01', periods=365, freq='D')
base_traffic = 1000 + np.arange(365) * 0.5
weekly_pattern = np.sin(2 * np.pi * np.arange(365) / 7) * 100
traffic = base_traffic + weekly_pattern + np.random.randn(365) * 50

df = pd.DataFrame({'ds': dates, 'y': traffic})

# Model with holiday effects
model = Seer(weekly_seasonality=True, yearly_seasonality=False)

# Add Black Friday
model.add_holidays(
    'blackfriday',
    ['2020-11-27'],
    lower_window=-2,
    upper_window=2,
    prior_scale=20.0
)

model.fit(df)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
```

### Example 4: Custom Monthly Seasonality

```python
# Data with monthly patterns
dates = pd.date_range('2019-01-01', periods=730, freq='D')
monthly = np.sin(2 * np.pi * np.arange(730) / 30.5) * 15
y = 50 + monthly + np.random.randn(730) * 5

df = pd.DataFrame({'ds': dates, 'y': y})

# Add custom monthly seasonality
model = Seer(yearly_seasonality=False, weekly_seasonality=False)
model.add_seasonality(
    name='monthly',
    period=30.5,
    fourier_order=5
)

model.fit(df)
future = model.make_future_dataframe(periods=60)
forecast = model.predict(future)
```

### Example 5: Logistic Growth with Capacity

```python
# Growth with a natural limit
dates = pd.date_range('2020-01-01', periods=100, freq='D')
capacity = 1000
y = capacity / (1 + np.exp(-0.05 * (np.arange(100) - 50))) + np.random.randn(100) * 10

df = pd.DataFrame({
    'ds': dates,
    'y': y,
    'cap': [capacity] * 100  # Capacity limit
})

# Logistic growth model
model = Seer(
    growth='logistic',
    yearly_seasonality=False,
    weekly_seasonality=False
)
model.fit(df)

future = model.make_future_dataframe(periods=50)
forecast = model.predict(future)
```

### Example 6: Method Chaining

```python
# Build complex model with method chaining
forecast = (Seer(yearly_seasonality=False, weekly_seasonality=False)
    .add_seasonality('monthly', period=30.5, fourier_order=5)
    .add_seasonality('quarterly', period=91.25, fourier_order=3)
    .add_holidays('christmas', ['2020-12-25'], lower_window=-2, upper_window=2)
    .add_holidays('newyear', ['2020-01-01'], lower_window=-1, upper_window=1)
    .fit(df)
    .predict(model.make_future_dataframe(periods=30)))
```

### Example 7: Model Persistence

```python
# Train model
model = Seer()
model.fit(df)

# Save for later use
model.save('sales_forecast_model.json')

# Later... load and use
loaded_model = Seer.load('sales_forecast_model.json')
forecast = loaded_model.predict(future)
```

## Data Format

Your input data must be a pandas DataFrame with:
- **`ds`** column: Dates (datetime or string in 'YYYY-MM-DD' format)
- **`y`** column: Values to forecast (numeric)
- **`cap`** column (optional): Capacity for logistic growth

Example:
```python
df = pd.DataFrame({
    'ds': pd.date_range('2020-01-01', periods=100),
    'y': [100, 102, 105, ...],
    'cap': [1000, 1000, 1000, ...]  # optional
})
```

## Output Format

Predictions are returned as a pandas DataFrame with:
- **`ds`**: Dates
- **`yhat`**: Predicted values
- **`yhat_lower`**: Lower uncertainty bound
- **`yhat_upper`**: Upper uncertainty bound
- **`trend`**: Trend component
- **`yearly`**: Yearly seasonality (if enabled)
- **`weekly`**: Weekly seasonality (if enabled)

## Performance Tips

### For Fast Fitting
```python
model = Seer(
    n_changepoints=10,  # Fewer changepoints
)
```

### For Best Accuracy
```python
model = Seer(
    n_changepoints=50,   # More changepoints
)
```

### For Large Datasets
- Use daily or weekly aggregation
- Reduce `n_changepoints`
- Disable unnecessary seasonalities
- Consider sampling for initial exploration

## Common Patterns

### Weekly Business Data
```python
model = Seer(
    weekly_seasonality=True,
    yearly_seasonality=False,
    daily_seasonality=False
)
```

### Seasonal Products
```python
model = Seer(
    yearly_seasonality=True,
    weekly_seasonality=False
)
```

### Hourly Data
```python
model = Seer(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False
)
```

## Troubleshooting

### Issue: "Model must be fitted before prediction"
**Solution**: Call `model.fit(df)` before `model.predict()`

### Issue: "DataFrame must have 'ds' and 'y' columns"
**Solution**: Rename your columns to match:
```python
df = df.rename(columns={'date': 'ds', 'value': 'y'})
```

### Issue: Poor predictions
**Solutions**:
- Add relevant seasonalities
- Increase `n_changepoints`
- Add holiday effects
- Check for data quality issues

### Issue: Predictions are flat
**Solutions**:
- Check if `growth='flat'` (change to `'linear'`)
- Increase `changepoint_prior_scale`
- Verify input data has variation

## Comparison with Prophet

Seer is inspired by Facebook's Prophet but offers:
- ✅ **Faster**: Rust implementation
- ✅ **Simpler**: Fewer dependencies
- ✅ **Portable**: Easy deployment
- ✅ **Compatible**: Similar API

Migration from Prophet:
```python
# Prophet
from fbprophet import Prophet
m = Prophet()
m.fit(df)

# Seer
from seer import Seer
m = Seer()
m.fit(df)
```

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use Seer in academic work, please cite:
```bibtex
@software{seer2025,
  title={Seer: Fast Bayesian Time Series Forecasting},
  author={Bieber, Ryan},
  year={2025},
  url={https://github.com/ryanbieber/seer}
}
```

## Resources

- **GitHub**: https://github.com/ryanbieber/seer
- **Documentation**: [Full docs](https://seer.readthedocs.io)
- **Issues**: [Report bugs](https://github.com/ryanbieber/seer/issues)
- **Discussions**: [Ask questions](https://github.com/ryanbieber/seer/discussions)

## Acknowledgments

Inspired by Facebook's Prophet and built with:
- Rust for performance
- PyO3 for Python bindings
- Stan/BridgeStan for Bayesian inference

---

**Version**: 0.1.0  
**Status**: Beta  
**Python**: 3.7+  
**Rust**: 1.70+
