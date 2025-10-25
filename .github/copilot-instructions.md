# GitHub Copilot Instructions

Always use UV to run python code.

These instructions are targeted at automated coding assistants (Copilot-style agents) contributing code to the Farseer (seer) repository. They focus on repository-specific conventions, build/test workflows, and code patterns discovered in the codebase.

## General rules
- Prefer minimal, targeted edits. Keep public APIs stable (the Python package exports `farseer.Farseer`).
- Preserve Polars-first design: code should prefer `polars.DataFrame` where possible; conversions to/from pandas are explicit in `farseer/__init__.py`.
- When changing behavior, update or add tests under `tests/` (the repo has comprehensive PyTest tests like `tests/test_python_api.py`).

## Data Scaling and Preprocessing

### Y-Scaling (absmax scaling)
Farseer uses Prophet's absmax scaling approach for the target variable:
- **When floor is absent**: `y_scale = max(abs(y))`, scaled as `y_scaled = y / y_scale`
- **When floor is present** (logistic growth): `y_scale = max(abs(y - floor))`, scaled as `y_scaled = (y - floor) / y_scale`
- Minimum y_scale is 1.0 to avoid division issues
- **Cap scaling**: When using logistic growth, cap is also scaled: `cap_scaled = (cap - floor) / y_scale`
- **Prediction unscaling**: Predictions are unscaled as `yhat = yhat_scaled * y_scale + floor`
- **Not currently configurable**, but follows Prophet's default behavior
- **Future consideration**: Could add support for minmax scaling (like Prophet's `scaling="minmax"`)

### Regressor Standardization
Farseer automatically detects and standardizes regressors:
- **Auto-detection mode** (default, `standardize="auto"`):
  - Binary regressors (only 0 and 1 values) are NOT standardized (mu=0, std=1)
  - Continuous regressors ARE standardized using (x - mean) / std
  - Constant regressors are never standardized
- **Explicit modes**:
  - `standardize="true"`: Always standardize (even binary regressors)
  - `standardize="false"`: Never standardize
- **Implementation**: See `src/core/model.rs` lines 580-630 for standardization logic
- **Testing**: See `tests/test_regressors.py` for validation tests

Example:
```python
m = Farseer()
m.add_regressor("weekend", standardize="auto")  # Binary 0/1 won't be standardized
m.add_regressor("temperature", standardize="auto")  # Continuous will be standardized
m.add_regressor("manual_feature", standardize="false")  # Force no standardization
```

### Floor Parameter (Logistic Growth)
Farseer fully supports floor parameter for logistic growth with saturating minimum:
- **Usage**: Add `floor` column to dataframe (like `cap` for logistic growth)
- **Validation**: `cap` must be greater than `floor` for all data points
- **Scaling**: Both y and cap are scaled relative to floor: `(value - floor) / y_scale`
- **Prediction**: Floor is added back during prediction: `trend * y_scale + floor`
- **Implementation**:
  - Data structure: `TimeSeriesData` has `floor: Option<Vec<f64>>` field
  - Python API: Automatically extracts `floor` column from dataframe if present
  - Model: `logistic_floor` boolean flag tracks whether floor is being used
- **Testing**: See `tests/test_prophet_compatibility.py::TestProphetDataPrep::test_logistic_floor`

Example:
```python
m = Farseer(growth='logistic')
df['floor'] = 1.5
df['cap'] = 10.0
m.fit(df)
future['floor'] = 1.5
future['cap'] = 10.0
forecast = m.predict(future)
```

## Model Estimation and Optimization

### Estimation Mode (MAP via L-BFGS)
- **Current implementation**: Farseer uses Maximum A Posteriori (MAP) estimation via L-BFGS optimizer
- **Stan backend**: CmdStan optimizer (110x faster than BridgeStan according to codebase comments)
- **Not configurable**: Unlike Prophet which supports both optimization and MCMC sampling
- **Future enhancement**: Could add MCMC sampling support using Stan's sampling capabilities
- **Uncertainty**: Current uncertainty intervals are approximations based on sigma_obs, not full posterior samples
- **Performance**: Optimized for speed with parallel execution (uses `SEER_NUM_THREADS` environment variable)

### Holiday Prior Scales
- **Independent configuration**: Holidays use their own `prior_scale` parameter (default: 10.0)
- **Separate from seasonality**: Holiday priors are independent of seasonality prior scales
- **Per-holiday control**: Each holiday can have its own prior_scale via `add_holidays()`
- **Implementation**: See `src/core/model.rs` HolidayConfig struct
- **Default value**: 10.0 (same as Prophet's default)

Example:
```python
m = Farseer()
# Different prior scales for different holidays
m.add_holidays("christmas", dates=["2020-12-25"], prior_scale=20.0)  # Strong prior
m.add_holidays("minor_holiday", dates=["2020-03-17"], prior_scale=5.0)  # Weak prior
```

## Conditional Seasonalities
**Status**: Fully implemented

Conditional seasonalities allow seasonality components to be active only when a boolean condition is met. This matches Prophet's conditional seasonality feature.

### Usage
```python
m = Farseer()

# Add conditional seasonality
m.add_seasonality(name='weekly_on_weekday', period=7, fourier_order=3,
                  condition_name='is_weekday')

# Training data must include the condition column
df['is_weekday'] = (df['ds'].dt.weekday < 5)
m.fit(df)

# Future dataframe must also include the condition column
future = m.make_future_dataframe(periods=30)
future['is_weekday'] = (future['ds'].dt.weekday < 5)
forecast = m.predict(future)
```

### Implementation Details
- **Configuration**: `SeasonalityConfig` struct has `condition_name: Option<String>` field
- **Data storage**: `TimeSeriesData` stores conditions in `HashMap<String, Vec<bool>>`
- **Feature masking**: Fourier series features are multiplied by condition values (0.0 or 1.0) during matrix construction
- **Validation**: Missing condition columns raise clear errors during fit/predict
- **Multiple conditions**: Supports multiple conditional seasonalities with different condition columns
- **Serialization**: Condition names are preserved in saved models

### Testing
- See `tests/test_conditional_seasonality.py` for 13 comprehensive tests
- Tests cover: basic usage, multiple conditions, error handling, pandas/polars compatibility, serialization
- All tests validate that masking properly zeros out features when condition is False

## Feature Gaps and Future Enhancements

### Model Serialization
**Status**: Known limitation

Due to PyO3 design, models loaded via `Farseer.load()` or `Farseer.from_json()` return Rust `Farseer` objects instead of Python `Farseer` wrapper objects. This means:

- **Limitation**: Loaded models do not support Polars DataFrames directly
- **Workaround**: Convert Polars to pandas before calling `predict()` on loaded models
- **Core functionality**: All fitting/prediction logic works identically
- **Future fix**: Would require custom PyO3 wrapper or alternative serialization approach

Example:
```python
# Save model
m.save('model.json')

# Load model (returns Rust Farseer)
m2 = Farseer.load('model.json')

# Use pandas DataFrame with loaded model
future_pd = future.to_pandas()
forecast = m2.predict(future_pd)
```

## Build & Development Workflow (Quick Commands)
- Build Python extension (local development): `maturin develop --release` (pyproject.toml / Cargo.toml are configured to build the Rust extension). Ensure `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` for Python 3.13+ if needed.
- Run Python tests: install dev deps and run `pytest -q` from repo root. Tests exercise the Python API and expect a built extension if making changes that touch Rust/PyO3 layers.

## Key Project Patterns and Files to Reference
- Rust core: `src/lib.rs`, `src/core/*` (model, trend, seasonality, stan glue). Rust implements the heavy compute and is exposed via PyO3.
- Python wrapper: `farseer/__init__.py` — contains conversions between pandas/polars and small Python-side helpers. Keep conversions explicit and ensure datetime precision (ns) handling remains correct.
- Utilities: `farseer/utilities.py` — example of how model params are post-processed (e.g., `regressor_coefficients`). Follow its patterns for reading `m.params()`.
- Stan model: files under `stan/` and maturin include directives in `pyproject.toml`. Changes touching Stan models must be mirrored in packaging include lists.
- Examples & tests: `examples/` and `tests/` show canonical usage; emulate these for new helpers/APIs.

## API & Data Expectations
- Input frames: functions expect `ds` (date) and `y` columns. Optional columns: `cap`, `floor`, `weight`, custom regressors.
- Polars is primary: internal conversion often moves to pandas before calling the Rust layer; keep conversions localized in the Python wrapper and minimize cross-module conversion duplication.

## Testing Guidance
- Add a focused PyTest test under `tests/` for any Python-visible change. Use small DataFrames (pandas or polars) as in `tests/test_python_api.py`.
- For changes in the Rust code, run a local maturin build then run pytest to validate the Python integration.
- Key test files:
  - `tests/test_prophet_compatibility.py` - Prophet feature parity tests
  - `tests/test_regressors.py` - Regressor standardization and functionality
  - `tests/test_python_api.py` - Core Python API tests
  - `tests/test_utilities.py` - Helper function tests (includes scaling tests)

## Edge Cases and Gotchas
- Datetime precision: the wrapper forces nanosecond datetime (`Datetime("ns")`) to ensure compatibility with pandas and internal computations. When creating/transforming `ds`, maintain nanosecond precision.
- Import paths: the Rust extension can be named `farseer` or `_farseer` depending on build configuration; `farseer/__init__.py` tries both. When adding new extension modules, update import attempts accordingly.
- Polars versioning: the project expects Polars >= 0.20.0; prefer using Polars API patterns from `farseer/__init__.py`.
- Floor validation: When using logistic growth with floor, ensure cap > floor for all data points

## When in Doubt
- Mirror the style and tests already in the repository. For examples, see how `Farseer.fit`, `predict`, `make_future_dataframe`, and `add_seasonality` are exercised in `tests/test_python_api.py`.

If you need more context or a behavioral change that touches multiple components (Rust, Python wrapper, and tests), ask for approval before large refactors and provide a small migration plan.

## After Applying Changes
- Update or add tests, then run the test suite locally. Commit small, incremental patches with clear messages.

## Component Introspection

### Model Parameters via `m.params()`
The `params()` method returns a dictionary with fitted model parameters. Expand this if needed to expose more internal state:
- Current keys: `k`, `m`, `delta`, `beta`, `sigma_obs`, `fitted`, `logistic_floor`, etc.
- For regressor coefficients, use `farseer.utilities.regressor_coefficients(m)` helper
- When adding new model parameters, update the params() serialization in `src/core/model.rs`

## Request Feedback
If any instructions are unclear or a deeper explanation of a component is needed, reply with the specific area (Rust core, Python wrapper, packaging, or tests) you'd like expanded.
