// src/lib.rs - Full structure with Python bindings

// Allow clippy::useless_conversion for PyO3-related false positives
#![allow(clippy::useless_conversion)]

use numpy::ToPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub mod core;
pub use core::{Farseer as CoreFarseer, ForecastResult, TimeSeriesData, TrendType};

pub type Result<T> = std::result::Result<T, FarseerError>;

#[derive(Debug, thiserror::Error)]
pub enum FarseerError {
    #[error("Data validation error: {0}")]
    DataValidation(String),

    #[error("Prediction error: {0}")]
    Prediction(String),

    #[error("Stan model error: {0}")]
    StanError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

// Helper function to convert pandas dates to strings
fn convert_ds_to_strings(ds_series: Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if ds_series.hasattr("dt")? {
        // It's a datetime column
        // Check if first value has time component
        let first_val = ds_series.call_method1("iloc", (0,))?;

        let dt_accessor = ds_series.getattr("dt")?;
        let format_str = if first_val.hasattr("hour")? {
            let hour: i32 = first_val.getattr("hour")?.extract()?;
            let minute: i32 = first_val.getattr("minute")?.extract()?;
            let second: i32 = first_val.getattr("second")?.extract()?;

            if hour == 0 && minute == 0 && second == 0 {
                "%Y-%m-%d"
            } else {
                "%Y-%m-%d %H:%M:%S"
            }
        } else {
            "%Y-%m-%d"
        };

        dt_accessor
            .call_method1("strftime", (format_str,))?
            .call_method0("tolist")?
            .extract()
    } else {
        // Already strings or can be converted
        ds_series
            .call_method1("astype", ("str",))?
            .call_method0("tolist")?
            .extract()
    }
}

// Python wrapper class
#[pyclass(subclass)]
struct Farseer {
    inner: CoreFarseer,
}

#[pymethods]
impl Farseer {
    #[new]
    #[allow(unused_variables)] // Prophet compatibility parameters
    #[pyo3(signature = (
        growth="linear",
        n_changepoints=25,
        changepoint_range=0.8,
        changepoint_prior_scale=0.05,
        yearly_seasonality=true,
        weekly_seasonality=true,
        daily_seasonality=false,
        seasonality_mode="additive",
        interval_width=0.8,
        uncertainty_samples=1000,
        changepoints=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        growth: &str,
        n_changepoints: usize,
        changepoint_range: f64,
        changepoint_prior_scale: f64,
        yearly_seasonality: bool,
        weekly_seasonality: bool,
        daily_seasonality: bool,
        seasonality_mode: &str,
        interval_width: f64,
        uncertainty_samples: usize, // Accepted but not used (for Prophet compatibility)
        changepoints: Option<Vec<String>>, // Accepted but not used (for Prophet compatibility)
    ) -> PyResult<Self> {
        let trend = match growth {
            "linear" => TrendType::Linear,
            "logistic" => TrendType::Logistic,
            "flat" => TrendType::Flat,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "growth must be 'linear', 'logistic', or 'flat'",
                ))
            }
        };

        let mut seer = CoreFarseer::new()
            .with_trend(trend)
            .with_changepoints(n_changepoints)
            .with_changepoint_range(changepoint_range)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
            .with_changepoint_prior_scale(changepoint_prior_scale)
            .with_seasonality_mode(seasonality_mode)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
            .with_interval_width(interval_width);

        // Set seasonality based on explicit parameters
        if yearly_seasonality {
            seer = seer.with_yearly_seasonality();
        } else {
            seer = seer.without_yearly_seasonality();
        }

        if weekly_seasonality {
            seer = seer.with_weekly_seasonality();
        } else {
            seer = seer.without_weekly_seasonality();
        }

        if daily_seasonality {
            seer = seer.with_daily_seasonality();
        } else {
            seer = seer.without_daily_seasonality();
        }

        // Set manual changepoints if provided
        if let Some(cp_vec) = changepoints {
            seer = seer.with_manual_changepoints(cp_vec);
        }

        Ok(Farseer { inner: seer })
    }

    /// Fit the model to historical data
    fn fit(&mut self, py: Python, df: &Bound<'_, PyAny>) -> PyResult<()> {
        // Validate required columns exist
        let columns = df
            .getattr("columns")?
            .call_method0("tolist")?
            .extract::<Vec<String>>()?;
        if !columns.contains(&"ds".to_string()) || !columns.contains(&"y".to_string()) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Dataframe must have 'ds' and 'y' columns",
            ));
        }

        // Filter out rows with NaN values in 'y' column (Prophet compatibility)
        let df_copy = df.call_method0("copy")?;
        let y_notna = df_copy.getattr("y")?.call_method0("notna")?;
        let df_clean = df_copy.call_method1("__getitem__", (y_notna,))?;

        // Validate minimum data points (Prophet compatibility: requires at least 2 points)
        let n_rows: usize = df_clean.getattr("shape")?.get_item(0)?.extract()?;
        if n_rows < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Dataframe has less than 2 non-NaN rows. Found {} rows.",
                n_rows
            )));
        }

        // Convert ds column to strings, handling datetime objects
        let ds_series = df_clean.getattr("ds")?;
        let ds = convert_ds_to_strings(ds_series)?;

        let y: Vec<f64> = df_clean.getattr("y")?.call_method0("tolist")?.extract()?;

        let cap = if df_clean.hasattr("cap")? {
            Some(df_clean.getattr("cap")?.call_method0("tolist")?.extract()?)
        } else {
            None
        };

        let weights = if df_clean.hasattr("weight")? {
            Some(
                df_clean
                    .getattr("weight")?
                    .call_method0("tolist")?
                    .extract()?,
            )
        } else {
            None
        };

        let mut data = TimeSeriesData::new(ds, y, cap, weights)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Extract regressor columns if any are defined
        // We need to get the list of regressors from the inner model
        let regressor_names = self.inner.get_regressor_names();
        for regressor_name in regressor_names {
            if df_clean.hasattr(regressor_name.as_str())? {
                let regressor_series = df_clean.getattr(regressor_name.as_str())?;
                let regressor_values: Vec<f64> =
                    regressor_series.call_method0("tolist")?.extract()?;
                data = data
                    .with_regressor(regressor_name.clone(), regressor_values)
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Regressor '{}' not found in dataframe",
                    regressor_name
                )));
            }
        }

        py.detach(|| self.inner.fit(&data))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(())
    }

    /// Make predictions (df=None uses training data like Prophet)
    #[pyo3(signature = (df=None))]
    fn predict(&self, py: Python, df: Option<Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        use std::collections::HashMap;

        // If df is None, use history (training data) like Prophet
        let (ds, cap, regressors) = if let Some(df_val) = df {
            // Convert ds column to strings, handling datetime objects
            let ds_series = df_val.getattr("ds")?;
            let ds = convert_ds_to_strings(ds_series)?;

            // Extract cap if it exists in the dataframe (for logistic growth)
            let cap = if df_val.hasattr("cap")? {
                let cap_series = df_val.getattr("cap")?;
                let cap_list = cap_series.call_method0("tolist")?;
                Some(cap_list.extract::<Vec<f64>>()?)
            } else {
                None
            };

            // Extract regressors
            let regressor_names = self.inner.get_regressor_names();
            let mut regressors_map = HashMap::new();
            for regressor_name in regressor_names {
                if df_val.hasattr(regressor_name.as_str())? {
                    let regressor_series = df_val.getattr(regressor_name.as_str())?;
                    let regressor_values: Vec<f64> =
                        regressor_series.call_method0("tolist")?.extract()?;
                    regressors_map.insert(regressor_name.clone(), regressor_values);
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Regressor '{}' not found in prediction dataframe",
                        regressor_name
                    )));
                }
            }

            (ds, cap, regressors_map)
        } else {
            // Use history if df is None
            if let Some(history) = self.inner.get_history() {
                let ds = history.ds.clone();
                let cap = history.cap.clone();
                let regressors = history.regressors.clone();
                (ds, cap, regressors)
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Model must be fitted before calling predict() without a dataframe",
                ));
            }
        };

        let forecast = py
            .detach(|| self.inner.predict_with_data(&ds, cap, &regressors))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let pd = py.import("pandas")?;
        let dict = PyDict::new(py);

        // Add columns in Prophet's order
        dict.set_item("ds", forecast.ds)?;
        dict.set_item("trend", forecast.trend.to_pyarray(py))?;
        dict.set_item("yhat_lower", forecast.yhat_lower.to_pyarray(py))?;
        dict.set_item("yhat_upper", forecast.yhat_upper.to_pyarray(py))?;
        dict.set_item("trend_lower", forecast.trend_lower.to_pyarray(py))?;
        dict.set_item("trend_upper", forecast.trend_upper.to_pyarray(py))?;
        dict.set_item("additive_terms", forecast.additive_terms.to_pyarray(py))?;
        dict.set_item(
            "additive_terms_lower",
            forecast.additive_terms_lower.to_pyarray(py),
        )?;
        dict.set_item(
            "additive_terms_upper",
            forecast.additive_terms_upper.to_pyarray(py),
        )?;

        // Always add weekly component and its bounds (zeros if not enabled)
        dict.set_item("weekly", forecast.weekly.to_pyarray(py))?;
        dict.set_item("weekly_lower", forecast.weekly_lower.to_pyarray(py))?;
        dict.set_item("weekly_upper", forecast.weekly_upper.to_pyarray(py))?;

        // Always add yearly component and its bounds (zeros if not enabled)
        dict.set_item("yearly", forecast.yearly.to_pyarray(py))?;
        dict.set_item("yearly_lower", forecast.yearly_lower.to_pyarray(py))?;
        dict.set_item("yearly_upper", forecast.yearly_upper.to_pyarray(py))?;

        dict.set_item(
            "multiplicative_terms",
            forecast.multiplicative_terms.to_pyarray(py),
        )?;
        dict.set_item(
            "multiplicative_terms_lower",
            forecast.multiplicative_terms_lower.to_pyarray(py),
        )?;
        dict.set_item(
            "multiplicative_terms_upper",
            forecast.multiplicative_terms_upper.to_pyarray(py),
        )?;
        dict.set_item("yhat", forecast.yhat.to_pyarray(py))?;

        let df = pd.call_method1("DataFrame", (dict,))?;
        Ok(df.into())
    }

    /// Generate future dataframe
    #[pyo3(signature = (periods, freq=None, include_history=None))]
    fn make_future_dataframe(
        &self,
        py: Python,
        periods: usize,
        freq: Option<&str>,
        include_history: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let freq = freq.unwrap_or("D");
        let include_history = include_history.unwrap_or(true);

        let dates = self
            .inner
            .make_future_dates(periods, freq, include_history)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let pd = py.import("pandas")?;

        // Convert date strings to pandas datetime objects
        // Use format='mixed' to handle both daily and hourly formats
        let kwargs = PyDict::new(py);
        kwargs.set_item("format", "mixed")?;
        let dates_converted = pd.call_method("to_datetime", (dates,), Some(&kwargs))?;

        let dict = PyDict::new(py);
        dict.set_item("ds", dates_converted)?;

        let df = pd.call_method1("DataFrame", (dict,))?;
        Ok(df.into())
    }

    /// Add custom seasonality
    #[pyo3(signature = (name, period, fourier_order, prior_scale=None, mode=None))]
    fn add_seasonality(
        mut slf: PyRefMut<'_, Self>,
        name: &str,
        period: f64,
        fourier_order: usize,
        prior_scale: Option<f64>,
        mode: Option<&str>,
    ) -> PyResult<Py<Self>> {
        slf.inner
            .add_seasonality(name, period, fourier_order, prior_scale, mode)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(slf.into())
    }

    /// Add regressor
    #[pyo3(signature = (name, prior_scale=None, standardize=None, mode=None))]
    fn add_regressor(
        mut slf: PyRefMut<'_, Self>,
        name: &str,
        prior_scale: Option<f64>,
        standardize: Option<&str>,
        mode: Option<&str>,
    ) -> PyResult<Py<Self>> {
        slf.inner
            .add_regressor(name, prior_scale, standardize, mode)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(slf.into())
    }

    /// Add custom holidays
    #[pyo3(signature = (name, dates, lower_window=None, upper_window=None, prior_scale=None, mode=None))]
    fn add_holidays(
        mut slf: PyRefMut<'_, Self>,
        name: &str,
        dates: Vec<String>,
        lower_window: Option<i32>,
        upper_window: Option<i32>,
        prior_scale: Option<f64>,
        mode: Option<&str>,
    ) -> PyResult<Py<Self>> {
        slf.inner
            .add_holidays(name, dates, lower_window, upper_window, prior_scale, mode)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(slf.into())
    }

    /// Add country holidays
    fn add_country_holidays(mut slf: PyRefMut<'_, Self>, country_name: &str) -> PyResult<Py<Self>> {
        slf.inner
            .add_country_holidays(country_name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(slf.into())
    }

    /// Get model parameters
    fn params(&self, py: Python) -> PyResult<Py<PyAny>> {
        let params = self.inner.get_params();
        let json_str = serde_json::to_string(&params)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let json = py.import("json")?;
        Ok(json.call_method1("loads", (json_str,))?.into())
    }

    /// Get list of regressor names
    fn get_regressor_names(&self) -> Vec<String> {
        self.inner.get_regressor_names()
    }

    /// Save model to file
    fn save(&self, path: &str) -> PyResult<()> {
        let json_str = self
            .inner
            .to_json()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        std::fs::write(path, json_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
    }

    /// Load model from file
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let json_str = std::fs::read_to_string(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        let inner = CoreFarseer::from_json(&json_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Serialize model to JSON string
    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Deserialize model from JSON string
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let inner = CoreFarseer::from_json(json)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Farseer { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "Farseer(growth={:?}, n_changepoints={})",
            self.inner.trend_type(),
            self.inner.n_changepoints()
        )
    }
}

/// Python module definition
#[pymodule]
fn _farseer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Farseer>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
