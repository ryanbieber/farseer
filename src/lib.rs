// src/lib.rs - Full structure with Python bindings

use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::ToPyArray;

pub mod core;
pub use core::{Seer as CoreSeer, TimeSeriesData, ForecastResult, TrendType};

pub type Result<T> = std::result::Result<T, SeerError>;

#[derive(Debug, thiserror::Error)]
pub enum SeerError {
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
fn convert_ds_to_strings(py: Python, ds_series: &PyAny) -> PyResult<Vec<String>> {
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
#[pyclass]
struct Seer {
    inner: CoreSeer,
}

#[pymethods]
impl Seer {
    #[new]
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
    ))]
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
    ) -> PyResult<Self> {
        let trend = match growth {
            "linear" => TrendType::Linear,
            "logistic" => TrendType::Logistic,
            "flat" => TrendType::Flat,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "growth must be 'linear', 'logistic', or 'flat'"
            )),
        };
        
        let mut seer = CoreSeer::new()
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
        
        Ok(Seer { inner: seer })
    }
    
    /// Fit the model to historical data
    fn fit(&mut self, py: Python, df: &PyAny) -> PyResult<()> {
        // Filter out rows with NaN values in 'y' column (Prophet compatibility)
        let df_copy = df.call_method0("copy")?;
        let y_notna = df_copy.getattr("y")?.call_method0("notna")?;
        let df_clean = df_copy.call_method1("__getitem__", (y_notna,))?;
        
        // Convert ds column to strings, handling datetime objects
        let ds_series = df_clean.getattr("ds")?;
        let ds = convert_ds_to_strings(py, ds_series)?;
        
        let y: Vec<f64> = df_clean.getattr("y")?.call_method0("tolist")?.extract()?;
        
        let cap = if df_clean.hasattr("cap")? {
            Some(df_clean.getattr("cap")?.call_method0("tolist")?.extract()?)
        } else {
            None
        };
        
        let weights = if df_clean.hasattr("weight")? {
            Some(df_clean.getattr("weight")?.call_method0("tolist")?.extract()?)
        } else {
            None
        };
        
        let data = TimeSeriesData::new(ds, y, cap, weights)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        py.allow_threads(|| {
            self.inner.fit(&data)
        }).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        Ok(())
    }
    
    /// Make predictions
    fn predict(&self, py: Python, df: &PyAny) -> PyResult<PyObject> {
        // Convert ds column to strings, handling datetime objects
        let ds_series = df.getattr("ds")?;
        let ds = convert_ds_to_strings(py, ds_series)?;
        
        let forecast = py.allow_threads(|| {
            self.inner.predict(&ds)
        }).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let pd = py.import("pandas")?;
        let dict = PyDict::new(py);
        
        dict.set_item("ds", forecast.ds)?;
        dict.set_item("yhat", forecast.yhat.to_pyarray(py))?;
        dict.set_item("yhat_lower", forecast.yhat_lower.to_pyarray(py))?;
        dict.set_item("yhat_upper", forecast.yhat_upper.to_pyarray(py))?;
        dict.set_item("trend", forecast.trend.to_pyarray(py))?;
        
        if let Some(yearly) = forecast.yearly {
            dict.set_item("yearly", yearly.to_pyarray(py))?;
        }
        if let Some(weekly) = forecast.weekly {
            dict.set_item("weekly", weekly.to_pyarray(py))?;
        }
        
        let df = pd.call_method1("DataFrame", (dict,))?;
        Ok(df.into())
    }
    
    /// Generate future dataframe
    fn make_future_dataframe(
        &self,
        py: Python,
        periods: usize,
        freq: Option<&str>,
        include_history: Option<bool>,
    ) -> PyResult<PyObject> {
        let freq = freq.unwrap_or("D");
        let include_history = include_history.unwrap_or(true);
        
        let dates = self.inner.make_future_dates(periods, freq, include_history)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        let pd = py.import("pandas")?;
        
        // Convert date strings to pandas datetime objects
        let dates_converted = pd.call_method1("to_datetime", (dates,))?;
        
        let dict = PyDict::new(py);
        dict.set_item("ds", dates_converted)?;
        
        let df = pd.call_method1("DataFrame", (dict,))?;
        Ok(df.into())
    }
    
    /// Add custom seasonality
    fn add_seasonality(
        &mut self,
        name: &str,
        period: f64,
        fourier_order: usize,
        prior_scale: Option<f64>,
        mode: Option<&str>,
    ) -> PyResult<()> {
        self.inner.add_seasonality(name, period, fourier_order, prior_scale, mode)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
    
    /// Add custom holidays
    fn add_holidays(
        &mut self,
        name: &str,
        dates: Vec<String>,
        lower_window: Option<i32>,
        upper_window: Option<i32>,
        prior_scale: Option<f64>,
        mode: Option<&str>,
    ) -> PyResult<()> {
        self.inner.add_holidays(name, dates, lower_window, upper_window, prior_scale, mode)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
    
    /// Add country holidays
    fn add_country_holidays(&mut self, country_name: &str) -> PyResult<()> {
        self.inner.add_country_holidays(country_name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
    
    /// Get model parameters
    fn params(&self, py: Python) -> PyResult<PyObject> {
        let params = self.inner.get_params();
        let json_str = serde_json::to_string(&params)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let json = py.import("json")?;
        Ok(json.call_method1("loads", (json_str,))?.into())
    }
    
    /// Serialize model to JSON string
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
    
    /// Deserialize model from JSON string
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let inner = CoreSeer::from_json(json)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Seer { inner })
    }
    
    fn __repr__(&self) -> String {
        format!("Seer(growth={:?}, n_changepoints={})", 
                self.inner.trend_type(), self.inner.n_changepoints())
    }
}

/// Python module definition
#[pymodule]
fn seer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Seer>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
    
    