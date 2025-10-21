pub mod data;
pub mod model;
pub mod seasonality;
pub mod trend;

// Pure Rust Prophet implementation with autodiff
pub mod prophet_autodiff;
pub mod prophet_optimizer;

// Nightly reverse-mode autodiff implementation
#[cfg(feature = "nightly-ad")]
pub mod prophet_nightly_ad;

pub use data::{ForecastResult, TimeSeriesData};
pub use model::{Farseer, TrendType};
