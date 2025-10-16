pub mod cmdstan_optimizer;
pub mod data;
pub mod model;
pub mod seasonality;
pub mod stan; // Stan/BridgeStan integration
pub mod trend; // LBFGS optimizer using argmin-rs

pub use data::{ForecastResult, TimeSeriesData};
pub use model::{Farseer, TrendType};
