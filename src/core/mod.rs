pub mod model;
pub mod data;
pub mod trend;
pub mod seasonality;
pub mod stan;  // Stan/BridgeStan integration

pub use model::{Seer, TrendType};
pub use data::{TimeSeriesData, ForecastResult};