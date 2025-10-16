use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TimeSeriesData {
    pub ds: Vec<String>,
    pub y: Vec<f64>,
    pub cap: Option<Vec<f64>>,
    pub weights: Option<Vec<f64>>,
    pub regressors: HashMap<String, Vec<f64>>, // Additional regressor columns
}

impl TimeSeriesData {
    pub fn new(
        ds: Vec<String>,
        y: Vec<f64>,
        cap: Option<Vec<f64>>,
        weights: Option<Vec<f64>>,
    ) -> crate::Result<Self> {
        if ds.len() != y.len() {
            return Err(crate::FarseerError::DataValidation(
                "ds and y must have same length".to_string(),
            ));
        }

        if let Some(ref cap) = cap {
            if cap.len() != ds.len() {
                return Err(crate::FarseerError::DataValidation(
                    "cap must have same length as ds".to_string(),
                ));
            }
        }

        if let Some(ref weights) = weights {
            if weights.len() != ds.len() {
                return Err(crate::FarseerError::DataValidation(
                    "weights must have same length as ds".to_string(),
                ));
            }
            // Validate weights are non-negative
            if weights.iter().any(|&w| w < 0.0) {
                return Err(crate::FarseerError::DataValidation(
                    "weights must be non-negative".to_string(),
                ));
            }
        }

        Ok(Self {
            ds,
            y,
            cap,
            weights,
            regressors: HashMap::new(),
        })
    }

    pub fn with_regressor(mut self, name: String, values: Vec<f64>) -> crate::Result<Self> {
        if values.len() != self.ds.len() {
            return Err(crate::FarseerError::DataValidation(format!(
                "regressor '{}' must have same length as ds",
                name
            )));
        }
        self.regressors.insert(name, values);
        Ok(self)
    }

    pub fn len(&self) -> usize {
        self.ds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ds.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastResult {
    pub ds: Vec<String>,
    pub trend: Vec<f64>,
    pub yhat_lower: Vec<f64>,
    pub yhat_upper: Vec<f64>,
    pub trend_lower: Vec<f64>,
    pub trend_upper: Vec<f64>,
    pub additive_terms: Vec<f64>,
    pub additive_terms_lower: Vec<f64>,
    pub additive_terms_upper: Vec<f64>,
    pub weekly: Vec<f64>,
    pub weekly_lower: Vec<f64>,
    pub weekly_upper: Vec<f64>,
    pub yearly: Vec<f64>,
    pub yearly_lower: Vec<f64>,
    pub yearly_upper: Vec<f64>,
    pub multiplicative_terms: Vec<f64>,
    pub multiplicative_terms_lower: Vec<f64>,
    pub multiplicative_terms_upper: Vec<f64>,
    pub yhat: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeseries_data_new_valid() {
        let ds = vec!["2020-01-01".to_string(), "2020-01-02".to_string()];
        let y = vec![10.0, 11.0];
        let data = TimeSeriesData::new(ds.clone(), y.clone(), None, None);
        assert!(data.is_ok());
        let data = data.unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data.ds, ds);
        assert_eq!(data.y, y);
        assert!(data.cap.is_none());
        assert!(data.weights.is_none());
    }

    #[test]
    fn test_timeseries_data_with_cap() {
        let ds = vec!["2020-01-01".to_string(), "2020-01-02".to_string()];
        let y = vec![10.0, 11.0];
        let cap = vec![20.0, 22.0];
        let data = TimeSeriesData::new(ds.clone(), y.clone(), Some(cap.clone()), None);
        assert!(data.is_ok());
        let data = data.unwrap();
        assert_eq!(data.cap, Some(cap));
    }

    #[test]
    fn test_timeseries_data_with_weights() {
        let ds = vec!["2020-01-01".to_string(), "2020-01-02".to_string()];
        let y = vec![10.0, 11.0];
        let weights = vec![1.0, 0.5];
        let data = TimeSeriesData::new(ds.clone(), y.clone(), None, Some(weights.clone()));
        assert!(data.is_ok());
        let data = data.unwrap();
        assert_eq!(data.weights, Some(weights));
    }

    #[test]
    fn test_timeseries_data_length_mismatch() {
        let ds = vec!["2020-01-01".to_string(), "2020-01-02".to_string()];
        let y = vec![10.0]; // Wrong length
        let data = TimeSeriesData::new(ds, y, None, None);
        assert!(data.is_err());
    }

    #[test]
    fn test_timeseries_data_cap_length_mismatch() {
        let ds = vec!["2020-01-01".to_string(), "2020-01-02".to_string()];
        let y = vec![10.0, 11.0];
        let cap = vec![20.0]; // Wrong length
        let data = TimeSeriesData::new(ds, y, Some(cap), None);
        assert!(data.is_err());
    }

    #[test]
    fn test_timeseries_data_weights_length_mismatch() {
        let ds = vec!["2020-01-01".to_string(), "2020-01-02".to_string()];
        let y = vec![10.0, 11.0];
        let weights = vec![1.0]; // Wrong length
        let data = TimeSeriesData::new(ds, y, None, Some(weights));
        assert!(data.is_err());
    }

    #[test]
    fn test_timeseries_data_negative_weights() {
        let ds = vec!["2020-01-01".to_string(), "2020-01-02".to_string()];
        let y = vec![10.0, 11.0];
        let weights = vec![1.0, -0.5]; // Negative weight
        let data = TimeSeriesData::new(ds, y, None, Some(weights));
        assert!(data.is_err());
    }

    #[test]
    fn test_timeseries_data_empty() {
        let data = TimeSeriesData::new(vec![], vec![], None, None).unwrap();
        assert!(data.is_empty());
        assert_eq!(data.len(), 0);
    }
}
