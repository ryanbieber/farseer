use crate::core::data::{ForecastResult, TimeSeriesData};
use crate::core::seasonality::{fourier_series, holiday_features, hstack};
use crate::core::stan::StanModel;
use crate::core::trend::{
    changepoint_matrix, flat_trend, future_dates, parse_ds, piecewise_linear, piecewise_logistic,
    select_changepoints, time_scale,
};
use crate::Result;
use chrono::NaiveDateTime;

#[derive(Debug, Clone, Copy)]
pub enum TrendType {
    Linear,
    Logistic,
    Flat,
}

/// Configuration for a holiday component
#[derive(Debug, Clone)]
pub struct HolidayConfig {
    pub name: String,
    pub dates: Vec<String>, // Date strings in "YYYY-MM-DD" format
    pub lower_window: i32,  // Days before holiday to include
    pub upper_window: i32,  // Days after holiday to include
    pub prior_scale: f64,
    pub mode: SeasonalityMode, // Can be Additive or Multiplicative
}

impl HolidayConfig {
    pub fn new(name: &str, dates: Vec<String>) -> Self {
        Self {
            name: name.to_string(),
            dates,
            lower_window: 0,
            upper_window: 0,
            prior_scale: 10.0, // Prophet default
            mode: SeasonalityMode::Additive,
        }
    }

    pub fn with_windows(mut self, lower: i32, upper: i32) -> Self {
        self.lower_window = lower;
        self.upper_window = upper;
        self
    }

    pub fn with_prior_scale(mut self, scale: f64) -> Self {
        self.prior_scale = scale;
        self
    }

    pub fn with_mode(mut self, mode: SeasonalityMode) -> Self {
        self.mode = mode;
        self
    }
}

/// Metadata for a holiday feature block in the design matrix
#[derive(Debug, Clone)]
struct HolidayBlock {
    name: String,
    start: usize,
    end: usize,
}

/// Configuration for an additional regressor
#[derive(Debug, Clone)]
pub struct RegressorConfig {
    pub name: String,
    pub prior_scale: f64,
    pub standardize: String, // "auto", "true", or "false"
    pub mode: SeasonalityMode,
    pub mu: f64,  // Mean (for standardization)
    pub std: f64, // Std dev (for standardization)
}

impl RegressorConfig {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            prior_scale: 10.0, // Default to holidays_prior_scale
            standardize: "auto".to_string(),
            mode: SeasonalityMode::Additive,
            mu: 0.0,
            std: 1.0,
        }
    }

    pub fn with_prior_scale(mut self, scale: f64) -> Self {
        self.prior_scale = scale;
        self
    }

    pub fn with_standardize(mut self, standardize: &str) -> Self {
        self.standardize = standardize.to_string();
        self
    }

    pub fn with_mode(mut self, mode: SeasonalityMode) -> Self {
        self.mode = mode;
        self
    }
}

/// Metadata for a regressor feature block in the design matrix
#[derive(Debug, Clone)]
struct RegressorBlock {
    name: String,
    start: usize,
    end: usize,
}

pub struct Farseer {
    trend: TrendType,
    n_changepoints: usize,
    changepoint_range: f64,
    changepoint_prior_scale: f64,
    yearly_seasonality: bool,
    weekly_seasonality: bool,
    daily_seasonality: bool,
    seasonality_mode: SeasonalityMode,

    // Manual changepoints
    manual_changepoints: Option<Vec<String>>, // User-specified changepoint dates
    specified_changepoints: bool,             // Whether changepoints were manually specified

    // Custom seasonalities registry
    seasonalities: Vec<SeasonalityConfig>,

    // Holidays registry
    holidays: Vec<HolidayConfig>,
    country_holidays: Vec<String>, // Countries to fetch holidays for

    // Regressors registry
    regressors: Vec<RegressorConfig>,

    // Fitted parameters
    fitted: bool,
    history: Option<TimeSeriesData>,
    // Time scaling and changepoints
    t0: Option<NaiveDateTime>,
    t_scale: f64,
    t_change: Vec<f64>,
    // Y scaling (Prophet compatibility)
    y_scale: f64,
    #[allow(dead_code)]
    logistic_floor: bool,
    // Trend params
    k: f64,
    m: f64,
    delta: Vec<f64>,
    // Seasonality params
    beta: Vec<f64>,
    season_blocks: Vec<SeasonBlock>,
    // Holiday params
    gamma: Vec<f64>, // Holiday coefficients
    holiday_blocks: Vec<HolidayBlock>,
    // Regressor params
    regressor_blocks: Vec<RegressorBlock>,
    // Uncertainty params
    sigma_obs: f64,
    interval_width: f64,
}

#[derive(Debug, Clone)]
struct SeasonBlock {
    name: String,
    period: f64,
    order: usize,
    start: usize,
    end: usize, // exclusive
}

#[derive(Debug, Clone)]
pub struct SeasonalityConfig {
    pub name: String,
    pub period: f64,
    pub fourier_order: usize,
    pub prior_scale: f64,
    pub mode: SeasonalityMode,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SeasonalityMode {
    Additive,
    Multiplicative,
}

impl SeasonalityConfig {
    pub fn new(name: &str, period: f64, fourier_order: usize) -> Self {
        Self {
            name: name.to_string(),
            period,
            fourier_order,
            prior_scale: 10.0,
            mode: SeasonalityMode::Additive,
        }
    }

    pub fn with_prior_scale(mut self, scale: f64) -> Self {
        self.prior_scale = scale;
        self
    }

    pub fn with_mode(mut self, mode: SeasonalityMode) -> Self {
        self.mode = mode;
        self
    }
}

impl Farseer {
    pub fn new() -> Self {
        Self {
            trend: TrendType::Linear,
            n_changepoints: 25,
            changepoint_range: 0.8,
            changepoint_prior_scale: 0.05,
            yearly_seasonality: true,
            weekly_seasonality: true,
            daily_seasonality: false,
            seasonality_mode: SeasonalityMode::Additive,
            manual_changepoints: None,
            specified_changepoints: false,
            seasonalities: Vec::new(),
            holidays: Vec::new(),
            country_holidays: Vec::new(),
            regressors: Vec::new(),
            fitted: false,
            history: None,
            t0: None,
            t_scale: 1.0,
            t_change: Vec::new(),
            y_scale: 1.0,
            logistic_floor: false,
            k: 0.0,
            m: 0.0,
            delta: Vec::new(),
            beta: Vec::new(),
            season_blocks: Vec::new(),
            gamma: Vec::new(),
            holiday_blocks: Vec::new(),
            regressor_blocks: Vec::new(),
            sigma_obs: 1.0,
            interval_width: 0.80,
        }
    }

    pub fn with_trend(mut self, trend: TrendType) -> Self {
        self.trend = trend;
        self
    }

    pub fn with_changepoints(mut self, n: usize) -> Self {
        self.n_changepoints = n;
        self
    }

    pub fn with_changepoint_range(mut self, range: f64) -> Result<Self> {
        // Note: NaN passes this validation (both comparisons are false).
        // If you want to reject NaN, add an explicit check: if range.is_nan() { ... }
        if !(0.0..=1.0).contains(&range) {
            return Err(crate::FarseerError::DataValidation(format!(
                "changepoint_range must be between 0 and 1, got {}",
                range
            )));
        }
        self.changepoint_range = range;
        Ok(self)
    }

    pub fn with_changepoint_prior_scale(mut self, scale: f64) -> Self {
        self.changepoint_prior_scale = scale;
        self
    }

    /// Set manual changepoints (dates as strings in "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS" format)
    /// This will override automatic changepoint detection.
    /// Changepoints must be within the training data range (validated during fit).
    pub fn with_manual_changepoints(mut self, changepoints: Vec<String>) -> Self {
        if changepoints.is_empty() {
            self.manual_changepoints = Some(Vec::new());
            self.specified_changepoints = true;
            self.n_changepoints = 0;
        } else {
            self.manual_changepoints = Some(changepoints.clone());
            self.specified_changepoints = true;
            self.n_changepoints = changepoints.len();
        }
        self
    }

    pub fn with_yearly_seasonality(mut self) -> Self {
        self.yearly_seasonality = true;
        self
    }

    pub fn without_yearly_seasonality(mut self) -> Self {
        self.yearly_seasonality = false;
        self
    }

    pub fn with_weekly_seasonality(mut self) -> Self {
        self.weekly_seasonality = true;
        self
    }

    pub fn without_weekly_seasonality(mut self) -> Self {
        self.weekly_seasonality = false;
        self
    }

    pub fn with_daily_seasonality(mut self) -> Self {
        self.daily_seasonality = true;
        self
    }

    pub fn without_daily_seasonality(mut self) -> Self {
        self.daily_seasonality = false;
        self
    }

    pub fn with_seasonality_mode(mut self, mode: &str) -> Result<Self> {
        let seasonality_mode = match mode.to_lowercase().as_str() {
            "additive" => SeasonalityMode::Additive,
            "multiplicative" => SeasonalityMode::Multiplicative,
            _ => {
                return Err(crate::FarseerError::DataValidation(format!(
                    "Invalid seasonality mode: {}. Must be 'additive' or 'multiplicative'.",
                    mode
                )))
            }
        };
        self.seasonality_mode = seasonality_mode;
        Ok(self)
    }

    pub fn with_interval_width(mut self, width: f64) -> Self {
        self.interval_width = width;
        self
    }

    pub fn fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        // Parse ds -> timestamps
        let ts: Vec<NaiveDateTime> = data
            .ds
            .iter()
            .map(|s| {
                parse_ds(s).ok_or_else(|| {
                    crate::FarseerError::DataValidation(format!("Invalid date format: {}", s))
                })
            })
            .collect::<std::result::Result<_, _>>()?;

        // Time scaling t in [0,1]
        let (t_hist, t_scale, t0) = time_scale(&ts);

        // Auto-detect seasonality based on data span (Prophet compatibility)
        // If user set seasonality to 'auto' (true), we auto-detect
        // Otherwise we respect their explicit choice
        let data_span_days = (ts[ts.len() - 1] - ts[0]).num_days() as f64;

        let use_yearly = if self.yearly_seasonality {
            // Only use yearly if we have at least 2 years of data
            data_span_days >= 730.0
        } else {
            false
        };

        let use_weekly = self.weekly_seasonality; // Weekly is usually always on
        let use_daily = self.daily_seasonality;

        // Changepoints (locations in t units)
        // Use manual changepoints if specified, otherwise auto-detect
        let t_change = if self.specified_changepoints {
            // Manual changepoints were provided
            if let Some(ref manual_cps) = self.manual_changepoints {
                if manual_cps.is_empty() {
                    // Empty manual changepoints = no changepoints
                    Vec::new()
                } else {
                    // Parse manual changepoint dates and convert to t values
                    let cp_dates: Vec<NaiveDateTime> = manual_cps
                        .iter()
                        .filter_map(|ds_str| parse_ds(ds_str))
                        .collect();

                    // Note: Unlike Prophet, we don't strictly validate that changepoints
                    // are within the training data range. This allows users to specify
                    // changepoints in gaps in the data or even slightly outside for
                    // forecasting purposes. The changepoint will simply not have an
                    // effect if it's outside the data range.

                    // Convert to t values (normalized time)
                    let mut t_change_vals: Vec<f64> = cp_dates
                        .iter()
                        .map(|cp_date| {
                            let dt = *cp_date - t0;
                            let s = dt.num_microseconds().unwrap_or(0) as f64 / 1_000_000.0;
                            if t_scale > 0.0 {
                                s / t_scale
                            } else {
                                0.0
                            }
                        })
                        .collect();

                    // Sort and remove duplicates
                    t_change_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    t_change_vals.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

                    // Filter out changepoints that are outside the [0, 1] normalized time range
                    // (these would be outside the history's time span)
                    t_change_vals.retain(|&t| (0.0..=1.0).contains(&t));

                    t_change_vals
                }
            } else {
                // specified_changepoints is true but manual_changepoints is None
                // This shouldn't happen, but handle gracefully
                Vec::new()
            }
        } else {
            // Automatic changepoint detection
            select_changepoints(&t_hist, self.n_changepoints, self.changepoint_range)
        };

        // t in days since start for seasonality features
        let t_days: Vec<f64> = ts
            .iter()
            .map(|dt| (*dt - t0).num_seconds() as f64 / 86_400.0)
            .collect();

        // Build seasonality registry from toggles and custom seasonalities
        let mut all_seasonalities = Vec::new();
        if use_yearly {
            all_seasonalities.push(
                SeasonalityConfig::new("yearly", 365.25, 10).with_mode(self.seasonality_mode),
            );
        }
        if use_weekly {
            all_seasonalities
                .push(SeasonalityConfig::new("weekly", 7.0, 3).with_mode(self.seasonality_mode));
        }
        if use_daily {
            all_seasonalities
                .push(SeasonalityConfig::new("daily", 1.0, 4).with_mode(self.seasonality_mode));
        }
        // Add custom seasonalities
        all_seasonalities.extend(self.seasonalities.clone());

        let mut blocks_additive: Vec<Vec<Vec<f64>>> = Vec::new();
        let mut blocks_multiplicative: Vec<Vec<Vec<f64>>> = Vec::new();
        let mut season_blocks: Vec<SeasonBlock> = Vec::new();
        let mut col_start = 0usize;

        for config in &all_seasonalities {
            let fb = fourier_series(&t_days, config.period, config.fourier_order);
            let cols = if fb.is_empty() { 0 } else { fb[0].len() };
            if cols > 0 {
                match config.mode {
                    SeasonalityMode::Additive => blocks_additive.push(fb),
                    SeasonalityMode::Multiplicative => blocks_multiplicative.push(fb),
                }
                let col_end = col_start + cols;
                season_blocks.push(SeasonBlock {
                    name: config.name.clone(),
                    period: config.period,
                    order: config.fourier_order,
                    start: col_start,
                    end: col_end,
                });
                col_start = col_end;
            }
        }

        // Combine additive and multiplicative blocks
        let mut blocks = blocks_additive;
        blocks.extend(blocks_multiplicative);

        // Add holiday features
        let mut holiday_blocks_additive: Vec<Vec<Vec<f64>>> = Vec::new();
        let mut holiday_blocks_multiplicative: Vec<Vec<Vec<f64>>> = Vec::new();
        let mut holiday_blocks: Vec<HolidayBlock> = Vec::new();

        for config in &self.holidays {
            let hf = holiday_features(&ts, &config.dates, config.lower_window, config.upper_window);
            let cols = if hf.is_empty() { 0 } else { hf[0].len() };
            if cols > 0 {
                match config.mode {
                    SeasonalityMode::Additive => holiday_blocks_additive.push(hf),
                    SeasonalityMode::Multiplicative => holiday_blocks_multiplicative.push(hf),
                }
                let col_end = col_start + cols;
                holiday_blocks.push(HolidayBlock {
                    name: config.name.clone(),
                    start: col_start,
                    end: col_end,
                });
                col_start = col_end;
            }
        }

        // Combine holiday blocks with seasonality blocks
        blocks.extend(holiday_blocks_additive);
        blocks.extend(holiday_blocks_multiplicative);

        // Add regressor features
        let mut regressor_blocks_additive: Vec<Vec<Vec<f64>>> = Vec::new();
        let mut regressor_blocks_multiplicative: Vec<Vec<Vec<f64>>> = Vec::new();
        let mut regressor_blocks: Vec<RegressorBlock> = Vec::new();
        let mut updated_regressors = Vec::new();

        for config in &self.regressors {
            // Check if regressor column exists in data
            if let Some(regressor_values) = data.regressors.get(&config.name) {
                // Compute standardization parameters
                let mut updated_config = config.clone();

                // Check for unique values (for binary detection)
                let mut unique_vals = Vec::new();
                for &val in regressor_values {
                    if !unique_vals.iter().any(|&v: &f64| (v - val).abs() < 1e-10) {
                        unique_vals.push(val);
                        if unique_vals.len() > 2 {
                            break; // More than 2 unique values, not binary
                        }
                    }
                }

                let mut should_standardize = false;

                match config.standardize.to_lowercase().as_str() {
                    "auto" => {
                        // Auto: standardize unless binary (only 0 and 1)
                        if unique_vals.len() >= 2 {
                            let mut sorted_unique = unique_vals.clone();
                            sorted_unique.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            // Check if it's binary 0/1
                            should_standardize = !(sorted_unique.len() == 2
                                && (sorted_unique[0] - 0.0).abs() < 1e-10
                                && (sorted_unique[1] - 1.0).abs() < 1e-10);
                        }
                    }
                    "true" => should_standardize = true,
                    "false" => should_standardize = false,
                    _ => should_standardize = false,
                }

                // Special case: if regressor is constant, don't standardize
                if unique_vals.len() < 2 {
                    should_standardize = false;
                }

                let (mu, std) = if should_standardize {
                    let mean = regressor_values.iter().sum::<f64>() / regressor_values.len() as f64;
                    let variance = regressor_values
                        .iter()
                        .map(|&x| (x - mean).powi(2))
                        .sum::<f64>()
                        / regressor_values.len() as f64;
                    let std_dev = variance.sqrt();
                    (mean, std_dev.max(1.0)) // Ensure std is at least 1.0 to avoid division by zero
                } else {
                    (0.0, 1.0)
                };

                updated_config.mu = mu;
                updated_config.std = std;

                // Standardize regressor values
                let standardized: Vec<f64> =
                    regressor_values.iter().map(|&x| (x - mu) / std).collect();

                // Add as a single column feature
                let regressor_feature: Vec<Vec<f64>> =
                    standardized.iter().map(|&v| vec![v]).collect();

                let cols = 1; // Regressors are always single columns
                match updated_config.mode {
                    SeasonalityMode::Additive => regressor_blocks_additive.push(regressor_feature),
                    SeasonalityMode::Multiplicative => {
                        regressor_blocks_multiplicative.push(regressor_feature)
                    }
                }

                let col_end = col_start + cols;
                regressor_blocks.push(RegressorBlock {
                    name: config.name.clone(),
                    start: col_start,
                    end: col_end,
                });
                col_start = col_end;

                updated_regressors.push(updated_config);
            } else {
                return Err(crate::FarseerError::DataValidation(format!(
                    "Regressor '{}' not found in dataframe",
                    config.name
                )));
            }
        }

        // Update regressor configs with computed mu/std
        self.regressors = updated_regressors;

        // Combine regressor blocks
        blocks.extend(regressor_blocks_additive);
        blocks.extend(regressor_blocks_multiplicative);

        // Prepare design matrix X for Stan/OLS
        let x_matrix = if !blocks.is_empty() {
            hstack(&blocks)
        } else {
            vec![vec![]; data.y.len()]
        };

        // Prepare capacity for logistic growth
        let cap = if let Some(ref cap_data) = data.cap {
            cap_data.clone()
        } else {
            vec![1e9; data.y.len()] // Large default capacity
        };

        // Determine trend indicator for Stan
        let trend_indicator = match self.trend {
            TrendType::Linear => 0,
            TrendType::Logistic => 1,
            TrendType::Flat => 2,
        };

        // Build feature mode indicators (additive vs multiplicative)
        let total_features = if x_matrix.is_empty() {
            0
        } else {
            x_matrix[0].len()
        };
        let mut s_a = vec![0.0; total_features];
        let mut s_m = vec![0.0; total_features];

        // Set mode indicators for seasonality blocks
        for block in &season_blocks {
            for i in block.start..block.end {
                // Find the corresponding seasonality config
                let config = all_seasonalities
                    .iter()
                    .find(|c| c.name == block.name)
                    .unwrap();
                match config.mode {
                    SeasonalityMode::Additive => s_a[i] = 1.0,
                    SeasonalityMode::Multiplicative => s_m[i] = 1.0,
                }
            }
        }

        // Set mode indicators for holiday blocks
        for block in &holiday_blocks {
            for i in block.start..block.end {
                // Find the corresponding holiday config
                let config = self.holidays.iter().find(|c| c.name == block.name).unwrap();
                match config.mode {
                    SeasonalityMode::Additive => s_a[i] = 1.0,
                    SeasonalityMode::Multiplicative => s_m[i] = 1.0,
                }
            }
        }

        // Set mode indicators for regressor blocks
        for block in &regressor_blocks {
            for i in block.start..block.end {
                // Find the corresponding regressor config
                let config = self
                    .regressors
                    .iter()
                    .find(|c| c.name == block.name)
                    .unwrap();
                match config.mode {
                    SeasonalityMode::Additive => s_a[i] = 1.0,
                    SeasonalityMode::Multiplicative => s_m[i] = 1.0,
                }
            }
        }

        // Store regressor blocks for prediction
        self.regressor_blocks = regressor_blocks;

        // Use Stan for parameter estimation
        let stan_model = StanModel::new()?;

        // Scale y (Prophet uses absmax scaling by default)
        // y_scale = max(abs(y)) to keep predictions interpretable
        let y_scale = data
            .y
            .iter()
            .map(|&v| v.abs())
            .fold(0.0_f64, f64::max)
            .max(1.0); // Minimum 1.0 to avoid division issues

        let y_scaled: Vec<f64> = data.y.iter().map(|&v| v / y_scale).collect();

        // Scale cap if logistic growth
        let cap_scaled: Vec<f64> = cap.iter().map(|&v| v / y_scale).collect();

        // Prepare prior scales for regressors
        let mut sigmas = vec![10.0; total_features]; // Default prior scale

        // Set custom prior scales for seasonality features
        for block in &season_blocks {
            let config = all_seasonalities
                .iter()
                .find(|c| c.name == block.name)
                .unwrap();
            sigmas[block.start..block.end].fill(config.prior_scale);
        }

        // Set custom prior scales for holiday features
        for block in &holiday_blocks {
            let config = self.holidays.iter().find(|c| c.name == block.name).unwrap();
            sigmas[block.start..block.end].fill(config.prior_scale);
        }

        // Set custom prior scales for regressor features
        for block in &self.regressor_blocks {
            let config = self
                .regressors
                .iter()
                .find(|c| c.name == block.name)
                .unwrap();
            sigmas[block.start..block.end].fill(config.prior_scale);
        }

        // Use CmdStan optimizer (110x faster than BridgeStan)
        // Set LD_LIBRARY_PATH=./stan to find TBB libraries
        let result = stan_model.optimize_with_cmdstan(
            &t_hist,
            &y_scaled,   // Use scaled y
            &cap_scaled, // Use scaled cap
            &x_matrix,
            &sigmas,
            self.changepoint_prior_scale,
            trend_indicator,
            &s_a,
            &s_m,
            &t_change,
            data.weights.as_deref(), // Pass weights
        )?;

        // Parameters are in scaled space - we'll unscale during prediction
        let k = result.k;
        let m = result.m;
        let delta = result.delta;
        let beta = result.beta;
        let sigma_obs = result.sigma_obs;

        // Save state
        self.history = Some(data.clone());
        self.fitted = true;
        self.t0 = Some(t0);
        self.t_scale = t_scale;
        self.t_change = t_change;
        self.y_scale = y_scale; // Save scaling factor
        self.k = k;
        self.m = m;
        self.delta = delta;
        self.beta = beta;
        self.season_blocks = season_blocks;
        self.gamma = Vec::new(); // Holiday coefficients included in beta
        self.holiday_blocks = holiday_blocks;
        self.sigma_obs = sigma_obs;
        Ok(())
    }

    pub fn predict(&self, ds: &[String]) -> Result<ForecastResult> {
        self.predict_with_data(ds, None, &std::collections::HashMap::new())
    }

    pub fn predict_with_cap(&self, ds: &[String], cap: Option<Vec<f64>>) -> Result<ForecastResult> {
        self.predict_with_data(ds, cap, &std::collections::HashMap::new())
    }

    pub fn predict_with_data(
        &self,
        ds: &[String],
        cap: Option<Vec<f64>>,
        regressors: &std::collections::HashMap<String, Vec<f64>>,
    ) -> Result<ForecastResult> {
        if !self.fitted {
            return Err(crate::FarseerError::Prediction(
                "Model must be fitted before prediction".to_string(),
            ));
        }

        // Build t in model units for incoming ds
        let t0 = self.t0.unwrap();
        let mut t: Vec<f64> = Vec::with_capacity(ds.len());
        for s in ds.iter() {
            let dt = parse_ds(s).ok_or_else(|| {
                crate::FarseerError::Prediction(format!("Invalid date format: {}", s))
            })?;
            let us = (dt - t0).num_microseconds().unwrap_or(0) as f64 / 1_000_000.0;
            let ti = if self.t_scale > 0.0 {
                us / self.t_scale
            } else {
                0.0
            };
            t.push(ti);
        }

        // Changepoint matrix and piecewise trend
        let a = changepoint_matrix(&t, &self.t_change);

        let trend = match self.trend {
            TrendType::Linear => {
                piecewise_linear(self.k, self.m, &self.delta, &t, &a, &self.t_change)
            }
            TrendType::Logistic => {
                // Use cap from parameter if provided, otherwise use training cap if available
                let cap_vec_unscaled = if let Some(cap_provided) = cap {
                    // Validate cap length matches ds length
                    if cap_provided.len() != ds.len() {
                        return Err(crate::FarseerError::Prediction(format!(
                            "Cap length ({}) must match ds length ({})",
                            cap_provided.len(),
                            ds.len()
                        )));
                    }
                    cap_provided
                } else if let Some(history) = self.history.as_ref() {
                    if let Some(ref cap_data) = history.cap {
                        // Extend cap to predict length if needed (use last cap value)
                        let last_cap = cap_data.last().copied().unwrap_or(1e9);
                        let mut cap_full = cap_data.clone();
                        while cap_full.len() < ds.len() {
                            cap_full.push(last_cap);
                        }
                        cap_full.truncate(ds.len());
                        cap_full
                    } else {
                        vec![1e9; ds.len()]
                    }
                } else {
                    // No history available (e.g., deserialized model) - use default large cap
                    vec![1e9; ds.len()]
                };
                // Scale cap by y_scale (model parameters are in scaled space)
                let cap_vec: Vec<f64> =
                    cap_vec_unscaled.iter().map(|&v| v / self.y_scale).collect();
                piecewise_logistic(
                    self.k,
                    self.m,
                    &self.delta,
                    &t,
                    &cap_vec,
                    &a,
                    &self.t_change,
                )
            }
            TrendType::Flat => flat_trend(self.m, ds.len()),
        };

        // Seasonality contributions
        let mut yearly_comp: Option<Vec<f64>> = None;
        let mut weekly_comp: Option<Vec<f64>> = None;
        let mut yearly_is_multiplicative = false;
        let mut weekly_is_multiplicative = false;
        let mut seasonal_additive = vec![0.0; ds.len()];
        let mut seasonal_multiplicative = vec![0.0; ds.len()];

        if !self.beta.is_empty() && !self.season_blocks.is_empty() {
            let t0 = self.t0.unwrap();
            let t_days: Vec<f64> = ds
                .iter()
                .map(|s| {
                    let dt = parse_ds(s).unwrap_or(t0);
                    (dt - t0).num_seconds() as f64 / 86_400.0
                })
                .collect();

            // Build seasonality registry to determine mode for each component
            let mut all_seasonalities = Vec::new();
            if self.yearly_seasonality {
                all_seasonalities.push(
                    SeasonalityConfig::new("yearly", 365.25, 10).with_mode(self.seasonality_mode),
                );
            }
            if self.weekly_seasonality {
                all_seasonalities.push(
                    SeasonalityConfig::new("weekly", 7.0, 3).with_mode(self.seasonality_mode),
                );
            }
            if self.daily_seasonality {
                all_seasonalities
                    .push(SeasonalityConfig::new("daily", 1.0, 4).with_mode(self.seasonality_mode));
            }
            all_seasonalities.extend(self.seasonalities.clone());

            for (idx, block) in self.season_blocks.iter().enumerate() {
                if block.start == block.end {
                    continue;
                }
                let x_block = fourier_series(&t_days, block.period, block.order);
                let beta_slice = &self.beta[block.start..block.end];
                let mut comp = vec![0.0; ds.len()];
                for i in 0..x_block.len() {
                    let row = &x_block[i];
                    let mut v = 0.0;
                    for (j, &xij) in row.iter().enumerate() {
                        v += xij * beta_slice[j];
                    }
                    comp[i] = v;
                }

                // Determine mode for this component
                let mode = if idx < all_seasonalities.len() {
                    all_seasonalities[idx].mode
                } else {
                    SeasonalityMode::Additive
                };

                match mode {
                    SeasonalityMode::Additive => {
                        for i in 0..ds.len() {
                            seasonal_additive[i] += comp[i];
                        }
                    }
                    SeasonalityMode::Multiplicative => {
                        for i in 0..ds.len() {
                            seasonal_multiplicative[i] += comp[i];
                        }
                    }
                }

                if block.name == "yearly" {
                    yearly_comp = Some(comp.clone());
                    yearly_is_multiplicative = matches!(mode, SeasonalityMode::Multiplicative);
                }
                if block.name == "weekly" {
                    weekly_comp = Some(comp.clone());
                    weekly_is_multiplicative = matches!(mode, SeasonalityMode::Multiplicative);
                }
            }
        }

        // Holiday contributions
        if !self.holiday_blocks.is_empty() {
            let t0 = self.t0.unwrap();
            let future_ts: Vec<NaiveDateTime> =
                ds.iter().map(|s| parse_ds(s).unwrap_or(t0)).collect();

            for (idx, h_block) in self.holiday_blocks.iter().enumerate() {
                if h_block.start == h_block.end {
                    continue;
                }

                // Get holiday config for this block
                if idx < self.holidays.len() {
                    let config = &self.holidays[idx];
                    let hf = holiday_features(
                        &future_ts,
                        &config.dates,
                        config.lower_window,
                        config.upper_window,
                    );
                    let beta_slice = &self.beta[h_block.start..h_block.end];

                    let mut comp = vec![0.0; ds.len()];
                    for i in 0..hf.len() {
                        let row = &hf[i];
                        let mut v = 0.0;
                        for (j, &xij) in row.iter().enumerate() {
                            v += xij * beta_slice[j];
                        }
                        comp[i] = v;
                    }

                    // Apply based on mode
                    match config.mode {
                        SeasonalityMode::Additive => {
                            for i in 0..ds.len() {
                                seasonal_additive[i] += comp[i];
                            }
                        }
                        SeasonalityMode::Multiplicative => {
                            for i in 0..ds.len() {
                                seasonal_multiplicative[i] += comp[i];
                            }
                        }
                    }
                }
            }
        }

        // Regressor contributions
        if !self.regressor_blocks.is_empty() {
            for r_block in &self.regressor_blocks {
                if r_block.start == r_block.end {
                    continue;
                }

                // Get regressor config for this block
                if let Some(config) = self.regressors.iter().find(|r| r.name == r_block.name) {
                    // Check if regressor data is provided
                    if let Some(regressor_values) = regressors.get(&config.name) {
                        // Validate length
                        if regressor_values.len() != ds.len() {
                            return Err(crate::FarseerError::Prediction(format!(
                                "Regressor '{}' length ({}) must match prediction length ({})",
                                config.name,
                                regressor_values.len(),
                                ds.len()
                            )));
                        }

                        // Standardize regressor values using stored mu and std
                        let standardized: Vec<f64> = regressor_values
                            .iter()
                            .map(|&x| (x - config.mu) / config.std)
                            .collect();

                        // Get beta coefficients for this regressor
                        let beta_slice = &self.beta[r_block.start..r_block.end];

                        // Compute contribution (regressor is a single column)
                        let mut comp = vec![0.0; ds.len()];
                        for i in 0..standardized.len() {
                            comp[i] = standardized[i] * beta_slice[0];
                        }

                        // Apply based on mode
                        match config.mode {
                            SeasonalityMode::Additive => {
                                for i in 0..ds.len() {
                                    seasonal_additive[i] += comp[i];
                                }
                            }
                            SeasonalityMode::Multiplicative => {
                                for i in 0..ds.len() {
                                    seasonal_multiplicative[i] += comp[i];
                                }
                            }
                        }
                    } else {
                        return Err(crate::FarseerError::Prediction(format!(
                            "Regressor '{}' data not provided for prediction",
                            config.name
                        )));
                    }
                }
            }
        }

        // Combine trend with seasonality, holiday, and regressor components
        // yhat = trend * (1 + seasonal_multiplicative) + seasonal_additive
        let yhat_scaled: Vec<f64> = (0..ds.len())
            .map(|i| trend[i] * (1.0 + seasonal_multiplicative[i]) + seasonal_additive[i])
            .collect();

        // Unscale predictions back to original scale
        let yhat: Vec<f64> = yhat_scaled.iter().map(|&v| v * self.y_scale).collect();
        let trend: Vec<f64> = trend.iter().map(|&v| v * self.y_scale).collect();

        // Uncertainty intervals using sigma_obs and interval_width
        // Approximate z-score for 80% interval: ~1.28, for 95%: ~1.96
        let z_score = match (self.interval_width * 100.0).round() as i32 {
            80 => 1.28,
            90 => 1.645,
            95 => 1.96,
            99 => 2.576,
            _ => 1.28, // default to 80%
        };
        // Unscale sigma_obs for uncertainty intervals
        let margin = z_score * self.sigma_obs * self.y_scale;
        let yhat_lower: Vec<f64> = yhat.iter().map(|&y| y - margin).collect();
        let yhat_upper: Vec<f64> = yhat.iter().map(|&y| y + margin).collect();

        // Trend uncertainty intervals (same margin as yhat for now)
        let trend_lower: Vec<f64> = trend.iter().map(|&t| t - margin).collect();
        let trend_upper: Vec<f64> = trend.iter().map(|&t| t + margin).collect();

        // Unscale seasonal components
        // For multiplicative seasonality, components should remain as fractional values (not scaled)
        // For additive seasonality, components should be scaled to original units
        // Always include yearly and weekly columns (as zeros if not enabled) for Prophet compatibility
        let yearly_unscaled = if let Some(v) = yearly_comp.as_ref() {
            if yearly_is_multiplicative {
                // Multiplicative: keep as fractional values (e.g., 0.2 = 20% increase)
                v.clone()
            } else {
                // Additive: scale to original units
                v.iter().map(|&x| x * self.y_scale).collect()
            }
        } else {
            // No yearly component - return zeros for Prophet compatibility
            vec![0.0; ds.len()]
        };

        let weekly_unscaled = if let Some(v) = weekly_comp.as_ref() {
            if weekly_is_multiplicative {
                // Multiplicative: keep as fractional values
                v.clone()
            } else {
                // Additive: scale to original units
                v.iter().map(|&x| x * self.y_scale).collect()
            }
        } else {
            // No weekly component - return zeros for Prophet compatibility
            vec![0.0; ds.len()]
        };

        // Compute uncertainty intervals for yearly component
        let yearly_lower: Vec<f64> = yearly_unscaled.iter().map(|&y| y - margin).collect();
        let yearly_upper: Vec<f64> = yearly_unscaled.iter().map(|&y| y + margin).collect();

        // Compute uncertainty intervals for weekly component
        let weekly_lower: Vec<f64> = weekly_unscaled.iter().map(|&w| w - margin).collect();
        let weekly_upper: Vec<f64> = weekly_unscaled.iter().map(|&w| w + margin).collect();

        // Compute additive and multiplicative terms
        let additive_terms: Vec<f64> = seasonal_additive
            .iter()
            .map(|&v| v * self.y_scale)
            .collect();
        let multiplicative_terms: Vec<f64> = seasonal_multiplicative.clone();

        // Uncertainty intervals for additive and multiplicative terms
        let additive_terms_lower: Vec<f64> = additive_terms.iter().map(|&v| v - margin).collect();
        let additive_terms_upper: Vec<f64> = additive_terms.iter().map(|&v| v + margin).collect();
        let multiplicative_terms_lower: Vec<f64> =
            multiplicative_terms.iter().map(|&v| v - margin).collect();
        let multiplicative_terms_upper: Vec<f64> =
            multiplicative_terms.iter().map(|&v| v + margin).collect();

        Ok(ForecastResult {
            ds: ds.to_vec(),
            trend,
            yhat_lower,
            yhat_upper,
            trend_lower,
            trend_upper,
            additive_terms,
            additive_terms_lower,
            additive_terms_upper,
            weekly: weekly_unscaled,
            weekly_lower,
            weekly_upper,
            yearly: yearly_unscaled,
            yearly_lower,
            yearly_upper,
            multiplicative_terms,
            multiplicative_terms_lower,
            multiplicative_terms_upper,
            yhat,
        })
    }

    pub fn make_future_dates(
        &self,
        periods: usize,
        freq: &str,
        include_history: bool,
    ) -> Result<Vec<String>> {
        if !self.fitted {
            return Err(crate::FarseerError::Prediction(
                "Model must be fitted before making future dates".to_string(),
            ));
        }

        let history = self.history.as_ref().unwrap();
        let mut out = Vec::new();
        if include_history {
            out.extend_from_slice(&history.ds);
        }

        // Determine last timestamp from history
        let last_ts = history.ds.last().and_then(|s| parse_ds(s)).ok_or_else(|| {
            crate::FarseerError::Prediction("Unable to parse last history date".to_string())
        })?;

        // Support multiple frequencies: H (hourly), D (daily), W (weekly), M (monthly), Y (yearly)
        let fut = future_dates(last_ts, periods, freq);
        out.extend(fut);
        Ok(out)
    }

    pub fn trend_type(&self) -> TrendType {
        self.trend
    }

    pub fn n_changepoints(&self) -> usize {
        self.n_changepoints
    }

    pub fn add_seasonality(
        &mut self,
        name: &str,
        period: f64,
        fourier_order: usize,
        prior_scale: Option<f64>,
        mode: Option<&str>,
    ) -> Result<()> {
        // Validate fourier_order
        if fourier_order == 0 {
            return Err(crate::FarseerError::DataValidation(
                "Fourier order must be greater than 0".to_string(),
            ));
        }

        // Check for duplicate names
        if self.seasonalities.iter().any(|s| s.name == name) {
            return Err(crate::FarseerError::DataValidation(format!(
                "Seasonality with name '{}' already exists",
                name
            )));
        }

        // Check against built-in seasonality names
        if name == "yearly" || name == "weekly" || name == "daily" {
            return Err(crate::FarseerError::DataValidation(format!(
                "Cannot use reserved seasonality name '{}'",
                name
            )));
        }

        let mut config = SeasonalityConfig::new(name, period, fourier_order);

        if let Some(scale) = prior_scale {
            config = config.with_prior_scale(scale);
        }

        if let Some(mode_str) = mode {
            let seasonality_mode = match mode_str.to_lowercase().as_str() {
                "additive" => SeasonalityMode::Additive,
                "multiplicative" => SeasonalityMode::Multiplicative,
                _ => {
                    return Err(crate::FarseerError::DataValidation(format!(
                        "Invalid seasonality mode: {}. Must be 'additive' or 'multiplicative'.",
                        mode_str
                    )))
                }
            };
            config = config.with_mode(seasonality_mode);
        }

        self.seasonalities.push(config);
        Ok(())
    }

    /// Add custom holidays.
    ///
    /// # Arguments
    /// * `name` - Name of the holiday (e.g., "christmas", "thanksgiving")
    /// * `dates` - Vector of date strings in "YYYY-MM-DD" format
    /// * `lower_window` - Days before the holiday to include (default 0)
    /// * `upper_window` - Days after the holiday to include (default 0)
    /// * `prior_scale` - Regularization parameter (default 10.0)
    /// * `mode` - "additive" or "multiplicative" (default None = additive)
    pub fn add_holidays(
        &mut self,
        name: &str,
        dates: Vec<String>,
        lower_window: Option<i32>,
        upper_window: Option<i32>,
        prior_scale: Option<f64>,
        mode: Option<&str>,
    ) -> Result<()> {
        let mut config = HolidayConfig::new(name, dates);

        if let Some(lower) = lower_window {
            config.lower_window = lower;
        }
        if let Some(upper) = upper_window {
            config.upper_window = upper;
        }
        if let Some(scale) = prior_scale {
            config = config.with_prior_scale(scale);
        }
        if let Some(mode_str) = mode {
            let holiday_mode = match mode_str.to_lowercase().as_str() {
                "additive" => SeasonalityMode::Additive,
                "multiplicative" => SeasonalityMode::Multiplicative,
                _ => {
                    return Err(crate::FarseerError::DataValidation(format!(
                        "Invalid holiday mode: {}. Must be 'additive' or 'multiplicative'.",
                        mode_str
                    )))
                }
            };
            config = config.with_mode(holiday_mode);
        }

        self.holidays.push(config);
        Ok(())
    }

    /// Add country holidays using the Python holidays package.
    /// This method stores the country name; actual holiday dates are fetched
    /// by the Python layer and passed back via add_holidays().
    pub fn add_country_holidays(&mut self, country: &str) -> Result<()> {
        self.country_holidays.push(country.to_string());
        Ok(())
    }

    /// Add an additional regressor to be used for fitting and predicting.
    ///
    /// # Arguments
    /// * `name` - Name of the regressor (must match a column in the dataframe)
    /// * `prior_scale` - Regularization parameter (default 10.0)
    /// * `standardize` - "auto", "true", or "false" (default "auto": standardize unless binary)
    /// * `mode` - "additive" or "multiplicative" (default: model's seasonality_mode)
    pub fn add_regressor(
        &mut self,
        name: &str,
        prior_scale: Option<f64>,
        standardize: Option<&str>,
        mode: Option<&str>,
    ) -> Result<()> {
        if self.fitted {
            return Err(crate::FarseerError::DataValidation(
                "Regressors must be added prior to model fitting".to_string(),
            ));
        }

        // Check for duplicate names
        if self.regressors.iter().any(|r| r.name == name) {
            return Err(crate::FarseerError::DataValidation(format!(
                "Regressor with name '{}' already exists",
                name
            )));
        }

        let mut config = RegressorConfig::new(name);

        if let Some(scale) = prior_scale {
            if scale <= 0.0 {
                return Err(crate::FarseerError::DataValidation(
                    "Prior scale must be > 0".to_string(),
                ));
            }
            config = config.with_prior_scale(scale);
        }

        if let Some(std) = standardize {
            if !["auto", "true", "false"].contains(&std.to_lowercase().as_str()) {
                return Err(crate::FarseerError::DataValidation(
                    "standardize must be 'auto', 'true', or 'false'".to_string(),
                ));
            }
            config = config.with_standardize(std);
        }

        if let Some(mode_str) = mode {
            let regressor_mode = match mode_str.to_lowercase().as_str() {
                "additive" => SeasonalityMode::Additive,
                "multiplicative" => SeasonalityMode::Multiplicative,
                _ => {
                    return Err(crate::FarseerError::DataValidation(format!(
                        "Invalid regressor mode: {}. Must be 'additive' or 'multiplicative'.",
                        mode_str
                    )))
                }
            };
            config = config.with_mode(regressor_mode);
        } else {
            // Use model's default seasonality mode
            config = config.with_mode(self.seasonality_mode);
        }

        self.regressors.push(config);
        Ok(())
    }

    /// Get reference to training history (for predict without df)
    pub fn get_history(&self) -> Option<&TimeSeriesData> {
        self.history.as_ref()
    }

    /// Get the list of regressor names
    pub fn get_regressor_names(&self) -> Vec<String> {
        self.regressors.iter().map(|r| r.name.clone()).collect()
    }

    pub fn get_params(&self) -> serde_json::Value {
        let growth_str = match self.trend {
            TrendType::Linear => "linear",
            TrendType::Logistic => "logistic",
            TrendType::Flat => "flat",
        };

        serde_json::json!({
            "version": "0.1.0",
            "fitted": self.fitted,

            // Trend configuration
            "growth": growth_str,
            "trend": format!("{:?}", self.trend),
            "n_changepoints": self.n_changepoints,
            "changepoint_range": self.changepoint_range,
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "changepoints": self.manual_changepoints.as_ref().unwrap_or(&vec![]),
            "specified_changepoints": self.specified_changepoints,

            // Seasonality configuration
            "yearly_seasonality": self.yearly_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "daily_seasonality": self.daily_seasonality,
            "seasonality_mode": format!("{:?}", self.seasonality_mode),

            // Custom seasonalities
            "seasonalities": self.seasonalities.iter().map(|s| serde_json::json!({
                "name": s.name,
                "period": s.period,
                "fourier_order": s.fourier_order,
                "prior_scale": s.prior_scale,
                "mode": format!("{:?}", s.mode),
            })).collect::<Vec<_>>(),

            // Holidays
            "holidays": self.holidays.iter().map(|h| serde_json::json!({
                "name": h.name,
                "dates": h.dates,
                "lower_window": h.lower_window,
                "upper_window": h.upper_window,
                "prior_scale": h.prior_scale,
                "mode": format!("{:?}", h.mode),
            })).collect::<Vec<_>>(),

            "country_holidays": self.country_holidays,

            // Uncertainty configuration
            "interval_width": self.interval_width,

            // Fitted parameters (only if fitted)
            "t0": self.t0.as_ref().map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string()),
            "t_scale": self.t_scale,
            "y_scale": self.y_scale,
            "t_change": self.t_change,
            "k": self.k,
            "m": self.m,
            "delta": self.delta,
            "beta": self.beta,
            "gamma": self.gamma,
            "sigma_obs": self.sigma_obs,

            // Component metadata
            "season_blocks": self.season_blocks.iter().map(|b| serde_json::json!({
                "name": b.name,
                "period": b.period,
                "order": b.order,
                "start": b.start,
                "end": b.end,
            })).collect::<Vec<_>>(),

            "holiday_blocks": self.holiday_blocks.iter().map(|b| serde_json::json!({
                "name": b.name,
                "start": b.start,
                "end": b.end,
            })).collect::<Vec<_>>(),

            // Regressors
            "regressors": self.regressors.iter().map(|r| serde_json::json!({
                "name": r.name,
                "prior_scale": r.prior_scale,
                "standardize": r.standardize,
                "mode": format!("{:?}", r.mode),
                "mu": r.mu,
                "std": r.std,
            })).collect::<Vec<_>>(),

            "regressor_blocks": self.regressor_blocks.iter().map(|b| serde_json::json!({
                "name": b.name,
                "start": b.start,
                "end": b.end,
            })).collect::<Vec<_>>(),
        })
    }

    /// Serialize model to JSON string
    pub fn to_json(&self) -> Result<String> {
        let params = self.get_params();
        serde_json::to_string_pretty(&params)
            .map_err(|e| crate::FarseerError::DataValidation(format!("Serialization error: {}", e)))
    }

    /// Deserialize model from JSON string
    pub fn from_json(json: &str) -> Result<Self> {
        let params: serde_json::Value = serde_json::from_str(json).map_err(|e| {
            crate::FarseerError::DataValidation(format!("Deserialization error: {}", e))
        })?;

        // Parse trend type
        let trend_str = params["trend"].as_str().ok_or_else(|| {
            crate::FarseerError::DataValidation("Missing trend field".to_string())
        })?;
        let trend = match trend_str {
            "Linear" => TrendType::Linear,
            "Logistic" => TrendType::Logistic,
            "Flat" => TrendType::Flat,
            _ => {
                return Err(crate::FarseerError::DataValidation(format!(
                    "Invalid trend type: {}",
                    trend_str
                )))
            }
        };

        // Parse seasonality mode
        let seasonality_mode_str = params["seasonality_mode"].as_str().unwrap_or("Additive");
        let seasonality_mode = match seasonality_mode_str {
            "Additive" => SeasonalityMode::Additive,
            "Multiplicative" => SeasonalityMode::Multiplicative,
            _ => SeasonalityMode::Additive,
        };

        // Parse custom seasonalities
        let mut seasonalities = Vec::new();
        if let Some(seas_array) = params["seasonalities"].as_array() {
            for s in seas_array {
                let mode_str = s["mode"].as_str().unwrap_or("Additive");
                let mode = match mode_str {
                    "Additive" => SeasonalityMode::Additive,
                    "Multiplicative" => SeasonalityMode::Multiplicative,
                    _ => SeasonalityMode::Additive,
                };

                seasonalities.push(SeasonalityConfig {
                    name: s["name"].as_str().unwrap_or("").to_string(),
                    period: s["period"].as_f64().unwrap_or(1.0),
                    fourier_order: s["fourier_order"].as_u64().unwrap_or(3) as usize,
                    prior_scale: s["prior_scale"].as_f64().unwrap_or(10.0),
                    mode,
                });
            }
        }

        // Parse holidays
        let mut holidays = Vec::new();
        if let Some(hol_array) = params["holidays"].as_array() {
            for h in hol_array {
                let mode_str = h["mode"].as_str().unwrap_or("Additive");
                let mode = match mode_str {
                    "Additive" => SeasonalityMode::Additive,
                    "Multiplicative" => SeasonalityMode::Multiplicative,
                    _ => SeasonalityMode::Additive,
                };

                let dates: Vec<String> = h["dates"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default();

                holidays.push(HolidayConfig {
                    name: h["name"].as_str().unwrap_or("").to_string(),
                    dates,
                    lower_window: h["lower_window"].as_i64().unwrap_or(0) as i32,
                    upper_window: h["upper_window"].as_i64().unwrap_or(0) as i32,
                    prior_scale: h["prior_scale"].as_f64().unwrap_or(10.0),
                    mode,
                });
            }
        }

        // Parse season blocks
        let mut season_blocks = Vec::new();
        if let Some(blocks_array) = params["season_blocks"].as_array() {
            for b in blocks_array {
                season_blocks.push(SeasonBlock {
                    name: b["name"].as_str().unwrap_or("").to_string(),
                    period: b["period"].as_f64().unwrap_or(1.0),
                    order: b["order"].as_u64().unwrap_or(3) as usize,
                    start: b["start"].as_u64().unwrap_or(0) as usize,
                    end: b["end"].as_u64().unwrap_or(0) as usize,
                });
            }
        }

        // Parse holiday blocks
        let mut holiday_blocks = Vec::new();
        if let Some(blocks_array) = params["holiday_blocks"].as_array() {
            for b in blocks_array {
                holiday_blocks.push(HolidayBlock {
                    name: b["name"].as_str().unwrap_or("").to_string(),
                    start: b["start"].as_u64().unwrap_or(0) as usize,
                    end: b["end"].as_u64().unwrap_or(0) as usize,
                });
            }
        }

        // Extract arrays
        let t_change: Vec<f64> = params["t_change"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
            .unwrap_or_default();

        let delta: Vec<f64> = params["delta"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
            .unwrap_or_default();

        let beta: Vec<f64> = params["beta"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
            .unwrap_or_default();

        let gamma: Vec<f64> = params["gamma"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
            .unwrap_or_default();

        let country_holidays: Vec<String> = params["country_holidays"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        Ok(Self {
            trend,
            n_changepoints: params["n_changepoints"].as_u64().unwrap_or(25) as usize,
            changepoint_range: params["changepoint_range"].as_f64().unwrap_or(0.8),
            changepoint_prior_scale: params["changepoint_prior_scale"].as_f64().unwrap_or(0.05),
            yearly_seasonality: params["yearly_seasonality"].as_bool().unwrap_or(true),
            weekly_seasonality: params["weekly_seasonality"].as_bool().unwrap_or(true),
            daily_seasonality: params["daily_seasonality"].as_bool().unwrap_or(false),
            seasonality_mode,
            manual_changepoints: None, // Manual changepoints not yet serialized
            specified_changepoints: false, // Will be false when deserializing
            seasonalities,
            holidays,
            country_holidays,
            regressors: Vec::new(), // Regressors not yet serialized
            fitted: params["fitted"].as_bool().unwrap_or(false),
            history: None, // History not serialized
            t0: params["t0"].as_str().and_then(parse_ds),
            t_scale: params["t_scale"].as_f64().unwrap_or(1.0),
            t_change,
            y_scale: params["y_scale"].as_f64().unwrap_or(1.0),
            logistic_floor: params["logistic_floor"].as_bool().unwrap_or(false),
            k: params["k"].as_f64().unwrap_or(0.0),
            m: params["m"].as_f64().unwrap_or(0.0),
            delta,
            beta,
            season_blocks,
            gamma,
            holiday_blocks,
            regressor_blocks: Vec::new(), // Regressor blocks not yet serialized
            sigma_obs: params["sigma_obs"].as_f64().unwrap_or(1.0),
            interval_width: params["interval_width"].as_f64().unwrap_or(0.8),
        })
    }
}

impl Default for Farseer {
    fn default() -> Self {
        Self::new()
    }
}
