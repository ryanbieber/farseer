use crate::core::data::{TimeSeriesData, ForecastResult};
use crate::core::trend::{
    parse_ds, time_scale, select_changepoints, changepoint_matrix,
    ols_linear_trend, piecewise_linear, piecewise_logistic, flat_trend, future_dates,
};
use crate::core::seasonality::{fourier_series, hstack, holiday_features};
use crate::core::stan::StanModel;
use chrono::NaiveDateTime;
use crate::Result;

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

pub struct Seer {
    trend: TrendType,
    n_changepoints: usize,
    changepoint_range: f64,
    changepoint_prior_scale: f64,
    yearly_seasonality: bool,
    weekly_seasonality: bool,
    daily_seasonality: bool,
    seasonality_mode: SeasonalityMode,
    use_stan: bool,  // NEW: Use Stan for parameter estimation instead of OLS
    
    // Custom seasonalities registry
    seasonalities: Vec<SeasonalityConfig>,
    
    // Holidays registry
    holidays: Vec<HolidayConfig>,
    country_holidays: Vec<String>, // Countries to fetch holidays for
    
    // Fitted parameters
    fitted: bool,
    history: Option<TimeSeriesData>,
    // Time scaling and changepoints
    t0: Option<NaiveDateTime>,
    t_scale: f64,
    t_change: Vec<f64>,
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

impl Seer {
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
            use_stan: false,  // Default to OLS for backwards compatibility
            seasonalities: Vec::new(),
            holidays: Vec::new(),
            country_holidays: Vec::new(),
            fitted: false,
            history: None,
            t0: None,
            t_scale: 1.0,
            t_change: Vec::new(),
            k: 0.0,
            m: 0.0,
            delta: Vec::new(),
            beta: Vec::new(),
            season_blocks: Vec::new(),
            gamma: Vec::new(),
            holiday_blocks: Vec::new(),
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
    
    pub fn with_changepoint_range(mut self, range: f64) -> Self {
        self.changepoint_range = range;
        self
    }
    
    pub fn with_changepoint_prior_scale(mut self, scale: f64) -> Self {
        self.changepoint_prior_scale = scale;
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
            _ => return Err(crate::SeerError::DataValidation(
                format!("Invalid seasonality mode: {}. Must be 'additive' or 'multiplicative'.", mode)
            )),
        };
        self.seasonality_mode = seasonality_mode;
        Ok(self)
    }
    
    pub fn with_interval_width(mut self, width: f64) -> Self {
        self.interval_width = width;
        self
    }
    
    pub fn with_stan(mut self, use_stan: bool) -> Self {
        self.use_stan = use_stan;
        self
    }
    
    pub fn fit(&mut self, data: &TimeSeriesData) -> Result<()> {
        // Parse ds -> timestamps
        let ts: Vec<NaiveDateTime> = data
            .ds
            .iter()
            .map(|s| parse_ds(s).ok_or_else(|| crate::SeerError::DataValidation(format!("Invalid date format: {}", s))))
            .collect::<std::result::Result<_, _>>()?;

        // Time scaling t in [0,1]
        let (t_hist, t_scale, t0) = time_scale(&ts);

        // Changepoints (locations in t units)
        let t_change = select_changepoints(&t_hist, self.n_changepoints, self.changepoint_range);

        // t in days since start for seasonality features
        let t_days: Vec<f64> = ts
            .iter()
            .map(|dt| (*dt - t0).num_seconds() as f64 / 86_400.0)
            .collect();

        // Build seasonality registry from toggles and custom seasonalities
        let mut all_seasonalities = Vec::new();
        if self.yearly_seasonality {
            all_seasonalities.push(SeasonalityConfig::new("yearly", 365.25, 10).with_mode(self.seasonality_mode));
        }
        if self.weekly_seasonality {
            all_seasonalities.push(SeasonalityConfig::new("weekly", 7.0, 3).with_mode(self.seasonality_mode));
        }
        if self.daily_seasonality {
            all_seasonalities.push(SeasonalityConfig::new("daily", 1.0, 4).with_mode(self.seasonality_mode));
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
                    end: col_end 
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
        let total_features = if x_matrix.is_empty() { 0 } else { x_matrix[0].len() };
        let mut s_a = vec![0.0; total_features];
        let mut s_m = vec![0.0; total_features];
        
        for block in &season_blocks {
            for i in block.start..block.end {
                // Find the corresponding seasonality config
                let config = all_seasonalities.iter()
                    .find(|c| c.name == block.name)
                    .unwrap();
                match config.mode {
                    SeasonalityMode::Additive => s_a[i] = 1.0,
                    SeasonalityMode::Multiplicative => s_m[i] = 1.0,
                }
            }
        }

        // Use Stan for fitting if enabled, otherwise use OLS
        let (k, m, delta, beta, sigma_obs) = if self.use_stan {
            // Stan-based parameter estimation
            let stan_model = StanModel::new()?;
            
            // Prepare prior scales for regressors
            let sigmas = vec![10.0; total_features]; // Default prior scale
            
            let result = stan_model.optimize(
                &t_hist,
                &data.y,
                &cap,
                &x_matrix,
                &sigmas,
                self.changepoint_prior_scale,
                trend_indicator,
                &s_a,
                &s_m,
                &t_change,
            )?;
            
            (result.k, result.m, result.delta, result.beta, result.sigma_obs)
        } else {
            // Fall back to OLS for speed
            let (k_ols, m_ols) = ols_linear_trend(&t_hist, &data.y);
            let delta_ols = vec![0.0; t_change.len()];
            
            // Fit seasonality on residuals
            let trend_hist = {
                let a_hist = changepoint_matrix(&t_hist, &t_change);
                piecewise_linear(k_ols, m_ols, &delta_ols, &t_hist, &a_hist, &t_change)
            };
            let y_resid: Vec<f64> = data.y.iter().zip(trend_hist.iter()).map(|(y, tr)| y - tr).collect();
            
            let beta_ols = if !x_matrix.is_empty() {
                ols_multi(&x_matrix, &y_resid)
            } else {
                Vec::new()
            };
            
            // Estimate sigma_obs from residuals
            let yhat_hist: Vec<f64> = trend_hist.iter().enumerate().map(|(i, &tr)| {
                let mut seas = 0.0;
                if !beta_ols.is_empty() && i < x_matrix.len() {
                    for (j, &xij) in x_matrix[i].iter().enumerate() {
                        if j < beta_ols.len() {
                            seas += xij * beta_ols[j];
                        }
                    }
                }
                tr + seas
            }).collect();
            let resid: Vec<f64> = data.y.iter().zip(yhat_hist.iter()).map(|(y, yh)| y - yh).collect();
            let sigma = (resid.iter().map(|r| r * r).sum::<f64>() / resid.len().max(1) as f64).sqrt().max(1e-6);
            
            (k_ols, m_ols, delta_ols, beta_ols, sigma)
        };

        // Save state
        self.history = Some(data.clone());
        self.fitted = true;
        self.t0 = Some(t0);
        self.t_scale = t_scale;
        self.t_change = t_change;
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
        if !self.fitted {
            return Err(crate::SeerError::Prediction(
                "Model must be fitted before prediction".to_string()
            ));
        }
        // Build t in model units for incoming ds
        let t0 = self.t0.unwrap();
        let mut t: Vec<f64> = Vec::with_capacity(ds.len());
        for s in ds.iter() {
            let dt = parse_ds(s)
                .ok_or_else(|| crate::SeerError::Prediction(format!("Invalid date format: {}", s)))?;
            let us = (dt - t0).num_microseconds().unwrap_or(0) as f64 / 1_000_000.0;
            let ti = if self.t_scale > 0.0 { us / self.t_scale } else { 0.0 };
            t.push(ti);
        }

        // Changepoint matrix and piecewise trend
        let a = changepoint_matrix(&t, &self.t_change);
        let history = self.history.as_ref().unwrap();
        
        let trend = match self.trend {
            TrendType::Linear => {
                piecewise_linear(self.k, self.m, &self.delta, &t, &a, &self.t_change)
            }
            TrendType::Logistic => {
                // Use cap from data if available, otherwise default to large value
                let cap_vec = if let Some(ref cap_data) = history.cap {
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
                };
                piecewise_logistic(self.k, self.m, &self.delta, &t, &cap_vec, &a, &self.t_change)
            }
            TrendType::Flat => {
                flat_trend(self.m, ds.len())
            }
        };

        // Seasonality contributions
        let mut yearly_comp: Option<Vec<f64>> = None;
        let mut weekly_comp: Option<Vec<f64>> = None;
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
                all_seasonalities.push(SeasonalityConfig::new("yearly", 365.25, 10).with_mode(self.seasonality_mode));
            }
            if self.weekly_seasonality {
                all_seasonalities.push(SeasonalityConfig::new("weekly", 7.0, 3).with_mode(self.seasonality_mode));
            }
            if self.daily_seasonality {
                all_seasonalities.push(SeasonalityConfig::new("daily", 1.0, 4).with_mode(self.seasonality_mode));
            }
            all_seasonalities.extend(self.seasonalities.clone());
            
            for (idx, block) in self.season_blocks.iter().enumerate() {
                if block.start == block.end { continue; }
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
                
                if block.name == "yearly" { yearly_comp = Some(comp.clone()); }
                if block.name == "weekly" { weekly_comp = Some(comp.clone()); }
            }
        }
        
        // Holiday contributions
        if !self.holiday_blocks.is_empty() {
            let t0 = self.t0.unwrap();
            let future_ts: Vec<NaiveDateTime> = ds
                .iter()
                .map(|s| parse_ds(s).unwrap_or(t0))
                .collect();
            
            for (idx, h_block) in self.holiday_blocks.iter().enumerate() {
                if h_block.start == h_block.end { continue; }
                
                // Get holiday config for this block
                if idx < self.holidays.len() {
                    let config = &self.holidays[idx];
                    let hf = holiday_features(&future_ts, &config.dates, config.lower_window, config.upper_window);
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

        // Combine trend with seasonality and holiday components
        // yhat = trend * (1 + seasonal_multiplicative) + seasonal_additive
        let yhat: Vec<f64> = (0..ds.len())
            .map(|i| trend[i] * (1.0 + seasonal_multiplicative[i]) + seasonal_additive[i])
            .collect();
        
        // Uncertainty intervals using sigma_obs and interval_width
        // Approximate z-score for 80% interval: ~1.28, for 95%: ~1.96
        let z_score = match (self.interval_width * 100.0).round() as i32 {
            80 => 1.28,
            90 => 1.645,
            95 => 1.96,
            99 => 2.576,
            _ => 1.28, // default to 80%
        };
        let margin = z_score * self.sigma_obs;
        let yhat_lower: Vec<f64> = yhat.iter().map(|&y| y - margin).collect();
        let yhat_upper: Vec<f64> = yhat.iter().map(|&y| y + margin).collect();

        Ok(ForecastResult {
            ds: ds.to_vec(),
            yhat,
            yhat_lower,
            yhat_upper,
            trend,
            yearly: yearly_comp,
            weekly: weekly_comp,
        })
    }
    
    pub fn make_future_dates(
        &self,
        periods: usize,
        freq: &str,
        include_history: bool,
    ) -> Result<Vec<String>> {
        if !self.fitted {
            return Err(crate::SeerError::Prediction(
                "Model must be fitted before making future dates".to_string()
            ));
        }
        
        let history = self.history.as_ref().unwrap();
        let mut out = Vec::new();
        if include_history {
            out.extend_from_slice(&history.ds);
        }

        // Determine last timestamp from history
        let last_ts = history
            .ds
            .last()
            .and_then(|s| parse_ds(s))
            .ok_or_else(|| crate::SeerError::Prediction("Unable to parse last history date".to_string()))?;

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
        let mut config = SeasonalityConfig::new(name, period, fourier_order);
        
        if let Some(scale) = prior_scale {
            config = config.with_prior_scale(scale);
        }
        
        if let Some(mode_str) = mode {
            let seasonality_mode = match mode_str.to_lowercase().as_str() {
                "additive" => SeasonalityMode::Additive,
                "multiplicative" => SeasonalityMode::Multiplicative,
                _ => return Err(crate::SeerError::DataValidation(
                    format!("Invalid seasonality mode: {}. Must be 'additive' or 'multiplicative'.", mode_str)
                )),
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
                _ => return Err(crate::SeerError::DataValidation(
                    format!("Invalid holiday mode: {}. Must be 'additive' or 'multiplicative'.", mode_str)
                )),
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
    
    pub fn get_params(&self) -> serde_json::Value {
        serde_json::json!({
            "version": "0.1.0",
            "fitted": self.fitted,
            
            // Trend configuration
            "trend": format!("{:?}", self.trend),
            "n_changepoints": self.n_changepoints,
            "changepoint_range": self.changepoint_range,
            "changepoint_prior_scale": self.changepoint_prior_scale,
            
            // Fitting method
            "use_stan": self.use_stan,
            
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
            "t_scale": self.t_scale,
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
        })
    }
    
    /// Serialize model to JSON string
    pub fn to_json(&self) -> Result<String> {
        let params = self.get_params();
        serde_json::to_string_pretty(&params)
            .map_err(|e| crate::SeerError::DataValidation(format!("Serialization error: {}", e)))
    }
    
    /// Deserialize model from JSON string
    pub fn from_json(json: &str) -> Result<Self> {
        let params: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| crate::SeerError::DataValidation(format!("Deserialization error: {}", e)))?;
        
        // Parse trend type
        let trend_str = params["trend"].as_str().ok_or_else(|| 
            crate::SeerError::DataValidation("Missing trend field".to_string()))?;
        let trend = match trend_str {
            "Linear" => TrendType::Linear,
            "Logistic" => TrendType::Logistic,
            "Flat" => TrendType::Flat,
            _ => return Err(crate::SeerError::DataValidation(format!("Invalid trend type: {}", trend_str))),
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
                
                let dates: Vec<String> = h["dates"].as_array()
                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
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
        let t_change: Vec<f64> = params["t_change"].as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
            .unwrap_or_default();
        
        let delta: Vec<f64> = params["delta"].as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
            .unwrap_or_default();
        
        let beta: Vec<f64> = params["beta"].as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
            .unwrap_or_default();
        
        let gamma: Vec<f64> = params["gamma"].as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
            .unwrap_or_default();
        
        let country_holidays: Vec<String> = params["country_holidays"].as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
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
            use_stan: params["use_stan"].as_bool().unwrap_or(false),  // Load Stan setting
            seasonalities,
            holidays,
            country_holidays,
            fitted: params["fitted"].as_bool().unwrap_or(false),
            history: None, // History not serialized
            t0: None, // Reconstructed from history if needed
            t_scale: params["t_scale"].as_f64().unwrap_or(1.0),
            t_change,
            k: params["k"].as_f64().unwrap_or(0.0),
            m: params["m"].as_f64().unwrap_or(0.0),
            delta,
            beta,
            season_blocks,
            gamma,
            holiday_blocks,
            sigma_obs: params["sigma_obs"].as_f64().unwrap_or(1.0),
            interval_width: params["interval_width"].as_f64().unwrap_or(0.8),
        })
    }
}

/// Solve (X^T X + λI) beta = X^T y via naive Gaussian elimination (λ small ridge)
fn ols_multi(x: &Vec<Vec<f64>>, y: &Vec<f64>) -> Vec<f64> {
    let n = x.len();
    if n == 0 { return Vec::new(); }
    let p = x[0].len();
    if p == 0 { return Vec::new(); }
    let mut xtx = vec![vec![0.0; p]; p];
    let mut xty = vec![0.0; p];
    for i in 0..n {
        for a in 0..p {
            let xa = x[i][a];
            xty[a] += xa * y[i];
            for b in 0..p {
                xtx[a][b] += xa * x[i][b];
            }
        }
    }
    // ridge
    let lambda = 1e-8;
    for d in 0..p { xtx[d][d] += lambda; }
    // Gaussian elimination
    let mut a = xtx;
    let mut b = xty;
    for i in 0..p {
        // pivot
        let mut max_r = i;
        let mut max_v = a[i][i].abs();
        for r in (i+1)..p { if a[r][i].abs() > max_v { max_v = a[r][i].abs(); max_r = r; } }
        if max_r != i { a.swap(i, max_r); b.swap(i, max_r); }
        let piv = a[i][i];
        if piv.abs() < 1e-12 { continue; }
        let inv_piv = 1.0 / piv;
        for j in i..p { a[i][j] *= inv_piv; }
        b[i] *= inv_piv;
        for r in 0..p {
            if r == i { continue; }
            let factor = a[r][i];
            if factor == 0.0 { continue; }
            for j in i..p { a[r][j] -= factor * a[i][j]; }
            b[r] -= factor * b[i];
        }
    }
    b
}

impl Default for Seer {
    fn default() -> Self {
        Self::new()
    }
}