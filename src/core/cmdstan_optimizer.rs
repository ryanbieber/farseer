// LBFGS Optimizer using argmin-rs
// Replaces CmdStan with native Rust optimization

use crate::Result;
use argmin::core::{CostFunction, Executor, Gradient};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use ndarray::{Array1, Array2, ArrayView1};

/// Configuration for LBFGS optimization
pub struct CmdStanConfig {
    /// Maximum number of iterations
    pub iter: usize,
    /// Convergence tolerance on gradient norm
    pub tol_grad: f64,
    /// Convergence tolerance on changes in objective function
    pub tol_obj: f64,
    /// History size for L-BFGS (m parameter)
    pub history_size: usize,
    /// Line search condition parameter
    pub linesearch_condition: f64,
}

impl Default for CmdStanConfig {
    fn default() -> Self {
        Self {
            iter: 10000,
            tol_grad: 1e-8,
            tol_obj: 1e-12,
            history_size: 5,
            linesearch_condition: 1e-4,
        }
    }
}

/// Result of LBFGS optimization
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub k: f64,
    pub m: f64,
    pub sigma_obs: f64,
    pub delta: Vec<f64>,
    pub beta: Vec<f64>,
    pub trend: Vec<f64>,
}

/// Prophet model cost function for optimization
struct ProphetCostFunction {
    t: Vec<f64>,
    y: Vec<f64>,
    cap: Vec<f64>,
    x: Vec<Vec<f64>>,
    sigmas: Vec<f64>,
    tau: f64,
    trend_indicator: i32,
    s_a: Vec<f64>,
    s_m: Vec<f64>,
    t_change: Vec<f64>,
    weights: Vec<f64>,
    n: usize,
    k: usize,
    s: usize,
    changepoint_matrix: Array2<f64>,
}

impl ProphetCostFunction {
    fn new(
        t: Vec<f64>,
        y: Vec<f64>,
        cap: Vec<f64>,
        x: Vec<Vec<f64>>,
        sigmas: Vec<f64>,
        tau: f64,
        trend_indicator: i32,
        s_a: Vec<f64>,
        s_m: Vec<f64>,
        t_change: Vec<f64>,
        weights: Vec<f64>,
    ) -> Self {
        let n = t.len();
        let k = x.first().map(|v| v.len()).unwrap_or(0);
        let s = t_change.len();
        
        // Precompute changepoint matrix
        let changepoint_matrix = Self::get_changepoint_matrix(&t, &t_change, n, s);
        
        Self {
            t,
            y,
            cap,
            x,
            sigmas,
            tau,
            trend_indicator,
            s_a,
            s_m,
            t_change,
            weights,
            n,
            k,
            s,
            changepoint_matrix,
        }
    }
    
    fn get_changepoint_matrix(t: &[f64], t_change: &[f64], n: usize, s: usize) -> Array2<f64> {
        let mut a = Array2::<f64>::zeros((n, s));
        for (i, &t_i) in t.iter().enumerate() {
            for (j, &t_c) in t_change.iter().enumerate() {
                if t_i >= t_c {
                    a[[i, j]] = 1.0;
                }
            }
        }
        a
    }
    
    fn unpack_params<'a>(&self, params: &'a [f64]) -> (f64, f64, f64, ArrayView1<'a, f64>, ArrayView1<'a, f64>) {
        // Parameter layout: [k, m, sigma_obs, delta[s], beta[k]]
        let k = params[0];
        let m = params[1];
        let sigma_obs = params[2];
        let delta = ArrayView1::from(&params[3..3 + self.s]);
        let beta = ArrayView1::from(&params[3 + self.s..3 + self.s + self.k]);
        (k, m, sigma_obs, delta, beta)
    }
    
    fn compute_trend(&self, k: f64, m: f64, delta: &ArrayView1<f64>) -> Vec<f64> {
        let delta_arr = Array1::from_vec(delta.to_vec());
        
        match self.trend_indicator {
            0 => {
                // Flat trend
                vec![m; self.n]
            }
            1 => {
                // Logistic growth
                let gamma = self.logistic_gamma(k, m, &delta_arr);
                let k_vec = Array1::from_elem(self.n, k);
                let k_t = &k_vec + self.changepoint_matrix.dot(&delta_arr);
                let m_vec = Array1::from_elem(self.n, m);
                let m_t = &m_vec + self.changepoint_matrix.dot(&gamma);
                
                let t_arr = Array1::from_vec(self.t.clone());
                let cap_arr = Array1::from_vec(self.cap.clone());
                
                cap_arr.iter()
                    .zip(k_t.iter())
                    .zip(t_arr.iter())
                    .zip(m_t.iter())
                    .map(|(((c, k), t), m)| c / (1.0 + (-k * (t - m)).exp()))
                    .collect()
            }
            _ => {
                // Linear growth
                let delta_arr = Array1::from_vec(delta.to_vec());
                let k_vec = Array1::from_elem(self.n, k);
                let k_t = &k_vec + self.changepoint_matrix.dot(&delta_arr);
                
                let t_arr = Array1::from_vec(self.t.clone());
                let t_change_arr = Array1::from_vec(self.t_change.clone());
                let m_vec = Array1::from_elem(self.n, m);
                let m_t = &m_vec - self.changepoint_matrix.dot(&(&t_change_arr * &delta_arr));
                
                (k_t * &t_arr + m_t).to_vec()
            }
        }
    }
    
    fn logistic_gamma(&self, k: f64, m: f64, delta: &Array1<f64>) -> Array1<f64> {
        let s = delta.len();
        let mut gamma = Array1::<f64>::zeros(s);
        let mut k_s = vec![k];
        
        for &d in delta.iter() {
            k_s.push(k_s.last().unwrap() + d);
        }
        
        let mut m_pr = m;
        for i in 0..s {
            gamma[i] = (self.t_change[i] - m_pr) * (1.0 - k_s[i] / k_s[i + 1]);
            m_pr += gamma[i];
        }
        
        gamma
    }
    
    fn compute_seasonality(&self, beta: &ArrayView1<f64>) -> Vec<f64> {
        let mut seasonality = vec![0.0; self.n];
        
        for i in 0..self.n {
            let mut s = 0.0;
            for (j, &b) in beta.iter().enumerate() {
                if j < self.s_a.len() {
                    s += b * self.s_a[j];
                } else if j - self.s_a.len() < self.s_m.len() {
                    s += b * self.s_m[j - self.s_a.len()];
                }
            }
            seasonality[i] = s;
        }
        
        seasonality
    }
    
    fn negative_log_posterior(&self, params: &[f64]) -> f64 {
        let (k, m, sigma_obs, delta, beta) = self.unpack_params(params);
        
        // Ensure sigma_obs is positive
        if sigma_obs <= 0.0 {
            return 1e10;
        }
        
        // Compute trend
        let trend = self.compute_trend(k, m, &delta);
        
        // Compute seasonality
        let seasonality = self.compute_seasonality(&beta);
        
        // Compute regression component
        let mut regression = vec![0.0; self.n];
        for i in 0..self.n {
            for (j, &b) in beta.iter().enumerate() {
                if j < self.x.len() && i < self.x[j].len() {
                    regression[i] += b * self.x[j][i];
                }
            }
        }
        
        // Compute likelihood
        let mut log_likelihood = 0.0;
        for i in 0..self.n {
            let mu = trend[i] + seasonality[i] + regression[i];
            let residual = self.y[i] - mu;
            log_likelihood -= 0.5 * self.weights[i] * (residual / sigma_obs).powi(2);
        }
        log_likelihood -= (self.n as f64) * sigma_obs.ln();
        
        // Priors
        let mut log_prior = 0.0;
        
        // Prior on delta (Laplace with scale tau)
        for &d in delta.iter() {
            log_prior -= d.abs() / self.tau;
        }
        
        // Prior on sigma_obs
        for &sigma in self.sigmas.iter() {
            log_prior -= 0.5 * (sigma_obs / sigma).powi(2);
        }
        
        // Return negative log posterior (since we minimize)
        -(log_likelihood + log_prior)
    }
}

impl CostFunction for ProphetCostFunction {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> std::result::Result<Self::Output, argmin::core::Error> {
        Ok(self.negative_log_posterior(params))
    }
}

impl Gradient for ProphetCostFunction {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, params: &Self::Param) -> std::result::Result<Self::Gradient, argmin::core::Error> {
        // Use finite differences for gradient calculation
        let eps = 1e-8;
        let mut grad = vec![0.0; params.len()];
        
        for i in 0..params.len() {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            
            params_plus[i] += eps;
            params_minus[i] -= eps;
            
            let f_plus = self.negative_log_posterior(&params_plus);
            let f_minus = self.negative_log_posterior(&params_minus);
            
            grad[i] = (f_plus - f_minus) / (2.0 * eps);
        }
        
        Ok(grad)
    }
}

/// LBFGS optimizer using argmin-rs
pub struct CmdStanOptimizer {
    config: CmdStanConfig,
}

impl CmdStanOptimizer {
    pub fn new(config: CmdStanConfig) -> Self {
        Self { config }
    }

    pub fn with_model_path(_model_path: impl Into<std::path::PathBuf>) -> Self {
        // Model path is not used in pure Rust implementation
        Self {
            config: CmdStanConfig::default(),
        }
    }

    /// Optimize the Prophet model with the given data and initial parameters
    pub fn optimize(
        &self,
        data: &serde_json::Value,
        init: &serde_json::Value,
    ) -> Result<OptimizationResult> {
        // Extract data
        let t: Vec<f64> = serde_json::from_value(data["t"].clone())
            .map_err(|e| crate::SeerError::StanError(format!("Failed to parse t: {}", e)))?;
        let y: Vec<f64> = serde_json::from_value(data["y"].clone())
            .map_err(|e| crate::SeerError::StanError(format!("Failed to parse y: {}", e)))?;
        let cap: Vec<f64> = serde_json::from_value(data["cap"].clone())
            .map_err(|e| crate::SeerError::StanError(format!("Failed to parse cap: {}", e)))?;
        let x: Vec<Vec<f64>> = serde_json::from_value(data["X"].clone())
            .map_err(|e| crate::SeerError::StanError(format!("Failed to parse X: {}", e)))?;
        let sigmas: Vec<f64> = serde_json::from_value(data["sigmas"].clone())
            .map_err(|e| crate::SeerError::StanError(format!("Failed to parse sigmas: {}", e)))?;
        let tau: f64 = serde_json::from_value(data["tau"].clone())
            .map_err(|e| crate::SeerError::StanError(format!("Failed to parse tau: {}", e)))?;
        let trend_indicator: i32 = serde_json::from_value(data["trend_indicator"].clone())
            .map_err(|e| crate::SeerError::StanError(format!("Failed to parse trend_indicator: {}", e)))?;
        let s_a: Vec<f64> = serde_json::from_value(data["s_a"].clone())
            .map_err(|e| crate::SeerError::StanError(format!("Failed to parse s_a: {}", e)))?;
        let s_m: Vec<f64> = serde_json::from_value(data["s_m"].clone())
            .map_err(|e| crate::SeerError::StanError(format!("Failed to parse s_m: {}", e)))?;
        let t_change: Vec<f64> = serde_json::from_value(data["t_change"].clone())
            .map_err(|e| crate::SeerError::StanError(format!("Failed to parse t_change: {}", e)))?;
        let weights: Vec<f64> = serde_json::from_value(data["weights"].clone())
            .map_err(|e| crate::SeerError::StanError(format!("Failed to parse weights: {}", e)))?;

        // Extract initial parameters
        let k_init: f64 = serde_json::from_value(init["k"].clone())
            .map_err(|e| crate::SeerError::StanError(format!("Failed to parse k: {}", e)))?;
        let m_init: f64 = serde_json::from_value(init["m"].clone())
            .map_err(|e| crate::SeerError::StanError(format!("Failed to parse m: {}", e)))?;
        let delta_init: Vec<f64> = serde_json::from_value(init["delta"].clone())
            .map_err(|e| crate::SeerError::StanError(format!("Failed to parse delta: {}", e)))?;
        let beta_init: Vec<f64> = serde_json::from_value(init["beta"].clone())
            .map_err(|e| crate::SeerError::StanError(format!("Failed to parse beta: {}", e)))?;
        let sigma_obs_init: f64 = serde_json::from_value(init["sigma_obs"].clone())
            .map_err(|e| crate::SeerError::StanError(format!("Failed to parse sigma_obs: {}", e)))?;

        // Pack parameters into a single vector
        let mut init_params = vec![k_init, m_init, sigma_obs_init];
        init_params.extend_from_slice(&delta_init);
        init_params.extend_from_slice(&beta_init);

        // Create cost function
        let cost_fn = ProphetCostFunction::new(
            t.clone(),
            y.clone(),
            cap,
            x,
            sigmas,
            tau,
            trend_indicator,
            s_a,
            s_m,
            t_change.clone(),
            weights,
        );

        // Create LBFGS solver with line search
        let linesearch = MoreThuenteLineSearch::new();
        let lbfgs = LBFGS::new(linesearch, self.config.history_size);

        // Run optimization
        let result = Executor::new(cost_fn, lbfgs)
            .configure(|state| {
                state
                    .param(init_params)
                    .max_iters(self.config.iter as u64)
            })
            .run()
            .map_err(|e| crate::SeerError::StanError(format!("Optimization failed: {}", e)))?;

        // Extract optimized parameters using the State trait
        let best_params = result.state().param
            .as_ref()
            .ok_or_else(|| crate::SeerError::StanError("No parameters found".to_string()))?;

        let k = best_params[0];
        let m = best_params[1];
        let sigma_obs = best_params[2];
        let s = delta_init.len();
        let delta = best_params[3..3 + s].to_vec();
        let beta = best_params[3 + s..].to_vec();

        // Compute final trend
        let cost_fn = ProphetCostFunction::new(
            t,
            y,
            vec![],
            vec![],
            vec![],
            tau,
            trend_indicator,
            vec![],
            vec![],
            t_change,
            vec![],
        );
        let delta_view = ArrayView1::from(&delta);
        let trend = cost_fn.compute_trend(k, m, &delta_view);

        Ok(OptimizationResult {
            k,
            m,
            sigma_obs,
            delta,
            beta,
            trend,
        })
    }
}
