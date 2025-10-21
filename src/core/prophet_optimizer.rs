// Optimizer integration for autodiff-based Prophet model
// Integrates with argmin L-BFGS optimizer to mimic CmdStanPy's optimization

use super::prophet_autodiff::{ProphetModel, ProphetParams};

/// Optimization configuration
#[derive(Clone, Debug)]
pub struct OptimizationConfig {
    pub max_iters: u64,
    pub tolerance: f64,
    pub history_size: usize, // For L-BFGS
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            max_iters: 10_000,
            tolerance: 1e-10,
            history_size: 5,
        }
    }
}

/// Optimization result
#[derive(Debug, Clone)]
    pub struct OptimizationResult {
    pub params: ProphetParams,
    pub neg_log_prob: f64,
    pub iterations: u64,
    pub converged: bool,
}

mod optimizer_impl {
    use super::*;
    use argmin::core::{CostFunction, Error as ArgminError, Executor, Gradient, State};
    use argmin::solver::linesearch::MoreThuenteLineSearch;
    use argmin::solver::quasinewton::LBFGS;
    use ndarray::Array1;    /// Wrapper for argmin optimization
    struct ProphetProblem {
        model: ProphetModel,
        call_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl CostFunction for ProphetProblem {
        type Param = Array1<f64>;
        type Output = f64;

        fn cost(&self, params: &Self::Param) -> Result<Self::Output, ArgminError> {
            let params_slice = params.as_slice().unwrap();
            let cost = self.model.neg_log_prob(params_slice);
            
            let count = self.call_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if count == 0 || count % 5 == 0 {
                // Print k, m, delta values and first few beta values for debugging
                let k_val = params_slice[0];
                let m_val = params_slice[1];
                let delta_start = 2;
                let sigma_idx = 2 + self.model.s;  // k, m, delta[s], then sigma_obs
                let beta_start = 3 + self.model.s;  // k, m, delta[s], sigma_obs, then beta
                
                let num_delta_to_print = self.model.s.min(3);
                let mut delta_str = String::new();
                for i in 0..num_delta_to_print {
                    if delta_start + i < params_slice.len() {
                        delta_str.push_str(&format!(" δ{}={:.4}", i, params_slice[delta_start + i]));
                    }
                }
                
                let num_beta_to_print = self.model.k.min(4);
                let mut beta_str = String::new();
                for i in 0..num_beta_to_print {
                    if beta_start + i < params_slice.len() {
                        beta_str.push_str(&format!(" β{}={:.4}", i, params_slice[beta_start + i]));
                    }
                }
                let sigma_obs = if sigma_idx < params_slice.len() {
                    params_slice[sigma_idx]
                } else {
                    0.0
                };
                println!("  Iteration {}: cost = {:.6}  k={:.4} m={:.4}{} σ={:.6}{}", 
                         count, cost, k_val, m_val, delta_str, sigma_obs, beta_str);
            }
            
            Ok(cost)
        }
    }

    impl Gradient for ProphetProblem {
        type Param = Array1<f64>;
        type Gradient = Array1<f64>;

        fn gradient(&self, params: &Self::Param) -> Result<Self::Gradient, ArgminError> {
            let params_slice = params.as_slice().unwrap();
            let grad = self.model.gradient(params_slice);
            
            // Debug: Print gradient norm
            let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            println!("    Gradient norm: {:.6e}", grad_norm);
            
            Ok(Array1::from_vec(grad))
        }
    }

    /// Optimize Prophet model using L-BFGS
    pub fn optimize_prophet(
        model: ProphetModel,
        init_params: Vec<f64>,
        config: OptimizationConfig,
    ) -> Result<OptimizationResult, Box<dyn std::error::Error>> {
        eprintln!("=== OPTIMIZE_PROPHET CALLED === with {} params", init_params.len());
        println!("Starting optimization with {} parameters, max_iters={}", 
                 init_params.len(), config.max_iters);
        
        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let problem = ProphetProblem { 
            model: model.clone(),
            call_count: call_count.clone(),
        };

        // Configure line search with conservative parameters to avoid overshooting
        // c1: Armijo condition (sufficient decrease), c2: curvature condition
        let linesearch = MoreThuenteLineSearch::new()
            .with_c(1e-4, 0.9)?;  // Standard Wolfe conditions: c1=1e-4 (loose), c2=0.9 (tight)
        
        println!("DEBUG: Created line search");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        
        // Configure LBFGS with very strict convergence criteria to match CmdStan/Prophet
        // Prophet uses CmdStan defaults: tol_grad=1e-8, tol_rel_grad=1e4, tol_obj/tol_rel_obj=1e-12
        let solver = LBFGS::new(linesearch, config.history_size)
            .with_tolerance_grad(1e-8)?      // Match CmdStan default (very strict)
            .with_tolerance_cost(1e-12)?;    // Match CmdStan tol_obj (very strict)
        
        println!("DEBUG: Created LBFGS solver with history_size={}", config.history_size);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let init = Array1::from_vec(init_params.clone());
        println!("DEBUG: Initial params array length: {}", init.len());
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let executor = Executor::new(problem, solver)
            .configure(|state| {
                state
                    .param(init)
                    .max_iters(config.max_iters)
                    .target_cost(f64::NEG_INFINITY)  // Never stop based on absolute cost value
            });
        
        println!("DEBUG: Executor configured with max_iters={}", config.max_iters);
        
        let start_time = std::time::Instant::now();
        println!("DEBUG: Starting executor.run()...");
        let result = executor.run()?;
        println!("DEBUG: executor.run() returned");
        let elapsed = start_time.elapsed();
        
        println!("Optimization completed in {:.2}s", elapsed.as_secs_f64());

        let state = result.state();
        let optimized_params = state
            .get_best_param()
            .ok_or("No parameters found")?
            .clone();

        let neg_log_prob = state.get_best_cost();
        let iterations = state.get_iter();
        
        println!("DEBUG: L-BFGS completed {} iterations, final cost={}", iterations, neg_log_prob);

        let params = ProphetParams::from_vec(
            optimized_params.as_slice().unwrap(),
            model.s,
            model.k,
        );

        // Check convergence based on whether we hit max iterations
        let converged = iterations < config.max_iters;

        Ok(OptimizationResult {
            params,
            neg_log_prob,
            iterations,
            converged,
        })
    }

    /// Initialize parameters with sensible defaults
    pub fn initialize_params(model: &ProphetModel, y_mean: f64, y_std: f64) -> Vec<f64> {
        let mut params = Vec::new();

        // k: base growth rate - estimate from data
        let k_init = if model.trend_indicator == 1 {
            // Logistic: use small positive value
            0.1
        } else if model.trend_indicator == 0 {
            // Linear: estimate slope using linear regression instead of first-last points
            // This gives a better initial slope for curved data
            if model.n > 1 {
                let t_slice = model.t.as_slice().unwrap();
                let y_slice = model.y.as_slice().unwrap();
                
                // Compute linear regression: k = Σ((t - t_mean)(y - y_mean)) / Σ((t - t_mean)²)
                let t_mean = t_slice.iter().sum::<f64>() / model.n as f64;
                let y_mean_calc = y_slice.iter().sum::<f64>() / model.n as f64;
                
                let mut numerator = 0.0;
                let mut denominator = 0.0;
                for i in 0..model.n {
                    let t_diff = t_slice[i] - t_mean;
                    let y_diff = y_slice[i] - y_mean_calc;
                    numerator += t_diff * y_diff;
                    denominator += t_diff * t_diff;
                }
                
                if denominator > 1e-10 {
                    numerator / denominator
                } else {
                    // Fallback to first-last if regression fails
                    let t_range = t_slice[model.n - 1] - t_slice[0];
                    if t_range > 0.0 {
                        (y_slice[model.n - 1] - y_slice[0]) / t_range
                    } else {
                        0.0
                    }
                }
            } else {
                0.0
            }
        } else {
            // Flat trend: k not used
            0.0
        };
        params.push(k_init);

        // m: offset - use mean of y
        params.push(y_mean);

        // delta: changepoint adjustments - initialize to zero
        for _ in 0..model.s {
            params.push(0.0);
        }

        // sigma_obs: observation noise - use std of y or default
        let sigma_init = if y_std > 0.0 { y_std } else { 0.1 };
        params.push(sigma_init);

        // beta: seasonality coefficients - initialize to zero
        for _ in 0..model.k {
            params.push(0.0);
        }

        params
    }
}

pub use optimizer_impl::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_config_default() {
        let config = OptimizationConfig::default();
        assert_eq!(config.max_iters, 10_000);
        assert_eq!(config.tolerance, 1e-10);
        assert_eq!(config.history_size, 5);
    }
}
