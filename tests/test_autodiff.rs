// Unit tests for autodiff-based Prophet implementation

#[cfg(test)]
mod tests {
    use farseer::core::prophet_autodiff::{ProphetModel, ProphetParams};
    use farseer::core::prophet_optimizer::{initialize_params, optimize_prophet, OptimizationConfig};
    use ndarray::{Array1, Array2};

    /// Create a simple test model with linear trend
    fn create_simple_linear_model() -> ProphetModel {
        let n = 10;
        let t = Array1::linspace(0.0, 1.0, n);
        
        // y = 2*t + 1 + noise
        let mut y = Array1::zeros(n);
        for i in 0..n {
            y[i] = 2.0 * t[i] + 1.0;
        }

        ProphetModel {
            n,
            k: 0, // No seasonality
            s: 0, // No changepoints
            t,
            y,
            cap: Array1::ones(n) * 10.0, // Not used for linear trend
            x: Array2::zeros((n, 0)),    // No seasonality features
            t_change: Array1::zeros(0),
            sigmas: Array1::zeros(0),
            tau: 0.1,
            s_a: Array1::zeros(0),
            s_m: Array1::zeros(0),
            weights: Array1::ones(n),
            trend_indicator: 0, // Linear trend
        }
    }

    /// Create a model with changepoints
    fn create_model_with_changepoints() -> ProphetModel {
        let n = 20;
        let t = Array1::linspace(0.0, 1.0, n);
        
        // Piecewise linear: y = t for t < 0.5, y = 0.5 + 2*(t-0.5) for t >= 0.5
        let mut y = Array1::zeros(n);
        for i in 0..n {
            if t[i] < 0.5 {
                y[i] = t[i];
            } else {
                y[i] = 0.5 + 2.0 * (t[i] - 0.5);
            }
        }

        ProphetModel {
            n,
            k: 0,
            s: 1, // One changepoint
            t,
            y,
            cap: Array1::ones(n) * 10.0,
            x: Array2::zeros((n, 0)),
            t_change: Array1::from_vec(vec![0.5]),
            sigmas: Array1::zeros(0),
            tau: 0.1,
            s_a: Array1::zeros(0),
            s_m: Array1::zeros(0),
            weights: Array1::ones(n),
            trend_indicator: 0,
        }
    }

    /// Create a model with seasonality
    fn create_model_with_seasonality() -> ProphetModel {
        let n = 30;
        let t = Array1::linspace(0.0, 1.0, n);
        
        // Linear trend + sinusoidal seasonality
        let mut y = Array1::zeros(n);
        let mut x = Array2::zeros((n, 2)); // cos and sin features
        
        for i in 0..n {
            let period_val = 2.0 * std::f64::consts::PI * t[i] * 4.0; // 4 periods
            x[[i, 0]] = period_val.cos();
            x[[i, 1]] = period_val.sin();
            
            y[i] = 1.0 + 0.5 * t[i] + 0.3 * x[[i, 0]] + 0.2 * x[[i, 1]];
        }

        ProphetModel {
            n,
            k: 2, // Two seasonality features
            s: 0,
            t,
            y,
            cap: Array1::ones(n) * 10.0,
            x,
            t_change: Array1::zeros(0),
            sigmas: Array1::from_vec(vec![1.0, 1.0]), // Priors for beta
            tau: 0.1,
            s_a: Array1::from_vec(vec![1.0, 1.0]), // Both additive
            s_m: Array1::from_vec(vec![0.0, 0.0]),
            weights: Array1::ones(n),
            trend_indicator: 0,
        }
    }

    #[test]
    fn test_param_conversion() {
        let params = ProphetParams {
            k: 1.5,
            m: 2.0,
            delta: Array1::from_vec(vec![0.1, 0.2]),
            sigma_obs: 0.5,
            beta: Array1::from_vec(vec![0.3, 0.4, 0.5]),
        };

        let vec = params.to_vec();
        assert_eq!(vec.len(), 8); // k, m, 2 deltas, sigma, 3 betas = 1+1+2+1+3 = 8

        let params2 = ProphetParams::from_vec(&vec, 2, 3);
        assert_eq!(params2.k, 1.5);
        assert_eq!(params2.m, 2.0);
        assert_eq!(params2.delta.len(), 2);
        assert_eq!(params2.sigma_obs, 0.5);
        assert_eq!(params2.beta.len(), 3);
    }

    #[test]
    fn test_neg_log_prob_simple() {
        let model = create_simple_linear_model();
        let params = vec![2.0, 1.0, 0.1]; // k=2, m=1, sigma=0.1 (should fit well)

        let neg_log_prob = model.neg_log_prob(&params);
        
        println!("neg_log_prob = {}", neg_log_prob);
        
        // Should be finite and reasonable
        assert!(neg_log_prob.is_finite());
        // Negative log prob can be negative with perfect fit and strong priors
        // The test should just verify it's finite
    }

    #[test]
    fn test_gradient_computation() {
        let model = create_simple_linear_model();
        let params = vec![1.0, 0.5, 0.5]; // k, m, sigma
        
        let grad = model.gradient(&params);
        
        // Gradient should have correct length
        assert_eq!(grad.len(), 3);
        
        // All gradient components should be finite
        for g in &grad {
            assert!(g.is_finite(), "Gradient component is not finite: {}", g);
        }
        
        // At least one component should be non-zero
        assert!(grad.iter().any(|&g| g.abs() > 1e-10));
    }

    #[test]
    fn test_gradient_numerical_check() {
        // Verify autodiff gradient against numerical gradient
        let model = create_simple_linear_model();
        let params = vec![1.5, 1.0, 0.3];
        
        let grad_auto = model.gradient(&params);
        
        // Numerical gradient with finite differences
        let eps = 1e-6;
        let mut grad_num = vec![0.0; params.len()];
        
        for i in 0..params.len() {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            params_plus[i] += eps;
            params_minus[i] -= eps;
            
            let f_plus = model.neg_log_prob(&params_plus);
            let f_minus = model.neg_log_prob(&params_minus);
            
            grad_num[i] = (f_plus - f_minus) / (2.0 * eps);
        }
        
        // Compare autodiff and numerical gradients
        for i in 0..params.len() {
            let rel_error: f64 = if grad_num[i].abs() > 1e-10 {
                (grad_auto[i] - grad_num[i]).abs() / grad_num[i].abs()
            } else {
                (grad_auto[i] - grad_num[i]).abs()
            };
            
            assert!(
                rel_error < 1e-4,
                "Gradient mismatch at index {}: auto={}, num={}, rel_error={}",
                i, grad_auto[i], grad_num[i], rel_error
            );
        }
    }

    #[test]
    fn test_optimization_simple_linear() {
        let model = create_simple_linear_model();
        let y_mean = model.y.mean().unwrap();
        let y_std = model.y.std(0.0);
        
        let init_params = initialize_params(&model, y_mean, y_std);
        
        let config = OptimizationConfig {
            max_iters: 100, // Reduced for faster testing
            tolerance: 1e-6, // Relaxed tolerance
            history_size: 5,
        };
        
        let result = optimize_prophet(model, init_params, config);
        
        assert!(result.is_ok(), "Optimization failed: {:?}", result.err());
        
        let result = result.unwrap();
        
        // Check that parameters are reasonable
        // True model: y = 2*t + 1
        assert!((result.params.k - 2.0).abs() < 0.5, "k should be close to 2.0, got {}", result.params.k);
        assert!((result.params.m - 1.0).abs() < 0.5, "m should be close to 1.0, got {}", result.params.m);
        assert!(result.params.sigma_obs > 0.0, "sigma should be positive");
        
        println!("Optimization result: k={}, m={}, sigma={}, iterations={}", 
                 result.params.k, result.params.m, result.params.sigma_obs, result.iterations);
    }

    #[test]
    fn test_optimization_with_changepoints() {
        let model = create_model_with_changepoints();
        let y_mean = model.y.mean().unwrap();
        let y_std = model.y.std(0.0);
        
        let init_params = initialize_params(&model, y_mean, y_std);
        
        let config = OptimizationConfig {
            max_iters: 200, // Reduced for faster testing
            tolerance: 1e-6, // Relaxed tolerance
            history_size: 5,
        };
        
        let result = optimize_prophet(model, init_params, config);
        
        assert!(result.is_ok(), "Optimization with changepoints failed");
        
        let result = result.unwrap();
        
        // Should detect the changepoint effect
        assert!(result.params.delta.len() == 1);
        assert!(result.params.delta[0].abs() > 0.1, "Delta should be significant");
        
        println!("Changepoint optimization: k={}, m={}, delta={:?}, iterations={}", 
                 result.params.k, result.params.m, result.params.delta, result.iterations);
    }

    #[test]
    fn test_optimization_with_seasonality() {
        let model = create_model_with_seasonality();
        let y_mean = model.y.mean().unwrap();
        let y_std = model.y.std(0.0);
        
        let init_params = initialize_params(&model, y_mean, y_std);
        
        let config = OptimizationConfig {
            max_iters: 200, // Reduced for faster testing
            tolerance: 1e-6, // Relaxed tolerance
            history_size: 5,
        };
        
        let result = optimize_prophet(model, init_params, config);
        
        assert!(result.is_ok(), "Optimization with seasonality failed");
        
        let result = result.unwrap();
        
        // Should detect seasonality
        assert!(result.params.beta.len() == 2);
        assert!(result.params.beta.iter().any(|&b| b.abs() > 0.05), 
                "At least one beta should be significant");
        
        println!("Seasonality optimization: k={}, m={}, beta={:?}, iterations={}", 
                 result.params.k, result.params.m, result.params.beta, result.iterations);
    }

    #[test]
    fn test_initialize_params() {
        let model = create_simple_linear_model();
        let y_mean = 1.5;
        let y_std = 0.5;
        
        let params = initialize_params(&model, y_mean, y_std);
        
        // Should have correct length: k, m, delta[s], sigma, beta[k]
        let expected_len = 2 + model.s + 1 + model.k;
        assert_eq!(params.len(), expected_len);
        
        // m should be initialized to y_mean
        assert_eq!(params[1], y_mean);
        
        // sigma should be initialized to y_std
        let sigma_idx = 2 + model.s;
        assert_eq!(params[sigma_idx], y_std);
    }

    #[test]
    fn test_flat_trend() {
        let n = 10;
        let model = ProphetModel {
            n,
            k: 0,
            s: 0,
            t: Array1::linspace(0.0, 1.0, n),
            y: Array1::ones(n) * 5.0, // Constant
            cap: Array1::ones(n) * 10.0,
            x: Array2::zeros((n, 0)),
            t_change: Array1::zeros(0),
            sigmas: Array1::zeros(0),
            tau: 0.1,
            s_a: Array1::zeros(0),
            s_m: Array1::zeros(0),
            weights: Array1::ones(n),
            trend_indicator: 2, // Flat trend
        };

        let params = vec![0.0, 5.0, 0.1]; // k (unused), m=5, sigma
        let neg_log_prob = model.neg_log_prob(&params);
        
        assert!(neg_log_prob.is_finite());
        
        // With flat trend, predictions should be constant = m
        // So residuals should be small, leading to good likelihood
    }
}
