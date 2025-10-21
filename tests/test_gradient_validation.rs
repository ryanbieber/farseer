#[cfg(test)]
#[cfg(feature = "autodiff-backend")]
mod gradient_tests {
    use farseer::core::prophet_autodiff::ProphetModel;
    use ndarray::{Array1, Array2};

    fn create_simple_model() -> ProphetModel {
        // Create a very simple model for testing gradients
        let n = 10;
        let k = 2; // 2 seasonality features
        let s = 0; // No changepoints for simplicity
        
        // Simple linear time
        let t = Array1::from_vec((0..n).map(|i| i as f64 / n as f64).collect());
        
        // Simple y values
        let y = Array1::from_vec((0..n).map(|i| 0.5 + 0.1 * (i as f64 / n as f64)).collect());
        
        println!("Test data:");
        println!("  t: {:?}", t.as_slice().unwrap());
        println!("  y: {:?}", y.as_slice().unwrap());
        
        // Simple Fourier features
        let mut x_vec = Vec::new();
        for i in 0..n {
            let phase = 2.0 * std::f64::consts::PI * (i as f64 / n as f64);
            x_vec.push(phase.sin());
            x_vec.push(phase.cos());
        }
        let x = Array2::from_shape_vec((n, k), x_vec).unwrap();
        
        println!("  X matrix first 3 rows:");
        for i in 0..3.min(n) {
            println!("    [{:.4}, {:.4}]", x[[i, 0]], x[[i, 1]]);
        }
        
        ProphetModel {
            n,
            k,
            s,
            t,
            y,
            cap: Array1::from_vec(vec![1.0; n]),
            x,
            t_change: Array1::from_vec(vec![]),
            sigmas: Array1::from_vec(vec![10.0; k]),
            tau: 0.05,
            s_a: Array1::from_vec(vec![1.0; k]),
            s_m: Array1::from_vec(vec![0.0; k]),
            weights: Array1::from_vec(vec![1.0; n]),
            trend_indicator: 0, // Linear trend
        }
    }

    #[test]
    fn test_gradient_vs_finite_diff() {
        let model = create_simple_model();
        
        // Test parameters: k, m, sigma_obs, beta[0], beta[1]
        let params = vec![0.1, 0.5, 0.1, 0.01, 0.02];
        
        // Compute gradient using autodiff
        let grad = model.gradient(&params);
        
        println!("Autodiff gradient: {:?}", grad);
        
        // Compute gradient using finite differences
        let eps = 1e-5;
        let mut fd_grad = vec![0.0; params.len()];
        
        for i in 0..params.len() {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            params_plus[i] += eps;
            params_minus[i] -= eps;
            
            let f_plus = model.neg_log_prob(&params_plus);
            let f_minus = model.neg_log_prob(&params_minus);
            
            fd_grad[i] = (f_plus - f_minus) / (2.0 * eps);
        }
        
        println!("Finite diff gradient: {:?}", fd_grad);
        
        // Check that they match
        for i in 0..params.len() {
            let rel_error = if fd_grad[i].abs() > 1e-6 {
                (grad[i] - fd_grad[i]).abs() / fd_grad[i].abs()
            } else {
                (grad[i] - fd_grad[i]).abs()
            };
            
            println!("Param {}: autodiff={:.6}, fd={:.6}, rel_error={:.6}", 
                     i, grad[i], fd_grad[i], rel_error);
            
            assert!(rel_error < 0.01, 
                    "Gradient mismatch for param {}: autodiff={}, fd={}, rel_error={}",
                    i, grad[i], fd_grad[i], rel_error);
        }
        
        // Also test at beta=0 (initialization)
        println!("\n=== Testing at beta=0 (initialization) ===");
        let params_init = vec![0.1, 0.5, 0.1, 0.0, 0.0];
        let grad_init = model.gradient(&params_init);
        let cost_init = model.neg_log_prob(&params_init);
        println!("Cost at init: {:.6}", cost_init);
        println!("Gradient at init: {:?}", grad_init);
        println!("Beta gradients: beta[0]={:.6}, beta[1]={:.6}", grad_init[3], grad_init[4]);
        
        // Try small non-zero betas
        let params_small = vec![0.1, 0.5, 0.1, 0.001, 0.001];
        let cost_small = model.neg_log_prob(&params_small);
        println!("\nCost with small betas (0.001): {:.6}", cost_small);
        println!("Cost difference: {:.6}", cost_small - cost_init);
    }
}
