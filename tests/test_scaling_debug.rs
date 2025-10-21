// Test to debug scaling issues in autodiff implementation

#[cfg(test)]
mod tests {
    use farseer::core::prophet_autodiff::{ProphetModel, ProphetParams};
    use farseer::core::prophet_optimizer::{initialize_params, optimize_prophet, OptimizationConfig};
    use ndarray::{Array1, Array2};

    /// Create a model with known values to test scaling
    fn create_test_model_with_scaling() -> ProphetModel {
        let n = 100;
        let t = Array1::linspace(0.0, 1.0, n);
        
        // Create known trend: y = 100 + 50*t (matches the airpassenger-like scale)
        // Add simple yearly seasonality
        let mut y = Array1::zeros(n);
        let mut x = Array2::zeros((n, 2)); // cos and sin for yearly
        
        for i in 0..n {
            // Yearly seasonality (1 full period over the time range)
            let phase = 2.0 * std::f64::consts::PI * t[i];
            x[[i, 0]] = phase.cos();
            x[[i, 1]] = phase.sin();
            
            // Trend + seasonality
            let trend_val = 100.0 + 50.0 * t[i];
            let seasonal_val = 10.0 * x[[i, 0]] + 5.0 * x[[i, 1]];
            y[i] = trend_val + seasonal_val;
        }

        // Calculate y_scale (as done in model.rs fit method)
        let y_scale = y.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(1.0);
        println!("📊 Original y range: [{:.2}, {:.2}]", y.iter().fold(f64::INFINITY, |a, &b| a.min(b)), y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
        println!("📊 y_scale: {:.2}", y_scale);
        
        // Scale y (as done in fit)
        let y_scaled = y.mapv(|v| v / y_scale);
        println!("📊 Scaled y range: [{:.6}, {:.6}]", y_scaled.iter().fold(f64::INFINITY, |a, &b| a.min(b)), y_scaled.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));

        ProphetModel {
            n,
            k: 2, // Two seasonality features (cos, sin)
            s: 0, // No changepoints for simplicity
            t: t.clone(),
            y: y_scaled.clone(),
            cap: Array1::ones(n) * 10.0,
            x: x.clone(),
            t_change: Array1::zeros(0),
            sigmas: Array1::from_vec(vec![10.0, 10.0]), // Standard prior
            tau: 0.05,
            s_a: Array1::from_vec(vec![1.0, 1.0]), // Both additive
            s_m: Array1::from_vec(vec![0.0, 0.0]), // Not multiplicative
            weights: Array1::ones(n),
            trend_indicator: 0, // Linear trend
        }
    }

    #[test]
    fn test_scaling_in_autodiff() {
        println!("\n🔍 Testing scaling in autodiff implementation...\n");
        
        let model = create_test_model_with_scaling();
        
        // Compute initial negative log probability
        let y_mean = model.y.iter().sum::<f64>() / model.y.len() as f64;
        let y_variance = model.y.iter().map(|&y| (y - y_mean).powi(2)).sum::<f64>() / model.y.len() as f64;
        let y_std = y_variance.sqrt();
        
        println!("📊 Scaled y statistics:");
        println!("   Mean: {:.6}", y_mean);
        println!("   Std: {:.6}", y_std);
        
        let init_params_vec = initialize_params(&model, y_mean, y_std);
        let init_params = ProphetParams::from_vec(&init_params_vec, model.s, model.k);
        println!("\n📊 Initial parameters:");
        println!("   k (growth rate): {:.6}", init_params.k);
        println!("   m (offset): {:.6}", init_params.m);
        println!("   sigma_obs: {:.6}", init_params.sigma_obs);
        println!("   beta[0]: {:.6}", init_params.beta[0]);
        println!("   beta[1]: {:.6}", init_params.beta[1]);
        
        let init_neg_log_prob = model.neg_log_prob(&init_params_vec);
        println!("\n📊 Initial neg_log_prob: {:.6}", init_neg_log_prob);
        
        // Optimize
        let config = OptimizationConfig {
            max_iters: 1000,
            tolerance: 1e-8,
            history_size: 10,
        };
        
        println!("\n🔧 Starting optimization...");
        let result = optimize_prophet(model.clone(), init_params_vec, config).expect("Optimization failed");
        
        println!("\n📊 Optimized parameters:");
        println!("   k (growth rate): {:.6}", result.params.k);
        println!("   m (offset): {:.6}", result.params.m);
        println!("   sigma_obs: {:.6}", result.params.sigma_obs);
        println!("   beta[0]: {:.6}", result.params.beta[0]);
        println!("   beta[1]: {:.6}", result.params.beta[1]);
        println!("   final neg_log_prob: {:.6}", result.neg_log_prob);
        
        // Now manually compute predictions to check scaling
        println!("\n🔍 Computing predictions manually...");
        
        // Trend: k*t + m (in scaled space)
        let k = result.params.k;
        let m = result.params.m;
        let beta = &result.params.beta;
        
        println!("\n📊 Sample predictions (first 5 points):");
        for i in 0..5 {
            let t_i = model.t[i];
            let trend_scaled = k * t_i + m;
            
            // Seasonality (additive)
            let seasonal_scaled = beta[0] * model.x[[i, 0]] + beta[1] * model.x[[i, 1]];
            
            let yhat_scaled = trend_scaled + seasonal_scaled;
            
            println!("   Point {}: t={:.3}, trend_scaled={:.6}, seasonal_scaled={:.6}, yhat_scaled={:.6}, y_actual_scaled={:.6}", 
                     i, t_i, trend_scaled, seasonal_scaled, yhat_scaled, model.y[i]);
        }
        
        // Expected behavior: 
        // - k should be close to 50/y_scale (slope in scaled space)
        // - m should be close to 100/y_scale (intercept in scaled space)
        // - beta should capture the seasonality scaled appropriately
        
        println!("\n✅ Test completed. Check if scaled values match expectations.");
    }
}
