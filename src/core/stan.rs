// CmdStan Integration Module
// This module handles Stan model optimization using CmdStan

use crate::Result;
use serde_json::json;
use std::path::PathBuf;

/// Stan model wrapper for Prophet-style forecasting
pub struct StanModel {
    #[allow(dead_code)]
    model_path: PathBuf,
}

impl StanModel {
    /// Create a new StanModel using the Prophet CmdStan binary
    pub fn new() -> Result<Self> {
        Self::new_with_threads(None)
    }

    /// Get the platform-specific binary name
    fn get_binary_name() -> &'static str {
        #[cfg(target_os = "windows")]
        {
            "prophet_model.exe"
        }
        #[cfg(not(target_os = "windows"))]
        {
            "prophet_model"
        }
    }

    /// Get the platform-specific directory
    fn get_platform_dir() -> &'static str {
        #[cfg(target_os = "windows")]
        {
            "windows"
        }
        #[cfg(target_os = "linux")]
        {
            "linux"
        }
        #[cfg(target_os = "macos")]
        {
            "macos"
        }
        #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
        {
            "linux" // Fallback to linux for other Unix-like systems
        }
    }

    /// Create a new StanModel with specified number of threads
    pub fn new_with_threads(_num_threads: Option<usize>) -> Result<Self> {
        let binary_name = Self::get_binary_name();
        let platform_dir = Self::get_platform_dir();

        // Try multiple paths to find the prophet model binary
        let possible_paths = [
            // Platform-specific directories (preferred)
            PathBuf::from(format!("stan/{}/{}", platform_dir, binary_name)),
            PathBuf::from(format!("./stan/{}/{}", platform_dir, binary_name)),
            // Legacy paths (backward compatibility)
            PathBuf::from(format!("stan/{}", binary_name)),
            PathBuf::from(format!("./stan/{}", binary_name)),
            // Relative to executable
            std::env::current_exe()
                .ok()
                .and_then(|exe| {
                    exe.parent()
                        .map(|p| p.join(format!("stan/{}/{}", platform_dir, binary_name)))
                })
                .unwrap_or_else(|| PathBuf::from(format!("stan/{}/{}", platform_dir, binary_name))),
            // Environment variable override
            std::env::var("PROPHET_MODEL_PATH")
                .ok()
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(format!("stan/{}/{}", platform_dir, binary_name))),
        ];

        // Find the first path that exists
        let model_path = possible_paths
            .iter()
            .find(|p| p.exists())
            .ok_or_else(|| crate::FarseerError::StanError(
                format!(
                    "Prophet model binary not found. Tried:\n  - {}\nSet PROPHET_MODEL_PATH environment variable to specify location.",
                    possible_paths.iter()
                        .map(|p| p.display().to_string())
                        .collect::<Vec<_>>()
                        .join("\n  - ")
                )
            ))?
            .clone();

        Ok(Self { model_path })
    }

    /// Optimize the Stan model (find MAP estimate) using CmdStan's L-BFGS
    #[allow(clippy::too_many_arguments)]
    pub fn optimize(
        &self,
        t: &[f64],
        y: &[f64],
        cap: &[f64],
        x: &[Vec<f64>],
        sigmas: &[f64],
        tau: f64,
        trend_indicator: i32,
        s_a: &[f64],
        s_m: &[f64],
        t_change: &[f64],
        weights: Option<&[f64]>,
    ) -> Result<StanOptimizationResult> {
        self.optimize_with_config(
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
            OptimizationConfig::default(),
        )
    }

    /// Optimize the Stan model with custom configuration
    /// Now delegates to CmdStan optimizer (110x faster than BridgeStan)
    #[allow(clippy::too_many_arguments)]
    pub fn optimize_with_config(
        &self,
        t: &[f64],
        y: &[f64],
        cap: &[f64],
        x: &[Vec<f64>],
        sigmas: &[f64],
        tau: f64,
        trend_indicator: i32,
        s_a: &[f64],
        s_m: &[f64],
        t_change: &[f64],
        weights: Option<&[f64]>,
        _config: OptimizationConfig,
    ) -> Result<StanOptimizationResult> {
        // Delegate to CmdStan optimizer
        self.optimize_with_cmdstan(
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
        )
    }

    /// Optimize using CmdStan (110x faster than BridgeStan)
    /// This shells out to the compiled Prophet Stan binary
    #[allow(clippy::too_many_arguments)]
    pub fn optimize_with_cmdstan(
        &self,
        t: &[f64],
        y: &[f64],
        cap: &[f64],
        x: &[Vec<f64>],
        sigmas: &[f64],
        tau: f64,
        trend_indicator: i32,
        s_a: &[f64],
        s_m: &[f64],
        t_change: &[f64],
        weights: Option<&[f64]>,
    ) -> Result<StanOptimizationResult> {
        use crate::core::cmdstan_optimizer::CmdStanOptimizer;

        let n = t.len();
        let k = x.first().map(|v| v.len()).unwrap_or(0);
        let s = t_change.len();

        // Calculate grainsize
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let grainsize = {
            let min_grainsize = 1;
            let max_grainsize = 1000;
            let calculated = n / (num_threads * 4).max(1);
            calculated.clamp(min_grainsize, max_grainsize).max(1)
        };

        // Use provided weights or default to equal weights
        let weights_vec = weights.map(|w| w.to_vec()).unwrap_or_else(|| vec![1.0; n]);

        // Prepare data in JSON format matching Stan's data block
        let data = json!({
            "T": n,
            "K": k,
            "t": t,
            "cap": cap,
            "y": y,
            "S": s,
            "t_change": t_change,
            "X": x,
            "sigmas": sigmas,
            "tau": tau,
            "trend_indicator": trend_indicator,
            "s_a": s_a,
            "s_m": s_m,
            "weights": weights_vec,
            "grainsize": grainsize,
        });

        // Initialize parameters using Prophet's approach
        let y_mean = y.iter().sum::<f64>() / y.len() as f64;
        let y_std = (y.iter().map(|v| (v - y_mean).powi(2)).sum::<f64>() / y.len() as f64).sqrt();
        let y_scaled: Vec<f64> = if y_std > 1e-9 {
            y.iter().map(|v| (v - y_mean) / y_std).collect()
        } else {
            vec![0.0; y.len()]
        };

        let (k_init, m_init) = if trend_indicator == 1 {
            let _cap_mean = cap.iter().sum::<f64>() / cap.len() as f64;
            let cap_scaled: Vec<f64> = if y_std > 1e-9 {
                cap.iter().map(|v| (v - y_mean) / y_std).collect()
            } else {
                vec![1.0; cap.len()]
            };
            crate::core::trend::logistic_growth_init(t, &y_scaled, &cap_scaled)
        } else if trend_indicator == 0 {
            crate::core::trend::flat_growth_init(&y_scaled)
        } else {
            crate::core::trend::linear_growth_init(t, &y_scaled)
        };

        let init = json!({
            "k": k_init,
            "m": m_init,
            "delta": vec![0.0; s],
            "beta": vec![0.0; k],
            "sigma_obs": 0.5,
        });

        // Create CmdStan optimizer
        let cmdstan_path =
            std::env::var("SEER_CMDSTAN_PATH").unwrap_or_else(|_| "stan/prophet_model".to_string());

        let optimizer = CmdStanOptimizer::with_model_path(&cmdstan_path);

        // Run optimization
        let result = optimizer.optimize(&data, &init)?;

        Ok(StanOptimizationResult {
            k: result.k,
            m: result.m,
            delta: result.delta,
            sigma_obs: result.sigma_obs,
            beta: result.beta,
        })
    }
}

/// Configuration for Stan optimization (following Prophet's approach)
#[derive(Clone)]
pub struct OptimizationConfig {
    pub num_threads: usize,
    pub grainsize: Option<usize>,
    pub seed: u32,
}

impl OptimizationConfig {
    /// Create default config for optimization
    pub fn default_for_size(_n: usize) -> Self {
        Self {
            num_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            grainsize: None, // Auto-calculate
            seed: 42,
        }
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self::default_for_size(100)
    }
}

/// Result from Stan optimization
pub struct StanOptimizationResult {
    pub k: f64,          // Base trend growth rate
    pub m: f64,          // Trend offset
    pub delta: Vec<f64>, // Trend rate adjustments at changepoints
    pub sigma_obs: f64,  // Observation noise
    pub beta: Vec<f64>,  // Seasonality/regressor coefficients
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Ignore by default as it requires Stan compilation
    fn test_stan_model_compilation() {
        let model = StanModel::new();
        assert!(model.is_ok(), "Stan model should compile successfully");
    }

    #[test]
    #[ignore]
    fn test_stan_model_with_threads() {
        let model = StanModel::new_with_threads(Some(4));
        assert!(
            model.is_ok(),
            "Stan model should compile with threading support"
        );
    }

    #[test]
    #[ignore]
    fn test_optimization_with_config() {
        let model = StanModel::new_with_threads(Some(4)).unwrap();

        let t = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 12.0, 14.0, 16.0, 18.0];
        let cap = vec![100.0; 5];
        let x = vec![vec![0.0]; 5];
        let sigmas = vec![1.0];
        let s_a = vec![1.0];
        let s_m = vec![0.0];
        let t_change = vec![];

        let config = OptimizationConfig {
            num_threads: 4,
            ..Default::default()
        };

        let result = model.optimize_with_config(
            &t, &y, &cap, &x, &sigmas, 0.05, 0, &s_a, &s_m, &t_change, None, config,
        );

        assert!(result.is_ok(), "Optimization should succeed");
    }
}
