// BridgeStan Integration Module
// This module handles Stan model compilation, optimization, and sampling

use bridgestan::{Model, StanLibrary, open_library, compile_model, download_bridgestan_src};
use serde_json::json;
use crate::Result;
use std::ffi::CString;
use std::path::{Path, PathBuf};

/// Stan model wrapper for Prophet
pub struct StanModel {
    library: StanLibrary,
}

impl StanModel {
    /// Compile and load the Prophet Stan model
    pub fn new() -> Result<Self> {
        // Try multiple paths to find prophet.stan
        let possible_paths = vec![
            // 1. Relative to current directory (when running from project root)
            PathBuf::from("stan/prophet.stan"),
            // 2. In the same directory as the executable
            std::env::current_exe()
                .ok()
                .and_then(|exe| exe.parent().map(|p| p.join("stan/prophet.stan")))
                .unwrap_or_else(|| PathBuf::from("stan/prophet.stan")),
            // 3. Environment variable override
            std::env::var("SEER_STAN_FILE")
                .ok()
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("stan/prophet.stan")),
        ];
        
        // Find the first path that exists
        let stan_file = possible_paths
            .iter()
            .find(|p| p.exists())
            .ok_or_else(|| crate::SeerError::StanError(
                format!(
                    "Stan file not found. Tried:\n  - {}\nSet SEER_STAN_FILE environment variable to specify location.",
                    possible_paths.iter()
                        .map(|p| p.display().to_string())
                        .collect::<Vec<_>>()
                        .join("\n  - ")
                )
            ))?;
        
        // Get BridgeStan source path
        // First try environment variable, then download
        let bridgestan_path = if let Ok(bs_path) = std::env::var("BRIDGESTAN") {
            PathBuf::from(bs_path)
        } else {
            // Download BridgeStan source code
            download_bridgestan_src()
                .map_err(|e| crate::SeerError::StanError(
                    format!("Failed to download BridgeStan source: {:?}", e)
                ))?
        };
        
        // Compile the Stan model
        // compile_model signature: (&bridgestan_src, &stan_file, &stanc_args, &make_args)
        let model_path = compile_model(
            &bridgestan_path,           // BridgeStan source directory
            stan_file.as_path(),        // Path to .stan file
            &[],                        // stanc args
            &[]                         // make args
        )
            .map_err(|e| crate::SeerError::StanError(
                format!("Failed to compile Stan model: {:?}", e)
            ))?;
        
        // Load the compiled model library
        let library = open_library(&model_path)
            .map_err(|e| crate::SeerError::StanError(
                format!("Failed to load Stan library: {:?}", e)
            ))?;
        
        Ok(Self { library })
    }
    
    /// Optimize the Stan model (find MAP estimate)
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
    ) -> Result<StanOptimizationResult> {
        let n = t.len();
        let k = x.first().map(|v| v.len()).unwrap_or(0);
        let s = t_change.len();
        
        // X is already in correct format (Vec<Vec<f64>>), each row is one observation
        // Stan expects matrix[T,K] as an array of T rows, each with K elements
        
        // Prepare data in JSON format matching Stan's data block
        let data = json!({
            "T": n,
            "K": k,
            "t": t,
            "cap": cap,
            "y": y,
            "S": s,
            "t_change": t_change,
            "X": x,  // Stan expects array of arrays for matrix[T,K]
            "sigmas": sigmas,
            "tau": tau,
            "trend_indicator": trend_indicator,
            "s_a": s_a,
            "s_m": s_m,
            "weights": vec![1.0; n],  // Equal weights for all observations
        });
        
        let data_str = serde_json::to_string(&data)
            .map_err(|e| crate::SeerError::StanError(
                format!("Failed to serialize Stan data: {}", e)
            ))?;
        
        // Convert to CString for FFI
        let data_cstr = CString::new(data_str)
            .map_err(|e| crate::SeerError::StanError(
                format!("Failed to create CString: {}", e)
            ))?;
        
        // Create model instance with data
        let _model = Model::new(&self.library, Some(&data_cstr), 42)  // seed=42
            .map_err(|e| crate::SeerError::StanError(
                format!("Failed to create model instance: {:?}", e)
            ))?;
        
        // For now, return placeholder values
        // TODO: Implement actual optimization using BridgeStan's optimization API
        // This requires understanding the exact optimization method available
        Ok(StanOptimizationResult {
            k: 0.1,  // Placeholder - will be replaced with actual optimization
            m: 10.0,  // Placeholder
            delta: vec![0.0; s],
            sigma_obs: 1.0,
            beta: vec![0.0; k],
        })
    }
}

/// Result from Stan optimization
pub struct StanOptimizationResult {
    pub k: f64,           // Base trend growth rate
    pub m: f64,           // Trend offset
    pub delta: Vec<f64>,  // Trend rate adjustments at changepoints
    pub sigma_obs: f64,   // Observation noise
    pub beta: Vec<f64>,   // Seasonality/regressor coefficients
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[ignore]  // Ignore by default as it requires Stan compilation
    fn test_stan_model_compilation() {
        let model = StanModel::new();
        assert!(model.is_ok(), "Stan model should compile successfully");
    }
}
