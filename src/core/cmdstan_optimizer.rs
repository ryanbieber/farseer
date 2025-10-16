// LBFGS Optimizer using CmdStan
// Directly calls CmdStan binary for maximum performance

use crate::Result;

/// Configuration for CmdStan optimization (kept for compatibility)
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
            iter: 10000,    // Keep high max iterations (rarely hit)
            tol_grad: 1e-8, // Keep strict gradient tolerance for accuracy
            tol_obj: 1e-12, // Keep strict objective tolerance
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

/// CmdStan optimizer - calls CmdStan binary directly
pub struct CmdStanOptimizer {
    _config: CmdStanConfig,
    model_path: std::path::PathBuf,
}

impl CmdStanOptimizer {
    pub fn new(config: CmdStanConfig) -> Self {
        let binary_name = Self::get_binary_name();
        let platform_dir = Self::get_platform_dir();
        let default_path = format!("stan/{}/{}", platform_dir, binary_name);

        let model_path =
            Self::find_model_binary().unwrap_or_else(|| std::path::PathBuf::from(default_path));

        Self {
            _config: config,
            model_path,
        }
    }

    pub fn with_model_path(model_path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            _config: CmdStanConfig::default(),
            model_path: model_path.into(),
        }
    }

    /// Find the prophet model binary in common locations
    fn find_model_binary() -> Option<std::path::PathBuf> {
        // Get platform-specific binary name and directory
        let binary_name = Self::get_binary_name();
        let platform_dir = Self::get_platform_dir();

        // Check environment variable first
        if let Ok(path) = std::env::var("PROPHET_MODEL_PATH") {
            let p = std::path::PathBuf::from(&path);
            if p.exists() {
                return Some(p);
            }
        }

        // Try various relative and absolute paths
        let candidates = vec![
            // Platform-specific directories (preferred)
            format!("stan/{}/{}", platform_dir, binary_name),
            format!("./stan/{}/{}", platform_dir, binary_name),
            format!("../stan/{}/{}", platform_dir, binary_name),
            // Legacy paths (backward compatibility)
            format!("stan/{}", binary_name),
            format!("./stan/{}", binary_name),
            // Try to find via cargo manifest dir (compile-time)
            format!(
                "{}/stan/{}/{}",
                env!("CARGO_MANIFEST_DIR"),
                platform_dir,
                binary_name
            ),
        ];

        for candidate in candidates {
            let path = std::path::PathBuf::from(candidate);
            if path.exists() {
                return Some(path);
            }
        }

        None
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

    /// Optimize the Prophet model with the given data and initial parameters
    /// This now calls CmdStan directly for maximum performance
    pub fn optimize(
        &self,
        data: &serde_json::Value,
        init: &serde_json::Value,
    ) -> Result<OptimizationResult> {
        use std::io::Write;
        use std::process::Command;

        // Check if model binary exists
        if !self.model_path.exists() {
            let candidates = [
                "stan/prophet_model",
                "./stan/prophet_model",
                concat!(env!("CARGO_MANIFEST_DIR"), "/stan/prophet_model"),
                concat!(env!("CARGO_MANIFEST_DIR"), "/stan/prophet_model"),
            ];
            return Err(crate::FarseerError::StanError(format!(
                "Prophet model binary not found. Tried:\n  - {}\n  - {}\nSet PROPHET_MODEL_PATH environment variable to specify location.",
                self.model_path.display(),
                candidates.join("\n  - ")
            )));
        }

        // Create temporary files for data and init with proper extensions
        let mut data_file = tempfile::Builder::new()
            .suffix(".json")
            .tempfile()
            .map_err(crate::FarseerError::Io)?;
        let mut init_file = tempfile::Builder::new()
            .suffix(".json")
            .tempfile()
            .map_err(crate::FarseerError::Io)?;
        let output_file = tempfile::Builder::new()
            .suffix(".csv")
            .tempfile()
            .map_err(crate::FarseerError::Io)?;

        // Write data in JSON format (CmdStan's preferred input format)
        let data_content = serde_json::to_string_pretty(data).map_err(|e| {
            crate::FarseerError::StanError(format!("Failed to serialize data: {}", e))
        })?;

        data_file
            .write_all(data_content.as_bytes())
            .map_err(crate::FarseerError::Io)?;
        data_file.flush().map_err(crate::FarseerError::Io)?;

        // Write init in JSON format
        let init_content = serde_json::to_string_pretty(init).map_err(|e| {
            crate::FarseerError::StanError(format!("Failed to serialize init: {}", e))
        })?;

        init_file
            .write_all(init_content.as_bytes())
            .map_err(crate::FarseerError::Io)?;
        init_file.flush().map_err(crate::FarseerError::Io)?;

        // Set LD_LIBRARY_PATH for TBB libraries
        let model_dir = self
            .model_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("stan"));

        let ld_library_path = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
        let new_ld_library_path = if ld_library_path.is_empty() {
            model_dir.to_string_lossy().to_string()
        } else {
            format!("{}:{}", model_dir.to_string_lossy(), ld_library_path)
        };

        // Determine number of threads to use
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        // Run CmdStan optimization
        let output = Command::new(&self.model_path)
            .env("LD_LIBRARY_PATH", new_ld_library_path)
            .arg("optimize")
            .arg("algorithm=lbfgs")
            .arg("data")
            .arg(format!("file={}", data_file.path().display()))
            .arg(format!("init={}", init_file.path().display()))
            .arg("output")
            .arg(format!("file={}", output_file.path().display()))
            .arg(format!("num_threads={}", num_threads))
            .output()
            .map_err(|e| crate::FarseerError::StanError(format!("Failed to run CmdStan: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(crate::FarseerError::StanError(format!(
                "CmdStan optimization failed.\nStderr: {}\nStdout: {}",
                stderr, stdout
            )));
        }

        // Parse the output CSV file
        self.parse_cmdstan_output(output_file.path())
    }

    /// Parse CmdStan output CSV file
    fn parse_cmdstan_output(&self, path: &std::path::Path) -> Result<OptimizationResult> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(path).map_err(crate::FarseerError::Io)?;
        let reader = BufReader::new(file);

        let mut header = Vec::new();
        let mut last_line = String::new();

        for line in reader.lines() {
            let line = line.map_err(crate::FarseerError::Io)?;

            // Skip comments
            if line.starts_with('#') {
                continue;
            }

            // First non-comment line is the header
            if header.is_empty() {
                header = line.split(',').map(|s| s.trim().to_string()).collect();
                continue;
            }

            // Keep track of the last data line (final optimized values)
            if !line.is_empty() {
                last_line = line;
            }
        }

        if last_line.is_empty() {
            return Err(crate::FarseerError::StanError(
                "No optimization output found".to_string(),
            ));
        }

        // Parse the last line
        let values: Vec<f64> = last_line
            .split(',')
            .filter_map(|s| s.trim().parse::<f64>().ok())
            .collect();

        // Find parameter indices in header
        let find_index = |name: &str| header.iter().position(|h| h == name);

        let k_idx = find_index("k")
            .ok_or_else(|| crate::FarseerError::StanError("k not found in output".to_string()))?;
        let m_idx = find_index("m")
            .ok_or_else(|| crate::FarseerError::StanError("m not found in output".to_string()))?;
        let sigma_obs_idx = find_index("sigma_obs").ok_or_else(|| {
            crate::FarseerError::StanError("sigma_obs not found in output".to_string())
        })?;

        // Extract delta and beta arrays
        let mut delta = Vec::new();
        let mut beta = Vec::new();

        for (i, h) in header.iter().enumerate() {
            if h.starts_with("delta.") {
                if i < values.len() {
                    delta.push(values[i]);
                }
            } else if h.starts_with("beta.") && i < values.len() {
                beta.push(values[i]);
            }
        }

        Ok(OptimizationResult {
            k: values[k_idx],
            m: values[m_idx],
            sigma_obs: values[sigma_obs_idx],
            delta,
            beta,
            trend: Vec::new(), // Will be computed later if needed
        })
    }
}
