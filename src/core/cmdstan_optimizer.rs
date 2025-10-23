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
        self.optimize_internal(data, init, None)
    }

    /// Optimize with an explicit thread count override (used by the Stan wrapper)
    pub fn optimize_with_thread_count(
        &self,
        data: &serde_json::Value,
        init: &serde_json::Value,
        num_threads: usize,
    ) -> Result<OptimizationResult> {
        self.optimize_internal(data, init, Some(num_threads))
    }

    fn optimize_internal(
        &self,
        data: &serde_json::Value,
        init: &serde_json::Value,
        num_threads_override: Option<usize>,
    ) -> Result<OptimizationResult> {
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
        let data_file = tempfile::Builder::new()
            .suffix(".json")
            .tempfile()
            .map_err(crate::FarseerError::Io)?;
        let init_file = tempfile::Builder::new()
            .suffix(".json")
            .tempfile()
            .map_err(crate::FarseerError::Io)?;
        let output_file = tempfile::Builder::new()
            .suffix(".csv")
            .tempfile()
            .map_err(crate::FarseerError::Io)?;

        // Serialize data in compact JSON format (not pretty-printed for speed)
        let data_content = serde_json::to_string(data).map_err(|e| {
            crate::FarseerError::StanError(format!("Failed to serialize data: {}", e))
        })?;

        // Serialize init in compact JSON format (not pretty-printed for speed)
        let init_content = serde_json::to_string(init).map_err(|e| {
            crate::FarseerError::StanError(format!("Failed to serialize init: {}", e))
        })?;

        // Write files in parallel using scoped threads for better performance
        let data_path = data_file.path().to_path_buf();
        let init_path = init_file.path().to_path_buf();

        std::thread::scope(|s| {
            let data_handle = s.spawn(|| {
                use std::io::Write;
                let mut file = std::fs::OpenOptions::new()
                    .write(true)
                    .open(&data_path)
                    .map_err(crate::FarseerError::Io)?;
                file.write_all(data_content.as_bytes())
                    .map_err(crate::FarseerError::Io)?;
                file.flush().map_err(crate::FarseerError::Io)?;
                Ok::<(), crate::FarseerError>(())
            });

            let init_handle = s.spawn(|| {
                use std::io::Write;
                let mut file = std::fs::OpenOptions::new()
                    .write(true)
                    .open(&init_path)
                    .map_err(crate::FarseerError::Io)?;
                file.write_all(init_content.as_bytes())
                    .map_err(crate::FarseerError::Io)?;
                file.flush().map_err(crate::FarseerError::Io)?;
                Ok::<(), crate::FarseerError>(())
            });

            // Wait for both to complete
            data_handle.join().unwrap()?;
            init_handle.join().unwrap()?;
            Ok::<(), crate::FarseerError>(())
        })?;

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
        let num_threads = num_threads_override.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        });
        let num_threads = num_threads.max(1);

        // Run CmdStan optimization
        let output = Command::new(&self.model_path)
            .env("LD_LIBRARY_PATH", new_ld_library_path)
            .env("STAN_NUM_THREADS", num_threads.to_string())
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

    /// Parse CmdStan output CSV file (optimized version)
    fn parse_cmdstan_output(&self, path: &std::path::Path) -> Result<OptimizationResult> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(path).map_err(crate::FarseerError::Io)?;
        let reader = BufReader::with_capacity(8192, file); // Larger buffer for faster reading

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

        // Parse the last line with pre-allocated capacity
        let mut values = Vec::with_capacity(header.len());
        for s in last_line.split(',') {
            if let Ok(v) = s.trim().parse::<f64>() {
                values.push(v);
            }
        }

        // Find parameter indices in header (cache lookups)
        let k_idx = header
            .iter()
            .position(|h| h == "k")
            .ok_or_else(|| crate::FarseerError::StanError("k not found in output".to_string()))?;
        let m_idx = header
            .iter()
            .position(|h| h == "m")
            .ok_or_else(|| crate::FarseerError::StanError("m not found in output".to_string()))?;
        let sigma_obs_idx = header
            .iter()
            .position(|h| h == "sigma_obs")
            .ok_or_else(|| {
                crate::FarseerError::StanError("sigma_obs not found in output".to_string())
            })?;

        // Pre-count delta and beta sizes for allocation
        let delta_count = header.iter().filter(|h| h.starts_with("delta.")).count();
        let beta_count = header.iter().filter(|h| h.starts_with("beta.")).count();

        // Extract delta and beta arrays with pre-allocation
        let mut delta = Vec::with_capacity(delta_count);
        let mut beta = Vec::with_capacity(beta_count);

        for (i, h) in header.iter().enumerate() {
            if i < values.len() {
                if h.starts_with("delta.") {
                    delta.push(values[i]);
                } else if h.starts_with("beta.") {
                    beta.push(values[i]);
                }
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
