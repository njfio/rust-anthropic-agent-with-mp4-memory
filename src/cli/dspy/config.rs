//! DSPy CLI Configuration Management
//!
//! This module handles configuration loading, validation, and management for DSPy CLI operations.

use crate::cli::dspy::error::{DspyCliError, DspyCliResult};
use crate::dspy::DspyConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::{debug, info, warn};

/// DSPy CLI configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DspyCliConfig {
    /// Configuration schema version
    pub version: String,
    /// Global DSPy settings
    pub dspy: DspyGlobalConfig,
    /// Path configuration
    pub paths: DspyPathConfig,
    /// Module management configuration
    pub modules: DspyModulesConfig,
    /// Benchmarking configuration
    pub benchmark: DspyBenchmarkConfig,
    /// Optimization configuration
    pub optimization: DspyOptimizationConfig,
    /// Pipeline configuration
    pub pipeline: DspyPipelineConfig,
    /// Development tools configuration
    pub dev: DspyDevConfig,
    /// Security configuration
    pub security: DspySecurityConfig,
    /// API configuration
    pub api: DspyApiConfig,
    /// Output configuration
    pub output: DspyOutputConfig,
    /// Logging configuration
    pub logging: DspyLoggingConfig,
    /// Cache configuration
    pub cache: DspyCacheConfig,
    /// Performance configuration
    pub performance: DspyPerformanceConfig,
}

/// Global DSPy framework settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DspyGlobalConfig {
    pub default_strategy: String,
    pub enable_caching: bool,
    pub cache_ttl_seconds: u64,
    pub max_concurrent_operations: usize,
    pub log_level: String,
    pub timeout_seconds: u64,
    pub auto_save_results: bool,
    pub backup_before_changes: bool,
}

/// Directory and file path configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DspyPathConfig {
    pub base_dir: PathBuf,
    pub registry_path: PathBuf,
    pub templates_path: PathBuf,
    pub cache_path: PathBuf,
    pub logs_path: PathBuf,
    pub backups_path: PathBuf,
    pub temp_path: PathBuf,
}

/// Module management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DspyModulesConfig {
    pub auto_validate: bool,
    pub template_auto_update: bool,
    pub default_template: String,
    pub signature_validation: String,
    pub metadata_required: Vec<String>,
    pub max_module_size_mb: usize,
    pub compression_enabled: bool,
    pub templates: HashMap<String, ModuleTemplateConfig>,
}

/// Module template configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleTemplateConfig {
    pub description: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

/// Benchmarking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DspyBenchmarkConfig {
    pub default_iterations: usize,
    pub warmup_iterations: usize,
    pub timeout_seconds: u64,
    pub output_format: String,
    pub save_results: bool,
    pub results_retention_days: u32,
    pub parallel_execution: bool,
    pub max_parallel_benchmarks: usize,
    pub memory_monitoring: bool,
    pub cpu_monitoring: bool,
    pub metrics: BenchmarkMetricsConfig,
}

/// Benchmark metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetricsConfig {
    pub default_metrics: Vec<String>,
    pub latency_percentiles: Vec<u8>,
    pub accuracy_threshold: f64,
    pub performance_baseline: String,
}

/// Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DspyOptimizationConfig {
    pub max_iterations: usize,
    pub convergence_threshold: f64,
    pub save_history: bool,
    pub history_retention_days: u32,
    pub auto_apply_best: bool,
    pub validation_split: f64,
    pub early_stopping_patience: usize,
    pub checkpoint_frequency: usize,
    pub strategies: HashMap<String, serde_json::Value>,
}

/// Pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DspyPipelineConfig {
    pub execution_timeout: u64,
    pub parallel_execution: bool,
    pub max_parallel_stages: usize,
    pub save_logs: bool,
    pub log_retention_days: u32,
    pub checkpoint_enabled: bool,
    pub auto_retry_failed_stages: bool,
    pub max_retries: usize,
    pub monitoring: PipelineMonitoringConfig,
}

/// Pipeline monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineMonitoringConfig {
    pub enable_metrics: bool,
    pub metrics_interval_seconds: u64,
    pub alert_on_failure: bool,
    pub alert_on_slow_execution: bool,
    pub slow_execution_threshold: u64,
}

/// Development tools configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DspyDevConfig {
    pub auto_format_code: bool,
    pub include_documentation: bool,
    pub validation_on_save: bool,
    pub debug_mode: bool,
    pub test_coverage_threshold: f64,
    pub generate_examples: bool,
    pub templates: DevTemplatesConfig,
}

/// Development templates configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevTemplatesConfig {
    pub code_style: String,
    pub include_tests: bool,
    pub include_benchmarks: bool,
    pub license: String,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DspySecurityConfig {
    pub validate_inputs: bool,
    pub sanitize_file_paths: bool,
    pub max_file_size_mb: usize,
    pub allowed_file_extensions: Vec<String>,
    pub rate_limiting: bool,
    pub max_operations_per_minute: usize,
}

/// API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DspyApiConfig {
    pub anthropic_timeout_seconds: u64,
    pub max_retries: usize,
    pub retry_delay_seconds: u64,
    pub connection_pool_size: usize,
    pub enable_compression: bool,
    pub user_agent: String,
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DspyOutputConfig {
    pub default_format: String,
    pub color_output: String,
    pub progress_indicators: String,
    pub table_max_width: usize,
    pub json_pretty_print: bool,
    pub include_timestamps: bool,
    pub include_metadata: bool,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DspyLoggingConfig {
    pub level: String,
    pub format: String,
    pub output: String,
    pub file_path: PathBuf,
    pub max_file_size_mb: usize,
    pub max_files: usize,
    pub include_source_location: bool,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DspyCacheConfig {
    pub enabled: bool,
    pub cache_type: String,
    pub max_size_mb: usize,
    pub ttl_seconds: u64,
    pub compression: bool,
    pub cleanup_interval_seconds: u64,
    pub eviction_policy: String,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DspyPerformanceConfig {
    pub enable_monitoring: bool,
    pub metrics_collection: bool,
    pub profiling_enabled: bool,
    pub memory_limit_mb: usize,
    pub cpu_limit_percent: f64,
    pub disk_space_threshold_mb: usize,
}

impl Default for DspyCliConfig {
    fn default() -> Self {
        let base_dir = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("memvid-agent")
            .join("dspy");

        Self {
            version: "1.0.0".to_string(),
            dspy: DspyGlobalConfig::default(),
            paths: DspyPathConfig::new(&base_dir),
            modules: DspyModulesConfig::default(),
            benchmark: DspyBenchmarkConfig::default(),
            optimization: DspyOptimizationConfig::default(),
            pipeline: DspyPipelineConfig::default(),
            dev: DspyDevConfig::default(),
            security: DspySecurityConfig::default(),
            api: DspyApiConfig::default(),
            output: DspyOutputConfig::default(),
            logging: DspyLoggingConfig::new(&base_dir),
            cache: DspyCacheConfig::default(),
            performance: DspyPerformanceConfig::default(),
        }
    }
}

impl DspyPathConfig {
    fn new(base_dir: &Path) -> Self {
        Self {
            base_dir: base_dir.to_path_buf(),
            registry_path: base_dir.join("modules"),
            templates_path: base_dir.join("templates"),
            cache_path: base_dir.join("cache"),
            logs_path: base_dir.join("logs"),
            backups_path: base_dir.join("backups"),
            temp_path: base_dir.join("temp"),
        }
    }
}

impl DspyLoggingConfig {
    fn new(base_dir: &Path) -> Self {
        Self {
            level: "info".to_string(),
            format: "structured".to_string(),
            output: "file".to_string(),
            file_path: base_dir.join("logs").join("dspy.log"),
            max_file_size_mb: 10,
            max_files: 5,
            include_source_location: false,
        }
    }
}

// Default implementations for all config structs
impl Default for DspyGlobalConfig {
    fn default() -> Self {
        Self {
            default_strategy: "mipro_v2".to_string(),
            enable_caching: true,
            cache_ttl_seconds: 3600,
            max_concurrent_operations: 4,
            log_level: "info".to_string(),
            timeout_seconds: 300,
            auto_save_results: true,
            backup_before_changes: true,
        }
    }
}

impl Default for DspyModulesConfig {
    fn default() -> Self {
        let mut templates = HashMap::new();
        templates.insert("predict".to_string(), ModuleTemplateConfig {
            description: "Basic prediction module".to_string(),
            inputs: vec!["text".to_string()],
            outputs: vec!["response".to_string()],
        });
        templates.insert("chain_of_thought".to_string(), ModuleTemplateConfig {
            description: "Chain of thought reasoning".to_string(),
            inputs: vec!["question".to_string()],
            outputs: vec!["answer".to_string(), "reasoning".to_string()],
        });

        Self {
            auto_validate: true,
            template_auto_update: false,
            default_template: "predict".to_string(),
            signature_validation: "strict".to_string(),
            metadata_required: vec!["name".to_string(), "description".to_string(), "version".to_string()],
            max_module_size_mb: 100,
            compression_enabled: true,
            templates,
        }
    }
}

impl Default for DspyBenchmarkConfig {
    fn default() -> Self {
        Self {
            default_iterations: 100,
            warmup_iterations: 5,
            timeout_seconds: 300,
            output_format: "table".to_string(),
            save_results: true,
            results_retention_days: 30,
            parallel_execution: true,
            max_parallel_benchmarks: 2,
            memory_monitoring: true,
            cpu_monitoring: true,
            metrics: BenchmarkMetricsConfig::default(),
        }
    }
}

impl Default for BenchmarkMetricsConfig {
    fn default() -> Self {
        Self {
            default_metrics: vec!["latency".to_string(), "throughput".to_string(), "accuracy".to_string()],
            latency_percentiles: vec![50, 90, 95, 99],
            accuracy_threshold: 0.8,
            performance_baseline: "previous".to_string(),
        }
    }
}

impl Default for DspyOptimizationConfig {
    fn default() -> Self {
        let mut strategies = HashMap::new();
        strategies.insert("mipro_v2".to_string(), serde_json::json!({
            "max_candidates": 50,
            "max_bootstrapped_demos": 20,
            "max_labeled_demos": 10,
            "num_trials": 100
        }));

        Self {
            max_iterations: 50,
            convergence_threshold: 0.01,
            save_history: true,
            history_retention_days: 90,
            auto_apply_best: false,
            validation_split: 0.2,
            early_stopping_patience: 5,
            checkpoint_frequency: 10,
            strategies,
        }
    }
}

impl Default for DspyPipelineConfig {
    fn default() -> Self {
        Self {
            execution_timeout: 600,
            parallel_execution: true,
            max_parallel_stages: 3,
            save_logs: true,
            log_retention_days: 14,
            checkpoint_enabled: true,
            auto_retry_failed_stages: true,
            max_retries: 3,
            monitoring: PipelineMonitoringConfig::default(),
        }
    }
}

impl Default for PipelineMonitoringConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            metrics_interval_seconds: 30,
            alert_on_failure: true,
            alert_on_slow_execution: true,
            slow_execution_threshold: 300,
        }
    }
}

impl Default for DspyDevConfig {
    fn default() -> Self {
        Self {
            auto_format_code: true,
            include_documentation: true,
            validation_on_save: true,
            debug_mode: false,
            test_coverage_threshold: 0.8,
            generate_examples: true,
            templates: DevTemplatesConfig::default(),
        }
    }
}

impl Default for DevTemplatesConfig {
    fn default() -> Self {
        Self {
            code_style: "rust_standard".to_string(),
            include_tests: true,
            include_benchmarks: false,
            license: "MIT".to_string(),
        }
    }
}

impl Default for DspySecurityConfig {
    fn default() -> Self {
        Self {
            validate_inputs: true,
            sanitize_file_paths: true,
            max_file_size_mb: 50,
            allowed_file_extensions: vec![".json".to_string(), ".yaml".to_string(), ".toml".to_string(), ".txt".to_string()],
            rate_limiting: true,
            max_operations_per_minute: 60,
        }
    }
}

impl Default for DspyApiConfig {
    fn default() -> Self {
        Self {
            anthropic_timeout_seconds: 30,
            max_retries: 3,
            retry_delay_seconds: 1,
            connection_pool_size: 10,
            enable_compression: true,
            user_agent: "memvid-agent-dspy/1.0".to_string(),
        }
    }
}

impl Default for DspyOutputConfig {
    fn default() -> Self {
        Self {
            default_format: "auto".to_string(),
            color_output: "auto".to_string(),
            progress_indicators: "auto".to_string(),
            table_max_width: 120,
            json_pretty_print: true,
            include_timestamps: true,
            include_metadata: false,
        }
    }
}

impl Default for DspyCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cache_type: "disk".to_string(),
            max_size_mb: 500,
            ttl_seconds: 3600,
            compression: true,
            cleanup_interval_seconds: 300,
            eviction_policy: "lru".to_string(),
        }
    }
}

impl Default for DspyPerformanceConfig {
    fn default() -> Self {
        Self {
            enable_monitoring: true,
            metrics_collection: true,
            profiling_enabled: false,
            memory_limit_mb: 1024,
            cpu_limit_percent: 80.0,
            disk_space_threshold_mb: 1024,
        }
    }
}

impl DspyCliConfig {
    /// Validate the configuration
    pub fn validate(&self) -> DspyCliResult<()> {
        // Validate version
        if self.version.is_empty() {
            return Err(DspyCliError::Config {
                message: "Configuration version cannot be empty".to_string(),
                suggestion: Some("Set version to '1.0.0'".to_string()),
                config_path: None,
                line_number: None,
            });
        }

        // Validate timeout values
        if self.dspy.timeout_seconds == 0 {
            return Err(DspyCliError::Config {
                message: "Timeout seconds must be greater than 0".to_string(),
                suggestion: Some("Set timeout_seconds to a positive value (e.g., 300)".to_string()),
                config_path: None,
                line_number: None,
            });
        }

        // Validate paths exist or can be created
        if !self.paths.base_dir.exists() {
            if let Err(e) = std::fs::create_dir_all(&self.paths.base_dir) {
                return Err(DspyCliError::Resource {
                    resource: "filesystem".to_string(),
                    message: format!("Cannot create base directory {}: {}", self.paths.base_dir.display(), e),
                    current_usage: None,
                    limit: None,
                    suggestion: Some("Check directory permissions".to_string()),
                });
            }
        }

        // Validate optimization settings
        if !(0.0..=1.0).contains(&self.optimization.validation_split) {
            return Err(DspyCliError::Config {
                message: "Validation split must be between 0.0 and 1.0".to_string(),
                suggestion: Some("Set validation_split to a value like 0.2".to_string()),
                config_path: None,
                line_number: None,
            });
        }

        Ok(())
    }

    /// Convert to DSPy framework configuration
    pub fn to_dspy_config(&self) -> DspyConfig {
        DspyConfig {
            enable_optimization: true,
            max_optimization_iterations: self.optimization.max_iterations as u32,
            optimization_timeout_seconds: self.dspy.timeout_seconds,
            enable_module_caching: self.cache.enabled,
            cache_ttl_seconds: self.cache.ttl_seconds,
            enable_monitoring: self.performance.enable_monitoring,
            max_examples: 100, // Default value
            min_confidence_threshold: self.benchmark.metrics.accuracy_threshold,
        }
    }
}

/// Configuration manager for DSPy CLI
pub struct DspyConfigManager {
    config_path: PathBuf,
}

impl DspyConfigManager {
    /// Create a new configuration manager
    pub fn new() -> Self {
        let config_path = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("memvid-agent")
            .join("dspy.toml");

        Self { config_path }
    }

    /// Create configuration manager with custom path
    pub fn with_path(config_path: PathBuf) -> Self {
        Self { config_path }
    }

    /// Load configuration from file
    pub async fn load_config(&self) -> DspyCliResult<DspyCliConfig> {
        if !self.config_path.exists() {
            debug!("Configuration file not found, using defaults: {}", self.config_path.display());
            return Ok(DspyCliConfig::default());
        }

        let content = fs::read_to_string(&self.config_path).await
            .map_err(|e| DspyCliError::Config {
                message: format!("Failed to read configuration file: {}", e),
                suggestion: Some("Check file permissions and path".to_string()),
                config_path: Some(self.config_path.clone()),
                line_number: None,
            })?;

        let config: DspyCliConfig = toml::from_str(&content)
            .map_err(|e| DspyCliError::Config {
                message: format!("Failed to parse configuration file: {}", e),
                suggestion: Some("Check TOML syntax and structure".to_string()),
                config_path: Some(self.config_path.clone()),
                line_number: None, // TOML error doesn't provide line info in this version
            })?;

        config.validate()?;
        info!("Configuration loaded from: {}", self.config_path.display());
        Ok(config)
    }

    /// Save configuration to file
    pub async fn save_config(&self, config: &DspyCliConfig) -> DspyCliResult<()> {
        // Validate before saving
        config.validate()?;

        // Ensure parent directory exists
        if let Some(parent) = self.config_path.parent() {
            fs::create_dir_all(parent).await
                .map_err(|e| DspyCliError::Resource {
                    resource: "filesystem".to_string(),
                    message: format!("Failed to create config directory: {}", e),
                    current_usage: None,
                    limit: None,
                    suggestion: Some("Check directory permissions".to_string()),
                })?;
        }

        let content = toml::to_string_pretty(config)
            .map_err(|e| DspyCliError::Internal {
                message: format!("Failed to serialize configuration: {}", e),
                error_id: uuid::Uuid::new_v4().to_string(),
                context: std::collections::HashMap::new(),
            })?;

        fs::write(&self.config_path, content).await
            .map_err(|e| DspyCliError::Config {
                message: format!("Failed to write configuration file: {}", e),
                suggestion: Some("Check file permissions and disk space".to_string()),
                config_path: Some(self.config_path.clone()),
                line_number: None,
            })?;

        info!("Configuration saved to: {}", self.config_path.display());
        Ok(())
    }

    /// Ensure configuration file exists with defaults
    pub async fn ensure_config_exists(&self) -> DspyCliResult<()> {
        if !self.config_path.exists() {
            let default_config = DspyCliConfig::default();
            self.save_config(&default_config).await?;
            info!("Created default configuration file: {}", self.config_path.display());
        }
        Ok(())
    }

    /// Get configuration file path
    pub fn config_path(&self) -> &Path {
        &self.config_path
    }
}

impl Default for DspyConfigManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_default_config_validation() {
        let config = DspyCliConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_to_dspy_config() {
        let config = DspyCliConfig::default();
        let dspy_config = config.to_dspy_config();
        
        assert_eq!(dspy_config.max_optimization_iterations, config.optimization.max_iterations as u32);
        assert_eq!(dspy_config.enable_module_caching, config.cache.enabled);
    }

    #[tokio::test]
    async fn test_config_manager_save_load() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("test_config.toml");
        
        let manager = DspyConfigManager::with_path(config_path);
        let original_config = DspyCliConfig::default();
        
        // Save config
        manager.save_config(&original_config).await.unwrap();
        
        // Load config
        let loaded_config = manager.load_config().await.unwrap();
        
        // Compare (basic check)
        assert_eq!(original_config.version, loaded_config.version);
        assert_eq!(original_config.dspy.default_strategy, loaded_config.dspy.default_strategy);
    }
}
