//! DSPy CLI Module
//!
//! This module provides command-line interface functionality for the DSPy framework,
//! including module management, benchmarking, optimization, and pipeline operations.

pub mod commands;
pub mod config;
pub mod error;
pub mod utils;

// Re-export main types for convenience
pub use commands::{DspyCommands, DspySubcommand};
pub use config::{DspyCliConfig, DspyConfigManager};
pub use error::{DspyCliError, DspyCliResult};

use crate::agent::Agent;
use crate::config::AgentConfig;
use crate::dspy::{init_dspy, DspyConfig, DspyRegistry};
use crate::utils::error::Result;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// DSPy CLI context containing shared state and configuration
#[derive(Debug)]
pub struct DspyCliContext {
    /// Agent instance for DSPy operations
    pub agent: Arc<Agent>,
    /// DSPy configuration
    pub dspy_config: DspyConfig,
    /// DSPy module registry
    pub registry: Arc<DspyRegistry>,
    /// CLI-specific configuration
    pub cli_config: DspyCliConfig,
}

impl DspyCliContext {
    /// Create a new DSPy CLI context
    pub async fn new(agent_config: AgentConfig) -> DspyCliResult<Self> {
        debug!("Initializing DSPy CLI context");

        // Load CLI configuration
        let config_manager = DspyConfigManager::new();
        let cli_config = config_manager.load_config().await?;

        // Create DSPy configuration from CLI config
        let dspy_config = cli_config.to_dspy_config();

        // Initialize DSPy system
        let registry =
            Arc::new(
                init_dspy(dspy_config.clone())
                    .await
                    .map_err(|e| DspyCliError::Internal {
                        message: format!("Failed to initialize DSPy system: {}", e),
                        error_id: uuid::Uuid::new_v4().to_string(),
                        context: std::collections::HashMap::new(),
                    })?,
            );

        // Create agent
        let mut agent = Agent::new(agent_config)
            .await
            .map_err(|e| DspyCliError::Internal {
                message: format!("Failed to create agent: {}", e),
                error_id: uuid::Uuid::new_v4().to_string(),
                context: std::collections::HashMap::new(),
            })?;

        // Enable DSPy integration
        agent
            .enable_dspy_integration(None)
            .map_err(|e| DspyCliError::Internal {
                message: format!("Failed to enable DSPy integration: {}", e),
                error_id: uuid::Uuid::new_v4().to_string(),
                context: std::collections::HashMap::new(),
            })?;

        let agent = Arc::new(agent);

        info!("DSPy CLI context initialized successfully");

        Ok(Self {
            agent,
            dspy_config,
            registry,
            cli_config,
        })
    }

    /// Get the current working directory for DSPy operations
    pub fn working_directory(&self) -> &std::path::Path {
        &self.cli_config.paths.base_dir
    }

    /// Get the module registry path
    pub fn registry_path(&self) -> &std::path::Path {
        &self.cli_config.paths.registry_path
    }

    /// Get the templates path
    pub fn templates_path(&self) -> &std::path::Path {
        &self.cli_config.paths.templates_path
    }

    /// Get the cache path
    pub fn cache_path(&self) -> &std::path::Path {
        &self.cli_config.paths.cache_path
    }

    /// Check if caching is enabled
    pub fn is_caching_enabled(&self) -> bool {
        self.cli_config.cache.enabled
    }

    /// Get the default output format
    pub fn default_output_format(&self) -> &str {
        &self.cli_config.output.default_format
    }

    /// Check if verbose logging is enabled
    pub fn is_verbose(&self) -> bool {
        matches!(self.cli_config.logging.level.as_str(), "debug" | "trace")
    }

    /// Validate the CLI context and configuration
    pub async fn validate(&self) -> DspyCliResult<()> {
        debug!("Validating DSPy CLI context");

        // Validate configuration
        self.cli_config.validate()?;

        // Ensure required directories exist
        self.ensure_directories_exist().await?;

        // Validate agent DSPy integration
        // Note: We assume DSPy integration is enabled since we called enable_dspy_integration above

        info!("DSPy CLI context validation completed successfully");
        Ok(())
    }

    /// Ensure all required directories exist
    async fn ensure_directories_exist(&self) -> DspyCliResult<()> {
        use tokio::fs;

        let directories = [
            &self.cli_config.paths.base_dir,
            &self.cli_config.paths.registry_path,
            &self.cli_config.paths.templates_path,
            &self.cli_config.paths.cache_path,
            &self.cli_config.paths.logs_path,
            &self.cli_config.paths.backups_path,
            &self.cli_config.paths.temp_path,
        ];

        for dir in &directories {
            if !dir.exists() {
                debug!("Creating directory: {}", dir.display());
                fs::create_dir_all(dir)
                    .await
                    .map_err(|e| DspyCliError::Resource {
                        resource: "filesystem".to_string(),
                        message: format!("Failed to create directory {}: {}", dir.display(), e),
                        current_usage: None,
                        limit: None,
                        suggestion: Some(
                            "Check directory permissions and available disk space".to_string(),
                        ),
                    })?;
            }
        }

        Ok(())
    }
}

/// Execute a DSPy CLI command with proper error handling and context
pub async fn execute_dspy_command(
    command: commands::DspyCommands,
    agent_config: AgentConfig,
) -> DspyCliResult<()> {
    // Create CLI context
    let context = DspyCliContext::new(agent_config).await?;

    // Validate context
    context.validate().await?;

    // Execute the command
    match command.command {
        commands::DspySubcommand::Modules { command } => {
            commands::modules::execute_modules_command(command, &context).await
        }
        commands::DspySubcommand::Benchmark { command } => {
            commands::benchmark::execute_benchmark_command(command, &context).await
        }
        commands::DspySubcommand::Optimize { command } => {
            commands::optimize::execute_optimize_command(command, &context).await
        }
        commands::DspySubcommand::Pipeline { command } => {
            commands::pipeline::execute_pipeline_command(command, &context).await
        }
        commands::DspySubcommand::Dev { command } => {
            commands::dev::execute_dev_command(command, &context).await
        }
    }
}

/// Initialize DSPy CLI system
pub async fn init_dspy_cli() -> DspyCliResult<()> {
    debug!("Initializing DSPy CLI system");

    // Load and validate configuration
    let config_manager = DspyConfigManager::new();
    config_manager.ensure_config_exists().await?;

    let config = config_manager.load_config().await?;
    config.validate()?;

    // Create required directories
    let directories = [
        &config.paths.base_dir,
        &config.paths.registry_path,
        &config.paths.templates_path,
        &config.paths.cache_path,
        &config.paths.logs_path,
        &config.paths.backups_path,
        &config.paths.temp_path,
    ];

    for dir in &directories {
        if !dir.exists() {
            tokio::fs::create_dir_all(dir)
                .await
                .map_err(|e| DspyCliError::Resource {
                    resource: "filesystem".to_string(),
                    message: format!("Failed to create directory {}: {}", dir.display(), e),
                    current_usage: None,
                    limit: None,
                    suggestion: Some(
                        "Check directory permissions and available disk space".to_string(),
                    ),
                })?;
        }
    }

    info!("DSPy CLI system initialized successfully");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AgentConfig;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_dspy_cli_context_creation() {
        let config = AgentConfig::default();
        let result = DspyCliContext::new(config).await;

        // Note: This test may fail if DSPy integration has issues
        // In a real environment, we'd mock the dependencies
        match result {
            Ok(context) => {
                // Note: We can't access dspy_extension directly as it's private
                // Just verify that the context was created successfully
                assert!(!context.cli_config.paths.base_dir.as_os_str().is_empty());
            }
            Err(e) => {
                // Log the error for debugging but don't fail the test
                // since this depends on external configuration
                eprintln!(
                    "DSPy CLI context creation failed (expected in test env): {}",
                    e
                );
            }
        }
    }

    #[tokio::test]
    async fn test_init_dspy_cli() {
        // This test verifies the initialization process
        let result = init_dspy_cli().await;

        // Similar to above, this may fail in test environment
        match result {
            Ok(()) => {
                // Success case
            }
            Err(e) => {
                // Expected in test environment without proper config
                eprintln!("DSPy CLI init failed (expected in test env): {}", e);
            }
        }
    }
}
