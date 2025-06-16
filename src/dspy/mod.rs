//! # DSPy Integration for Rust MemVid Agent
//!
//! This module provides a comprehensive implementation of DSPy (Stanford NLP's framework for
//! programming foundation models) concepts in Rust. It enables systematic prompt engineering,
//! automatic optimization, and modular LLM programming while maintaining Rust's performance
//! and safety guarantees.
//!
//! ## Core Components
//!
//! - **Signatures**: Type-safe input/output definitions for LLM tasks
//! - **Modules**: Composable units that define specific LLM operations
//! - **Optimization**: Automatic prompt optimization and compilation
//! - **Evaluation**: Comprehensive metrics and evaluation framework
//!
//! ## Example Usage
//!
//! ```rust
//! use rust_memvid_agent::dspy::{Signature, Module, Predict};
//! use rust_memvid_agent::anthropic::AnthropicClient;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Serialize, Deserialize)]
//! struct QuestionInput {
//!     question: String,
//! }
//!
//! #[derive(Serialize, Deserialize)]
//! struct AnswerOutput {
//!     answer: String,
//!     reasoning: String,
//! }
//!
//! // Define a signature for the task
//! let signature = Signature::<QuestionInput, AnswerOutput>::new()
//!     .with_input_field("question", "The question to answer")
//!     .with_output_field("answer", "The final answer")
//!     .with_output_field("reasoning", "Step-by-step reasoning");
//!
//! // Create a prediction module
//! let predict_module = Predict::new(signature, anthropic_client);
//! ```

pub mod error;
pub mod module;
pub mod predictor;
pub mod signature;

#[cfg(test)]
mod tests;

// Re-export core types for convenience
pub use error::{DspyError, DspyResult};
pub use module::{Module, ModuleMetadata, ModuleStats};
pub use predictor::{Predict, PredictConfig};
pub use signature::{Field, FieldType, Signature, SignatureBuilder};

use crate::utils::error::{AgentError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

/// DSPy module registry for managing and discovering modules
#[derive(Debug, Default)]
pub struct DspyRegistry {
    modules: HashMap<String, Box<dyn ModuleInfo>>,
}

/// Information about a registered DSPy module
pub trait ModuleInfo: Send + Sync + std::fmt::Debug {
    /// Get the module name
    fn name(&self) -> &str;

    /// Get the module description
    fn description(&self) -> Option<&str>;

    /// Get the module version
    fn version(&self) -> &str;

    /// Get module capabilities
    fn capabilities(&self) -> Vec<String>;
}

impl DspyRegistry {
    /// Create a new DSPy registry
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
        }
    }

    /// Register a module in the registry
    pub fn register_module<T: ModuleInfo + 'static>(&mut self, module: T) -> Result<()> {
        let name = module.name().to_string();

        if self.modules.contains_key(&name) {
            return Err(AgentError::invalid_input(format!(
                "Module '{}' is already registered",
                name
            )));
        }

        debug!("Registering DSPy module: {}", name);
        self.modules.insert(name, Box::new(module));

        Ok(())
    }

    /// Get information about a registered module
    pub fn get_module_info(&self, name: &str) -> Option<&dyn ModuleInfo> {
        self.modules.get(name).map(|m| m.as_ref())
    }

    /// List all registered modules
    pub fn list_modules(&self) -> Vec<&str> {
        self.modules.keys().map(|k| k.as_str()).collect()
    }

    /// Get the number of registered modules
    pub fn module_count(&self) -> usize {
        self.modules.len()
    }

    /// Check if a module is registered
    pub fn has_module(&self, name: &str) -> bool {
        self.modules.contains_key(name)
    }

    /// Remove a module from the registry
    pub fn unregister_module(&mut self, name: &str) -> Result<()> {
        if self.modules.remove(name).is_some() {
            debug!("Unregistered DSPy module: {}", name);
            Ok(())
        } else {
            Err(AgentError::invalid_input(format!(
                "Module '{}' is not registered",
                name
            )))
        }
    }

    /// Clear all registered modules
    pub fn clear(&mut self) {
        debug!("Clearing all DSPy modules from registry");
        self.modules.clear();
    }
}

/// DSPy system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DspyConfig {
    /// Enable optimization features
    pub enable_optimization: bool,

    /// Maximum number of optimization iterations
    pub max_optimization_iterations: u32,

    /// Optimization timeout in seconds
    pub optimization_timeout_seconds: u64,

    /// Enable caching of compiled modules
    pub enable_module_caching: bool,

    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,

    /// Enable performance monitoring
    pub enable_monitoring: bool,

    /// Maximum number of examples for optimization
    pub max_examples: usize,

    /// Minimum confidence threshold for optimization
    pub min_confidence_threshold: f64,
}

impl Default for DspyConfig {
    fn default() -> Self {
        Self {
            enable_optimization: true,
            max_optimization_iterations: 10,
            optimization_timeout_seconds: 300,
            enable_module_caching: true,
            cache_ttl_seconds: 3600,
            enable_monitoring: true,
            max_examples: 100,
            min_confidence_threshold: 0.7,
        }
    }
}

impl DspyConfig {
    /// Create a new DSPy configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set optimization enabled
    pub fn with_optimization(mut self, enabled: bool) -> Self {
        self.enable_optimization = enabled;
        self
    }

    /// Set maximum optimization iterations
    pub fn with_max_iterations(mut self, iterations: u32) -> Self {
        self.max_optimization_iterations = iterations;
        self
    }

    /// Set optimization timeout
    pub fn with_timeout(mut self, timeout_seconds: u64) -> Self {
        self.optimization_timeout_seconds = timeout_seconds;
        self
    }

    /// Set caching enabled
    pub fn with_caching(mut self, enabled: bool) -> Self {
        self.enable_module_caching = enabled;
        self
    }

    /// Set cache TTL
    pub fn with_cache_ttl(mut self, ttl_seconds: u64) -> Self {
        self.cache_ttl_seconds = ttl_seconds;
        self
    }

    /// Set monitoring enabled
    pub fn with_monitoring(mut self, enabled: bool) -> Self {
        self.enable_monitoring = enabled;
        self
    }

    /// Set maximum examples for optimization
    pub fn with_max_examples(mut self, max_examples: usize) -> Self {
        self.max_examples = max_examples;
        self
    }

    /// Set minimum confidence threshold
    pub fn with_min_confidence(mut self, threshold: f64) -> Self {
        self.min_confidence_threshold = threshold;
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.max_optimization_iterations == 0 {
            return Err(AgentError::config(
                "max_optimization_iterations must be greater than 0",
            ));
        }

        if self.optimization_timeout_seconds == 0 {
            return Err(AgentError::config(
                "optimization_timeout_seconds must be greater than 0",
            ));
        }

        if self.cache_ttl_seconds == 0 {
            return Err(AgentError::config(
                "cache_ttl_seconds must be greater than 0",
            ));
        }

        if self.max_examples == 0 {
            return Err(AgentError::config("max_examples must be greater than 0"));
        }

        if !(0.0..=1.0).contains(&self.min_confidence_threshold) {
            return Err(AgentError::config(
                "min_confidence_threshold must be between 0.0 and 1.0",
            ));
        }

        Ok(())
    }
}

/// Initialize the DSPy system with the given configuration
pub async fn init_dspy(config: DspyConfig) -> Result<DspyRegistry> {
    info!("Initializing DSPy system");

    // Validate configuration
    config.validate()?;

    // Create registry
    let registry = DspyRegistry::new();

    info!("DSPy system initialized successfully");
    Ok(registry)
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_dspy_system_initialization() {
        let config = DspyConfig::default();
        let registry = init_dspy(config).await.unwrap();

        assert_eq!(registry.module_count(), 0);
        assert!(registry.list_modules().is_empty());
    }

    #[tokio::test]
    async fn test_dspy_config_validation() {
        let mut config = DspyConfig::default();
        assert!(config.validate().is_ok());

        config.max_optimization_iterations = 0;
        assert!(config.validate().is_err());

        config = DspyConfig::default();
        config.min_confidence_threshold = 1.5;
        assert!(config.validate().is_err());
    }
}
