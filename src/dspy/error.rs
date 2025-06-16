//! Error types and handling for the DSPy integration module
//!
//! This module provides comprehensive error handling for DSPy operations,
//! integrating with the existing AgentError framework while providing
//! DSPy-specific error types and context.

use crate::utils::error::AgentError;
use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// Result type alias for DSPy operations
pub type DspyResult<T> = std::result::Result<T, DspyError>;

/// Comprehensive error types for DSPy operations
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum DspyError {
    /// Signature definition errors
    #[error("Signature error: {message}")]
    Signature { message: String },

    /// Module execution errors
    #[error("Module error: {module_name}: {message}")]
    Module {
        module_name: String,
        message: String,
    },

    /// Optimization and compilation errors
    #[error("Optimization error: {strategy}: {message}")]
    Optimization { strategy: String, message: String },

    /// Evaluation and metrics errors
    #[error("Evaluation error: {metric_name}: {message}")]
    Evaluation {
        metric_name: String,
        message: String,
    },

    /// Type validation errors
    #[error("Type validation error: {field_name}: {message}")]
    TypeValidation { field_name: String, message: String },

    /// Serialization/deserialization errors
    #[error("Serialization error: {context}: {message}")]
    Serialization { context: String, message: String },

    /// Configuration errors
    #[error("Configuration error: {parameter}: {message}")]
    Configuration { parameter: String, message: String },

    /// Resource errors (memory, timeout, etc.)
    #[error("Resource error: {resource}: {message}")]
    Resource { resource: String, message: String },

    /// Compilation errors
    #[error("Compilation error: {phase}: {message}")]
    Compilation { phase: String, message: String },

    /// Cache-related errors
    #[error("Cache error: {operation}: {message}")]
    Cache { operation: String, message: String },

    /// Integration errors with other systems
    #[error("Integration error: {system}: {message}")]
    Integration { system: String, message: String },

    /// Chain execution errors
    #[error("Chain execution error in {module_name} ({stage}): {source}")]
    ChainExecution {
        module_name: String,
        stage: String,
        source: Box<DspyError>,
    },

    /// IO errors
    #[error("IO error: {message}")]
    Io { message: String },
}

impl DspyError {
    /// Create a new signature error
    pub fn signature<S: Into<String>>(message: S) -> Self {
        Self::Signature {
            message: message.into(),
        }
    }

    /// Create a new module error
    pub fn module<S: Into<String>>(module_name: S, message: S) -> Self {
        Self::Module {
            module_name: module_name.into(),
            message: message.into(),
        }
    }

    /// Create a new optimization error
    pub fn optimization<S: Into<String>>(strategy: S, message: S) -> Self {
        Self::Optimization {
            strategy: strategy.into(),
            message: message.into(),
        }
    }

    /// Create a new evaluation error
    pub fn evaluation<S: Into<String>>(metric_name: S, message: S) -> Self {
        Self::Evaluation {
            metric_name: metric_name.into(),
            message: message.into(),
        }
    }

    /// Create a new type validation error
    pub fn type_validation<S: Into<String>>(field_name: S, message: S) -> Self {
        Self::TypeValidation {
            field_name: field_name.into(),
            message: message.into(),
        }
    }

    /// Create a new serialization error
    pub fn serialization<S: Into<String>>(context: S, message: S) -> Self {
        Self::Serialization {
            context: context.into(),
            message: message.into(),
        }
    }

    /// Create a new configuration error
    pub fn configuration<S: Into<String>>(parameter: S, message: S) -> Self {
        Self::Configuration {
            parameter: parameter.into(),
            message: message.into(),
        }
    }

    /// Create a new resource error
    pub fn resource<S: Into<String>>(resource: S, message: S) -> Self {
        Self::Resource {
            resource: resource.into(),
            message: message.into(),
        }
    }

    /// Create a new compilation error
    pub fn compilation<S: Into<String>>(phase: S, message: S) -> Self {
        Self::Compilation {
            phase: phase.into(),
            message: message.into(),
        }
    }

    /// Create a new cache error
    pub fn cache<S: Into<String>>(operation: S, message: S) -> Self {
        Self::Cache {
            operation: operation.into(),
            message: message.into(),
        }
    }

    /// Create a new integration error
    pub fn integration<S: Into<String>>(system: S, message: S) -> Self {
        Self::Integration {
            system: system.into(),
            message: message.into(),
        }
    }

    /// Create a new chain execution error
    pub fn chain_execution<S: Into<String>>(module_name: S, stage: S, source: DspyError) -> Self {
        Self::ChainExecution {
            module_name: module_name.into(),
            stage: stage.into(),
            source: Box::new(source),
        }
    }

    /// Create a new IO error
    pub fn io<S: Into<String>>(message: S) -> Self {
        Self::Io {
            message: message.into(),
        }
    }

    /// Create a new validation error
    pub fn validation<S: Into<String>>(field: S, message: S) -> Self {
        Self::TypeValidation {
            field_name: field.into(),
            message: message.into(),
        }
    }

    /// Create a new invalid input error (alias for validation)
    pub fn invalid_input<S: Into<String>>(message: S) -> Self {
        let msg = message.into();
        Self::validation("input", &msg)
    }

    /// Get the error category for logging and monitoring
    pub fn category(&self) -> &'static str {
        match self {
            Self::Signature { .. } => "signature",
            Self::Module { .. } => "module",
            Self::Optimization { .. } => "optimization",
            Self::Evaluation { .. } => "evaluation",
            Self::TypeValidation { .. } => "type_validation",
            Self::Serialization { .. } => "serialization",
            Self::Configuration { .. } => "configuration",
            Self::Resource { .. } => "resource",
            Self::Compilation { .. } => "compilation",
            Self::Cache { .. } => "cache",
            Self::Integration { .. } => "integration",
            Self::ChainExecution { .. } => "chain_execution",
            Self::Io { .. } => "io",
        }
    }

    /// Check if the error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::Resource { .. }
                | Self::Cache { .. }
                | Self::Integration { .. }
                | Self::Optimization { .. }
                | Self::ChainExecution { .. }
        )
    }

    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::Configuration { .. } => ErrorSeverity::Critical,
            Self::TypeValidation { .. } => ErrorSeverity::High,
            Self::Signature { .. } => ErrorSeverity::High,
            Self::Module { .. } => ErrorSeverity::Medium,
            Self::Compilation { .. } => ErrorSeverity::Medium,
            Self::Optimization { .. } => ErrorSeverity::Low,
            Self::Evaluation { .. } => ErrorSeverity::Low,
            Self::Serialization { .. } => ErrorSeverity::Medium,
            Self::Resource { .. } => ErrorSeverity::Medium,
            Self::Cache { .. } => ErrorSeverity::Low,
            Self::Integration { .. } => ErrorSeverity::Medium,
            Self::ChainExecution { .. } => ErrorSeverity::High,
            Self::Io { .. } => ErrorSeverity::Medium,
        }
    }

    /// Convert to AgentError for integration with existing error handling
    pub fn into_agent_error(self) -> AgentError {
        match self {
            Self::Signature { message } => {
                AgentError::invalid_input(format!("DSPy signature error: {}", message))
            }
            Self::Module {
                module_name,
                message,
            } => AgentError::tool(module_name, format!("DSPy module error: {}", message)),
            Self::Optimization { strategy, message } => AgentError::Generic(anyhow::anyhow!(
                "DSPy optimization error in {}: {}",
                strategy,
                message
            )),
            Self::Evaluation {
                metric_name,
                message,
            } => AgentError::Generic(anyhow::anyhow!(
                "DSPy evaluation error in {}: {}",
                metric_name,
                message
            )),
            Self::TypeValidation {
                field_name,
                message,
            } => AgentError::validation(format!(
                "DSPy type validation error in {}: {}",
                field_name, message
            )),
            Self::Serialization { context, message } => AgentError::Generic(anyhow::anyhow!(
                "DSPy serialization error in {}: {}",
                context,
                message
            )),
            Self::Configuration { parameter, message } => AgentError::config(format!(
                "DSPy configuration error in {}: {}",
                parameter, message
            )),
            Self::Resource { resource, message } => AgentError::Generic(anyhow::anyhow!(
                "DSPy resource error with {}: {}",
                resource,
                message
            )),
            Self::Compilation { phase, message } => AgentError::Generic(anyhow::anyhow!(
                "DSPy compilation error in {}: {}",
                phase,
                message
            )),
            Self::Cache { operation, message } => AgentError::Generic(anyhow::anyhow!(
                "DSPy cache error during {}: {}",
                operation,
                message
            )),
            Self::Integration { system, message } => AgentError::Generic(anyhow::anyhow!(
                "DSPy integration error with {}: {}",
                system,
                message
            )),
            Self::ChainExecution {
                module_name,
                stage,
                source,
            } => AgentError::Generic(anyhow::anyhow!(
                "DSPy chain execution error in {} ({}): {}",
                module_name,
                stage,
                source
            )),
            Self::Io { message } => AgentError::Generic(anyhow::anyhow!(
                "DSPy IO error: {}",
                message
            )),
        }
    }
}

impl From<serde_json::Error> for DspyError {
    fn from(error: serde_json::Error) -> Self {
        Self::serialization("json", &error.to_string())
    }
}

/// Error severity levels for monitoring and alerting
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Critical errors that prevent system operation
    Critical,
    /// High priority errors that significantly impact functionality
    High,
    /// Medium priority errors that may impact some operations
    Medium,
    /// Low priority errors that have minimal impact
    Low,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Critical => write!(f, "CRITICAL"),
            Self::High => write!(f, "HIGH"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::Low => write!(f, "LOW"),
        }
    }
}

/// Error context for enhanced debugging and monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    /// Error ID for tracking
    pub error_id: String,
    /// Timestamp when error occurred
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Module or component where error occurred
    pub component: String,
    /// Operation being performed when error occurred
    pub operation: String,
    /// Additional context data
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

impl ErrorContext {
    /// Create a new error context
    pub fn new<S: Into<String>>(component: S, operation: S) -> Self {
        Self {
            error_id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
            component: component.into(),
            operation: operation.into(),
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add metadata to the error context
    pub fn with_metadata<K: Into<String>, V: Into<serde_json::Value>>(
        mut self,
        key: K,
        value: V,
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Enhanced error type with context for debugging and monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualDspyError {
    /// The underlying DSPy error
    pub error: DspyError,
    /// Additional context for debugging
    pub context: ErrorContext,
}

impl ContextualDspyError {
    /// Create a new contextual error
    pub fn new(error: DspyError, context: ErrorContext) -> Self {
        Self { error, context }
    }

    /// Create a contextual error with basic context
    pub fn with_context<S: Into<String>>(error: DspyError, component: S, operation: S) -> Self {
        let context = ErrorContext::new(component, operation);
        Self::new(error, context)
    }
}

impl fmt::Display for ContextualDspyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} in {}.{}: {}",
            self.error.severity(),
            self.error.category(),
            self.context.component,
            self.context.operation,
            self.error
        )
    }
}

impl std::error::Error for ContextualDspyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dspy_error_creation() {
        let error = DspyError::signature("Invalid field type");
        assert_eq!(error.category(), "signature");
        assert_eq!(error.severity(), ErrorSeverity::High);
        assert!(!error.is_recoverable());
    }

    #[test]
    fn test_error_severity_display() {
        assert_eq!(ErrorSeverity::Critical.to_string(), "CRITICAL");
        assert_eq!(ErrorSeverity::High.to_string(), "HIGH");
        assert_eq!(ErrorSeverity::Medium.to_string(), "MEDIUM");
        assert_eq!(ErrorSeverity::Low.to_string(), "LOW");
    }

    #[test]
    fn test_error_context_creation() {
        let context = ErrorContext::new("test_component", "test_operation")
            .with_metadata("key1", "value1")
            .with_metadata("key2", 42);

        assert_eq!(context.component, "test_component");
        assert_eq!(context.operation, "test_operation");
        assert_eq!(context.metadata.len(), 2);
        assert!(context.metadata.contains_key("key1"));
        assert!(context.metadata.contains_key("key2"));
    }

    #[test]
    fn test_contextual_error() {
        let error = DspyError::module("test_module", "Test error message");
        let contextual =
            ContextualDspyError::with_context(error, "test_component", "test_operation");

        assert_eq!(contextual.error.category(), "module");
        assert_eq!(contextual.context.component, "test_component");
        assert_eq!(contextual.context.operation, "test_operation");
    }

    #[test]
    fn test_error_conversion_to_agent_error() {
        let dspy_error = DspyError::signature("Test signature error");
        let agent_error = dspy_error.into_agent_error();

        match agent_error {
            AgentError::InvalidInput { message } => {
                assert!(message.contains("DSPy signature error"));
                assert!(message.contains("Test signature error"));
            }
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[test]
    fn test_error_recoverability() {
        assert!(DspyError::resource("memory", "Out of memory").is_recoverable());
        assert!(DspyError::cache("get", "Cache miss").is_recoverable());
        assert!(!DspyError::signature("Invalid type").is_recoverable());
        assert!(!DspyError::configuration("invalid_param", "Bad value").is_recoverable());
    }
}
