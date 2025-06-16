//! DSPy CLI Error Handling
//!
//! This module provides comprehensive error handling for DSPy CLI operations,
//! including user-friendly error messages and actionable suggestions.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use thiserror::Error;

/// Result type for DSPy CLI operations
pub type DspyCliResult<T> = Result<T, DspyCliError>;

/// Comprehensive error types for DSPy CLI operations
#[derive(Error, Debug)]
pub enum DspyCliError {
    /// Configuration-related errors
    #[error("Configuration error: {message}")]
    Config {
        message: String,
        suggestion: Option<String>,
        config_path: Option<PathBuf>,
        line_number: Option<usize>,
    },

    /// Input validation errors
    #[error("Validation error in {field}: {message}")]
    Validation {
        field: String,
        message: String,
        expected: Option<String>,
        actual: Option<String>,
        suggestions: Vec<String>,
    },

    /// Execution and runtime errors
    #[error("Execution error in {operation}: {message}")]
    Execution {
        operation: String,
        message: String,
        stage: Option<String>,
        retry_possible: bool,
        context: HashMap<String, String>,
    },

    /// Network and API errors
    #[error("Network error: {message}")]
    Network {
        message: String,
        error_code: Option<u16>,
        retry_after: Option<Duration>,
        endpoint: Option<String>,
    },

    /// Resource-related errors (memory, disk, CPU)
    #[error("Resource error - {resource}: {message}")]
    Resource {
        resource: String,
        message: String,
        current_usage: Option<String>,
        limit: Option<String>,
        suggestion: Option<String>,
    },

    /// Permission and access errors
    #[error("Permission error: {message}")]
    Permission {
        message: String,
        path: Option<PathBuf>,
        required_permission: Option<String>,
        current_user: Option<String>,
    },

    /// Timeout errors
    #[error("Timeout error: {operation} timed out after {duration:?}")]
    Timeout {
        operation: String,
        duration: Duration,
        suggestion: Option<String>,
    },

    /// User cancellation
    #[error("User cancelled operation: {operation}")]
    UserCancelled {
        operation: String,
        partial_completion: bool,
        cleanup_required: bool,
    },

    /// Dependency and external tool errors
    #[error("Dependency error: {dependency} - {message}")]
    Dependency {
        dependency: String,
        message: String,
        version_required: Option<String>,
        version_found: Option<String>,
        install_suggestion: Option<String>,
    },

    /// Internal errors and bugs
    #[error("Internal error: {message}")]
    Internal {
        message: String,
        error_id: String,
        context: HashMap<String, String>,
    },
}

/// Exit codes for CLI operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitCode {
    Success = 0,
    GeneralError = 1,
    ConfigurationError = 2,
    ValidationError = 3,
    NetworkError = 4,
    ResourceError = 5,
    PermissionError = 6,
    TimeoutError = 7,
    UserCancelled = 8,
    DependencyError = 9,
    InternalError = 10,
}

impl From<DspyCliError> for ExitCode {
    fn from(error: DspyCliError) -> Self {
        match error {
            DspyCliError::Config { .. } => ExitCode::ConfigurationError,
            DspyCliError::Validation { .. } => ExitCode::ValidationError,
            DspyCliError::Execution { .. } => ExitCode::GeneralError,
            DspyCliError::Network { .. } => ExitCode::NetworkError,
            DspyCliError::Resource { .. } => ExitCode::ResourceError,
            DspyCliError::Permission { .. } => ExitCode::PermissionError,
            DspyCliError::Timeout { .. } => ExitCode::TimeoutError,
            DspyCliError::UserCancelled { .. } => ExitCode::UserCancelled,
            DspyCliError::Dependency { .. } => ExitCode::DependencyError,
            DspyCliError::Internal { .. } => ExitCode::InternalError,
        }
    }
}

impl DspyCliError {
    /// Create a configuration error with suggestion
    pub fn config_error(message: impl Into<String>, suggestion: impl Into<String>) -> Self {
        Self::Config {
            message: message.into(),
            suggestion: Some(suggestion.into()),
            config_path: None,
            line_number: None,
        }
    }

    /// Create a validation error with suggestions
    pub fn validation_error(
        field: impl Into<String>,
        message: impl Into<String>,
        suggestions: Vec<String>,
    ) -> Self {
        Self::Validation {
            field: field.into(),
            message: message.into(),
            expected: None,
            actual: None,
            suggestions,
        }
    }

    /// Create an execution error with context
    pub fn execution_error(
        operation: impl Into<String>,
        message: impl Into<String>,
        retry_possible: bool,
    ) -> Self {
        Self::Execution {
            operation: operation.into(),
            message: message.into(),
            stage: None,
            retry_possible,
            context: HashMap::new(),
        }
    }

    /// Create a network error with retry information
    pub fn network_error(message: impl Into<String>, retry_after: Option<Duration>) -> Self {
        Self::Network {
            message: message.into(),
            error_code: None,
            retry_after,
            endpoint: None,
        }
    }

    /// Create a resource error with usage information
    pub fn resource_error(
        resource: impl Into<String>,
        message: impl Into<String>,
        suggestion: impl Into<String>,
    ) -> Self {
        Self::Resource {
            resource: resource.into(),
            message: message.into(),
            current_usage: None,
            limit: None,
            suggestion: Some(suggestion.into()),
        }
    }

    /// Create an internal error with unique ID
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
            error_id: uuid::Uuid::new_v4().to_string(),
            context: HashMap::new(),
        }
    }

    /// Add context information to the error
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        match &mut self {
            Self::Execution { context, .. } => {
                context.insert(key.into(), value.into());
            }
            Self::Internal { context, .. } => {
                context.insert(key.into(), value.into());
            }
            _ => {} // Other error types don't support context
        }
        self
    }

    /// Add suggestion to applicable error types
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        let suggestion = suggestion.into();
        match &mut self {
            Self::Config {
                suggestion: ref mut s,
                ..
            } => *s = Some(suggestion),
            Self::Resource {
                suggestion: ref mut s,
                ..
            } => *s = Some(suggestion),
            Self::Timeout {
                suggestion: ref mut s,
                ..
            } => *s = Some(suggestion),
            _ => {} // Other error types don't support suggestions
        }
        self
    }

    /// Get user-friendly error message with suggestions
    pub fn user_message(&self) -> String {
        match self {
            Self::Config {
                message,
                suggestion,
                config_path,
                ..
            } => {
                let mut msg = format!("Configuration Error: {}", message);
                if let Some(path) = config_path {
                    msg.push_str(&format!("\nFile: {}", path.display()));
                }
                if let Some(suggestion) = suggestion {
                    msg.push_str(&format!("\nSuggestion: {}", suggestion));
                }
                msg
            }
            Self::Validation {
                field,
                message,
                expected,
                actual,
                suggestions,
            } => {
                let mut msg = format!("Validation Error in '{}': {}", field, message);
                if let (Some(expected), Some(actual)) = (expected, actual) {
                    msg.push_str(&format!("\nExpected: {}\nActual: {}", expected, actual));
                }
                if !suggestions.is_empty() {
                    msg.push_str("\nSuggestions:");
                    for suggestion in suggestions {
                        msg.push_str(&format!("\n  • {}", suggestion));
                    }
                }
                msg
            }
            Self::Execution {
                operation,
                message,
                stage,
                retry_possible,
                ..
            } => {
                let mut msg = format!("Execution Error in '{}': {}", operation, message);
                if let Some(stage) = stage {
                    msg.push_str(&format!("\nStage: {}", stage));
                }
                if *retry_possible {
                    msg.push_str("\nRetry: This operation can be retried");
                }
                msg
            }
            Self::Network {
                message,
                retry_after,
                endpoint,
                ..
            } => {
                let mut msg = format!("Network Error: {}", message);
                if let Some(endpoint) = endpoint {
                    msg.push_str(&format!("\nEndpoint: {}", endpoint));
                }
                if let Some(retry_after) = retry_after {
                    msg.push_str(&format!("\nRetry after: {:?}", retry_after));
                }
                msg
            }
            Self::Resource {
                resource,
                message,
                current_usage,
                limit,
                suggestion,
            } => {
                let mut msg = format!("Resource Error ({}): {}", resource, message);
                if let (Some(usage), Some(limit)) = (current_usage, limit) {
                    msg.push_str(&format!("\nUsage: {} / {}", usage, limit));
                }
                if let Some(suggestion) = suggestion {
                    msg.push_str(&format!("\nSuggestion: {}", suggestion));
                }
                msg
            }
            Self::Permission {
                message,
                path,
                required_permission,
                ..
            } => {
                let mut msg = format!("Permission Error: {}", message);
                if let Some(path) = path {
                    msg.push_str(&format!("\nPath: {}", path.display()));
                }
                if let Some(permission) = required_permission {
                    msg.push_str(&format!("\nRequired Permission: {}", permission));
                }
                msg
            }
            Self::Timeout {
                operation,
                duration,
                suggestion,
            } => {
                let mut msg = format!(
                    "Timeout Error: '{}' timed out after {:?}",
                    operation, duration
                );
                if let Some(suggestion) = suggestion {
                    msg.push_str(&format!("\nSuggestion: {}", suggestion));
                }
                msg
            }
            Self::UserCancelled {
                operation,
                partial_completion,
                cleanup_required,
            } => {
                let mut msg = format!("Operation Cancelled: '{}'", operation);
                if *partial_completion {
                    msg.push_str("\nPartial completion detected");
                }
                if *cleanup_required {
                    msg.push_str("\nCleanup may be required");
                }
                msg
            }
            Self::Dependency {
                dependency,
                message,
                version_required,
                version_found,
                install_suggestion,
            } => {
                let mut msg = format!("Dependency Error ({}): {}", dependency, message);
                if let (Some(required), Some(found)) = (version_required, version_found) {
                    msg.push_str(&format!("\nRequired: {}\nFound: {}", required, found));
                }
                if let Some(suggestion) = install_suggestion {
                    msg.push_str(&format!("\nInstall: {}", suggestion));
                }
                msg
            }
            Self::Internal {
                message, error_id, ..
            } => {
                format!(
                    "Internal Error: {}\nError ID: {}\nPlease report this issue",
                    message, error_id
                )
            }
        }
    }

    /// Get machine-readable error information
    pub fn error_info(&self) -> ErrorInfo {
        ErrorInfo {
            error_type: self.error_type(),
            error_code: self.error_code(),
            message: self.to_string(),
            severity: self.severity(),
            category: self.category(),
            retry_possible: self.is_retry_possible(),
            help_url: self.help_url(),
        }
    }

    /// Get error type as string
    pub fn error_type(&self) -> &'static str {
        match self {
            Self::Config { .. } => "ConfigurationError",
            Self::Validation { .. } => "ValidationError",
            Self::Execution { .. } => "ExecutionError",
            Self::Network { .. } => "NetworkError",
            Self::Resource { .. } => "ResourceError",
            Self::Permission { .. } => "PermissionError",
            Self::Timeout { .. } => "TimeoutError",
            Self::UserCancelled { .. } => "UserCancelledError",
            Self::Dependency { .. } => "DependencyError",
            Self::Internal { .. } => "InternalError",
        }
    }

    /// Get error code
    pub fn error_code(&self) -> String {
        match self {
            Self::Config { .. } => "DSPY_CFG_001".to_string(),
            Self::Validation { .. } => "DSPY_VAL_001".to_string(),
            Self::Execution { .. } => "DSPY_EXE_001".to_string(),
            Self::Network { .. } => "DSPY_NET_001".to_string(),
            Self::Resource { .. } => "DSPY_RES_001".to_string(),
            Self::Permission { .. } => "DSPY_PER_001".to_string(),
            Self::Timeout { .. } => "DSPY_TIM_001".to_string(),
            Self::UserCancelled { .. } => "DSPY_CAN_001".to_string(),
            Self::Dependency { .. } => "DSPY_DEP_001".to_string(),
            Self::Internal { .. } => "DSPY_INT_001".to_string(),
        }
    }

    /// Get error severity
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::Config { .. } => ErrorSeverity::Error,
            Self::Validation { .. } => ErrorSeverity::Error,
            Self::Execution { .. } => ErrorSeverity::Error,
            Self::Network { .. } => ErrorSeverity::Warning,
            Self::Resource { .. } => ErrorSeverity::Error,
            Self::Permission { .. } => ErrorSeverity::Error,
            Self::Timeout { .. } => ErrorSeverity::Warning,
            Self::UserCancelled { .. } => ErrorSeverity::Info,
            Self::Dependency { .. } => ErrorSeverity::Error,
            Self::Internal { .. } => ErrorSeverity::Critical,
        }
    }

    /// Get error category
    pub fn category(&self) -> &'static str {
        match self {
            Self::Config { .. } => "configuration",
            Self::Validation { .. } => "validation",
            Self::Execution { .. } => "execution",
            Self::Network { .. } => "network",
            Self::Resource { .. } => "resource",
            Self::Permission { .. } => "permission",
            Self::Timeout { .. } => "timeout",
            Self::UserCancelled { .. } => "user_action",
            Self::Dependency { .. } => "dependency",
            Self::Internal { .. } => "internal",
        }
    }

    /// Check if operation can be retried
    pub fn is_retry_possible(&self) -> bool {
        match self {
            Self::Execution { retry_possible, .. } => *retry_possible,
            Self::Network { .. } => true,
            Self::Timeout { .. } => true,
            Self::Resource { .. } => false,
            _ => false,
        }
    }

    /// Get help URL for error type
    pub fn help_url(&self) -> Option<String> {
        let base_url = "https://docs.memvid-agent.com/dspy/errors";
        match self {
            Self::Config { .. } => Some(format!("{}/configuration", base_url)),
            Self::Validation { .. } => Some(format!("{}/validation", base_url)),
            Self::Execution { .. } => Some(format!("{}/execution", base_url)),
            Self::Network { .. } => Some(format!("{}/network", base_url)),
            Self::Resource { .. } => Some(format!("{}/resources", base_url)),
            Self::Permission { .. } => Some(format!("{}/permissions", base_url)),
            _ => None,
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Machine-readable error information
#[derive(Debug, Clone)]
pub struct ErrorInfo {
    pub error_type: &'static str,
    pub error_code: String,
    pub message: String,
    pub severity: ErrorSeverity,
    pub category: &'static str,
    pub retry_possible: bool,
    pub help_url: Option<String>,
}

/// Convert from common error types
impl From<std::io::Error> for DspyCliError {
    fn from(error: std::io::Error) -> Self {
        Self::Resource {
            resource: "filesystem".to_string(),
            message: error.to_string(),
            current_usage: None,
            limit: None,
            suggestion: Some("Check file permissions and disk space".to_string()),
        }
    }
}

impl From<toml::de::Error> for DspyCliError {
    fn from(error: toml::de::Error) -> Self {
        Self::Config {
            message: format!("TOML parsing error: {}", error),
            suggestion: Some("Check configuration file syntax".to_string()),
            config_path: None,
            line_number: None, // TOML error doesn't provide line info in this version
        }
    }
}

impl From<serde_json::Error> for DspyCliError {
    fn from(error: serde_json::Error) -> Self {
        Self::Validation {
            field: "json".to_string(),
            message: format!("JSON parsing error: {}", error),
            expected: None,
            actual: None,
            suggestions: vec!["Check JSON syntax and structure".to_string()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = DspyCliError::config_error("Test message", "Test suggestion");
        assert!(matches!(error, DspyCliError::Config { .. }));
        assert_eq!(error.error_type(), "ConfigurationError");
        assert_eq!(error.error_code(), "DSPY_CFG_001");
    }

    #[test]
    fn test_exit_code_conversion() {
        let error = DspyCliError::config_error("Test", "Test");
        let exit_code: ExitCode = error.into();
        assert_eq!(exit_code, ExitCode::ConfigurationError);
    }

    #[test]
    fn test_error_context() {
        let error = DspyCliError::execution_error("test_op", "test message", true)
            .with_context("key", "value");

        if let DspyCliError::Execution { context, .. } = error {
            assert_eq!(context.get("key"), Some(&"value".to_string()));
        } else {
            panic!("Expected execution error");
        }
    }

    #[test]
    fn test_user_message() {
        let error = DspyCliError::validation_error(
            "test_field",
            "test message",
            vec!["suggestion 1".to_string(), "suggestion 2".to_string()],
        );

        let message = error.user_message();
        assert!(message.contains("Validation Error"));
        assert!(message.contains("test_field"));
        assert!(message.contains("suggestion 1"));
    }
}
