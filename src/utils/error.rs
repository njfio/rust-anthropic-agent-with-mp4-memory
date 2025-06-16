use thiserror::Error;

/// Result type alias for the agent system
pub type Result<T> = std::result::Result<T, AgentError>;

/// Comprehensive error types for the agent system
#[derive(Error, Debug)]
pub enum AgentError {
    /// Anthropic API related errors
    #[error("Anthropic API error: {message}")]
    AnthropicApi { message: String },

    /// HTTP request errors
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// JSON serialization/deserialization errors
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Configuration errors
    #[error("Configuration error: {message}")]
    Config { message: String },

    /// Memory system errors
    #[error("Memory error: {message}")]
    Memory { message: String },

    /// Tool execution errors
    #[error("Tool error: {tool_name}: {message}")]
    Tool { tool_name: String, message: String },

    /// File system errors
    #[error("File system error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid input errors
    #[error("Invalid input: {message}")]
    InvalidInput { message: String },

    /// Authentication errors
    #[error("Authentication error: {message}")]
    Authentication { message: String },

    /// Rate limiting errors
    #[error("Rate limit exceeded: {message}")]
    RateLimit { message: String },

    /// Plugin system errors
    #[error("Plugin error: {message}")]
    Plugin { message: String },

    /// Validation errors
    #[error("Validation error: {message}")]
    Validation { message: String },

    /// Security errors
    #[error("Security error: {message}")]
    Security { message: String },

    /// DSPy module errors
    #[error("DSPy module error: {module_name}: {message}")]
    DspyModule {
        module_name: String,
        message: String,
    },

    /// DSPy signature errors
    #[error("DSPy signature error: {message}")]
    DspySignature { message: String },

    /// DSPy optimization errors
    #[error("DSPy optimization error: {strategy}: {message}")]
    DspyOptimization { strategy: String, message: String },

    /// DSPy evaluation errors
    #[error("DSPy evaluation error: {metric}: {message}")]
    DspyEvaluation { metric: String, message: String },

    /// DSPy compilation errors
    #[error("DSPy compilation error: {phase}: {message}")]
    DspyCompilation { phase: String, message: String },

    /// Generic errors
    #[error("Agent error: {0}")]
    Generic(#[from] anyhow::Error),
}

impl AgentError {
    /// Create a new Anthropic API error
    pub fn anthropic_api<S: Into<String>>(message: S) -> Self {
        Self::AnthropicApi {
            message: message.into(),
        }
    }

    /// Create a new configuration error
    pub fn config<S: Into<String>>(message: S) -> Self {
        Self::Config {
            message: message.into(),
        }
    }

    /// Create a new memory error
    pub fn memory<S: Into<String>>(message: S) -> Self {
        Self::Memory {
            message: message.into(),
        }
    }

    /// Create a new tool error
    pub fn tool<S: Into<String>>(tool_name: S, message: S) -> Self {
        Self::Tool {
            tool_name: tool_name.into(),
            message: message.into(),
        }
    }

    /// Create a new invalid input error
    pub fn invalid_input<S: Into<String>>(message: S) -> Self {
        Self::InvalidInput {
            message: message.into(),
        }
    }

    /// Create a new authentication error
    pub fn authentication<S: Into<String>>(message: S) -> Self {
        Self::Authentication {
            message: message.into(),
        }
    }

    /// Create a new rate limit error
    pub fn rate_limit<S: Into<String>>(message: S) -> Self {
        Self::RateLimit {
            message: message.into(),
        }
    }

    /// Create a new plugin error
    pub fn plugin<S: Into<String>>(message: S) -> Self {
        Self::Plugin {
            message: message.into(),
        }
    }

    /// Create a new validation error
    pub fn validation<S: Into<String>>(message: S) -> Self {
        Self::Validation {
            message: message.into(),
        }
    }

    /// Create a new security error
    pub fn security<S: Into<String>>(message: S) -> Self {
        Self::Security {
            message: message.into(),
        }
    }

    /// Create a new DSPy module error
    pub fn dspy_module<S: Into<String>>(module_name: S, message: S) -> Self {
        Self::DspyModule {
            module_name: module_name.into(),
            message: message.into(),
        }
    }

    /// Create a new DSPy signature error
    pub fn dspy_signature<S: Into<String>>(message: S) -> Self {
        Self::DspySignature {
            message: message.into(),
        }
    }

    /// Create a new DSPy optimization error
    pub fn dspy_optimization<S: Into<String>>(strategy: S, message: S) -> Self {
        Self::DspyOptimization {
            strategy: strategy.into(),
            message: message.into(),
        }
    }

    /// Create a new DSPy evaluation error
    pub fn dspy_evaluation<S: Into<String>>(metric: S, message: S) -> Self {
        Self::DspyEvaluation {
            metric: metric.into(),
            message: message.into(),
        }
    }

    /// Create a new DSPy compilation error
    pub fn dspy_compilation<S: Into<String>>(phase: S, message: S) -> Self {
        Self::DspyCompilation {
            phase: phase.into(),
            message: message.into(),
        }
    }

    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            AgentError::Http(_) | AgentError::RateLimit { .. } | AgentError::AnthropicApi { .. }
        )
    }

    /// Check if this error is due to authentication
    pub fn is_auth_error(&self) -> bool {
        matches!(self, AgentError::Authentication { .. })
    }

    /// Check if this error is due to rate limiting
    pub fn is_rate_limit(&self) -> bool {
        matches!(self, AgentError::RateLimit { .. })
    }

    /// Check if this error is due to security
    pub fn is_security_error(&self) -> bool {
        matches!(self, AgentError::Security { .. })
    }
}
