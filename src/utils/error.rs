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
}
