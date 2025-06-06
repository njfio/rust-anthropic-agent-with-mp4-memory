use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use crate::utils::error::{AgentError, Result};

/// Initialize logging with default configuration
pub fn init_logging() -> Result<()> {
    init_logging_with_level(tracing::Level::INFO)
}

/// Initialize logging with specified level
pub fn init_logging_with_level(level: tracing::Level) -> Result<()> {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(level.to_string()));

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(false)
                .with_thread_ids(true)
                .with_file(true)
                .with_line_number(true),
        )
        .with(env_filter)
        .try_init()
        .map_err(|e| AgentError::config(format!("Failed to initialize logging: {}", e)))?;

    Ok(())
}

/// Initialize JSON logging for structured output
pub fn init_json_logging() -> Result<()> {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .json()
                .with_target(false)
                .with_thread_ids(true)
                .with_file(true)
                .with_line_number(true),
        )
        .with(env_filter)
        .try_init()
        .map_err(|e| AgentError::config(format!("Failed to initialize JSON logging: {}", e)))?;

    Ok(())
}
