//! # Rust MemVid Agent
//!
//! A comprehensive AI agent system in Rust that integrates with Anthropic's Claude API
//! and uses MP4-based memory storage for persistent conversations and context.
//!
//! ## Features
//!
//! - **Full Anthropic API Integration**: Support for all latest tools including code execution,
//!   web search, and text editor tools
//! - **MP4 Memory Storage**: Persistent memory using rust-mp4-memory for conversation history
//!   and semantic search
//! - **Extensible Tool System**: Built-in tools and framework for custom tool development
//! - **Async/Await Architecture**: High-performance async operations throughout
//! - **Type Safety**: Comprehensive type system for API interactions and tool definitions
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use rust_memvid_agent::{Agent, AgentConfig};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Initialize the agent with memory
//!     let config = AgentConfig::default()
//!         .with_anthropic_key("your-api-key")
//!         .with_memory_path("agent_memory.mp4");
//!     
//!     let mut agent = Agent::new(config).await?;
//!     
//!     // Start a conversation
//!     let response = agent.chat("Hello! Can you help me write some Rust code?").await?;
//!     println!("{}", response);
//!     
//!     Ok(())
//! }
//! ```

pub mod agent;
pub mod anthropic;
pub mod config;
pub mod memory;
pub mod tools;
pub mod utils;

// Re-export main types for convenience
pub use agent::{Agent, AgentBuilder};
pub use config::{AgentConfig, AnthropicConfig, MemoryConfig};
pub use utils::error::{AgentError, Result};

// Re-export tool types
pub use tools::{Tool, ToolRegistry, ToolResult};

// Re-export memory types
pub use memory::{MemoryManager, SearchResult, MemoryEntry, MemoryStats};

/// Initialize the agent system with default logging
pub async fn init() -> Result<()> {
    utils::logging::init_logging()?;
    Ok(())
}

/// Initialize the agent system with custom logging configuration
pub async fn init_with_logging(level: tracing::Level) -> Result<()> {
    utils::logging::init_logging_with_level(level)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_init() {
        init().await.unwrap();
    }
}
